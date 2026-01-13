"""
Orchestrator for SuperCoder engine.

Ties together model adapter, beam search, and entropy-gated branching.

Supports THREE modes:
1. BASELINE: Simple entropy threshold (entropy >= threshold → branch)
2. SAEG: Semantic-Adaptive Entropy Gating (novel algorithm)
3. SABS: Security-Aware Beam Search (use_sabs=True)

Use --use-saeg flag to enable SAEG for experiments.
Use --use-sabs flag to enable SABS security verification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import time
import numpy as np
import json

from uuid import uuid4
from engine.model_adapter import ModelAdapter
from engine.search import BeamManager, SearchConfig, BeamStatus
from engine.constraints import ConstraintTracker
from engine.saeg import SAEGConfig, SAEGTracker, saeg_branch_decision
from engine.security_controller import (
    SABSConfig, SecurityController, SABSAction, extract_code_only
)


@dataclass
class OrchestratorConfig:
    model_path: str
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    n_batch: int = 512
    beam_width: int = 4
    max_branches: int = 8
    max_tokens: int = 512
    checkpoint_interval: int = 16
    length_penalty: float = 1.0
    max_depth: int = 3
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.95
    entropy_threshold: float = 6.0
    branch_count: int = 2
    log_every: int = 1

    # SAEG configuration (novel algorithm)
    use_saeg: bool = False
    saeg_alpha: float = 1.0      # Entropy weight
    saeg_beta: float = 0.5       # Semantic uncertainty weight
    saeg_gamma: float = 0.3      # Position weight
    saeg_mu: float = 0.2         # Syntactic complexity modifier
    saeg_lambda: float = 0.1     # Position decay rate
    saeg_k: int = 5              # Top-k for semantic uncertainty

    # Experiment tracking
    export_decisions: bool = False
    export_path: Optional[str] = None

    # SABS configuration (Security-Aware Beam Search)
    use_sabs: bool = False
    sabs_lambda: float = 0.5       # Security penalty weight
    sabs_hard_fail: bool = True    # Hard-fail beams with CRITICAL issues
    sabs_max_calls: int = 50       # Max verifier calls per generation
    sabs_min_tokens: int = 8       # Min tokens between security checks
    sabs_log_path: Optional[str] = None  # Path for SABS decision log (JSONL)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def entropy_and_varentropy_from_logits(logits: np.ndarray) -> Tuple[float, float]:
    """
    Compute entropy and varentropy (variance of surprisal) from logits.

    High entropy = model is uncertain -> good place to branch
    """
    p = _softmax(logits)
    p = np.clip(p, 1e-12, 1.0)
    surprisal = -np.log(p)
    ent = float(np.sum(p * surprisal))
    varent = float(np.sum(p * (surprisal - ent) ** 2))
    return ent, varent


class Orchestrator:
    """
    Main orchestrator for code generation.

    Supports THREE modes:
    1. BASELINE: entropy >= threshold → branch
    2. SAEG: Semantic-Adaptive Entropy Gating (use_saeg=True)
    3. SABS: Security-Aware Beam Search (use_sabs=True)

    SAEG considers:
    - Entropy (distributional uncertainty)
    - Semantic uncertainty (embedding distance between top-k)
    - Position weight (early tokens matter more)
    - Adaptive threshold (respects syntactic context)

    SABS adds:
    - Per-beam constraint tracking for syntactic boundaries
    - Incremental security verification during generation
    - Graduated penalties (soft) or termination (hard) for security issues
    """

    def __init__(self, cfg: OrchestratorConfig):
        self.cfg = cfg
        self.model = ModelAdapter(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_gpu_layers=cfg.n_gpu_layers,
            n_batch=cfg.n_batch,
            verbose=False,
        )
        caps = self.model.load()
        if not caps.all_critical_ok:
            raise RuntimeError(f"Model capabilities insufficient:\n{caps.summary()}")

        self.search = BeamManager(
            config=SearchConfig(
                beam_width=cfg.beam_width,
                max_branches=cfg.max_branches,
                max_tokens=cfg.max_tokens,
                checkpoint_interval=cfg.checkpoint_interval,
                length_penalty=cfg.length_penalty,
                max_depth=cfg.max_depth,
                security_lambda=cfg.sabs_lambda if cfg.use_sabs else 0.0,  # SABS
            ),
            model=self.model,
        )

        # Global constraint tracker for SAEG (per-beam trackers used for SABS)
        self.constraint_tracker = ConstraintTracker()

        # SAEG setup
        self.saeg_config: Optional[SAEGConfig] = None
        self.saeg_tracker: Optional[SAEGTracker] = None

        if cfg.use_saeg:
            self.saeg_config = SAEGConfig(
                alpha=cfg.saeg_alpha,
                beta=cfg.saeg_beta,
                gamma=cfg.saeg_gamma,
                base_threshold=cfg.entropy_threshold,
                mu=cfg.saeg_mu,
                lambda_=cfg.saeg_lambda,
                k=cfg.saeg_k,
                log_decisions=cfg.log_every > 0,
            )
            self.saeg_tracker = SAEGTracker(self.saeg_config)
            print(f"SAEG enabled: alpha={cfg.saeg_alpha}, beta={cfg.saeg_beta}, gamma={cfg.saeg_gamma}")

        # SABS setup
        self.security_controller: Optional[SecurityController] = None

        if cfg.use_sabs:
            sabs_cfg = SABSConfig(
                lambda_weight=cfg.sabs_lambda,
                hard_fail_on_critical=cfg.sabs_hard_fail,
                max_verifier_calls=cfg.sabs_max_calls,
                min_tokens_between_checks=cfg.sabs_min_tokens,
                log_path=cfg.sabs_log_path,
            )
            self.security_controller = SecurityController(sabs_cfg)
            print(f"SABS enabled: λ={cfg.sabs_lambda}, hard_fail={cfg.sabs_hard_fail}")

    def generate(self, prompt: str) -> str:
        """
        Generate code from prompt using entropy-gated beam search.

        If use_saeg=True, uses Semantic-Adaptive Entropy Gating.
        If use_sabs=True, uses Security-Aware Beam Search.
        Otherwise, uses baseline entropy threshold.

        Returns the best beam's output.
        """
        self.search.reset()
        self.constraint_tracker.reset()

        if self.saeg_tracker:
            self.saeg_tracker.reset()

        # SABS: Reset security controller
        gen_id = str(uuid4())[:8]
        if self.security_controller:
            self.security_controller.reset(gen_id)

        prompt_tokens = self.model.tokenize(prompt)
        self.model.clear_runtime_state()
        if prompt_tokens:
            self.model.eval(prompt_tokens)

        self.search.create_initial_beam(prompt_tokens)

        # SABS: Initialize per-beam constraint trackers
        if self.security_controller:
            for beam in self.search.active_beams.values():
                beam.constraint_tracker = ConstraintTracker()

        step_idx = 0

        strategy = "SABS" if self.cfg.use_sabs else ("SAEG" if self.cfg.use_saeg else "BASELINE")
        print(f"Thinking... (engine loop started, strategy={strategy})")
        start_t = time.time()

        while self.search.has_active_beams:
            step_idx += 1
            active = list(self.search.active_beams.values())
            if not active:
                break

            for beam in active:
                if beam.status != BeamStatus.ACTIVE:
                    continue
                if beam.num_generated >= self.cfg.max_tokens:
                    self.search.complete_beam(beam, reason="max_tokens")
                    continue

                # CRITICAL: Restore model state for this beam
                self.search.prepare_beam_step(beam)
                logits = self.model.get_logits()

                ent, varent = entropy_and_varentropy_from_logits(logits)

                # Checkpoint if needed
                if self.search.should_checkpoint(beam, at_boundary=False):
                    self.search.add_checkpoint_to_beam(
                        beam, entropy=ent, varentropy=varent, save_state=True
                    )

                # BRANCHING DECISION
                if self.cfg.use_saeg and self.saeg_config:
                    # SAEG: Semantic-Adaptive Entropy Gating
                    decision = saeg_branch_decision(
                        logits=logits,
                        position=beam.num_generated,
                        constraint_state=self.constraint_tracker.state,
                        config=self.saeg_config,
                        embeddings=None,  # TODO: expose embeddings from model
                    )

                    if self.saeg_tracker:
                        self.saeg_tracker.record(decision)

                    do_branch = (
                        decision.should_branch
                        and self.search.branches_remaining > 0
                        and beam.depth < self.search.config.max_depth
                    )

                    if step_idx % self.cfg.log_every == 0:
                        action = "BRANCH" if do_branch else "FLOW"
                        print(
                            f"[{step_idx}] beam={beam.beam_id} "
                            f"score={decision.branch_score:.3f} "
                            f"thresh={decision.adaptive_threshold:.3f} "
                            f"ent={ent:.3f} sem={decision.semantic_uncertainty:.3f} "
                            f"action={action}"
                        )
                else:
                    # BASELINE: Simple entropy threshold
                    do_branch = (
                        ent >= self.cfg.entropy_threshold
                        and self.search.branches_remaining > 0
                        and beam.depth < self.search.config.max_depth
                    )

                    if step_idx % self.cfg.log_every == 0:
                        action = "BRANCH" if do_branch else "FLOW"
                        print(f"[{step_idx}] beam={beam.beam_id} ent={ent:.3f} var={varent:.3f} action={action}")

                if do_branch:
                    new_beams = self.search.branch_beam(
                        beam=beam,
                        logits=logits,
                        num_branches=self.cfg.branch_count,
                        temperature=max(self.cfg.temperature, 1e-6),
                    )

                    # Check if any new beam hit EOS
                    for nb in new_beams:
                        if (self.model.eos_token is not None
                            and nb.tokens
                            and nb.tokens[-1] == self.model.eos_token):
                            self.search.complete_beam(nb, reason="eos")

                        # Update global constraint tracker (for SAEG)
                        if nb.tokens:
                            token_text = self.model.detokenize([nb.tokens[-1]])
                            self.constraint_tracker.update(token_text)

                        # SABS: Initialize per-beam constraint tracker for new beams
                        if self.security_controller and nb.constraint_tracker is None:
                            nb.constraint_tracker = ConstraintTracker()
                            # Replay generated text to sync constraint state
                            generated_tokens = nb.tokens[nb.prompt_length:]
                            if generated_tokens:
                                gen_text = self.model.detokenize(generated_tokens)
                                nb.constraint_tracker.update(gen_text)

                        # SABS: Update per-beam constraint tracker
                        if nb.constraint_tracker and nb.tokens:
                            token_text = self.model.detokenize([nb.tokens[-1]])
                            nb.constraint_tracker.update(token_text)
                            nb.tokens_since_security_check += 1
                    continue

                # Normal flow: sample and step
                token_id, logp = self.model.sample(
                    logits=logits,
                    temperature=self.cfg.temperature,
                    top_k=self.cfg.top_k,
                    top_p=self.cfg.top_p,
                    rng=None,
                )
                self.search.step_beam(beam, token_id, logp)

                # Update global constraint tracker (for SAEG)
                token_text = self.model.detokenize([token_id])
                self.constraint_tracker.update(token_text)

                # SABS: Update per-beam constraint tracker
                if beam.constraint_tracker:
                    beam.constraint_tracker.update(token_text)
                    beam.tokens_since_security_check += 1

                # SABS: Security check at boundaries
                if self.security_controller and self.security_controller.should_check(beam):
                    code = extract_code_only(self.model, beam)
                    if code:
                        decision = self.security_controller.check_beam(
                            beam, code, gen_id, step_idx
                        )
                        self.security_controller.apply_decision(beam, decision, self.search)
                        if decision.should_terminate:
                            continue  # Skip to next beam (beam was hard-failed)

                # Check EOS
                if self.model.eos_token is not None and token_id == self.model.eos_token:
                    self.search.complete_beam(beam, reason="eos")

            # Prune and deduplicate
            self.search.deduplicate()
            self.search.prune_beams()

            # Safety limit
            if step_idx >= self.cfg.max_tokens * 4:
                break

        best = self.search.get_best_beam()
        if best is None:
            raise RuntimeError("No beam produced output")

        out = self.model.detokenize(best.tokens)

        dt = time.time() - start_t

        # Print summary
        summary = f"Thinking... (done in {dt:.2f}s) best_beam={best.beam_id} reason={best.completion_reason}"
        if self.saeg_tracker:
            stats = self.saeg_tracker.get_statistics()
            summary += f" | SAEG branch_rate={stats.get('branch_rate', 0):.2%}"
        if self.security_controller:
            stats = self.security_controller.get_statistics()
            summary += f" | SABS calls={stats['verifier_calls']} hard_fails={stats['hard_fails']}"
        print(summary)

        # Export decisions if requested
        if self.cfg.export_decisions and self.cfg.export_path and self.saeg_tracker:
            self._export_decisions(self.cfg.export_path)

        return out

    def _export_decisions(self, path: str) -> None:
        """Export SAEG decisions to JSON for analysis."""
        if not self.saeg_tracker:
            return

        data = {
            'config': {
                'use_saeg': self.cfg.use_saeg,
                'alpha': self.cfg.saeg_alpha,
                'beta': self.cfg.saeg_beta,
                'gamma': self.cfg.saeg_gamma,
                'base_threshold': self.cfg.entropy_threshold,
            },
            'statistics': self.saeg_tracker.get_statistics(),
            'decisions': self.saeg_tracker.to_csv_rows(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Exported {len(data['decisions'])} decisions to {path}")

    def get_experiment_results(self) -> Dict:
        """Get experiment results for comparison."""
        strategy = 'SABS' if self.cfg.use_sabs else ('SAEG' if self.cfg.use_saeg else 'BASELINE')
        results = {
            'strategy': strategy,
            'entropy_threshold': self.cfg.entropy_threshold,
        }

        if self.saeg_tracker:
            results['saeg_statistics'] = self.saeg_tracker.get_statistics()

        if self.security_controller:
            results['sabs_statistics'] = self.security_controller.get_statistics()

        return results
