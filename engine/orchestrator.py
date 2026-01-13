"""
Orchestrator for SuperCoder engine.

Ties together model adapter, beam search, and entropy-gated branching.
Simple hardcoded logic: high entropy -> branch, low entropy -> flow.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import time
import numpy as np

from engine.model_adapter import ModelAdapter
from engine.search import BeamManager, SearchConfig, BeamStatus


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

    Uses entropy-gated branching:
    - High entropy (uncertainty) -> create branches to explore alternatives
    - Low entropy (confidence) -> flow forward with sampling
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
            ),
            model=self.model,
        )

    def generate(self, prompt: str) -> str:
        """
        Generate code from prompt using entropy-gated beam search.

        Returns the best beam's output.
        """
        self.search.reset()

        prompt_tokens = self.model.tokenize(prompt)
        self.model.clear_runtime_state()
        if prompt_tokens:
            self.model.eval(prompt_tokens)

        self.search.create_initial_beam(prompt_tokens)
        step_idx = 0

        print("Thinking... (engine loop started)")
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

                # Entropy-gated branching decision
                do_branch = (
                    ent >= self.cfg.entropy_threshold
                    and self.search.branches_remaining > 0
                    and beam.depth < self.search.config.max_depth
                )

                if do_branch:
                    if step_idx % self.cfg.log_every == 0:
                        print(f"[{step_idx}] beam={beam.beam_id} ent={ent:.3f} var={varent:.3f} action=BRANCH")

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

                if step_idx % self.cfg.log_every == 0:
                    print(f"[{step_idx}] beam={beam.beam_id} ent={ent:.3f} var={varent:.3f} action=FLOW")

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
        print(f"Thinking... (done in {dt:.2f}s) best_beam={best.beam_id} reason={best.completion_reason}")

        return out
