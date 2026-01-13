"""
Beam search with structural checkpoint sharing.

Checkpoints DO NOT store tokens (structural sharing via position).
Beams own full token lists. Nearest-checkpoint lookup for efficient branching.

CRITICAL INVARIANT: Before stepping ANY beam, prepare_beam_step() MUST be called.
This restores the model to the correct state for that beam, ensuring isolation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, TYPE_CHECKING
from enum import Enum, auto
from uuid import uuid4
import numpy as np
import logging

if TYPE_CHECKING:
    from engine.model_adapter import ModelAdapter
    from engine.constraints import ConstraintTracker

logger = logging.getLogger(__name__)


class BeamStatus(Enum):
    ACTIVE = auto()
    COMPLETED = auto()
    PRUNED = auto()
    FAILED = auto()


@dataclass
class Checkpoint:
    """
    Lightweight checkpoint - NO token storage (structural sharing).

    Tokens are owned by the Beam; checkpoint just references position.
    The state_ref_id points to KV cache bytes in CheckpointStore.
    """
    checkpoint_id: str
    position: int                        # Absolute token index
    state_ref_id: Optional[str] = None   # Key into CheckpointStore (None = replay)
    entropy: float = 0.0
    varentropy: float = 0.0


@dataclass
class Beam:
    """
    Search beam with full token ownership.

    Beams own their complete token sequence. Checkpoints reference
    positions within this sequence but do not duplicate the tokens.

    SABS fields (Security-Aware Beam Search):
      - sec_penalty: Accumulated security penalty (>= 0)
      - constraint_tracker: Per-beam syntactic state tracker
      - tokens_since_security_check: Counter for verification scheduling
    """
    beam_id: str = field(default_factory=lambda: str(uuid4())[:8])
    tokens: List[int] = field(default_factory=list)
    prompt_length: int = 0
    log_prob: float = 0.0
    verification_score: float = 0.0  # Legacy, kept for compatibility
    checkpoints: List[Checkpoint] = field(default_factory=list)
    parent_beam_id: Optional[str] = None
    branch_checkpoint_id: Optional[str] = None
    depth: int = 0
    status: BeamStatus = BeamStatus.ACTIVE
    completion_reason: Optional[str] = None
    # SABS fields
    sec_penalty: float = 0.0  # Raw accumulated security penalty (>= 0)
    constraint_tracker: Optional["ConstraintTracker"] = None  # Per-beam syntax state
    tokens_since_security_check: int = 0  # Counter for verification scheduling

    @property
    def generated_tokens(self) -> List[int]:
        return self.tokens[self.prompt_length:]

    @property
    def num_generated(self) -> int:
        return len(self.tokens) - self.prompt_length

    @property
    def current_position(self) -> int:
        return len(self.tokens)

    def normalized_score(self, config: "SearchConfig") -> float:
        """
        Compute normalized beam score with length penalty and security penalty.

        Score = (log_prob / length_factor) - λ * sec_penalty

        Args:
            config: SearchConfig containing length_penalty and security_lambda

        Returns:
            Normalized score (higher is better)
        """
        gen_len = max(self.num_generated, 1)
        length_factor = ((5 + gen_len) / 6) ** config.length_penalty
        base = self.log_prob / length_factor
        # SABS: SUBTRACT security penalty (higher penalty = lower score)
        return base - config.security_lambda * self.sec_penalty

    def add_checkpoint(
        self,
        checkpoint_id: str,
        state_ref_id: Optional[str],
        entropy: float = 0.0,
        varentropy: float = 0.0
    ) -> Checkpoint:
        """Add checkpoint at current position."""
        cp = Checkpoint(
            checkpoint_id=checkpoint_id,
            position=self.current_position,
            state_ref_id=state_ref_id,
            entropy=entropy,
            varentropy=varentropy
        )
        self.checkpoints.append(cp)
        return cp

    def find_nearest_checkpoint(self, position: int) -> Optional[Checkpoint]:
        """Find latest checkpoint with position <= given position."""
        best = None
        for cp in self.checkpoints:
            if cp.position <= position:
                if best is None or cp.position > best.position:
                    best = cp
        return best

    def get_tokens_at(self, position: int) -> List[int]:
        """Get tokens up to position (exclusive)."""
        return self.tokens[:position]

    def clone(self, new_id: Optional[str] = None) -> "Beam":
        """
        Clone beam for branching.

        Preserves:
          - Token list (copied)
          - Checkpoint refs (copied)
          - sec_penalty (preserved for SABS)
          - constraint_tracker (cloned if present)
          - tokens_since_security_check (preserved)
        """
        return Beam(
            beam_id=new_id or str(uuid4())[:8],
            tokens=list(self.tokens),
            prompt_length=self.prompt_length,
            log_prob=self.log_prob,
            verification_score=self.verification_score,  # Preserve
            checkpoints=list(self.checkpoints),
            parent_beam_id=self.beam_id,
            depth=self.depth + 1,
            status=BeamStatus.ACTIVE,
            # SABS: Preserve security state
            sec_penalty=self.sec_penalty,
            constraint_tracker=self.constraint_tracker.clone() if self.constraint_tracker else None,
            tokens_since_security_check=self.tokens_since_security_check,
        )


@dataclass
class SearchConfig:
    """Beam search configuration."""
    beam_width: int = 4
    max_branches: int = 8
    max_tokens: int = 512
    checkpoint_interval: int = 16
    length_penalty: float = 1.0
    max_depth: int = 3
    security_lambda: float = 0.0  # SABS: 0.0 means disabled, higher = more security-focused


class BeamManager:
    """
    Manages beam search with nearest-checkpoint branching.

    CRITICAL INVARIANT: Before stepping ANY beam, call prepare_beam_step().
    This restores the model state to the beam's nearest checkpoint and
    replays any delta tokens, ensuring complete beam isolation.

    Failure to call prepare_beam_step() before generation will cause
    state contamination between beams.
    """

    def __init__(self, config: SearchConfig, model: "ModelAdapter"):
        self.config = config
        self.model = model
        self.active_beams: Dict[str, Beam] = {}
        self.completed_beams: List[Beam] = []
        self.pruned_beams: List[Beam] = []
        self._branches_used: int = 0
        self._rng = np.random.default_rng()
        self._next_checkpoint_id: int = 0

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset for new generation."""
        self.active_beams.clear()
        self.completed_beams.clear()
        self.pruned_beams.clear()
        self._branches_used = 0
        self._next_checkpoint_id = 0
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def _new_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        cid = f"cp_{self._next_checkpoint_id:04d}"
        self._next_checkpoint_id += 1
        return cid

    def create_initial_beam(self, prompt_tokens: List[int]) -> Beam:
        """Create initial beam from prompt."""
        beam = Beam(
            tokens=list(prompt_tokens),
            prompt_length=len(prompt_tokens)
        )
        self.active_beams[beam.beam_id] = beam

        # Create initial checkpoint (model already has prompt evaluated)
        cid = self._new_checkpoint_id()
        self.model.create_checkpoint(cid)
        beam.add_checkpoint(cid, cid, 0.0, 0.0)

        logger.debug(f"Created beam {beam.beam_id} with {len(prompt_tokens)} prompt tokens")
        return beam

    def prepare_beam_step(self, beam: Beam) -> None:
        """
        Restore model state for this beam before generation.

        CRITICAL: MUST be called before generating ANY tokens/logits for a beam.
        Ensures complete isolation between beams by:
        1. Finding the nearest checkpoint to current position
        2. Restoring KV cache state (or replaying from checkpoint tokens)
        3. Replaying any delta tokens between checkpoint and current position

        Args:
            beam: The beam to prepare for stepping

        Raises:
            AssertionError: (in debug mode) if called with invalid beam state
        """
        # Find nearest checkpoint to restore from
        nearest = beam.find_nearest_checkpoint(beam.current_position)

        if nearest is None:
            # No checkpoint - replay all tokens from beam
            logger.debug(f"No checkpoint for beam {beam.beam_id}, replaying all {len(beam.tokens)} tokens")
            self.model.clear_runtime_state()
            if beam.tokens:
                self.model.eval(beam.tokens)
            return

        # Get tokens up to checkpoint position from BEAM (authoritative)
        tokens_at_checkpoint = beam.get_tokens_at(nearest.position)
        checkpoint_id = nearest.state_ref_id or nearest.checkpoint_id

        # Restore from checkpoint with beam's tokens
        self.model.restore_checkpoint(checkpoint_id, tokens_at_checkpoint)

        # Replay delta tokens from checkpoint to current position
        if nearest.position < beam.current_position:
            delta = beam.tokens[nearest.position:beam.current_position]
            if delta:
                logger.debug(f"Replaying {len(delta)} delta tokens for beam {beam.beam_id}")
                self.model.eval(delta)

    def add_checkpoint_to_beam(
        self,
        beam: Beam,
        entropy: float = 0.0,
        varentropy: float = 0.0,
        save_state: bool = True
    ) -> Checkpoint:
        """
        Add checkpoint at beam's current position.

        Args:
            beam: Beam to checkpoint
            entropy: Entropy at this position
            varentropy: Varentropy at this position
            save_state: Whether to save KV cache state
        """
        cid = self._new_checkpoint_id()
        state_ref_id = None

        if save_state:
            try:
                self.model.create_checkpoint(cid)
                state_ref_id = cid
            except Exception as e:
                logger.warning(f"Checkpoint state save failed: {e}")

        cp = beam.add_checkpoint(cid, state_ref_id, entropy, varentropy)
        logger.debug(f"Checkpoint {cid} at position {cp.position}")
        return cp

    def should_checkpoint(self, beam: Beam, at_boundary: bool) -> bool:
        """Determine if checkpoint should be created."""
        if at_boundary:
            return True
        if not beam.checkpoints:
            return beam.num_generated >= self.config.checkpoint_interval
        last = beam.checkpoints[-1]
        return (beam.current_position - last.position) >= self.config.checkpoint_interval

    def branch_beam(
        self,
        beam: Beam,
        logits: np.ndarray,
        num_branches: int,
        temperature: float = 0.7
    ) -> List[Beam]:
        """
        Branch beam into multiple alternatives from nearest checkpoint.

        Performance: Uses top-k selection BEFORE softmax to avoid computing
        exp() over entire vocab. Softmax is computed only over the k candidates.

        Log probabilities are computed under the filtered (top-k) distribution.
        This is intentional for beam scoring comparability.

        When branching, cloned beams are truncated to checkpoint position
        before appending their branch token. This ensures clean divergence.
        """
        # Edge case: no branches requested
        if num_branches <= 0:
            return [beam]

        if self._branches_used >= self.config.max_branches:
            return [beam]

        if beam.depth >= self.config.max_depth:
            return [beam]

        nearest = beam.find_nearest_checkpoint(beam.current_position)
        if nearest is None:
            logger.warning(f"No checkpoint for branching beam {beam.beam_id}")
            return [beam]

        # Edge case: temperature <= 0 means greedy (no branching, just pick argmax)
        if temperature <= 0:
            token_id = int(np.argmax(logits))
            # Compute logprob under full softmax for greedy
            shifted = logits - np.max(logits)
            log_sum_exp = np.log(np.sum(np.exp(shifted)))
            logprob = float(shifted[token_id] - log_sum_exp)
            beam.tokens.append(token_id)
            beam.log_prob += logprob
            return [beam]

        # Determine how many branches we can create
        num_branches = min(num_branches, self.config.max_branches - self._branches_used)

        # TOP-K FIRST: Select top candidates from raw logits (O(n) via argpartition)
        scaled_logits = logits / temperature
        vocab_size = len(scaled_logits)
        k = min(num_branches, vocab_size)

        if k < vocab_size:
            top_k_indices = np.argpartition(scaled_logits, -k)[-k:]
            top_k_logits = scaled_logits[top_k_indices]
        else:
            top_k_indices = np.arange(vocab_size)
            top_k_logits = scaled_logits

        # SOFTMAX only over top-k subset
        shifted = top_k_logits - np.max(top_k_logits)
        exp_vals = np.exp(shifted)
        top_k_probs = exp_vals / np.sum(exp_vals)

        # Sort by probability descending
        order = np.argsort(top_k_probs)[::-1]
        top_k_indices = top_k_indices[order]
        top_k_probs = top_k_probs[order]

        new_beams = []

        for i in range(num_branches):
            token_id = int(top_k_indices[i])
            prob = top_k_probs[i]
            # Log probability under filtered distribution
            logprob = float(np.log(prob + 1e-10))

            if i == 0:
                # Continue original beam from current position
                new_beam = beam
            else:
                # Clone and branch from checkpoint position
                new_beam = beam.clone()
                new_beam.branch_checkpoint_id = nearest.checkpoint_id

                # CRITICAL: Truncate to checkpoint position for clean branch
                new_beam.tokens = beam.get_tokens_at(nearest.position)

                self.active_beams[new_beam.beam_id] = new_beam
                self._branches_used += 1

            new_beam.tokens.append(token_id)
            new_beam.log_prob += logprob
            new_beams.append(new_beam)

            logger.debug(f"Branch {i}: beam={new_beam.beam_id}, token={token_id}, pos={len(new_beam.tokens)}")

        return new_beams

    def step_beam(
        self,
        beam: Beam,
        token_id: int,
        logprob: float
    ) -> None:
        """Add token to beam."""
        beam.tokens.append(token_id)
        beam.log_prob += logprob

    def complete_beam(self, beam: Beam, reason: str = "eos") -> None:
        """Mark beam completed."""
        beam.status = BeamStatus.COMPLETED
        beam.completion_reason = reason
        self.active_beams.pop(beam.beam_id, None)
        self.completed_beams.append(beam)

    def fail_beam(self, beam: Beam, reason: str) -> None:
        """Mark beam failed."""
        beam.status = BeamStatus.FAILED
        beam.completion_reason = reason
        self.active_beams.pop(beam.beam_id, None)
        self.pruned_beams.append(beam)

    def prune_beams(self) -> List[Beam]:
        """Keep top beam_width beams (uses security_lambda from config)."""
        if len(self.active_beams) <= self.config.beam_width:
            return []

        sorted_beams = sorted(
            self.active_beams.values(),
            key=lambda b: b.normalized_score(self.config),  # Pass full config
            reverse=True
        )

        keep = {b.beam_id for b in sorted_beams[:self.config.beam_width]}
        pruned = []

        for bid in list(self.active_beams.keys()):
            if bid not in keep:
                beam = self.active_beams.pop(bid)
                beam.status = BeamStatus.PRUNED
                beam.completion_reason = "pruned"
                self.pruned_beams.append(beam)
                pruned.append(beam)

        return pruned

    def deduplicate(self) -> List[Beam]:
        """Remove duplicate token sequences (uses security_lambda from config)."""
        seen: Dict[Tuple[int, ...], Beam] = {}
        dups = []

        for beam in list(self.active_beams.values()):
            key = tuple(beam.tokens)
            if key in seen:
                existing = seen[key]
                if beam.normalized_score(self.config) > existing.normalized_score(self.config):
                    self.active_beams.pop(existing.beam_id, None)
                    existing.status = BeamStatus.PRUNED
                    dups.append(existing)
                    seen[key] = beam
                else:
                    self.active_beams.pop(beam.beam_id, None)
                    beam.status = BeamStatus.PRUNED
                    dups.append(beam)
            else:
                seen[key] = beam

        self.pruned_beams.extend(dups)
        return dups

    def get_best_beam(self) -> Optional[Beam]:
        """Get highest-scoring beam (uses security_lambda from config)."""
        all_beams = self.completed_beams + list(self.active_beams.values())
        if not all_beams:
            return None
        return max(all_beams, key=lambda b: b.normalized_score(self.config))

    @property
    def has_active_beams(self) -> bool:
        return len(self.active_beams) > 0

    @property
    def branches_remaining(self) -> int:
        return self.config.max_branches - self._branches_used
