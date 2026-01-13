"""
Model adapter with checkpoint store for correct beam isolation.

Implements:
- CheckpointStore with in-memory/disk spill support (NO token storage in StateRef)
- State restoration before each beam step
- Fast top-k sampling with log probability tracking

CRITICAL: Checkpoints store ONLY opaque KV cache bytes.
Tokens are owned by Beams (structural sharing via position).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import tempfile
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateRef:
    """
    Reference to saved model state.

    IMPORTANT: Does NOT store tokens (structural sharing).
    Only stores opaque KV cache bytes (memory or disk).
    """
    checkpoint_id: str
    state_bytes: Optional[bytes] = None
    disk_path: Optional[Path] = None

    @property
    def has_state(self) -> bool:
        return self.state_bytes is not None or self.disk_path is not None

    def get_bytes(self) -> Optional[bytes]:
        """Retrieve state bytes (from memory or disk)."""
        if self.state_bytes is not None:
            return self.state_bytes
        if self.disk_path is not None and self.disk_path.exists():
            return self.disk_path.read_bytes()
        return None

    def clear(self) -> None:
        """Clear state data."""
        self.state_bytes = None
        if self.disk_path is not None and self.disk_path.exists():
            try:
                self.disk_path.unlink()
            except OSError:
                pass
            self.disk_path = None


class CheckpointStore:
    """
    Manages checkpoint state storage with memory/disk spillover.

    IMPORTANT: Stores ONLY opaque KV cache bytes, NOT tokens.
    Handles OOM by spilling large states to disk when memory limit exceeded.
    """

    def __init__(
        self,
        memory_limit_mb: float = 512.0,
        spill_dir: Optional[Path] = None
    ):
        self.memory_limit_bytes = int(memory_limit_mb * 1024 * 1024)
        self.spill_dir = spill_dir or Path(tempfile.gettempdir()) / "supercoder_checkpoints"
        self._store: Dict[str, StateRef] = {}
        self._memory_used: int = 0

    def _ensure_spill_dir(self) -> None:
        """Create spill directory if needed."""
        if not self.spill_dir.exists():
            self.spill_dir.mkdir(parents=True, exist_ok=True)

    def add(
        self,
        checkpoint_id: str,
        state_bytes: Optional[bytes] = None
    ) -> None:
        """
        Store checkpoint state (KV cache bytes only, NO tokens).

        Args:
            checkpoint_id: Unique identifier
            state_bytes: KV cache state (may be None for token-replay fallback)
        """
        # Remove existing if present
        if checkpoint_id in self._store:
            self.remove(checkpoint_id)

        ref = StateRef(
            checkpoint_id=checkpoint_id,
            state_bytes=None,
            disk_path=None
        )

        if state_bytes is not None:
            size = len(state_bytes)

            # Check if we need to spill to disk
            if self._memory_used + size > self.memory_limit_bytes:
                self._ensure_spill_dir()
                disk_path = self.spill_dir / f"{checkpoint_id}.state"
                disk_path.write_bytes(state_bytes)
                ref.disk_path = disk_path
                logger.debug(f"Checkpoint {checkpoint_id} spilled to disk ({size} bytes)")
            else:
                ref.state_bytes = state_bytes
                self._memory_used += size
                logger.debug(f"Checkpoint {checkpoint_id} stored in memory ({size} bytes)")

        self._store[checkpoint_id] = ref

    def get(self, checkpoint_id: str) -> Optional[StateRef]:
        """Retrieve checkpoint reference."""
        return self._store.get(checkpoint_id)

    def remove(self, checkpoint_id: str) -> None:
        """Remove checkpoint and free resources."""
        ref = self._store.pop(checkpoint_id, None)
        if ref is not None:
            if ref.state_bytes is not None:
                self._memory_used -= len(ref.state_bytes)
            ref.clear()

    def clear_all(self) -> None:
        """Remove all checkpoints."""
        for ref in self._store.values():
            ref.clear()
        self._store.clear()
        self._memory_used = 0

    def __contains__(self, checkpoint_id: str) -> bool:
        return checkpoint_id in self._store


@dataclass
class Capabilities:
    """Model capability probe results."""
    can_tokenize: bool = False
    can_detokenize: bool = False
    can_eval: bool = False
    can_get_logits: bool = False
    logits_shape: Optional[Tuple[int, ...]] = None
    vocab_size: int = 0
    can_save_state: bool = False
    can_load_state: bool = False
    errors: List[str] = field(default_factory=list)

    @property
    def supports_state(self) -> bool:
        return self.can_save_state and self.can_load_state

    @property
    def all_critical_ok(self) -> bool:
        return all([
            self.can_tokenize,
            self.can_detokenize,
            self.can_eval,
            self.can_get_logits,
            self.vocab_size > 0
        ])

    def summary(self) -> str:
        lines = [
            f"Tokenize:    {'OK' if self.can_tokenize else 'FAIL'}",
            f"Detokenize:  {'OK' if self.can_detokenize else 'FAIL'}",
            f"Eval:        {'OK' if self.can_eval else 'FAIL'}",
            f"Logits:      {'OK' if self.can_get_logits else 'FAIL'} {self.logits_shape or ''}",
            f"Vocab:       {self.vocab_size}",
            f"Save State:  {'OK' if self.can_save_state else 'NO'}",
            f"Load State:  {'OK' if self.can_load_state else 'NO'}",
            f"Mode:        {'KV Cache' if self.supports_state else 'Token Replay'}",
        ]
        if self.errors:
            lines.append(f"Errors: {self.errors}")
        return "\n".join(lines)


def sample_from_logits(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.95,
    rng: Optional[np.random.Generator] = None
) -> Tuple[int, float]:
    """
    Sample token from logits with top-k then softmax optimization.

    Args:
        logits: Raw logits, shape (vocab_size,)
        temperature: Sampling temperature (0 = greedy)
        top_k: Number of top tokens to consider
        top_p: Nucleus sampling threshold
        rng: Random generator for reproducibility

    Returns:
        (token_id, search_logprob) where search_logprob is log probability
        under the filtered distribution (for beam scoring)
    """
    if rng is None:
        rng = np.random.default_rng()

    vocab_size = len(logits)

    # Greedy sampling
    if temperature <= 0:
        token_id = int(np.argmax(logits))
        # Compute logprob under softmax
        shifted = logits - np.max(logits)
        log_sum_exp = np.log(np.sum(np.exp(shifted)))
        logprob = shifted[token_id] - log_sum_exp
        return token_id, float(logprob)

    # Apply temperature
    scaled = logits / temperature

    # Top-K selection FIRST (before softmax for efficiency)
    k = min(top_k, vocab_size)
    if k < vocab_size:
        # Use argpartition for O(n) top-k
        top_k_indices = np.argpartition(scaled, -k)[-k:]
        top_k_logits = scaled[top_k_indices]
    else:
        top_k_indices = np.arange(vocab_size)
        top_k_logits = scaled

    # Softmax ONLY over top-k
    shifted = top_k_logits - np.max(top_k_logits)
    exp_vals = np.exp(shifted)
    probs = exp_vals / np.sum(exp_vals)

    # Top-P (nucleus) filtering over the top-k subset
    if top_p < 1.0:
        sorted_order = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_order]
        cumsum = np.cumsum(sorted_probs)

        # Find cutoff
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_probs))

        # Zero out tokens below threshold
        if cutoff_idx < len(probs):
            keep_mask = np.zeros(len(probs), dtype=bool)
            keep_mask[sorted_order[:cutoff_idx]] = True
            probs = np.where(keep_mask, probs, 0.0)
            probs = probs / np.sum(probs)

    # Sample from filtered distribution
    local_idx = rng.choice(len(probs), p=probs)
    token_id = int(top_k_indices[local_idx])

    # Compute log probability under filtered distribution
    search_logprob = float(np.log(probs[local_idx] + 1e-10))

    return token_id, search_logprob


class ModelAdapter:
    """
    Wrapper for llama-cpp-python with checkpoint-based beam isolation.

    Key features:
    - Capability probing at load
    - CheckpointStore for state management (NO token storage)
    - Correct isolation: restore state BEFORE each beam step
    - Fallback to token replay if save_state unsupported

    CRITICAL INVARIANT: restore_checkpoint() uses caller-provided tokens
    as the authoritative source (from Beam), not stored tokens.
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        checkpoint_memory_mb: float = 512.0,
        verbose: bool = False
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self._verbose = verbose

        self._model = None
        self._capabilities: Optional[Capabilities] = None
        self._eos_token: Optional[int] = None

        # Checkpoint management (NO token storage)
        self._checkpoint_store = CheckpointStore(memory_limit_mb=checkpoint_memory_mb)
        self._current_tokens: List[int] = []

    def load(self) -> Capabilities:
        """Load model and probe capabilities."""
        from llama_cpp import Llama

        logger.info(f"Loading model: {self.model_path}")
        caps = Capabilities()

        try:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_batch=self.n_batch,
                logits_all=True,
                verbose=self._verbose
            )
        except Exception as e:
            caps.errors.append(f"Load failed: {e}")
            self._capabilities = caps
            return caps

        self._probe_tokenization(caps)
        self._probe_eval_logits(caps)
        self._probe_state(caps)

        try:
            self._eos_token = self._model.token_eos()
        except Exception:
            self._eos_token = None

        self._capabilities = caps
        logger.info(f"Probe complete:\n{caps.summary()}")
        return caps

    def _probe_tokenization(self, caps: Capabilities) -> None:
        """Test tokenize/detokenize."""
        test = "def foo():\n    return 42"
        try:
            tokens = self._model.tokenize(test.encode('utf-8'))
            caps.can_tokenize = len(tokens) > 0
        except Exception as e:
            caps.errors.append(f"Tokenize: {e}")
            return
        try:
            decoded = self._model.detokenize(tokens).decode('utf-8', errors='replace')
            caps.can_detokenize = len(decoded) > 0
        except Exception as e:
            caps.errors.append(f"Detokenize: {e}")

    def _probe_eval_logits(self, caps: Capabilities) -> None:
        """Test eval and logits access."""
        try:
            tokens = self._model.tokenize(b"Hello")
            self._model.eval(tokens)
            caps.can_eval = True
        except Exception as e:
            caps.errors.append(f"Eval: {e}")
            return

        logits = self._try_get_logits()
        if logits is not None:
            caps.can_get_logits = True
            caps.logits_shape = logits.shape
            caps.vocab_size = logits.shape[-1]
        else:
            caps.errors.append("Logits access failed")

        self._model.reset()

    def _probe_state(self, caps: Capabilities) -> None:
        """Test save_state/load_state."""
        try:
            if not hasattr(self._model, 'save_state'):
                return

            tokens = self._model.tokenize(b"Test state")
            self._model.eval(tokens)

            state = self._model.save_state()
            if state is not None and len(state) > 0:
                caps.can_save_state = True

                if hasattr(self._model, 'load_state'):
                    try:
                        self._model.load_state(state)
                        caps.can_load_state = True
                    except Exception as e:
                        caps.errors.append(f"Load state: {e}")
        except Exception as e:
            caps.errors.append(f"State probe: {e}")

        try:
            self._model.reset()
        except Exception:
            pass

    def _try_get_logits(self) -> Optional[np.ndarray]:
        """Try multiple methods to get logits."""
        # Method 1: _scores
        if hasattr(self._model, '_scores') and self._model._scores is not None:
            scores = self._model._scores
            if isinstance(scores, np.ndarray) and scores.size > 0:
                return scores[-1].copy() if scores.ndim == 2 else scores.copy()

        # Method 2: scores property
        if hasattr(self._model, 'scores'):
            scores = self._model.scores
            if scores is not None and len(scores) > 0:
                return np.array(scores[-1]).copy()

        # Method 3: Low-level API
        try:
            import llama_cpp
            if hasattr(llama_cpp, 'llama_get_logits'):
                ctx = getattr(self._model, '_ctx', None)
                if ctx and hasattr(ctx, 'ctx'):
                    ptr = llama_cpp.llama_get_logits(ctx.ctx)
                    if ptr:
                        model_obj = getattr(self._model, '_model', None)
                        if model_obj and hasattr(model_obj, 'model'):
                            vocab = llama_cpp.llama_n_vocab(model_obj.model)
                            return np.ctypeslib.as_array(ptr, shape=(vocab,)).copy()
        except Exception:
            pass

        return None

    @property
    def capabilities(self) -> Optional[Capabilities]:
        return self._capabilities

    @property
    def eos_token(self) -> Optional[int]:
        return self._eos_token

    @property
    def vocab_size(self) -> int:
        return self._capabilities.vocab_size if self._capabilities else 0

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        return list(self._model.tokenize(text.encode('utf-8')))

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize tokens."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        return self._model.detokenize(tokens).decode('utf-8', errors='replace')

    def clear_runtime_state(self) -> None:
        """Reset KV cache and internal tracking."""
        if self._model:
            self._model.reset()
        self._current_tokens = []

    def create_checkpoint(self, checkpoint_id: str) -> None:
        """
        Create checkpoint at current state.

        Stores ONLY KV cache bytes, NOT tokens.
        Tokens are owned by Beams (structural sharing).

        Args:
            checkpoint_id: Unique identifier for this checkpoint
        """
        state_bytes = None

        if self._capabilities and self._capabilities.supports_state:
            try:
                state_bytes = self._model.save_state()
            except Exception as e:
                logger.warning(f"save_state failed: {e}")

        self._checkpoint_store.add(checkpoint_id, state_bytes)

    def restore_checkpoint(self, checkpoint_id: str, tokens: List[int]) -> None:
        """
        Restore model state from checkpoint.

        MUST be called before stepping any beam to ensure isolation.

        CRITICAL: Uses caller-provided tokens as authoritative source.
        The checkpoint only provides KV cache state; tokens come from Beam.

        Args:
            checkpoint_id: Checkpoint to restore from
            tokens: Authoritative tokens from Beam (for replay fallback)
        """
        ref = self._checkpoint_store.get(checkpoint_id)

        if ref is None:
            # No checkpoint found - full replay using provided tokens
            logger.debug(f"Checkpoint {checkpoint_id} not found, replaying {len(tokens)} tokens")
            self.clear_runtime_state()
            if tokens:
                self._model.eval(tokens)
                self._current_tokens = list(tokens)
            return

        # Try to restore from state bytes
        state_bytes = ref.get_bytes()
        if state_bytes is not None and self._capabilities.supports_state:
            try:
                self._model.load_state(state_bytes)
                self._current_tokens = list(tokens)  # Use caller's tokens
                logger.debug(f"Restored KV cache from {checkpoint_id}")
                return
            except Exception as e:
                logger.warning(f"load_state failed: {e}")

        # Fallback: replay using provided tokens (NOT stored tokens)
        logger.debug(f"Replaying {len(tokens)} tokens for {checkpoint_id}")
        self.clear_runtime_state()
        if tokens:
            self._model.eval(tokens)
            self._current_tokens = list(tokens)

    def eval(self, tokens: List[int]) -> None:
        """Evaluate tokens to update KV cache."""
        if not self._model:
            raise RuntimeError("Model not loaded")
        self._model.eval(tokens)
        self._current_tokens.extend(tokens)

    def get_logits(self) -> np.ndarray:
        """Get logits for last position. Returns copy."""
        logits = self._try_get_logits()
        if logits is None:
            raise RuntimeError("Failed to get logits")
        return logits

    def sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 40,
        top_p: float = 0.95,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, float]:
        """Sample next token with log probability."""
        return sample_from_logits(logits, temperature, top_k, top_p, rng)

    @property
    def current_position(self) -> int:
        return len(self._current_tokens)

    def cleanup(self) -> None:
        """Release all resources."""
        self._checkpoint_store.clear_all()
        self.clear_runtime_state()
