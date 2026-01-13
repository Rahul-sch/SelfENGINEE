"""
Semantic-Adaptive Entropy Gating (SAEG)

Novel branching strategy that considers:
1. Standard entropy (distributional uncertainty)
2. Semantic uncertainty (embedding distance between top-k candidates)
3. Position weight (early tokens matter more)
4. Adaptive threshold (respects syntactic context)

Paper: "Semantic-Adaptive Entropy Gating for Neural Code Generation"

Key insight: Branch on SEMANTIC uncertainty, not just DISTRIBUTIONAL uncertainty.
High entropy on variable names (x vs y) is cosmetic.
Low entropy on control flow (if vs while) is critical.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict
import numpy as np
import logging

from engine.constraints import SyntacticState

logger = logging.getLogger(__name__)


@dataclass
class SAEGConfig:
    """Configuration for Semantic-Adaptive Entropy Gating."""

    # Weights for branch score components
    alpha: float = 1.0      # Weight for entropy
    beta: float = 0.5       # Weight for semantic uncertainty
    gamma: float = 0.3      # Weight for position weight

    # Threshold parameters
    base_threshold: float = 6.0   # Base branching threshold
    mu: float = 0.2               # Syntactic complexity modifier
    lambda_: float = 0.1          # Position decay rate

    # Top-k for semantic uncertainty calculation
    k: int = 5

    # Logging
    log_decisions: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.alpha >= 0, "alpha must be non-negative"
        assert self.beta >= 0, "beta must be non-negative"
        assert self.gamma >= 0, "gamma must be non-negative"
        assert self.base_threshold > 0, "base_threshold must be positive"
        assert self.k >= 2, "k must be at least 2 for pairwise distances"


@dataclass
class SAEGDecision:
    """Result of SAEG branch decision with full diagnostics."""
    should_branch: bool
    branch_score: float
    adaptive_threshold: float

    # Components for analysis
    entropy: float
    semantic_uncertainty: float
    position_weight: float
    syntactic_complexity: float

    # Top-k analysis
    top_k_tokens: List[int] = field(default_factory=list)
    top_k_distances: List[float] = field(default_factory=list)

    def to_log_dict(self) -> Dict:
        """Convert to dict for logging/analysis."""
        return {
            'should_branch': self.should_branch,
            'branch_score': round(self.branch_score, 4),
            'threshold': round(self.adaptive_threshold, 4),
            'entropy': round(self.entropy, 4),
            'semantic_unc': round(self.semantic_uncertainty, 4),
            'pos_weight': round(self.position_weight, 4),
            'syntax_complexity': round(self.syntactic_complexity, 4),
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = logits - np.max(logits)
    ex = np.exp(x)
    s = np.sum(ex)
    if s <= 0:
        return np.ones_like(ex) / len(ex)
    return ex / s


def compute_entropy(logits: np.ndarray) -> float:
    """Compute Shannon entropy from logits."""
    probs = _softmax(logits)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def compute_semantic_uncertainty(
    logits: np.ndarray,
    embeddings: Optional[np.ndarray],
    k: int = 5
) -> Tuple[float, List[int], List[float]]:
    """
    Compute semantic uncertainty as mean pairwise embedding distance.

    Args:
        logits: Raw logits from model
        embeddings: Token embedding matrix (vocab_size, embed_dim)
                   If None, falls back to logit-based distance
        k: Number of top tokens to consider

    Returns:
        (semantic_uncertainty, top_k_indices, pairwise_distances)

    Intuition:
        - If top-k tokens have SIMILAR embeddings → semantically equivalent
          → Low semantic uncertainty → Don't branch
        - If top-k tokens have DISTANT embeddings → semantically different
          → High semantic uncertainty → Consider branching
    """
    vocab_size = len(logits)
    k = min(k, vocab_size)

    # Get top-k token indices (O(n) via argpartition)
    top_k_indices = np.argpartition(logits, -k)[-k:].tolist()

    if embeddings is None:
        # Fallback: use logit values as proxy for "embedding"
        # Less accurate but works without embedding matrix
        top_k_values = logits[top_k_indices]
        distances = []
        for i in range(k):
            for j in range(i + 1, k):
                dist = abs(top_k_values[i] - top_k_values[j])
                distances.append(float(dist))
        semantic_uncertainty = np.mean(distances) if distances else 0.0
        return semantic_uncertainty, top_k_indices, distances

    # Use actual embeddings
    top_k_embeddings = embeddings[top_k_indices]

    # Compute pairwise L2 distances
    distances = []
    for i in range(k):
        for j in range(i + 1, k):
            dist = float(np.linalg.norm(top_k_embeddings[i] - top_k_embeddings[j]))
            distances.append(dist)

    semantic_uncertainty = np.mean(distances) if distances else 0.0
    return semantic_uncertainty, top_k_indices, distances


def compute_position_weight(position: int, lambda_: float = 0.1) -> float:
    """
    Compute position-based weight for branching importance.

    Early positions (function signature) get higher weight.
    Later positions (body details) get lower weight.

    Formula: 1 / (1 + λ·log(position+1))

    Examples:
        position=0:  weight ≈ 1.0
        position=10: weight ≈ 0.81
        position=50: weight ≈ 0.72
        position=100: weight ≈ 0.68
    """
    return 1.0 / (1.0 + lambda_ * np.log(position + 1))


def compute_syntactic_complexity(state: SyntacticState) -> float:
    """
    Compute syntactic complexity score from constraint tracker state.

    Higher complexity → Higher threshold (less likely to branch)

    Intuition:
        - Deep in brackets (function args): cosmetic variations okay
        - At statement start: control flow decisions critical
    """
    complexity = float(state.bracket_depth)

    if state.in_string:
        complexity += 1.0  # Inside string, variations are just text

    if state.in_comment:
        complexity += 2.0  # Inside comment, never branch

    # Bonus: at statement start, REDUCE complexity (encourage branching)
    if state.at_statement_start:
        complexity = max(0.0, complexity - 1.0)

    return complexity


def compute_adaptive_threshold(
    base_threshold: float,
    syntactic_complexity: float,
    mu: float = 0.2
) -> float:
    """
    Compute context-adaptive branching threshold.

    Formula: θ_base · (1 + μ·complexity)

    Higher complexity → Higher threshold → Less branching
    """
    return base_threshold * (1.0 + mu * syntactic_complexity)


def saeg_branch_decision(
    logits: np.ndarray,
    position: int,
    constraint_state: SyntacticState,
    config: SAEGConfig,
    embeddings: Optional[np.ndarray] = None
) -> SAEGDecision:
    """
    Main SAEG decision function.

    Combines:
    1. Entropy (distributional uncertainty)
    2. Semantic uncertainty (embedding distance)
    3. Position weight (early tokens matter more)
    4. Adaptive threshold (syntactic context)

    Args:
        logits: Raw logits from model (vocab_size,)
        position: Current token position in sequence
        constraint_state: Syntactic state from ConstraintTracker
        config: SAEG configuration
        embeddings: Optional token embedding matrix (vocab_size, embed_dim)

    Returns:
        SAEGDecision with branch decision and diagnostics
    """
    # 1. Standard entropy
    entropy = compute_entropy(logits)

    # 2. Semantic uncertainty
    semantic_uncertainty, top_k_tokens, top_k_distances = compute_semantic_uncertainty(
        logits, embeddings, config.k
    )

    # 3. Position weight
    position_weight = compute_position_weight(position, config.lambda_)

    # 4. Syntactic complexity and adaptive threshold
    syntactic_complexity = compute_syntactic_complexity(constraint_state)
    adaptive_threshold = compute_adaptive_threshold(
        config.base_threshold, syntactic_complexity, config.mu
    )

    # 5. Combine into branch score
    branch_score = (
        config.alpha * entropy +
        config.beta * semantic_uncertainty +
        config.gamma * position_weight
    )

    # 6. Decision
    should_branch = branch_score > adaptive_threshold

    decision = SAEGDecision(
        should_branch=should_branch,
        branch_score=branch_score,
        adaptive_threshold=adaptive_threshold,
        entropy=entropy,
        semantic_uncertainty=semantic_uncertainty,
        position_weight=position_weight,
        syntactic_complexity=syntactic_complexity,
        top_k_tokens=top_k_tokens,
        top_k_distances=top_k_distances,
    )

    if config.log_decisions:
        action = "BRANCH" if should_branch else "FLOW"
        logger.debug(
            f"SAEG[pos={position}]: {action} | "
            f"score={branch_score:.3f} vs thresh={adaptive_threshold:.3f} | "
            f"ent={entropy:.3f} sem={semantic_uncertainty:.3f} pos={position_weight:.3f}"
        )

    return decision


class SAEGTracker:
    """
    Tracks SAEG decisions across generation for analysis.

    Collects statistics for:
    - How often we branch vs flow
    - What entropy/semantic uncertainty values trigger branches
    - Correlation between position and branching
    """

    def __init__(self, config: SAEGConfig):
        self.config = config
        self.decisions: List[SAEGDecision] = []
        self.branch_count: int = 0
        self.flow_count: int = 0

    def record(self, decision: SAEGDecision) -> None:
        """Record a decision for analysis."""
        self.decisions.append(decision)
        if decision.should_branch:
            self.branch_count += 1
        else:
            self.flow_count += 1

    def reset(self) -> None:
        """Reset tracker for new generation."""
        self.decisions.clear()
        self.branch_count = 0
        self.flow_count = 0

    @property
    def branch_rate(self) -> float:
        """Fraction of decisions that resulted in branching."""
        total = self.branch_count + self.flow_count
        return self.branch_count / total if total > 0 else 0.0

    def get_statistics(self) -> Dict:
        """Get summary statistics for paper/analysis."""
        if not self.decisions:
            return {}

        entropies = [d.entropy for d in self.decisions]
        semantic_uncs = [d.semantic_uncertainty for d in self.decisions]
        scores = [d.branch_score for d in self.decisions]
        thresholds = [d.adaptive_threshold for d in self.decisions]

        branch_decisions = [d for d in self.decisions if d.should_branch]
        flow_decisions = [d for d in self.decisions if not d.should_branch]

        return {
            'total_decisions': len(self.decisions),
            'branch_count': self.branch_count,
            'flow_count': self.flow_count,
            'branch_rate': round(self.branch_rate, 4),

            'entropy_mean': round(np.mean(entropies), 4),
            'entropy_std': round(np.std(entropies), 4),

            'semantic_unc_mean': round(np.mean(semantic_uncs), 4),
            'semantic_unc_std': round(np.std(semantic_uncs), 4),

            'score_mean': round(np.mean(scores), 4),
            'threshold_mean': round(np.mean(thresholds), 4),

            'branch_avg_entropy': round(np.mean([d.entropy for d in branch_decisions]), 4) if branch_decisions else 0,
            'flow_avg_entropy': round(np.mean([d.entropy for d in flow_decisions]), 4) if flow_decisions else 0,

            'branch_avg_semantic': round(np.mean([d.semantic_uncertainty for d in branch_decisions]), 4) if branch_decisions else 0,
            'flow_avg_semantic': round(np.mean([d.semantic_uncertainty for d in flow_decisions]), 4) if flow_decisions else 0,
        }

    def to_csv_rows(self) -> List[Dict]:
        """Export decisions as CSV-ready rows for analysis."""
        rows = []
        for i, d in enumerate(self.decisions):
            row = d.to_log_dict()
            row['step'] = i
            rows.append(row)
        return rows
