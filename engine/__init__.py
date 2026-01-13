"""
SuperCoder Engine - System-2 Reasoning for Code Generation

Core modules:
- constraints: Syntactic boundary detection for Python code
- search: Beam search with structural checkpoint sharing
- model_adapter: LLM wrapper with checkpoint-based beam isolation
- verify_static: AST-based security analysis
- orchestrator: Main orchestrator with entropy-gated branching
"""

from engine.constraints import (
    ConstraintTracker,
    SyntacticState,
    StringState,
    DECISION_KEYWORDS,
)

from engine.search import (
    BeamManager,
    Beam,
    Checkpoint,
    SearchConfig,
    BeamStatus,
)

from engine.model_adapter import (
    ModelAdapter,
    CheckpointStore,
    Capabilities,
    sample_from_logits,
)

from engine.verify_static import (
    analyze_code,
    is_code_safe,
    StaticAnalysisResult,
    SecurityIssue,
    Severity,
)

from engine.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    entropy_and_varentropy_from_logits,
)

from engine.saeg import (
    SAEGConfig,
    SAEGDecision,
    SAEGTracker,
    saeg_branch_decision,
    compute_entropy,
    compute_semantic_uncertainty,
    compute_position_weight,
    compute_adaptive_threshold,
)

__all__ = [
    # constraints
    'ConstraintTracker',
    'SyntacticState',
    'StringState',
    'DECISION_KEYWORDS',
    # search
    'BeamManager',
    'Beam',
    'Checkpoint',
    'SearchConfig',
    'BeamStatus',
    # model_adapter
    'ModelAdapter',
    'CheckpointStore',
    'Capabilities',
    'sample_from_logits',
    # verify_static
    'analyze_code',
    'is_code_safe',
    'StaticAnalysisResult',
    'SecurityIssue',
    'Severity',
    # orchestrator
    'Orchestrator',
    'OrchestratorConfig',
    'entropy_and_varentropy_from_logits',
    # saeg (novel algorithm)
    'SAEGConfig',
    'SAEGDecision',
    'SAEGTracker',
    'saeg_branch_decision',
    'compute_entropy',
    'compute_semantic_uncertainty',
    'compute_position_weight',
    'compute_adaptive_threshold',
]
