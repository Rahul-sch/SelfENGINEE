# SABS API Documentation

Security-Aware Beam Search (SABS) integrates AST-based security verification directly into the beam search decoding loop for code generation.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Examples](#examples)
7. [Algorithm Details](#algorithm-details)

---

## Overview

SABS extends the standard beam search algorithm with:

- **Boundary-triggered verification**: Security checks at syntactic boundaries
- **Graduated penalties**: Soft penalties for warnings, hard fail for critical issues
- **Per-beam tracking**: Independent constraint state for each beam
- **TRI-STATE parsing**: Handles incomplete code during generation

### Key Benefits

| Feature | Baseline | SABS |
|---------|----------|------|
| Security checking | Post-hoc | During generation |
| Wasted computation | High on unsafe beams | Low (early termination) |
| Penalty tuning | N/A | Lambda parameter |
| False positive handling | All-or-nothing | Graduated penalties |

---

## Quick Start

### CLI Usage

```bash
# Enable SABS with default settings
python -m cli.main --use-sabs "Write a function to process user input"

# Adjust security weight
python -m cli.main --use-sabs --sabs-lambda 0.8 "Execute a shell command"

# Disable hard-fail (soft penalties only)
python -m cli.main --use-sabs --sabs-no-hard-fail "Write file operations"

# Log SABS decisions
python -m cli.main --use-sabs --sabs-log ./sabs_decisions.jsonl "..."
```

### Programmatic Usage

```python
from engine.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    model_path="./models/model.gguf",
    use_sabs=True,
    sabs_lambda=0.5,
    sabs_hard_fail=True,
    sabs_max_calls=50,
    sabs_min_tokens=8,
)

orch = Orchestrator(config)
output = orch.generate("Write a file parser")

# Get SABS statistics
results = orch.get_experiment_results()
print(results['sabs_statistics'])
```

---

## Core Components

### 1. SecurityController

Orchestrates SABS verification during beam search.

```python
from engine.security_controller import SecurityController, SABSConfig

config = SABSConfig(
    lambda_weight=0.5,
    hard_fail_on_critical=True,
    max_verifier_calls=50,
    min_tokens_between_checks=8,
)

controller = SecurityController(config)
```

### 2. analyze_incremental

Incremental-safe security analysis with TRI-STATE parsing.

```python
from engine.verify_static import analyze_incremental, SecurityCache

cache = SecurityCache(max_size=1000)
result = analyze_incremental("import subprocess", cache)

print(f"State: {result.parse_state.name}")  # PARSE_OK
print(f"Penalty: {result.sec_penalty}")      # > 0
print(f"Issues: {len(result.issues)}")       # Issue count
```

### 3. ConstraintTracker

Tracks Python syntax for boundary detection.

```python
from engine.constraints import ConstraintTracker

tracker = ConstraintTracker()
boundaries = tracker.update("def foo():\n    return 1\n")

print(f"At boundary: {tracker.is_boundary_now()}")
print(f"Bracket depth: {tracker.state.bracket_depth}")
```

### 4. Beam Extensions

Beams include SABS-specific fields.

```python
from engine.search import Beam, SearchConfig

config = SearchConfig(security_lambda=0.5)
beam = Beam(tokens=[1,2,3], prompt_length=1, log_prob=-5.0)
beam.sec_penalty = 2.5

score = beam.normalized_score(config)
# score = (log_prob / length_factor) - lambda * sec_penalty
```

---

## Configuration

### SABSConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lambda_weight` | float | 0.5 | Security penalty weight in scoring |
| `hard_fail_on_critical` | bool | True | Terminate beams with CRITICAL issues |
| `max_verifier_calls` | int | 50 | Maximum verifier calls per generation |
| `min_tokens_between_checks` | int | 8 | Minimum tokens between security checks |
| `cache_enabled` | bool | True | Enable LRU cache for security results |
| `log_path` | str | None | Path for JSONL decision log |

### OrchestratorConfig SABS Fields

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_sabs` | bool | False | Enable SABS |
| `sabs_lambda` | float | 0.5 | Security penalty weight |
| `sabs_hard_fail` | bool | True | Hard-fail on CRITICAL |
| `sabs_max_calls` | int | 50 | Max verifier calls |
| `sabs_min_tokens` | int | 8 | Min tokens between checks |
| `sabs_log_path` | str | None | JSONL log path |

### SearchConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `security_lambda` | float | 0.0 | Lambda in beam scoring (0 = disabled) |

---

## API Reference

### engine.verify_static

#### ParseState (Enum)

```python
class ParseState(Enum):
    PARSE_OK = auto()         # AST succeeded; issues meaningful
    PARSE_INCOMPLETE = auto() # Syntax error near EOF; code incomplete
    PARSE_INVALID = auto()    # Syntax error early; code invalid
```

#### IncrementalResult (NamedTuple)

```python
class IncrementalResult(NamedTuple):
    parse_state: ParseState
    issues: List[SecurityIssue]
    sec_penalty: float
```

#### SEVERITY_WEIGHTS

```python
SEVERITY_WEIGHTS = {
    Severity.CRITICAL: 10.0,  # Hard fail candidate
    Severity.ERROR:    5.0,   # Strong penalty
    Severity.WARNING:  1.0,   # Soft penalty
    Severity.INFO:     0.1,   # Negligible
}
```

#### Functions

```python
def analyze_incremental(
    code: str,
    cache: Optional[SecurityCache] = None
) -> IncrementalResult:
    """
    Incremental-safe security analysis.

    Handles:
    - Empty/whitespace code -> PARSE_INCOMPLETE
    - Syntax error near end -> PARSE_INCOMPLETE (no penalty)
    - Syntax error early -> PARSE_INVALID (2.0 fixed penalty)
    - Successful parse -> PARSE_OK (run full analysis)
    """

def compute_sec_penalty(issues: List[SecurityIssue]) -> float:
    """Sum severity weights for all issues."""

def classify_parse_error(code: str, error: SyntaxError) -> ParseState:
    """Classify syntax error as INCOMPLETE or INVALID."""
```

#### SecurityCache

```python
class SecurityCache:
    """LRU cache for security results."""

    def __init__(self, max_size: int = 1000): ...
    def get(self, code: str) -> Optional[IncrementalResult]: ...
    def put(self, code: str, result: IncrementalResult) -> None: ...
    def clear(self) -> None: ...

    @property
    def stats(self) -> Dict[str, int]: ...
```

---

### engine.security_controller

#### SABSAction (Enum)

```python
class SABSAction(Enum):
    SKIP_INCOMPLETE = auto()    # Code incomplete, no penalty
    SKIP_BUDGET = auto()        # Verifier budget exhausted
    SKIP_MIN_TOKENS = auto()    # Not enough tokens since last check
    SKIP_NOT_BOUNDARY = auto()  # Not at syntactic boundary
    SOFT_PENALTY = auto()       # Applied graduated penalty
    HARD_FAIL = auto()          # Beam terminated (CRITICAL issue)
    NO_ISSUES = auto()          # Check passed, no issues found
```

#### SABSDecision

```python
@dataclass
class SABSDecision:
    action: SABSAction
    parse_state: ParseState
    issues: List[dict]
    sec_penalty_delta: float
    total_sec_penalty: float
    should_terminate: bool
    code_snippet: str = ""
```

#### SecurityController

```python
class SecurityController:
    def __init__(self, config: SABSConfig): ...

    def reset(self, gen_id: str = "") -> None:
        """Reset for new generation."""

    def should_check(self, beam: Beam) -> bool:
        """Check if beam should be security-checked."""

    def check_beam(
        self,
        beam: Beam,
        code: str,
        gen_id: str,
        step: int
    ) -> SABSDecision:
        """Run security check and compute penalty."""

    def apply_decision(
        self,
        beam: Beam,
        decision: SABSDecision,
        search: BeamManager
    ) -> None:
        """Apply decision to beam (penalty or hard-fail)."""

    def get_statistics(self) -> Dict:
        """Get SABS statistics for current generation."""
```

#### extract_code_only

```python
def extract_code_only(model: ModelAdapter, beam: Beam) -> str:
    """
    Extract generated code only (no prompt).
    Strips markdown fences and prose prefixes.
    """
```

---

### engine.constraints

#### ConstraintTracker

```python
class ConstraintTracker:
    def __init__(self): ...

    def reset(self) -> None:
        """Reset all state for new generation."""

    def update(self, text: str) -> List[int]:
        """
        Process text and return boundary offsets.

        Boundaries trigger on:
        1. Newline at top level (bracket_depth==0)
        2. Decision keyword at statement start
        """

    def is_boundary_now(self) -> bool:
        """Check if current position is at a boundary."""

    def clone(self) -> ConstraintTracker:
        """Deep copy for per-beam tracking."""

    @property
    def state(self) -> SyntacticState:
        """Current syntactic state."""
```

#### SyntacticState

```python
@dataclass
class SyntacticState:
    bracket_depth: int = 0
    string_state: StringState = StringState.NONE
    in_comment: bool = False
    at_line_start: bool = True
    line_indent_only: bool = True

    @property
    def in_string(self) -> bool: ...
    @property
    def at_statement_start(self) -> bool: ...
    @property
    def is_top_level(self) -> bool: ...
```

---

### engine.search (SABS Extensions)

#### Beam SABS Fields

```python
@dataclass
class Beam:
    # ... standard fields ...

    # SABS fields
    sec_penalty: float = 0.0
    constraint_tracker: Optional[ConstraintTracker] = None
    tokens_since_security_check: int = 0

    def normalized_score(self, config: SearchConfig) -> float:
        """
        Score = (log_prob / length_factor) - lambda * sec_penalty
        """
```

#### BeamManager.fail_beam

```python
def fail_beam(self, beam: Beam, reason: str) -> None:
    """Mark beam as FAILED and remove from active beams."""
```

---

## Examples

### Example 1: Basic SABS Usage

```python
from engine.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    model_path="./models/deepseek-coder-6.7b.Q4_K_M.gguf",
    use_sabs=True,
    sabs_lambda=0.5,
)

orch = Orchestrator(config)
code = orch.generate("Write a function to read a JSON file")

# Check statistics
stats = orch.get_experiment_results()['sabs_statistics']
print(f"Verifier calls: {stats['verifier_calls']}")
print(f"Hard fails: {stats['hard_fails']}")
print(f"Soft penalties: {stats['soft_penalties']}")
```

### Example 2: Analyzing Code Incrementally

```python
from engine.verify_static import analyze_incremental, ParseState

# Incomplete code - no penalty
result = analyze_incremental("def foo(")
assert result.parse_state == ParseState.PARSE_INCOMPLETE
assert result.sec_penalty == 0.0

# Complete dangerous code - penalty applied
result = analyze_incremental("import subprocess\nsubprocess.call(['ls'])")
assert result.parse_state == ParseState.PARSE_OK
assert result.sec_penalty > 0
```

### Example 3: Custom Security Checking

```python
from engine.security_controller import SecurityController, SABSConfig
from engine.search import Beam, BeamManager, SearchConfig

# Configure controller
sabs_config = SABSConfig(
    lambda_weight=0.8,
    hard_fail_on_critical=True,
)
controller = SecurityController(sabs_config)

# Create beam manager
search_config = SearchConfig(security_lambda=0.8)
manager = BeamManager(search_config, model)

# Check a beam
beam = Beam(tokens=[1,2,3,4,5], prompt_length=2)
beam.tokens_since_security_check = 10

code = "eval(user_input)"
decision = controller.check_beam(beam, code, "gen1", step=10)

if decision.should_terminate:
    manager.fail_beam(beam, "security_critical")
else:
    beam.sec_penalty += decision.sec_penalty_delta
```

---

## Algorithm Details

### Score Equation

```
normalized_score = (log_prob / length_factor) - lambda * sec_penalty

where:
  length_factor = ((5 + num_generated) / 6) ^ length_penalty
  sec_penalty >= 0 (accumulated from security issues)
  lambda in [0, 1] (security weight parameter)
```

### Penalty Calculation

```python
sec_penalty = sum(SEVERITY_WEIGHTS[issue.severity] for issue in issues)
```

| Severity | Weight | Action |
|----------|--------|--------|
| CRITICAL | 10.0 | Hard fail (if enabled) |
| ERROR | 5.0 | Soft penalty |
| WARNING | 1.0 | Soft penalty |
| INFO | 0.1 | Soft penalty |

### TRI-STATE Parsing

| State | Condition | Penalty |
|-------|-----------|---------|
| PARSE_OK | AST success | Computed from issues |
| PARSE_INCOMPLETE | Error near EOF | 0.0 (no penalty) |
| PARSE_INVALID | Error early | 2.0 (fixed) |

### Verification Triggers

A security check is triggered when ALL conditions are met:

1. `verifier_calls < max_verifier_calls` (budget not exhausted)
2. `tokens_since_last_check >= min_tokens_between_checks`
3. `constraint_tracker.is_boundary_now() == True`

---

## JSONL Log Format

When `sabs_log_path` is set, decisions are logged in JSONL format:

```json
{
  "gen_id": "abc123",
  "step": 42,
  "beam_id": "b1c2d3e4",
  "position": 156,
  "tokens_since_check": 12,
  "parse_state": "PARSE_OK",
  "issues": [
    {"severity": "WARNING", "code": "S007", "message": "open() with write mode", "lineno": 5}
  ],
  "sec_penalty_delta": 1.0,
  "total_sec_penalty": 3.0,
  "action": "SOFT_PENALTY",
  "hard_failed": false,
  "code_snippet": "f = open(path, 'w')",
  "timestamp": "2024-01-13T10:30:00Z"
}
```

---

## Troubleshooting

### High False Positive Rate

- Reduce `sabs_lambda` to make security less dominant
- Disable `hard_fail_on_critical` to use soft penalties only
- Increase `min_tokens_between_checks` for fewer checks

### Performance Issues

- Enable caching (default): `cache_enabled=True`
- Reduce `max_verifier_calls` to limit overhead
- Increase `min_tokens_between_checks`

### Missing Security Issues

- Increase `sabs_lambda` to weight security higher
- Decrease `min_tokens_between_checks` for more frequent checks
- Check that boundary detection is working correctly

---

## Version History

- **v1.0.0**: Initial SABS implementation
  - TRI-STATE parsing for incomplete code
  - Per-beam constraint tracking
  - Configurable lambda parameter
  - LRU cache for security results
