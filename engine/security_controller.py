"""
SecurityController: Orchestrates SABS verification during generation.

Security-Aware Beam Search (SABS) integrates AST-based security verification
directly into the beam search decoding loop.

Responsibilities:
- Decide when to run verifier (boundary + budget + min_tokens)
- Call analyze_incremental with caching
- Apply penalties or hard-fail beams
- Log decisions to JSONL

Usage:
    controller = SecurityController(SABSConfig(lambda_weight=0.5))
    if controller.should_check(beam):
        decision = controller.check_beam(beam, code, gen_id, step)
        controller.apply_decision(beam, decision, search)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, TYPE_CHECKING
from enum import Enum, auto
import json
import time
import re

if TYPE_CHECKING:
    from engine.search import Beam, BeamManager
    from engine.model_adapter import ModelAdapter

from engine.verify_static import (
    ParseState, IncrementalResult, SecurityCache,
    analyze_incremental, Severity, SecurityIssue
)


@dataclass
class SABSConfig:
    """
    Configuration for Security-Aware Beam Search.

    Attributes:
        lambda_weight: Security penalty weight (0.0 = disabled, higher = stricter)
        hard_fail_on_critical: Terminate beams with CRITICAL issues immediately
        max_verifier_calls: Maximum verifier calls per generation (budget)
        min_tokens_between_checks: Minimum tokens between security checks
        cache_enabled: Use LRU cache for security results
        log_path: Path for JSONL decision log (None = no logging)
    """
    lambda_weight: float = 0.5
    hard_fail_on_critical: bool = True
    max_verifier_calls: int = 50
    min_tokens_between_checks: int = 8
    cache_enabled: bool = True
    log_path: Optional[str] = None


class SABSAction(Enum):
    """Action taken by SABS on a beam check."""
    SKIP_INCOMPLETE = auto()    # Code incomplete, no penalty
    SKIP_BUDGET = auto()        # Verifier budget exhausted
    SKIP_MIN_TOKENS = auto()    # Not enough tokens since last check
    SKIP_NOT_BOUNDARY = auto()  # Not at syntactic boundary
    SOFT_PENALTY = auto()       # Applied graduated penalty
    HARD_FAIL = auto()          # Beam terminated (CRITICAL issue)
    NO_ISSUES = auto()          # Check passed, no issues found


@dataclass
class SABSDecision:
    """
    Result of a SABS check on a beam.

    Attributes:
        action: What action was taken
        parse_state: TRI-STATE parse outcome
        issues: List of issue dicts (severity, code, message, lineno)
        sec_penalty_delta: Penalty added this check
        total_sec_penalty: Beam's total penalty after this check
        should_terminate: Whether beam should be hard-failed
        code_snippet: Short snippet of checked code (for logging)
    """
    action: SABSAction
    parse_state: ParseState
    issues: List[dict]
    sec_penalty_delta: float
    total_sec_penalty: float
    should_terminate: bool
    code_snippet: str = ""


def extract_code_only(model: "ModelAdapter", beam: "Beam") -> str:
    """
    Extract generated code only (no prompt).
    Strip markdown fences and common prose patterns.

    Args:
        model: Model adapter for detokenization
        beam: Beam containing tokens

    Returns:
        Clean code string (may be empty)
    """
    # Get generated tokens only (exclude prompt)
    generated_tokens = beam.tokens[beam.prompt_length:]
    if not generated_tokens:
        return ""

    code = model.detokenize(generated_tokens)

    # Strip markdown fences
    code = re.sub(r'^```(?:python)?\s*\n', '', code)
    code = re.sub(r'\n```\s*$', '', code)

    # Strip common prose prefixes
    prose_patterns = [
        r'^Here(?:\'s| is) (?:a |the )?(?:Python )?(?:code|function|solution).*?:\s*\n',
        r'^(?:Sure|Certainly|Of course).*?:\s*\n',
        r'^I\'ll .*?:\s*\n',
    ]
    for pattern in prose_patterns:
        code = re.sub(pattern, '', code, flags=re.IGNORECASE)

    return code.strip()


class SecurityController:
    """
    Orchestrates SABS verification during beam search generation.

    Tracks verifier call budget, applies penalties, and logs decisions.
    """

    def __init__(self, config: SABSConfig):
        self.config = config
        self.cache = SecurityCache() if config.cache_enabled else None
        self.verifier_calls = 0
        self.decisions: List[dict] = []
        self._log_file = None
        self._gen_id: str = ""

        if config.log_path:
            try:
                self._log_file = open(config.log_path, 'w')
            except IOError as e:
                print(f"Warning: Could not open SABS log file: {e}")

    def reset(self, gen_id: str = "") -> None:
        """Reset for new generation."""
        self.verifier_calls = 0
        self.decisions.clear()
        self._gen_id = gen_id
        if self.cache:
            self.cache = SecurityCache()

    def should_check(self, beam: "Beam") -> bool:
        """
        Determine if this beam should be security-checked.

        Checks:
        1. Verifier budget not exhausted
        2. Enough tokens since last check
        3. At syntactic boundary (if constraint tracker available)

        Args:
            beam: Beam to potentially check

        Returns:
            True if check should proceed
        """
        # Budget check
        if self.verifier_calls >= self.config.max_verifier_calls:
            return False

        # Minimum tokens check
        if beam.tokens_since_security_check < self.config.min_tokens_between_checks:
            return False

        # Boundary check (if tracker available)
        if beam.constraint_tracker and not beam.constraint_tracker.is_boundary_now():
            return False

        return True

    def check_beam(
        self,
        beam: "Beam",
        code: str,
        gen_id: str,
        step: int
    ) -> SABSDecision:
        """
        Run security check on beam and compute penalty.

        Args:
            beam: Beam to check
            code: Extracted code string
            gen_id: Generation ID for logging
            step: Current step number

        Returns:
            SABSDecision with action, penalty, and termination flag
        """
        self.verifier_calls += 1
        old_penalty = beam.sec_penalty

        # Handle empty code
        if not code or not code.strip():
            decision = SABSDecision(
                action=SABSAction.SKIP_INCOMPLETE,
                parse_state=ParseState.PARSE_INCOMPLETE,
                issues=[],
                sec_penalty_delta=0.0,
                total_sec_penalty=beam.sec_penalty,
                should_terminate=False,
                code_snippet="",
            )
            self._log_decision(beam, decision, step)
            return decision

        # Run incremental analysis
        result = analyze_incremental(code, self.cache)

        # Handle incomplete code (no penalty)
        if result.parse_state == ParseState.PARSE_INCOMPLETE:
            decision = SABSDecision(
                action=SABSAction.SKIP_INCOMPLETE,
                parse_state=result.parse_state,
                issues=[],
                sec_penalty_delta=0.0,
                total_sec_penalty=beam.sec_penalty,
                should_terminate=False,
                code_snippet=code[:50] if code else "",
            )
            self._log_decision(beam, decision, step)
            return decision

        # Handle invalid code (small fixed penalty for early errors)
        if result.parse_state == ParseState.PARSE_INVALID:
            decision = SABSDecision(
                action=SABSAction.SOFT_PENALTY,
                parse_state=result.parse_state,
                issues=[],
                sec_penalty_delta=result.sec_penalty,  # 2.0 fixed
                total_sec_penalty=beam.sec_penalty + result.sec_penalty,
                should_terminate=False,
                code_snippet=code[:50] if code else "",
            )
            self._log_decision(beam, decision, step)
            return decision

        # PARSE_OK - process issues
        issues_dict = [
            {
                'severity': i.severity.name,
                'code': i.code,
                'message': i.message,
                'lineno': i.lineno,
            }
            for i in result.issues
        ]

        # Check for CRITICAL issues (hard fail)
        has_critical = any(i.severity == Severity.CRITICAL for i in result.issues)

        if has_critical and self.config.hard_fail_on_critical:
            decision = SABSDecision(
                action=SABSAction.HARD_FAIL,
                parse_state=result.parse_state,
                issues=issues_dict,
                sec_penalty_delta=result.sec_penalty,
                total_sec_penalty=beam.sec_penalty + result.sec_penalty,
                should_terminate=True,
                code_snippet=code[:100] if code else "",
            )
            self._log_decision(beam, decision, step)
            return decision

        # No critical issues - apply soft penalty
        if result.sec_penalty > 0:
            decision = SABSDecision(
                action=SABSAction.SOFT_PENALTY,
                parse_state=result.parse_state,
                issues=issues_dict,
                sec_penalty_delta=result.sec_penalty,
                total_sec_penalty=beam.sec_penalty + result.sec_penalty,
                should_terminate=False,
                code_snippet=code[:100] if code else "",
            )
        else:
            decision = SABSDecision(
                action=SABSAction.NO_ISSUES,
                parse_state=result.parse_state,
                issues=[],
                sec_penalty_delta=0.0,
                total_sec_penalty=beam.sec_penalty,
                should_terminate=False,
                code_snippet=code[:50] if code else "",
            )

        self._log_decision(beam, decision, step)
        return decision

    def apply_decision(
        self,
        beam: "Beam",
        decision: SABSDecision,
        search: "BeamManager"
    ) -> None:
        """
        Apply SABS decision to beam.

        Args:
            beam: Beam to modify
            decision: SABS decision result
            search: BeamManager for hard-fail handling
        """
        if decision.should_terminate:
            search.fail_beam(beam, "security_critical")
        else:
            beam.sec_penalty += decision.sec_penalty_delta
            beam.tokens_since_security_check = 0

    def _log_decision(self, beam: "Beam", decision: SABSDecision, step: int) -> None:
        """Log decision to JSONL file and internal list."""
        log_entry = {
            'gen_id': self._gen_id,
            'step': step,
            'beam_id': beam.beam_id,
            'position': beam.current_position,
            'tokens_since_check': beam.tokens_since_security_check,
            'parse_state': decision.parse_state.name,
            'issues': decision.issues,
            'sec_penalty_delta': decision.sec_penalty_delta,
            'total_sec_penalty': decision.total_sec_penalty,
            'action': decision.action.name,
            'hard_failed': decision.should_terminate,
            'code_snippet': decision.code_snippet,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }

        self.decisions.append(log_entry)

        if self._log_file:
            try:
                self._log_file.write(json.dumps(log_entry) + '\n')
                self._log_file.flush()
            except IOError:
                pass

    def get_statistics(self) -> Dict:
        """
        Get SABS statistics for current generation.

        Returns:
            Dict with verifier_calls, actions breakdown, penalties, etc.
        """
        if not self.decisions:
            return {
                'verifier_calls': self.verifier_calls,
                'total_decisions': 0,
                'hard_fails': 0,
                'soft_penalties': 0,
                'skipped': 0,
                'no_issues': 0,
            }

        actions = {}
        for d in self.decisions:
            action = d['action']
            actions[action] = actions.get(action, 0) + 1

        total_penalty = sum(d['sec_penalty_delta'] for d in self.decisions)
        hard_fails = sum(1 for d in self.decisions if d['hard_failed'])

        return {
            'verifier_calls': self.verifier_calls,
            'total_decisions': len(self.decisions),
            'hard_fails': hard_fails,
            'soft_penalties': actions.get('SOFT_PENALTY', 0),
            'skipped': (
                actions.get('SKIP_INCOMPLETE', 0) +
                actions.get('SKIP_BUDGET', 0) +
                actions.get('SKIP_MIN_TOKENS', 0) +
                actions.get('SKIP_NOT_BOUNDARY', 0)
            ),
            'no_issues': actions.get('NO_ISSUES', 0),
            'total_penalty_applied': total_penalty,
            'actions_breakdown': actions,
        }

    def close(self) -> None:
        """Close log file if open."""
        if self._log_file:
            try:
                self._log_file.close()
            except IOError:
                pass
            self._log_file = None
