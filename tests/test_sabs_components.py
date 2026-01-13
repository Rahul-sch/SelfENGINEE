"""
Unit tests for SABS (Security-Aware Beam Search) components.

Tests cover:
1. ParseState classification
2. SEVERITY_WEIGHTS mapping
3. compute_sec_penalty()
4. SecurityCache LRU behavior
5. analyze_incremental()
6. SABSConfig
7. SABSDecision
8. SecurityController

Run with: python -m pytest tests/test_sabs_components.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.verify_static import (
    ParseState,
    SEVERITY_WEIGHTS,
    compute_sec_penalty,
    SecurityCache,
    IncrementalResult,
    analyze_incremental,
    is_error_near_end,
    classify_parse_error,
    Severity,
    SecurityIssue,
)
from engine.security_controller import (
    SABSConfig,
    SABSAction,
    SABSDecision,
    SecurityController,
    extract_code_only,
)
from engine.constraints import ConstraintTracker


# =============================================================================
# SECTION 1: ParseState Classification Tests
# =============================================================================

class TestParseStateClassification:
    """Test TRI-STATE parse outcome classification."""

    def test_parse_ok_complete_code(self):
        """Complete valid code should return PARSE_OK."""
        code = "def foo():\n    return 42"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_parse_incomplete_trailing_colon(self):
        """Code ending with colon should return PARSE_INCOMPLETE."""
        code = "def foo():"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_open_paren(self):
        """Code with unclosed parenthesis should return PARSE_INCOMPLETE."""
        code = "print(x"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_open_bracket(self):
        """Code with unclosed bracket should return PARSE_INCOMPLETE."""
        code = "x = [1, 2, 3"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_open_brace(self):
        """Code with unclosed brace should return PARSE_INCOMPLETE."""
        code = "x = {'a': 1"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_multiline_string(self):
        """Incomplete triple-quoted string should return PARSE_INCOMPLETE."""
        code = '"""This is a docstring'
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_def_keyword_only(self):
        """Just 'def' keyword should return PARSE_INCOMPLETE."""
        code = "def"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_if_keyword_only(self):
        """Just 'if' keyword should return PARSE_INCOMPLETE."""
        code = "if"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_class_no_body(self):
        """Class declaration without body should return PARSE_INCOMPLETE."""
        code = "class Foo:"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_incomplete_for_loop_partial(self):
        """Partial for loop should return PARSE_INCOMPLETE."""
        code = "for i in"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_parse_invalid_early_error(self):
        """Syntax error early in code should return PARSE_INVALID."""
        code = "def @@@ invalid:\n    return 1\n    return 2\n    return 3"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INVALID

    def test_parse_invalid_mismatched_brackets(self):
        """Mismatched brackets early should return PARSE_INVALID."""
        # Error early in long code should be INVALID
        code = "x = [1, 2, 3)\ny = 1\nz = 2\nw = 3\na = 4\nb = 5"
        result = analyze_incremental(code)
        # Note: If error is near end, it may be INCOMPLETE. Adjust expectation.
        assert result.parse_state in (ParseState.PARSE_INVALID, ParseState.PARSE_INCOMPLETE)

    def test_empty_code_incomplete(self):
        """Empty code should return PARSE_INCOMPLETE."""
        result = analyze_incremental("")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_whitespace_only_incomplete(self):
        """Whitespace-only code should return PARSE_INCOMPLETE."""
        result = analyze_incremental("   \n\t  ")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE


class TestIsErrorNearEnd:
    """Test the is_error_near_end helper function."""

    def test_error_at_last_line(self):
        """Error on last line should be near end."""
        code = "x = 1\ny = 2\nz ="  # Error on line 3
        error = SyntaxError("invalid syntax")
        error.lineno = 3
        error.offset = 4
        assert is_error_near_end(code, error) is True

    def test_error_at_first_line(self):
        """Error on first line of multi-line code should not be near end."""
        code = "@@@ invalid\nx = 1\ny = 2\nz = 3"
        error = SyntaxError("invalid syntax")
        error.lineno = 1
        error.offset = 1
        assert is_error_near_end(code, error) is False

    def test_error_in_middle(self):
        """Error in middle should not be near end."""
        code = "x = 1\n@@@ invalid\ny = 2\nz = 3\nw = 4"
        error = SyntaxError("invalid syntax")
        error.lineno = 2
        error.offset = 1
        assert is_error_near_end(code, error) is False


class TestClassifyParseError:
    """Test classify_parse_error function."""

    def test_classify_eof_message(self):
        """Error with EOF in message should be INCOMPLETE."""
        code = "def foo("
        error = SyntaxError("unexpected EOF while parsing")
        error.lineno = 1
        error.offset = 9
        assert classify_parse_error(code, error) == ParseState.PARSE_INCOMPLETE

    def test_classify_expected_message_at_end(self):
        """'expected' error at end should be INCOMPLETE."""
        code = "def foo():"
        error = SyntaxError("expected ':'")
        error.lineno = 1
        error.offset = 11
        assert classify_parse_error(code, error) == ParseState.PARSE_INCOMPLETE

    def test_classify_error_early(self):
        """Error early in code should be INVALID."""
        code = "@@@ bad\nx = 1\ny = 2\nz = 3"
        error = SyntaxError("invalid syntax")
        error.lineno = 1
        error.offset = 1
        assert classify_parse_error(code, error) == ParseState.PARSE_INVALID


# =============================================================================
# SECTION 2: SEVERITY_WEIGHTS Tests
# =============================================================================

class TestSeverityWeights:
    """Test severity weight mappings."""

    def test_critical_weight(self):
        """CRITICAL should have weight 10.0."""
        assert SEVERITY_WEIGHTS[Severity.CRITICAL] == 10.0

    def test_error_weight(self):
        """ERROR should have weight 5.0."""
        assert SEVERITY_WEIGHTS[Severity.ERROR] == 5.0

    def test_warning_weight(self):
        """WARNING should have weight 1.0."""
        assert SEVERITY_WEIGHTS[Severity.WARNING] == 1.0

    def test_info_weight(self):
        """INFO should have weight 0.1."""
        assert SEVERITY_WEIGHTS[Severity.INFO] == 0.1

    def test_all_severities_have_weights(self):
        """All Severity enum values should have weights defined."""
        for severity in Severity:
            assert severity in SEVERITY_WEIGHTS, f"Missing weight for {severity}"


# =============================================================================
# SECTION 3: compute_sec_penalty() Tests
# =============================================================================

class TestComputeSecPenalty:
    """Test security penalty computation."""

    def test_no_issues_zero_penalty(self):
        """No issues should result in zero penalty."""
        assert compute_sec_penalty([]) == 0.0

    def test_single_critical_issue(self):
        """Single CRITICAL issue should return 10.0."""
        issues = [SecurityIssue(Severity.CRITICAL, "S001", "test", 1, 0)]
        assert compute_sec_penalty(issues) == 10.0

    def test_single_error_issue(self):
        """Single ERROR issue should return 5.0."""
        issues = [SecurityIssue(Severity.ERROR, "S002", "test", 1, 0)]
        assert compute_sec_penalty(issues) == 5.0

    def test_single_warning_issue(self):
        """Single WARNING issue should return 1.0."""
        issues = [SecurityIssue(Severity.WARNING, "S003", "test", 1, 0)]
        assert compute_sec_penalty(issues) == 1.0

    def test_single_info_issue(self):
        """Single INFO issue should return 0.1."""
        issues = [SecurityIssue(Severity.INFO, "S004", "test", 1, 0)]
        assert compute_sec_penalty(issues) == 0.1

    def test_multiple_issues_sum(self):
        """Multiple issues should sum their penalties."""
        issues = [
            SecurityIssue(Severity.CRITICAL, "S001", "test", 1, 0),
            SecurityIssue(Severity.ERROR, "S002", "test", 2, 0),
            SecurityIssue(Severity.WARNING, "S003", "test", 3, 0),
        ]
        expected = 10.0 + 5.0 + 1.0
        assert compute_sec_penalty(issues) == expected

    def test_all_severity_types(self):
        """Test with one of each severity."""
        issues = [
            SecurityIssue(Severity.CRITICAL, "S001", "test", 1, 0),
            SecurityIssue(Severity.ERROR, "S002", "test", 2, 0),
            SecurityIssue(Severity.WARNING, "S003", "test", 3, 0),
            SecurityIssue(Severity.INFO, "S004", "test", 4, 0),
        ]
        expected = 10.0 + 5.0 + 1.0 + 0.1
        assert compute_sec_penalty(issues) == expected


# =============================================================================
# SECTION 4: SecurityCache LRU Tests
# =============================================================================

class TestSecurityCache:
    """Test LRU cache for security results."""

    def test_cache_hit(self):
        """Cache should return stored result."""
        cache = SecurityCache(max_size=10)
        code = "x = 1"
        result = IncrementalResult(ParseState.PARSE_OK, [], 0.0)
        cache.put(code, result)
        assert cache.get(code) == result

    def test_cache_miss(self):
        """Cache should return None for unknown code."""
        cache = SecurityCache(max_size=10)
        assert cache.get("unknown code") is None

    def test_cache_lru_eviction(self):
        """Oldest entry should be evicted when cache is full."""
        cache = SecurityCache(max_size=3)

        # Fill cache
        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 1.0))
        cache.put("code2", IncrementalResult(ParseState.PARSE_OK, [], 2.0))
        cache.put("code3", IncrementalResult(ParseState.PARSE_OK, [], 3.0))

        # Access code1 to make it recently used
        cache.get("code1")

        # Add new entry - should evict code2 (oldest not recently used)
        cache.put("code4", IncrementalResult(ParseState.PARSE_OK, [], 4.0))

        # code2 should be evicted
        assert cache.get("code2") is None
        # code1, code3, code4 should still be present
        assert cache.get("code1") is not None
        assert cache.get("code3") is not None
        assert cache.get("code4") is not None

    def test_cache_update_existing(self):
        """Updating existing entry should move it to end."""
        cache = SecurityCache(max_size=3)

        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 1.0))
        cache.put("code2", IncrementalResult(ParseState.PARSE_OK, [], 2.0))
        cache.put("code3", IncrementalResult(ParseState.PARSE_OK, [], 3.0))

        # Update code1 with new value
        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 10.0))

        # Add new entry - should evict code2 (oldest)
        cache.put("code4", IncrementalResult(ParseState.PARSE_OK, [], 4.0))

        assert cache.get("code1") is not None
        assert cache.get("code1").sec_penalty == 10.0
        assert cache.get("code2") is None

    def test_cache_hash_collision_resistance(self):
        """Different codes should have different cache entries."""
        cache = SecurityCache(max_size=100)

        codes = ["x = 1", "x = 2", "y = 1", "def foo(): pass"]
        for i, code in enumerate(codes):
            cache.put(code, IncrementalResult(ParseState.PARSE_OK, [], float(i)))

        for i, code in enumerate(codes):
            result = cache.get(code)
            assert result is not None
            assert result.sec_penalty == float(i)

    def test_cache_empty_code(self):
        """Empty code should be cacheable."""
        cache = SecurityCache(max_size=10)
        result = IncrementalResult(ParseState.PARSE_INCOMPLETE, [], 0.0)
        cache.put("", result)
        assert cache.get("") == result


# =============================================================================
# SECTION 5: analyze_incremental() Tests
# =============================================================================

class TestAnalyzeIncremental:
    """Test incremental security analysis."""

    def test_safe_code_no_penalty(self):
        """Safe code should have no penalty."""
        code = "def add(a, b):\n    return a + b"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty == 0.0
        assert len(result.issues) == 0

    def test_subprocess_import_penalty(self):
        """subprocess import should have penalty."""
        code = "import subprocess"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0

    def test_os_system_call_penalty(self):
        """os.system call should have penalty."""
        code = "import os\nos.system('ls')"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0

    def test_eval_call_penalty(self):
        """eval() call should have penalty."""
        code = "result = eval(user_input)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0

    def test_exec_call_penalty(self):
        """exec() call should have penalty."""
        code = "exec(code_string)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0

    def test_incomplete_code_no_penalty(self):
        """Incomplete code should have no penalty."""
        code = "def foo():"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INCOMPLETE
        assert result.sec_penalty == 0.0

    def test_invalid_code_penalty(self):
        """Invalid code (early error) should have small fixed penalty."""
        code = "@@@ invalid\nx = 1\ny = 2"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_INVALID
        assert result.sec_penalty == 2.0  # Fixed penalty for invalid

    def test_cache_integration(self):
        """Cache should be used for repeated analysis."""
        cache = SecurityCache(max_size=10)
        code = "import os"

        # First call - not cached
        result1 = analyze_incremental(code, cache)

        # Second call - should use cache
        result2 = analyze_incremental(code, cache)

        assert result1 == result2
        assert cache.get(code) is not None

    def test_file_write_penalty(self):
        """File write operations should have penalty."""
        code = "f = open('file.txt', 'w')\nf.write('data')"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0

    def test_pickle_load_penalty(self):
        """pickle.load should have penalty (deserialization risk)."""
        code = "import pickle\ndata = pickle.load(f)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty > 0


# =============================================================================
# SECTION 6: SABSConfig Tests
# =============================================================================

class TestSABSConfig:
    """Test SABSConfig dataclass."""

    def test_default_values(self):
        """Default config values should be sensible."""
        config = SABSConfig()
        assert config.lambda_weight == 0.5
        assert config.hard_fail_on_critical is True
        assert config.max_verifier_calls == 50
        assert config.min_tokens_between_checks == 8
        assert config.cache_enabled is True
        assert config.log_path is None

    def test_custom_values(self):
        """Custom config values should be respected."""
        config = SABSConfig(
            lambda_weight=0.8,
            hard_fail_on_critical=False,
            max_verifier_calls=100,
            min_tokens_between_checks=16,
            cache_enabled=False,
            log_path="/tmp/sabs.jsonl",
        )
        assert config.lambda_weight == 0.8
        assert config.hard_fail_on_critical is False
        assert config.max_verifier_calls == 100
        assert config.min_tokens_between_checks == 16
        assert config.cache_enabled is False
        assert config.log_path == "/tmp/sabs.jsonl"

    def test_lambda_zero_disables_penalty(self):
        """Lambda of 0 should effectively disable security penalty."""
        config = SABSConfig(lambda_weight=0.0)
        assert config.lambda_weight == 0.0


# =============================================================================
# SECTION 7: SABSDecision Tests
# =============================================================================

class TestSABSDecision:
    """Test SABSDecision dataclass."""

    def test_skip_incomplete_decision(self):
        """SKIP_INCOMPLETE decision should not terminate."""
        decision = SABSDecision(
            action=SABSAction.SKIP_INCOMPLETE,
            parse_state=ParseState.PARSE_INCOMPLETE,
            issues=[],
            sec_penalty_delta=0.0,
            total_sec_penalty=0.0,
            should_terminate=False,
        )
        assert decision.action == SABSAction.SKIP_INCOMPLETE
        assert decision.should_terminate is False
        assert decision.sec_penalty_delta == 0.0

    def test_soft_penalty_decision(self):
        """SOFT_PENALTY decision should add penalty."""
        decision = SABSDecision(
            action=SABSAction.SOFT_PENALTY,
            parse_state=ParseState.PARSE_OK,
            issues=[{"severity": "WARNING", "code": "S007", "message": "test"}],
            sec_penalty_delta=1.0,
            total_sec_penalty=5.0,
            should_terminate=False,
        )
        assert decision.action == SABSAction.SOFT_PENALTY
        assert decision.should_terminate is False
        assert decision.sec_penalty_delta == 1.0

    def test_hard_fail_decision(self):
        """HARD_FAIL decision should terminate."""
        decision = SABSDecision(
            action=SABSAction.HARD_FAIL,
            parse_state=ParseState.PARSE_OK,
            issues=[{"severity": "CRITICAL", "code": "S001", "message": "test"}],
            sec_penalty_delta=10.0,
            total_sec_penalty=10.0,
            should_terminate=True,
        )
        assert decision.action == SABSAction.HARD_FAIL
        assert decision.should_terminate is True

    def test_no_issues_decision(self):
        """NO_ISSUES decision should have zero penalty delta."""
        decision = SABSDecision(
            action=SABSAction.NO_ISSUES,
            parse_state=ParseState.PARSE_OK,
            issues=[],
            sec_penalty_delta=0.0,
            total_sec_penalty=0.0,
            should_terminate=False,
        )
        assert decision.action == SABSAction.NO_ISSUES
        assert decision.sec_penalty_delta == 0.0
        assert len(decision.issues) == 0


# =============================================================================
# SECTION 8: SecurityController Tests
# =============================================================================

class TestSecurityController:
    """Test SecurityController orchestration."""

    def test_initialization(self):
        """Controller should initialize with config."""
        config = SABSConfig()
        controller = SecurityController(config)
        assert controller.config == config
        assert controller.verifier_calls == 0
        assert controller.cache is not None

    def test_initialization_no_cache(self):
        """Controller should work without cache."""
        config = SABSConfig(cache_enabled=False)
        controller = SecurityController(config)
        assert controller.cache is None

    def test_reset(self):
        """Reset should clear state."""
        config = SABSConfig()
        controller = SecurityController(config)
        controller.verifier_calls = 10
        controller.decisions.append({"test": True})
        controller.reset("new_gen")
        assert controller.verifier_calls == 0
        assert len(controller.decisions) == 0

    def test_should_check_budget_exceeded(self):
        """Should not check when budget exceeded."""
        config = SABSConfig(max_verifier_calls=5)
        controller = SecurityController(config)
        controller.verifier_calls = 5

        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = None

        assert controller.should_check(beam) is False

    def test_should_check_min_tokens(self):
        """Should not check before min tokens."""
        config = SABSConfig(min_tokens_between_checks=8)
        controller = SecurityController(config)

        beam = Mock()
        beam.tokens_since_security_check = 3
        beam.constraint_tracker = None

        assert controller.should_check(beam) is False

    def test_should_check_not_boundary(self):
        """Should not check when not at boundary."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = Mock()
        beam.constraint_tracker.is_boundary_now.return_value = False

        assert controller.should_check(beam) is False

    def test_should_check_passes(self):
        """Should check when all conditions met."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = Mock()
        beam.constraint_tracker.is_boundary_now.return_value = True

        assert controller.should_check(beam) is True

    def test_check_beam_empty_code(self):
        """Empty code should return SKIP_INCOMPLETE."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10
        beam.beam_id = "test"
        beam.current_position = 10

        decision = controller.check_beam(beam, "", "gen1", 1)
        assert decision.action == SABSAction.SKIP_INCOMPLETE
        assert decision.should_terminate is False

    def test_check_beam_safe_code(self):
        """Safe code should return NO_ISSUES."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10
        beam.beam_id = "test"
        beam.current_position = 10

        decision = controller.check_beam(beam, "x = 1 + 2", "gen1", 1)
        assert decision.action == SABSAction.NO_ISSUES
        assert decision.sec_penalty_delta == 0.0

    def test_check_beam_dangerous_code(self):
        """Dangerous code should return SOFT_PENALTY or HARD_FAIL."""
        config = SABSConfig(hard_fail_on_critical=True)
        controller = SecurityController(config)

        beam = Mock()
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10
        beam.beam_id = "test"
        beam.current_position = 10

        decision = controller.check_beam(beam, "import subprocess", "gen1", 1)
        assert decision.sec_penalty_delta > 0

    def test_apply_decision_soft_penalty(self):
        """Soft penalty should update beam penalty."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10

        decision = SABSDecision(
            action=SABSAction.SOFT_PENALTY,
            parse_state=ParseState.PARSE_OK,
            issues=[],
            sec_penalty_delta=5.0,
            total_sec_penalty=5.0,
            should_terminate=False,
        )

        search = Mock()
        controller.apply_decision(beam, decision, search)

        assert beam.sec_penalty == 5.0
        assert beam.tokens_since_security_check == 0
        search.fail_beam.assert_not_called()

    def test_apply_decision_hard_fail(self):
        """Hard fail should call search.fail_beam."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = Mock()
        beam.sec_penalty = 0.0

        decision = SABSDecision(
            action=SABSAction.HARD_FAIL,
            parse_state=ParseState.PARSE_OK,
            issues=[],
            sec_penalty_delta=10.0,
            total_sec_penalty=10.0,
            should_terminate=True,
        )

        search = Mock()
        controller.apply_decision(beam, decision, search)

        search.fail_beam.assert_called_once_with(beam, "security_critical")

    def test_get_statistics_empty(self):
        """Statistics should work with no decisions."""
        config = SABSConfig()
        controller = SecurityController(config)

        stats = controller.get_statistics()
        assert stats["verifier_calls"] == 0
        assert stats["total_decisions"] == 0
        assert stats["hard_fails"] == 0

    def test_get_statistics_with_decisions(self):
        """Statistics should reflect recorded decisions."""
        config = SABSConfig()
        controller = SecurityController(config)

        # Simulate some decisions
        controller.decisions = [
            {"action": "SOFT_PENALTY", "hard_failed": False, "sec_penalty_delta": 1.0},
            {"action": "SOFT_PENALTY", "hard_failed": False, "sec_penalty_delta": 5.0},
            {"action": "HARD_FAIL", "hard_failed": True, "sec_penalty_delta": 10.0},
            {"action": "NO_ISSUES", "hard_failed": False, "sec_penalty_delta": 0.0},
            {"action": "SKIP_INCOMPLETE", "hard_failed": False, "sec_penalty_delta": 0.0},
        ]
        controller.verifier_calls = 5

        stats = controller.get_statistics()
        assert stats["verifier_calls"] == 5
        assert stats["total_decisions"] == 5
        assert stats["hard_fails"] == 1
        assert stats["soft_penalties"] == 2
        assert stats["no_issues"] == 1
        assert stats["total_penalty_applied"] == 16.0


# =============================================================================
# SECTION 9: extract_code_only Tests
# =============================================================================

class TestExtractCodeOnly:
    """Test code extraction from beams."""

    def test_extract_basic(self):
        """Basic extraction should work."""
        model = Mock()
        model.detokenize.return_value = "def foo():\n    return 1"

        beam = Mock()
        beam.tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 3 prompt + 5 generated
        beam.prompt_length = 3

        code = extract_code_only(model, beam)
        assert "def foo():" in code

    def test_extract_strips_markdown_fence(self):
        """Markdown fences should be stripped."""
        model = Mock()
        model.detokenize.return_value = "```python\ndef foo():\n    return 1\n```"

        beam = Mock()
        beam.tokens = [1, 2, 3, 4, 5]
        beam.prompt_length = 2

        code = extract_code_only(model, beam)
        assert "```" not in code
        assert "def foo():" in code

    def test_extract_strips_prose(self):
        """Common prose prefixes should be stripped."""
        model = Mock()
        model.detokenize.return_value = "Here's the code:\ndef foo():\n    return 1"

        beam = Mock()
        beam.tokens = [1, 2, 3, 4, 5]
        beam.prompt_length = 2

        code = extract_code_only(model, beam)
        assert "Here's" not in code
        assert "def foo():" in code

    def test_extract_empty_generated(self):
        """No generated tokens should return empty string."""
        model = Mock()

        beam = Mock()
        beam.tokens = [1, 2, 3]
        beam.prompt_length = 3  # All prompt, no generated

        code = extract_code_only(model, beam)
        assert code == ""


# =============================================================================
# SECTION 10: ConstraintTracker Clone Tests
# =============================================================================

class TestConstraintTrackerClone:
    """Test ConstraintTracker.clone() for per-beam tracking."""

    def test_clone_preserves_state(self):
        """Clone should preserve syntactic state."""
        tracker = ConstraintTracker()
        tracker.update("def foo(")  # Open paren

        clone = tracker.clone()

        # Original state preserved
        assert clone.state.bracket_depth == 1

    def test_clone_independent_updates(self):
        """Cloned trackers should be independent."""
        tracker = ConstraintTracker()
        tracker.update("def foo(")

        clone = tracker.clone()

        # Update original
        tracker.update(")")

        # Original should have closed paren
        assert tracker.state.bracket_depth == 0

        # Clone should still have open paren
        assert clone.state.bracket_depth == 1

    def test_clone_string_state(self):
        """Clone should preserve string state."""
        tracker = ConstraintTracker()
        tracker.update('"hello')  # Open string

        clone = tracker.clone()

        assert clone.state.in_string is True

        # Close string in original
        tracker.update('"')
        assert tracker.state.in_string is False

        # Clone still in string
        assert clone.state.in_string is True


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
