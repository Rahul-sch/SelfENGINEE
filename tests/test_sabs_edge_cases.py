"""
Edge case tests for SABS (Security-Aware Beam Search).

Tests cover:
1. Unicode and encoding edge cases
2. Very long code strings
3. Cache boundary conditions
4. Empty/null inputs
5. Malformed syntax edge cases
6. Lambda extreme values (0, 1, very high)
7. Budget exhaustion scenarios
8. Constraint tracker edge cases

Run with: python -m pytest tests/test_sabs_edge_cases.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.search import Beam, BeamManager, SearchConfig, BeamStatus
from engine.constraints import ConstraintTracker, StringState
from engine.security_controller import (
    SABSConfig, SABSAction, SABSDecision, SecurityController, extract_code_only
)
from engine.verify_static import (
    ParseState, SEVERITY_WEIGHTS, SecurityCache, IncrementalResult,
    analyze_incremental, is_error_near_end, classify_parse_error,
    Severity, SecurityIssue, compute_sec_penalty
)


# =============================================================================
# SECTION 1: Unicode and Encoding Edge Cases
# =============================================================================

class TestUnicodeEdgeCases:
    """Test handling of unicode and special characters."""

    def test_unicode_identifiers(self):
        """Python 3 allows unicode identifiers."""
        code = "变量 = 42\nprint(变量)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_unicode_strings(self):
        """Unicode in string literals should work."""
        code = "s = '你好世界'\nprint(s)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_emoji_in_strings(self):
        """Emoji in strings should work."""
        code = "msg = '🎉 Success! 🎊'\nprint(msg)"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_mixed_encodings(self):
        """Mixed unicode should work."""
        code = "# Comment with émojis 🐍\ndef café():\n    return 'Ñoño'"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_null_bytes(self):
        """Null bytes should be handled gracefully."""
        code = "x = 1\x00\ny = 2"
        # Should either parse or be treated as incomplete/invalid
        result = analyze_incremental(code)
        assert result.parse_state in (ParseState.PARSE_OK, ParseState.PARSE_INCOMPLETE, ParseState.PARSE_INVALID)

    def test_cache_unicode_key(self):
        """Cache should handle unicode code as key."""
        cache = SecurityCache(max_size=10)
        code = "变量 = '你好'"
        result = IncrementalResult(ParseState.PARSE_OK, [], 0.0)
        cache.put(code, result)
        assert cache.get(code) == result


# =============================================================================
# SECTION 2: Very Long Code Strings
# =============================================================================

class TestLongCodeStrings:
    """Test handling of very long code."""

    def test_very_long_function(self):
        """Long function with many lines should work."""
        lines = ["def long_func():"]
        for i in range(100):
            lines.append(f"    x{i} = {i}")
        lines.append("    return x99")
        code = "\n".join(lines)

        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_long_string_literal(self):
        """Very long string literal should work."""
        code = f"s = '{'a' * 10000}'"
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_deeply_nested_brackets(self):
        """Deeply nested brackets should work."""
        depth = 50
        code = "x = " + "[" * depth + "1" + "]" * depth
        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK

    def test_many_dangerous_calls(self):
        """Many dangerous calls should accumulate penalties."""
        lines = ["import subprocess"]
        for i in range(5):
            lines.append(f"subprocess.call('cmd{i}')")
        code = "\n".join(lines)

        result = analyze_incremental(code)
        assert result.parse_state == ParseState.PARSE_OK
        # Multiple dangerous calls should accumulate
        assert result.sec_penalty > 0

    def test_cache_long_code(self):
        """Cache should handle long code."""
        cache = SecurityCache(max_size=10)
        long_code = "x = " + "1 + " * 1000 + "1"
        result = IncrementalResult(ParseState.PARSE_OK, [], 0.0)
        cache.put(long_code, result)
        assert cache.get(long_code) == result


# =============================================================================
# SECTION 3: Cache Boundary Conditions
# =============================================================================

class TestCacheBoundaryConditions:
    """Test cache edge cases."""

    def test_cache_size_one(self):
        """Cache with size 1 should work."""
        cache = SecurityCache(max_size=1)
        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 1.0))
        cache.put("code2", IncrementalResult(ParseState.PARSE_OK, [], 2.0))

        assert cache.get("code1") is None  # Evicted
        assert cache.get("code2") is not None

    def test_cache_exact_capacity(self):
        """Filling cache to exact capacity should work."""
        cache = SecurityCache(max_size=5)
        for i in range(5):
            cache.put(f"code{i}", IncrementalResult(ParseState.PARSE_OK, [], float(i)))

        for i in range(5):
            assert cache.get(f"code{i}") is not None

    def test_cache_overflow(self):
        """Cache overflow should evict oldest."""
        cache = SecurityCache(max_size=3)
        for i in range(5):
            cache.put(f"code{i}", IncrementalResult(ParseState.PARSE_OK, [], float(i)))

        # First 2 should be evicted
        assert cache.get("code0") is None
        assert cache.get("code1") is None
        # Last 3 should remain
        assert cache.get("code2") is not None
        assert cache.get("code3") is not None
        assert cache.get("code4") is not None

    def test_cache_stats(self):
        """Cache statistics should be accurate."""
        cache = SecurityCache(max_size=10)

        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 1.0))
        cache.get("code1")  # Hit
        cache.get("code2")  # Miss
        cache.get("code1")  # Hit

        stats = cache.stats
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['size'] == 1

    def test_cache_clear(self):
        """Cache clear should empty cache."""
        cache = SecurityCache(max_size=10)
        cache.put("code1", IncrementalResult(ParseState.PARSE_OK, [], 1.0))
        cache.put("code2", IncrementalResult(ParseState.PARSE_OK, [], 2.0))

        cache.clear()

        assert cache.get("code1") is None
        assert cache.get("code2") is None
        assert cache.stats['size'] == 0


# =============================================================================
# SECTION 4: Empty/Null Inputs
# =============================================================================

class TestEmptyNullInputs:
    """Test handling of empty and edge case inputs."""

    def test_empty_string(self):
        """Empty string should be PARSE_INCOMPLETE."""
        result = analyze_incremental("")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE
        assert result.sec_penalty == 0.0

    def test_whitespace_only(self):
        """Whitespace-only should be PARSE_INCOMPLETE."""
        result = analyze_incremental("   \n\t\n   ")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_single_newline(self):
        """Single newline should be PARSE_INCOMPLETE."""
        result = analyze_incremental("\n")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_comment_only(self):
        """Comment-only code should parse OK."""
        result = analyze_incremental("# This is a comment")
        assert result.parse_state == ParseState.PARSE_OK
        assert result.sec_penalty == 0.0

    def test_docstring_only(self):
        """Docstring-only should parse OK."""
        result = analyze_incremental('"""This is a docstring."""')
        assert result.parse_state == ParseState.PARSE_OK

    def test_extract_code_empty_beam(self):
        """Extract from beam with no generated tokens."""
        model = Mock()
        beam = Beam(tokens=[1, 2, 3], prompt_length=3)
        code = extract_code_only(model, beam)
        assert code == ""


# =============================================================================
# SECTION 5: Malformed Syntax Edge Cases
# =============================================================================

class TestMalformedSyntax:
    """Test handling of malformed syntax."""

    def test_single_operator(self):
        """Single operator should be incomplete."""
        result = analyze_incremental("+")
        assert result.parse_state in (ParseState.PARSE_INCOMPLETE, ParseState.PARSE_INVALID)

    def test_unmatched_quote(self):
        """Unmatched quote should be incomplete."""
        result = analyze_incremental("x = 'hello")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_unmatched_triple_quote(self):
        """Unmatched triple quote should be incomplete."""
        result = analyze_incremental('x = """hello')
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_double_colon(self):
        """Double colon (invalid syntax) should be handled."""
        result = analyze_incremental("if True::")
        assert result.parse_state in (ParseState.PARSE_INCOMPLETE, ParseState.PARSE_INVALID)

    def test_assignment_without_value(self):
        """Assignment without value should be incomplete."""
        result = analyze_incremental("x =")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_function_without_parens(self):
        """Function without parens should be incomplete."""
        result = analyze_incremental("def foo")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_class_without_colon(self):
        """Class without colon should be incomplete."""
        result = analyze_incremental("class Foo")
        assert result.parse_state == ParseState.PARSE_INCOMPLETE

    def test_indentation_error(self):
        """Indentation error should be handled."""
        code = "def foo():\nreturn 1"  # Missing indent
        result = analyze_incremental(code)
        # Could be incomplete or invalid depending on position
        assert result.parse_state in (ParseState.PARSE_INCOMPLETE, ParseState.PARSE_INVALID)

    def test_mixed_tabs_spaces(self):
        """Mixed tabs and spaces might cause issues."""
        code = "def foo():\n\treturn 1\n    return 2"
        result = analyze_incremental(code)
        # Python 3 is strict about mixing tabs/spaces
        assert result.parse_state in (ParseState.PARSE_OK, ParseState.PARSE_INCOMPLETE, ParseState.PARSE_INVALID)


# =============================================================================
# SECTION 6: Lambda Extreme Values
# =============================================================================

class TestLambdaExtremeValues:
    """Test security_lambda at extreme values."""

    def test_lambda_zero(self):
        """Lambda=0 should disable security penalty effect."""
        config = SearchConfig(security_lambda=0.0)
        beam = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-5.0)
        beam.sec_penalty = 1000.0  # Very high penalty

        score = beam.normalized_score(config)

        # Penalty should have no effect
        beam_no_penalty = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-5.0)
        beam_no_penalty.sec_penalty = 0.0
        score_no_penalty = beam_no_penalty.normalized_score(config)

        assert score == score_no_penalty

    def test_lambda_one(self):
        """Lambda=1 should apply full penalty."""
        config = SearchConfig(security_lambda=1.0)
        beam = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-5.0)
        beam.sec_penalty = 10.0

        score = beam.normalized_score(config)

        # Score should be reduced by exactly 10
        beam_no_penalty = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-5.0)
        beam_no_penalty.sec_penalty = 0.0
        score_no_penalty = beam_no_penalty.normalized_score(config)

        assert score == score_no_penalty - 10.0

    def test_lambda_very_high(self):
        """Very high lambda should make security dominant."""
        config = SearchConfig(security_lambda=100.0)

        beam_unsafe = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=0.0)  # Perfect log_prob
        beam_unsafe.sec_penalty = 1.0  # Small penalty

        beam_safe = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-10.0)  # Bad log_prob
        beam_safe.sec_penalty = 0.0  # No penalty

        # With high lambda, even small penalty makes score much worse
        assert beam_safe.normalized_score(config) > beam_unsafe.normalized_score(config)

    def test_lambda_negative_handled(self):
        """Negative lambda (unusual) should still compute."""
        config = SearchConfig(security_lambda=-0.5)
        beam = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-5.0)
        beam.sec_penalty = 10.0

        # Should compute without error (even though semantically odd)
        score = beam.normalized_score(config)
        assert isinstance(score, float)


# =============================================================================
# SECTION 7: Budget Exhaustion Scenarios
# =============================================================================

class TestBudgetExhaustion:
    """Test verifier budget exhaustion scenarios."""

    def test_budget_zero(self):
        """Budget=0 should never allow checks."""
        config = SABSConfig(max_verifier_calls=0)
        controller = SecurityController(config)

        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = None

        assert controller.should_check(beam) is False

    def test_budget_one(self):
        """Budget=1 should allow exactly one check."""
        config = SABSConfig(max_verifier_calls=1, min_tokens_between_checks=0)
        controller = SecurityController(config)

        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = None
        beam.sec_penalty = 0.0
        beam.beam_id = "test"
        beam.current_position = 10

        # First check allowed
        assert controller.should_check(beam) is True
        controller.check_beam(beam, "x = 1", "gen1", 1)

        # Second check blocked
        assert controller.should_check(beam) is False

    def test_budget_with_many_beams(self):
        """Budget should be shared across all beams."""
        config = SABSConfig(max_verifier_calls=3, min_tokens_between_checks=0)
        controller = SecurityController(config)

        for i in range(3):
            beam = Mock()
            beam.tokens_since_security_check = 100
            beam.constraint_tracker = None
            beam.sec_penalty = 0.0
            beam.beam_id = f"beam{i}"
            beam.current_position = 10

            assert controller.should_check(beam) is True
            controller.check_beam(beam, f"x{i} = {i}", "gen1", i)

        # 4th check for any beam should fail
        beam = Mock()
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = None
        assert controller.should_check(beam) is False


# =============================================================================
# SECTION 8: Constraint Tracker Edge Cases
# =============================================================================

class TestConstraintTrackerEdgeCases:
    """Test ConstraintTracker edge cases."""

    def test_empty_update(self):
        """Empty string update should be no-op."""
        tracker = ConstraintTracker()
        boundaries = tracker.update("")
        assert boundaries == []
        assert tracker.state.bracket_depth == 0

    def test_many_nested_brackets(self):
        """Many nested brackets should track correctly."""
        tracker = ConstraintTracker()
        # 5 parens + 4 brackets + 4 braces = 13
        tracker.update("((((([[[[{{{{")

        assert tracker.state.bracket_depth == 13

        # Close in reverse: 4 braces + 4 brackets + 5 parens = 13
        tracker.update("}}}}")
        assert tracker.state.bracket_depth == 9
        tracker.update("]]]]")
        assert tracker.state.bracket_depth == 5
        tracker.update(")))))")
        assert tracker.state.bracket_depth == 0

    def test_escaped_quote_in_string(self):
        """Escaped quotes should not end string."""
        tracker = ConstraintTracker()
        tracker.update("x = 'hello\\'world'")

        assert not tracker.state.in_string
        assert tracker.state.bracket_depth == 0

    def test_hash_in_string(self):
        """Hash in string should not start comment."""
        tracker = ConstraintTracker()
        tracker.update("x = 'hello # world'")

        assert not tracker.state.in_comment
        assert not tracker.state.in_string

    def test_quote_in_comment(self):
        """Quote in comment should not start string."""
        tracker = ConstraintTracker()
        tracker.update("# This is a 'comment'\nx = 1")

        assert not tracker.state.in_string
        assert not tracker.state.in_comment  # Comment ends at newline

    def test_triple_quote_boundary(self):
        """Triple quote should be tracked correctly."""
        tracker = ConstraintTracker()
        tracker.update('x = """hello')

        assert tracker.state.in_triple_string

        tracker.update(' world"""')

        assert not tracker.state.in_string

    def test_fstring_basic(self):
        """Basic f-string should work."""
        tracker = ConstraintTracker()
        tracker.update('x = f"hello {name}"')

        assert not tracker.state.in_string
        assert tracker.state.bracket_depth == 0

    def test_clone_deep_state(self):
        """Clone should preserve deep nested state."""
        tracker = ConstraintTracker()
        tracker.update('def foo(x, y, z={"a": [1,')

        clone = tracker.clone()

        # Original state
        assert tracker.state.bracket_depth == 3  # (, {, [
        assert clone.state.bracket_depth == 3

        # Modify original
        tracker.update("2]}")
        assert tracker.state.bracket_depth == 1  # Still in (

        # Clone should be unchanged
        assert clone.state.bracket_depth == 3

    def test_boundary_at_keyword(self):
        """Boundary should trigger at keyword start."""
        tracker = ConstraintTracker()
        boundaries = tracker.update("x = 1\ndef foo():")

        # Should have boundary after newline and after 'def'
        assert len(boundaries) >= 1

    def test_no_boundary_in_string(self):
        """No boundary should trigger inside string."""
        tracker = ConstraintTracker()
        boundaries = tracker.update("x = 'if True:\\ndef foo():'")

        # The newline and keywords are inside string
        assert len(boundaries) == 0


# =============================================================================
# SECTION 9: Security Issue Edge Cases
# =============================================================================

class TestSecurityIssueEdgeCases:
    """Test security detection edge cases."""

    def test_import_in_string(self):
        """Import in string should not trigger warning."""
        code = "x = 'import subprocess'"
        result = analyze_incremental(code)
        assert result.sec_penalty == 0.0

    def test_eval_as_variable_name(self):
        """'eval' as variable should be fine (it shadows, not calls)."""
        code = "eval = lambda x: x"
        result = analyze_incremental(code)
        # This shadows eval, which could be considered safe or risky
        # Current implementation may or may not flag this
        assert result.parse_state == ParseState.PARSE_OK

    def test_exec_in_comment(self):
        """exec in comment should not trigger."""
        code = "# exec('dangerous')\nx = 1"
        result = analyze_incremental(code)
        assert result.sec_penalty == 0.0

    def test_nested_eval_call(self):
        """Nested eval call should trigger."""
        code = "result = eval(eval(x))"
        result = analyze_incremental(code)
        assert result.sec_penalty > 0

    def test_method_named_eval(self):
        """Method named eval on object should not trigger false positive."""
        code = "class Foo:\n    def eval(self, x):\n        return x"
        result = analyze_incremental(code)
        # Method definition is safe
        assert result.parse_state == ParseState.PARSE_OK

    def test_pathlib_write(self):
        """Pathlib write methods should trigger."""
        code = "from pathlib import Path\nPath('x.txt').write_text('hello')"
        result = analyze_incremental(code)
        assert result.sec_penalty > 0


# =============================================================================
# SECTION 10: Beam Edge Cases
# =============================================================================

class TestBeamEdgeCases:
    """Test Beam edge cases."""

    def test_beam_zero_generated(self):
        """Beam with zero generated tokens."""
        beam = Beam(tokens=[1, 2, 3], prompt_length=3)
        assert beam.num_generated == 0
        assert beam.generated_tokens == []

    def test_beam_clone_no_tracker(self):
        """Clone without constraint tracker should work."""
        beam = Beam(tokens=[1, 2, 3], prompt_length=1)
        beam.constraint_tracker = None

        clone = beam.clone()

        assert clone.constraint_tracker is None

    def test_beam_normalized_score_zero_generated(self):
        """Score calculation with zero generated tokens."""
        config = SearchConfig(security_lambda=0.5)
        beam = Beam(tokens=[1, 2, 3], prompt_length=3, log_prob=-5.0)
        beam.sec_penalty = 1.0

        score = beam.normalized_score(config)
        assert isinstance(score, float)

    def test_beam_very_negative_log_prob(self):
        """Very negative log_prob should work."""
        config = SearchConfig(security_lambda=0.5)
        beam = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=-1000.0)
        beam.sec_penalty = 0.0

        score = beam.normalized_score(config)
        assert score < -100  # Still negative

    def test_beam_positive_log_prob(self):
        """Positive log_prob (unusual but possible) should work."""
        config = SearchConfig(security_lambda=0.5)
        beam = Beam(tokens=[1, 2, 3, 4, 5], prompt_length=2, log_prob=10.0)
        beam.sec_penalty = 0.0

        score = beam.normalized_score(config)
        assert score > 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
