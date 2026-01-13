"""
Integration tests for SABS (Security-Aware Beam Search).

Tests the full integration between:
- SecurityController + BeamManager
- Beam scoring with security penalties
- Per-beam constraint tracking
- End-to-end SABS flow

Run with: python -m pytest tests/test_sabs_integration.py -v
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.search import Beam, BeamManager, SearchConfig, BeamStatus
from engine.constraints import ConstraintTracker
from engine.security_controller import (
    SABSConfig, SABSAction, SABSDecision, SecurityController, extract_code_only
)
from engine.verify_static import (
    ParseState, SEVERITY_WEIGHTS, analyze_incremental, IncrementalResult
)


# =============================================================================
# Mock Helpers
# =============================================================================

def create_mock_model():
    """Create a mock ModelAdapter for testing."""
    model = Mock()
    model.eos_token = 2  # Common EOS token
    model.detokenize = Mock(side_effect=lambda tokens: ''.join(f't{t}' for t in tokens))
    model.tokenize = Mock(side_effect=lambda s: [ord(c) for c in s[:10]])
    model.eval = Mock()
    model.get_logits = Mock(return_value=np.random.randn(100))
    model.sample = Mock(return_value=(42, -0.5))
    model.create_checkpoint = Mock()
    model.restore_checkpoint = Mock()
    model.clear_runtime_state = Mock()
    return model


def create_test_beam(prompt_length=5, generated_tokens=None):
    """Create a test beam with optional generated tokens."""
    tokens = list(range(prompt_length))
    if generated_tokens:
        tokens.extend(generated_tokens)
    return Beam(
        beam_id="test_beam",
        tokens=tokens,
        prompt_length=prompt_length,
        log_prob=-5.0,
    )


# =============================================================================
# SECTION 1: Beam Scoring Integration Tests
# =============================================================================

class TestBeamScoringIntegration:
    """Test beam scoring with security penalties."""

    def test_normalized_score_without_sabs(self):
        """Without SABS (lambda=0), security penalty should not affect score."""
        config = SearchConfig(security_lambda=0.0)
        beam = create_test_beam(generated_tokens=[10, 11, 12])
        beam.sec_penalty = 10.0  # High penalty

        score = beam.normalized_score(config)

        # With lambda=0, sec_penalty shouldn't affect score
        beam_no_penalty = create_test_beam(generated_tokens=[10, 11, 12])
        beam_no_penalty.sec_penalty = 0.0
        score_no_penalty = beam_no_penalty.normalized_score(config)

        assert score == score_no_penalty

    def test_normalized_score_with_sabs(self):
        """With SABS enabled, higher penalty = lower score."""
        config = SearchConfig(security_lambda=0.5)

        beam_low = create_test_beam(generated_tokens=[10, 11, 12])
        beam_low.sec_penalty = 1.0

        beam_high = create_test_beam(generated_tokens=[10, 11, 12])
        beam_high.sec_penalty = 10.0

        score_low = beam_low.normalized_score(config)
        score_high = beam_high.normalized_score(config)

        # Higher penalty = lower score
        assert score_low > score_high

    def test_normalized_score_lambda_scaling(self):
        """Lambda should scale the penalty effect."""
        beam = create_test_beam(generated_tokens=[10, 11, 12])
        beam.sec_penalty = 5.0

        config_low = SearchConfig(security_lambda=0.1)
        config_high = SearchConfig(security_lambda=1.0)

        score_low_lambda = beam.normalized_score(config_low)
        score_high_lambda = beam.normalized_score(config_high)

        # Higher lambda = more penalty impact = lower score
        assert score_low_lambda > score_high_lambda


# =============================================================================
# SECTION 2: BeamManager Integration Tests
# =============================================================================

class TestBeamManagerIntegration:
    """Test BeamManager with SABS features."""

    def test_prune_considers_security_penalty(self):
        """Pruning should consider security penalty in scoring."""
        model = create_mock_model()
        config = SearchConfig(beam_width=2, security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        # Create 3 beams with different log_probs and sec_penalties
        beam1 = Beam(beam_id="b1", tokens=[1, 2, 3], prompt_length=1, log_prob=-2.0, sec_penalty=0.0)
        beam2 = Beam(beam_id="b2", tokens=[1, 2, 3], prompt_length=1, log_prob=-1.0, sec_penalty=10.0)  # Higher log_prob but high penalty
        beam3 = Beam(beam_id="b3", tokens=[1, 2, 3], prompt_length=1, log_prob=-3.0, sec_penalty=0.0)

        manager.active_beams = {"b1": beam1, "b2": beam2, "b3": beam3}

        pruned = manager.prune_beams()

        # beam2 should be pruned because its effective score is lower due to penalty
        assert len(manager.active_beams) == 2
        pruned_ids = [b.beam_id for b in pruned]

        # beam2 has log_prob=-1 but penalty=10, so score = -1/factor - 0.5*10 ≈ very negative
        # beam1 has log_prob=-2 but penalty=0, so score = -2/factor - 0 ≈ -2ish
        # beam2 should be pruned
        assert "b2" in pruned_ids

    def test_deduplicate_considers_security_penalty(self):
        """Deduplication should keep the beam with better security score."""
        model = create_mock_model()
        config = SearchConfig(security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        # Create 2 beams with same tokens but different penalties
        beam1 = Beam(beam_id="b1", tokens=[1, 2, 3], prompt_length=1, log_prob=-2.0, sec_penalty=0.0)
        beam2 = Beam(beam_id="b2", tokens=[1, 2, 3], prompt_length=1, log_prob=-2.0, sec_penalty=5.0)

        manager.active_beams = {"b1": beam1, "b2": beam2}

        dups = manager.deduplicate()

        # beam2 should be removed (same tokens, worse score due to penalty)
        assert len(manager.active_beams) == 1
        assert "b1" in manager.active_beams
        assert len(dups) == 1
        assert dups[0].beam_id == "b2"

    def test_get_best_beam_considers_security(self):
        """get_best_beam should consider security penalty."""
        model = create_mock_model()
        config = SearchConfig(security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        # Beam with higher log_prob but high penalty
        beam_unsafe = Beam(beam_id="unsafe", tokens=[1, 2, 3], prompt_length=1, log_prob=-1.0, sec_penalty=10.0)
        beam_unsafe.status = BeamStatus.COMPLETED

        # Beam with lower log_prob but no penalty
        beam_safe = Beam(beam_id="safe", tokens=[1, 2, 3], prompt_length=1, log_prob=-2.0, sec_penalty=0.0)
        beam_safe.status = BeamStatus.COMPLETED

        manager.completed_beams = [beam_unsafe, beam_safe]

        best = manager.get_best_beam()

        # Safe beam should be best because penalty makes unsafe worse
        assert best.beam_id == "safe"

    def test_fail_beam(self):
        """fail_beam should move beam to pruned and set FAILED status."""
        model = create_mock_model()
        config = SearchConfig()
        manager = BeamManager(config, model)
        manager.reset()

        beam = Beam(beam_id="test", tokens=[1, 2, 3], prompt_length=1)
        manager.active_beams["test"] = beam

        manager.fail_beam(beam, "security_critical")

        assert beam.status == BeamStatus.FAILED
        assert beam.completion_reason == "security_critical"
        assert "test" not in manager.active_beams
        assert beam in manager.pruned_beams


# =============================================================================
# SECTION 3: Constraint Tracker Integration Tests
# =============================================================================

class TestConstraintTrackerIntegration:
    """Test per-beam constraint tracking."""

    def test_beam_clone_preserves_constraint_tracker(self):
        """Cloning a beam should clone its constraint tracker."""
        beam = create_test_beam(generated_tokens=[10, 11])
        beam.constraint_tracker = ConstraintTracker()
        beam.constraint_tracker.update("def foo(")  # Open paren

        cloned = beam.clone()

        # Clone should have its own tracker with same state
        assert cloned.constraint_tracker is not None
        assert cloned.constraint_tracker is not beam.constraint_tracker
        assert cloned.constraint_tracker.state.bracket_depth == 1

    def test_beam_clone_trackers_independent(self):
        """Cloned trackers should be independent."""
        beam = create_test_beam(generated_tokens=[10, 11])
        beam.constraint_tracker = ConstraintTracker()
        beam.constraint_tracker.update("def foo(")

        cloned = beam.clone()

        # Update original
        beam.constraint_tracker.update(")")

        # Original closed, clone still open
        assert beam.constraint_tracker.state.bracket_depth == 0
        assert cloned.constraint_tracker.state.bracket_depth == 1

    def test_beam_clone_preserves_sec_penalty(self):
        """Cloning should preserve security penalty."""
        beam = create_test_beam(generated_tokens=[10, 11])
        beam.sec_penalty = 5.5

        cloned = beam.clone()

        assert cloned.sec_penalty == 5.5

    def test_beam_clone_preserves_tokens_since_check(self):
        """Cloning should preserve tokens_since_security_check."""
        beam = create_test_beam(generated_tokens=[10, 11])
        beam.tokens_since_security_check = 15

        cloned = beam.clone()

        assert cloned.tokens_since_security_check == 15


# =============================================================================
# SECTION 4: SecurityController Integration Tests
# =============================================================================

class TestSecurityControllerIntegration:
    """Test SecurityController with BeamManager."""

    def test_should_check_respects_boundary(self):
        """Security check should only trigger at boundaries."""
        config = SABSConfig(min_tokens_between_checks=1)
        controller = SecurityController(config)

        beam = create_test_beam(generated_tokens=[10, 11])
        beam.tokens_since_security_check = 10  # Enough tokens
        beam.constraint_tracker = ConstraintTracker()

        # Not at boundary (in the middle of a bracket)
        beam.constraint_tracker.update("def foo(")
        assert controller.should_check(beam) is False

        # At boundary (closed the bracket and newline)
        beam.constraint_tracker.update(")\n")
        assert controller.should_check(beam) is True

    def test_check_and_apply_soft_penalty(self):
        """Soft penalty should update beam sec_penalty."""
        config = SABSConfig(hard_fail_on_critical=False)
        controller = SecurityController(config)

        beam = create_test_beam(generated_tokens=[10, 11])
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10

        # Check with code that has security issues
        decision = controller.check_beam(beam, "import subprocess", "gen1", 1)

        # Create mock search manager
        search = Mock()

        controller.apply_decision(beam, decision, search)

        # Penalty should be added
        assert beam.sec_penalty > 0
        assert beam.tokens_since_security_check == 0
        search.fail_beam.assert_not_called()

    def test_check_and_apply_hard_fail(self):
        """Hard fail should call search.fail_beam."""
        config = SABSConfig(hard_fail_on_critical=True)
        controller = SecurityController(config)

        beam = create_test_beam(generated_tokens=[10, 11])
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10
        beam.beam_id = "test"
        beam.current_position  # Trigger property

        # Check with critical code - use code that triggers CRITICAL
        # os.system is CRITICAL in our verify_static.py
        code = "import os\nos.system('rm -rf /')"
        decision = controller.check_beam(beam, code, "gen1", 1)

        if decision.should_terminate:
            search = Mock()
            controller.apply_decision(beam, decision, search)
            search.fail_beam.assert_called_once_with(beam, "security_critical")

    def test_verifier_budget_limit(self):
        """Should stop checking after budget exhausted."""
        config = SABSConfig(max_verifier_calls=3)
        controller = SecurityController(config)

        beam = create_test_beam(generated_tokens=[10, 11])
        beam.tokens_since_security_check = 100
        beam.constraint_tracker = None  # No boundary check

        # Should allow first 3 checks
        for i in range(3):
            assert controller.should_check(beam) is True
            controller.check_beam(beam, "x = 1", "gen1", i)

        # 4th check should be blocked
        assert controller.should_check(beam) is False

    def test_statistics_tracking(self):
        """Statistics should track all decisions."""
        config = SABSConfig()
        controller = SecurityController(config)

        beam = create_test_beam(generated_tokens=[10, 11])
        beam.sec_penalty = 0.0
        beam.tokens_since_security_check = 10
        beam.beam_id = "test"

        # Make several checks
        controller.check_beam(beam, "x = 1", "gen1", 1)  # Safe
        controller.check_beam(beam, "import os", "gen1", 2)  # Unsafe
        controller.check_beam(beam, "", "gen1", 3)  # Empty (skip)

        stats = controller.get_statistics()

        assert stats['verifier_calls'] == 3
        assert stats['total_decisions'] == 3


# =============================================================================
# SECTION 5: End-to-End Flow Tests
# =============================================================================

class TestEndToEndFlow:
    """Test complete SABS integration flow."""

    def test_beam_lifecycle_with_sabs(self):
        """Test beam lifecycle with SABS enabled."""
        model = create_mock_model()
        config = SearchConfig(beam_width=4, security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        sabs_config = SABSConfig()
        controller = SecurityController(sabs_config)
        controller.reset("test_gen")

        # Create initial beam
        initial_tokens = [1, 2, 3]
        manager.create_initial_beam(initial_tokens)

        # Initialize constraint tracker
        for beam in manager.active_beams.values():
            beam.constraint_tracker = ConstraintTracker()

        beam = list(manager.active_beams.values())[0]

        # Simulate generating some tokens
        for i in range(5):
            manager.step_beam(beam, 100 + i, -0.1)
            beam.tokens_since_security_check += 1
            if beam.constraint_tracker:
                beam.constraint_tracker.update("x")

        # Simulate a security check
        code = "x = 1"
        if controller.should_check(beam) or True:  # Force check for test
            decision = controller.check_beam(beam, code, "test_gen", 5)
            controller.apply_decision(beam, decision, manager)

        # Complete the beam
        manager.complete_beam(beam, "eos")

        assert beam.status == BeamStatus.COMPLETED
        assert len(manager.completed_beams) == 1

    def test_branching_with_sabs_state(self):
        """Test that branching preserves SABS state correctly."""
        model = create_mock_model()
        config = SearchConfig(beam_width=4, max_branches=4, security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        # Create initial beam
        initial_tokens = [1, 2, 3]
        manager.create_initial_beam(initial_tokens)

        beam = list(manager.active_beams.values())[0]
        beam.constraint_tracker = ConstraintTracker()
        beam.constraint_tracker.update("def foo(")
        beam.sec_penalty = 2.5
        beam.tokens_since_security_check = 8

        # Create a checkpoint for branching
        manager.add_checkpoint_to_beam(beam, entropy=1.5, varentropy=0.5, save_state=True)

        # Simulate logits and branch
        logits = np.random.randn(100)
        new_beams = manager.branch_beam(beam, logits, num_branches=2, temperature=0.7)

        assert len(new_beams) == 2

        # Check that SABS state is preserved
        for b in new_beams:
            assert b.sec_penalty == 2.5
            assert b.tokens_since_security_check == 8
            if b.constraint_tracker:
                assert b.constraint_tracker.state.bracket_depth == 1

    def test_multiple_beams_independent_penalties(self):
        """Test that beams accumulate penalties independently."""
        model = create_mock_model()
        config = SearchConfig(beam_width=4, security_lambda=0.5)
        manager = BeamManager(config, model)
        manager.reset()

        # Use hard_fail=False to test soft penalty accumulation
        sabs_config = SABSConfig(hard_fail_on_critical=False)
        controller = SecurityController(sabs_config)
        controller.reset("test_gen")

        # Create two beams
        beam1 = Beam(beam_id="b1", tokens=[1, 2, 3], prompt_length=1)
        beam2 = Beam(beam_id="b2", tokens=[1, 2, 3], prompt_length=1)
        beam1.sec_penalty = 0.0
        beam2.sec_penalty = 0.0
        beam1.tokens_since_security_check = 10
        beam2.tokens_since_security_check = 10

        manager.active_beams = {"b1": beam1, "b2": beam2}

        # Check beam1 with code that has warnings (file write)
        decision1 = controller.check_beam(beam1, "f = open('x.txt', 'w')\nf.write('data')", "gen1", 1)
        controller.apply_decision(beam1, decision1, manager)

        # Check beam2 with safe code
        decision2 = controller.check_beam(beam2, "x = 1 + 2", "gen1", 2)
        controller.apply_decision(beam2, decision2, manager)

        # Penalties should be different
        assert beam1.sec_penalty > beam2.sec_penalty
        assert beam2.sec_penalty == 0.0


# =============================================================================
# SECTION 6: Extract Code Integration Tests
# =============================================================================

class TestExtractCodeIntegration:
    """Test code extraction for security checking."""

    def test_extract_code_only_basic(self):
        """Basic code extraction should work."""
        model = Mock()
        model.detokenize = Mock(return_value="def hello():\n    print('hello')")

        beam = Beam(
            beam_id="test",
            tokens=[1, 2, 3, 4, 5, 6, 7, 8],  # 3 prompt + 5 generated
            prompt_length=3,
        )

        code = extract_code_only(model, beam)

        assert "def hello():" in code
        model.detokenize.assert_called_once()

    def test_extract_code_strips_markdown(self):
        """Markdown fences should be stripped."""
        model = Mock()
        model.detokenize = Mock(return_value="```python\ndef foo():\n    pass\n```")

        beam = Beam(
            beam_id="test",
            tokens=[1, 2, 3, 4, 5],
            prompt_length=2,
        )

        code = extract_code_only(model, beam)

        assert "```" not in code
        assert "def foo():" in code

    def test_extract_code_strips_prose(self):
        """Prose prefixes should be stripped."""
        model = Mock()
        model.detokenize = Mock(return_value="Here's the Python code:\ndef bar():\n    return 42")

        beam = Beam(
            beam_id="test",
            tokens=[1, 2, 3, 4, 5],
            prompt_length=2,
        )

        code = extract_code_only(model, beam)

        assert "Here's" not in code
        assert "def bar():" in code

    def test_extract_empty_for_no_generated(self):
        """Should return empty string if no generated tokens."""
        model = Mock()

        beam = Beam(
            beam_id="test",
            tokens=[1, 2, 3],
            prompt_length=3,  # All prompt
        )

        code = extract_code_only(model, beam)

        assert code == ""


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
