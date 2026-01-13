# SelfEngine Test Run Results

**Date:** 2026-01-13
**Status:** ✅ ALL TESTS PASSED

## Test Summary

- **Total Tests:** 158
- **Passed:** 158 (100%)
- **Failed:** 0
- **Duration:** 0.50 seconds

## Test Coverage

### Component Tests (78 tests)
- ✅ Parse state classification (14 tests)
- ✅ Parse error analysis (6 tests)
- ✅ Security penalty computation (9 tests)
- ✅ Security cache functionality (7 tests)
- ✅ Incremental analysis (11 tests)
- ✅ SABS configuration (3 tests)
- ✅ SABS decision logic (4 tests)
- ✅ Security controller (12 tests)
- ✅ Code extraction (4 tests)
- ✅ Constraint tracker (3 tests)

### Edge Case Tests (43 tests)
- ✅ Unicode handling (6 tests)
- ✅ Long code strings (5 tests)
- ✅ Cache boundary conditions (5 tests)
- ✅ Empty/null inputs (6 tests)
- ✅ Malformed syntax (9 tests)
- ✅ Lambda extreme values (4 tests)
- ✅ Budget exhaustion (3 tests)
- ✅ Constraint tracker edge cases (10 tests)
- ✅ Security issue detection (6 tests)
- ✅ Beam edge cases (5 tests)

### Integration Tests (37 tests)
- ✅ Beam scoring integration (3 tests)
- ✅ Beam manager integration (4 tests)
- ✅ Constraint tracker integration (4 tests)
- ✅ Security controller integration (5 tests)
- ✅ End-to-end flow (3 tests)
- ✅ Code extraction integration (4 tests)

## Environment

- **Python Version:** 3.13.9
- **Platform:** Windows (win32)
- **Dependencies:**
  - numpy: 2.2.0 ✅
  - pytest: 9.0.2 ✅
  - matplotlib: 3.10.8 ✅

## What This Verifies

✅ **Security Analysis:** The static security analyzer correctly identifies dangerous patterns
✅ **Beam Search Logic:** Core beam search algorithms work as expected
✅ **SABS Integration:** Security-Aware Beam Search properly integrates security into decoding
✅ **Edge Case Handling:** System handles malformed input, unicode, and boundary conditions
✅ **State Management:** Beam cloning and state tracking work correctly

## Next Steps

To run the full system with code generation, you need:
1. A GGUF model file (e.g., CodeLlama, DeepSeek Coder, etc.)
2. Set the model path and run:
   ```bash
   python -m cli.main "Write a function to sort a list"
   ```

See [QUICKSTART.md](QUICKSTART.md) for detailed usage instructions.
