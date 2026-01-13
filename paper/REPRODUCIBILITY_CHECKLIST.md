# SABS Reproducibility Checklist

**Paper:** Security-Aware Beam Search for Neural Code Generation
**Version:** 1.0
**Date:** 2026-01-13

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.12+ |
| RAM | 8 GB | 16 GB |
| Disk Space | 5 GB | 10 GB |
| GPU | Not required | CUDA-compatible (optional) |

### Dependencies
```
numpy>=1.23.0,<2.0
pytest>=7.0.0
matplotlib>=3.7.0  # For figures only
llama-cpp-python>=0.2.0  # For model inference only
```

---

## Quick Verification Commands

### 1. Install & Test (5 minutes)
```bash
cd SelfEngine
pip install -r requirements.txt
pytest tests/ -v
```
**Expected:** 158 tests pass

### 2. Run Benchmarks (1 minute)
```bash
python -m eval.benchmark_sabs
```
**Expected output:**
- analyze_incremental_cached: ~300k+ ops/sec
- beam_normalized_score: ~2M+ ops/sec

### 3. Run Security Evaluation (1 minute)
```bash
python -m eval.eval_sabs
```
**Expected output:**
- Accuracy: 95%
- Precision: 100%
- Recall: 93.8%
- F1 Score: 96.8%

### 4. Generate Figures (30 seconds)
```bash
python paper/generate_figures.py
```
**Expected:** 6 figures in `paper/figures/`

---

## Full Experiment Reproduction

### HumanEval (requires model, ~45 min)
```bash
# Download model first
mkdir -p models
# Place deepseek-coder-6.7b-instruct.Q4_K_M.gguf in models/

# Run baseline
python -m cli.main --model models/*.gguf --humaneval data/HumanEval.jsonl --output baseline.json

# Run SABS lambda=0.5
python -m cli.main --model models/*.gguf --use-sabs --sabs-lambda 0.5 --humaneval data/HumanEval.jsonl --output sabs_0.5.json
```

### SecurityStress (no model needed)
```bash
python -m eval.eval_sabs --output security_results.csv
```

---

## Verification Checklist

### Code Quality
- [x] No hardcoded secrets (grep verified)
- [x] No unsafe eval/exec in engine/ (model.eval() is safe)
- [x] Input validation present
- [x] Dependencies documented

### Testing
- [x] 158 tests pass consistently
- [x] 80%+ coverage on SABS components
  - constraints.py: 82%
  - security_controller.py: 84%
  - verify_static.py: 80%
  - search.py: 81%
- [x] No flaky tests (3x runs pass)

### Reproducibility
- [x] Security evaluation deterministic
- [x] Same results across runs
- [x] Benchmarks stable (timing variance expected)

### Paper Integrity
- [x] All claims have data support
- [x] Figure data matches tables
- [x] All 7 citations verified (ACL/NAACL/EMNLP/IEEE S&P/arXiv)
- [x] Fixed claim: "Caching reduces latency by 24%" (was 32%)

---

## Expected Results Summary

| Metric | Value | Source |
|--------|-------|--------|
| Unit Tests | 158 pass | pytest tests/ |
| SABS Coverage | 80-84% | pytest --cov=engine |
| Eval Accuracy | 95% | eval.eval_sabs |
| Eval Precision | 100% | eval.eval_sabs |
| Eval Recall | 93.8% | eval.eval_sabs |
| Cache Speedup | ~6x | benchmark_sabs |

---

## Time to Reproduce

| Task | Time |
|------|------|
| Install dependencies | 2 min |
| Run all tests | 2 min |
| Run benchmarks | 1 min |
| Run security eval | 1 min |
| Generate figures | 30 sec |
| **Total (without HumanEval)** | **~7 min** |
| HumanEval (with model) | +45 min |

---

## Files Inventory

```
SelfEngine/
  engine/
    verify_static.py     # Security analysis + tri-state parsing
    security_controller.py # SABS orchestration
    constraints.py       # Syntactic boundary tracking
    search.py            # Beam scoring with sec_penalty
  tests/
    test_sabs_components.py    # 76 unit tests
    test_sabs_integration.py   # 23 integration tests
    test_sabs_edge_cases.py    # 59 edge case tests
  eval/
    benchmark_sabs.py    # Performance benchmarks
    eval_sabs.py         # Security evaluation
  paper/
    SABS_Research_Paper.md    # Full paper
    generate_figures.py       # Figure generation
    figures/                  # 6 publication figures
    REPRODUCIBILITY.md        # Detailed guide
    REPRODUCIBILITY_CHECKLIST.md  # This file
  docs/
    SABS_API.md          # API documentation
  requirements.txt       # Pinned dependencies
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install -r requirements.txt` |
| Test failures | Ensure Python 3.10+, check `pytest --version` |
| Model not found | Download GGUF to `models/` directory |
| Figures not generating | `pip install matplotlib` |

---

## Attestation

This reproducibility package was verified on:
- **Date:** 2026-01-13
- **Platform:** Windows 11
- **Python:** 3.13.9
- **pytest:** 9.0.2

All tests pass. All claims verified against data.

---

**SUPERCODER RESEARCH-READY FOR ARXIV**
