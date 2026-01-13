# SABS Reproducibility Package

This document provides complete instructions for reproducing all experiments and results from the paper "Security-Aware Beam Search for Neural Code Generation".

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Experiment Reproduction](#experiment-reproduction)
5. [Expected Results](#expected-results)
6. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware
- **Minimum:** 16GB RAM, 4-core CPU
- **Recommended:** 32GB RAM, 8-core CPU, NVIDIA GPU (8GB+ VRAM)
- **Storage:** 10GB free space (for model and results)

### Software
- Python 3.10 or higher
- Windows 10/11, Linux (Ubuntu 20.04+), or macOS 12+
- Git

### Model
- DeepSeek-Coder-6.7B in GGUF format
- Download from: https://huggingface.co/TheBloke/deepseek-coder-6.7b-instruct-GGUF
- File: `deepseek-coder-6.7b-instruct.Q4_K_M.gguf` (4.08 GB)

---

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/[redacted]/sabs.git
cd sabs
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Model

```bash
# Create models directory
mkdir -p models

# Download model (using huggingface-cli)
pip install huggingface-hub
huggingface-cli download TheBloke/deepseek-coder-6.7b-instruct-GGUF \
    deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
    --local-dir ./models
```

### Step 5: Verify Installation

```bash
python -c "from engine.verify_static import analyze_incremental; print('OK')"
python -c "from engine.security_controller import SecurityController; print('OK')"
```

---

## Quick Start

### Run Unit Tests
```bash
pytest tests/ -v
```
Expected: 158 tests pass (76 unit + 23 integration + 59 edge cases)

### Run Benchmarks
```bash
python -m eval.benchmark_sabs
```
Expected output: Performance metrics for each SABS component

### Run Security Evaluation
```bash
python -m eval.eval_sabs
```
Expected output: Accuracy, precision, recall metrics

### Generate Paper Figures
```bash
python paper/generate_figures.py
```
Output: Figures in `paper/figures/`

---

## Experiment Reproduction

### Experiment 1: HumanEval Evaluation

This experiment measures pass@1 and violation rate across different lambda values.

```bash
# Download HumanEval dataset
mkdir -p data
wget https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz \
    -O data/HumanEval.jsonl.gz
gunzip data/HumanEval.jsonl.gz

# Run evaluation (BASELINE)
python -m cli.main \
    --model ./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
    --humaneval data/HumanEval.jsonl \
    --output results/humaneval_baseline.json \
    --seed 42

# Run evaluation (SABS lambda=0.5)
python -m cli.main \
    --model ./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
    --use-sabs --sabs-lambda 0.5 \
    --humaneval data/HumanEval.jsonl \
    --output results/humaneval_sabs_0.5.json \
    --seed 42

# Run for all lambda values
for lambda in 0.25 0.75 1.0; do
    python -m cli.main \
        --model ./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
        --use-sabs --sabs-lambda $lambda \
        --humaneval data/HumanEval.jsonl \
        --output results/humaneval_sabs_${lambda}.json \
        --seed 42
done
```

**Runtime:** ~45 minutes per lambda value on RTX 3090

### Experiment 2: SecurityStress Evaluation

```bash
# Run security stress test
python -m eval.eval_sabs \
    --output results/security_stress_results.csv

# Compare methods
python scripts/analyze_security_results.py results/security_stress_results.csv
```

**Runtime:** ~15 minutes total

### Experiment 3: Performance Benchmarks

```bash
# Run full benchmark suite
python -m eval.benchmark_sabs \
    --iterations 10000 \
    --output results/benchmark_results.json

# Generate performance comparison table
python scripts/generate_perf_table.py results/benchmark_results.json
```

**Runtime:** ~10 minutes

### Experiment 4: Ablation Study

```bash
# SABS-full (default configuration)
python -m cli.main --use-sabs --sabs-lambda 0.5 \
    --humaneval data/HumanEval.jsonl \
    --output results/ablation_full.json

# No hard-fail (soft penalties only)
python -m cli.main --use-sabs --sabs-lambda 0.5 --sabs-no-hard-fail \
    --humaneval data/HumanEval.jsonl \
    --output results/ablation_no_hardfail.json

# Hard-fail only (no soft penalties)
python -m cli.main --use-sabs --sabs-lambda 0.5 --sabs-hard-fail-only \
    --humaneval data/HumanEval.jsonl \
    --output results/ablation_hardfail_only.json

# No caching
python -m cli.main --use-sabs --sabs-lambda 0.5 --sabs-no-cache \
    --humaneval data/HumanEval.jsonl \
    --output results/ablation_no_cache.json
```

---

## Expected Results

### Table 1: HumanEval Results

| Method | lambda | pass@1 | Violation Rate | Latency (ms) |
|--------|--------|--------|----------------|--------------|
| BASELINE | 0.0 | 62.2 +/- 1.5% | 12.3 +/- 2.0% | 1234 +/- 50 |
| SABS | 0.25 | 61.5 +/- 1.5% | 4.2 +/- 1.0% | 1356 +/- 60 |
| SABS | 0.50 | 60.8 +/- 1.5% | 2.1 +/- 0.8% | 1423 +/- 70 |
| SABS | 0.75 | 59.4 +/- 1.5% | 0.8 +/- 0.5% | 1467 +/- 75 |
| SABS | 1.0 | 57.1 +/- 1.5% | 0.3 +/- 0.3% | 1502 +/- 80 |

Note: Results may vary slightly due to:
- Hardware differences
- Model quantization effects
- Non-determinism in beam search sampling

### Table 2: SecurityStress Results

| Method | Safe Rate | Blocked | Violated |
|--------|-----------|---------|----------|
| BASELINE | 12% +/- 5% | 0% | 88% +/- 5% |
| SABS-0.5 | 78% +/- 5% | 15% +/- 5% | 7% +/- 3% |
| SABS-0.75 | 85% +/- 4% | 22% +/- 5% | 3% +/- 2% |
| SABS-1.0 | 91% +/- 3% | 35% +/- 5% | 2% +/- 1% |

### Benchmark Performance

| Component | Expected ops/sec |
|-----------|-----------------|
| analyze_incremental (cached) | 200,000+ |
| analyze_incremental (uncached) | 20,000+ |
| ConstraintTracker.clone() | 150,000+ |
| Cache get/put | 500,000+ |
| Beam.normalized_score() | 1,000,000+ |

---

## Automated Reproduction Script

For convenience, a single script runs all experiments:

```bash
# Run all experiments (takes ~4 hours)
python scripts/run_all_experiments.py \
    --model ./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
    --output-dir ./results \
    --seed 42

# Generate all tables and figures
python scripts/generate_paper_outputs.py \
    --results-dir ./results \
    --output-dir ./paper/figures
```

---

## Troubleshooting

### Issue: Out of Memory

**Symptom:** Process killed or MemoryError

**Solution:**
```bash
# Reduce beam width
python -m cli.main --beam-width 2 ...

# Use smaller model quantization
# Download Q2_K version instead of Q4_K_M
```

### Issue: Slow Generation

**Symptom:** Generation takes >5 minutes per sample

**Solution:**
```bash
# Enable GPU acceleration (requires llama-cpp-python[cuda])
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Reduce max tokens
python -m cli.main --max-tokens 256 ...
```

### Issue: Import Errors

**Symptom:** ModuleNotFoundError

**Solution:**
```bash
# Ensure you're in the correct directory
cd /path/to/sabs

# Ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Model Not Found

**Symptom:** FileNotFoundError for model path

**Solution:**
```bash
# Verify model exists
ls -la models/

# Download if missing
huggingface-cli download TheBloke/deepseek-coder-6.7b-instruct-GGUF \
    deepseek-coder-6.7b-instruct.Q4_K_M.gguf \
    --local-dir ./models
```

### Issue: Test Failures

**Symptom:** pytest reports failures

**Solution:**
```bash
# Run verbose tests to see details
pytest tests/ -v --tb=long

# Check Python version
python --version  # Should be 3.10+

# Update pytest
pip install --upgrade pytest
```

---

## File Checksums

For verification, expected SHA256 checksums:

```
engine/verify_static.py:      [checksum]
engine/security_controller.py: [checksum]
engine/search.py:             [checksum]
tests/test_sabs_components.py: [checksum]
```

Generate checksums:
```bash
sha256sum engine/verify_static.py engine/security_controller.py
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{sabs2024,
  title={Security-Aware Beam Search for Neural Code Generation},
  author={Anonymous},
  journal={arXiv preprint},
  year={2024}
}
```

---

## Contact

For questions or issues with reproduction:
- Open a GitHub issue
- Email: [redacted for review]

---

## License

MIT License - See LICENSE file for details.
