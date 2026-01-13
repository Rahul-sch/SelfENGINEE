# SelfEngine - Quick Start Guide

## Prerequisites
- Python 3.8+ (You have Python 3.13.9 ✓)
- A GGUF model file (for local inference)

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up your model:**
   - Place your GGUF model file in the project directory OR
   - Set the environment variable:
     ```bash
     set SUPERCODER_MODEL=C:\path\to\your\model.gguf
     ```
   - Or use the `--model` flag when running

## Running SelfEngine

### Basic Examples

**1. Baseline mode (entropy threshold):**
```bash
python -m cli.main "Write a python function to calculate fibonacci"
```

**2. SAEG mode (Semantic-Adaptive Entropy Gating - Novel Algorithm):**
```bash
python -m cli.main --use-saeg "Write a binary search function"
```

**3. SABS mode (Security-Aware Beam Search):**
```bash
python -m cli.main --use-sabs "Write a file reader function"
```

**4. SABS with custom security weight:**
```bash
python -m cli.main --use-sabs --sabs-lambda 1.0 "Write a shell command executor"
```

**5. Export decisions for analysis:**
```bash
python -m cli.main --use-saeg --export-decisions results.json "Write a JSON parser"
```

### Common Options

- `--model <path>` - Path to your GGUF model file
- `--max-tokens <n>` - Max tokens to generate (default: 512)
- `--temperature <f>` - Sampling temperature (default: 0.7)
- `--beam-width <n>` - Beam width (default: 4)
- `--use-saeg` - Enable SAEG algorithm
- `--use-sabs` - Enable Security-Aware Beam Search
- `--log-every 0` - Quiet mode (no step logging)

## Quick Test (without model)

To verify the system is working correctly without needing a model, run the test suite:
```bash
python -m pytest tests/ -v
```

All 158 tests should pass. This confirms:
- Security analysis engine works correctly
- Beam search components are functional
- SABS (Security-Aware Beam Search) logic is correct
- Edge cases are handled properly

## What Modes Do

- **Baseline**: Standard beam search with entropy threshold
- **SAEG**: Novel algorithm that adapts search based on semantic uncertainty and position
- **SABS**: Security-guided search that prunes unsafe code paths during generation

## Troubleshooting

**Error: "Model not found"**
- Make sure you have a GGUF model file
- Set `SUPERCODER_MODEL` environment variable
- Or use `--model` flag to specify path

**Missing dependencies:**
```bash
pip install -r requirements.txt
```
