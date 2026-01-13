# SuperCoder

**A Local System-2 Reasoning Engine for Code Generation**

SuperCoder is a sophisticated code generation engine that implements entropy-gated beam search with KV cache checkpointing. Unlike traditional autoregressive generation, SuperCoder can "think" by exploring multiple code paths when uncertain and pruning suboptimal solutions.

## Key Features

- **Entropy-Gated Branching**: Automatically detects high-uncertainty tokens and explores multiple alternatives
- **Beam Search with Structural Sharing**: Efficient memory usage through checkpoint position references (not token duplication)
- **KV Cache Checkpointing**: Save and restore model states for efficient backtracking
- **Static Security Analysis**: AST-based code verification with import alias tracking
- **Syntactic Boundary Detection**: Smart checkpointing at Python statement boundaries

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Orchestrator                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   Prompt    │──│  Tokenize   │──│  Entropy-Gated Loop     │ │
│  └─────────────┘  └─────────────┘  │                         │ │
│                                     │  ┌─────────────────┐   │ │
│                                     │  │ High Entropy?   │   │ │
│                                     │  │  → BRANCH       │   │ │
│                                     │  │ Low Entropy?    │   │ │
│                                     │  │  → FLOW         │   │ │
│                                     │  └─────────────────┘   │ │
│                                     └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ ModelAdapter  │    │  BeamManager  │    │ConstraintTracker│
│               │    │               │    │               │
│ • KV Cache    │    │ • Beam Search │    │ • Boundaries  │
│ • Checkpoints │    │ • Branching   │    │ • Syntax State│
│ • Sampling    │    │ • Pruning     │    │ • Keywords    │
└───────────────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────┐
│CheckpointStore│
│               │
│ • Memory Pool │
│ • Disk Spill  │
│ • NO Tokens   │
└───────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+
- A GGUF format LLM (e.g., CodeLlama, DeepSeek-Coder, Qwen2.5-Coder)
- `llama-cpp-python` with GPU support (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/SelfEngine.git
cd SelfEngine

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install numpy llama-cpp-python

# For GPU support (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

### Model Setup

Place your GGUF model in the `models/` directory:

```
SelfEngine/
└── models/
    └── model.gguf  # Your model here
```

Or set the environment variable:

```bash
# Windows
set SUPERCODER_MODEL=C:\path\to\your\model.gguf

# Linux/Mac
export SUPERCODER_MODEL=/path/to/your/model.gguf
```

---

## Usage

### Basic CLI Usage

```bash
python -m cli.main "Write a python function to parse JSON safely"
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `./models/model.gguf` | Path to GGUF model file |
| `--n-ctx` | `4096` | Context window size |
| `--n-gpu-layers` | `-1` | GPU layers (-1 = all) |
| `--max-tokens` | `512` | Maximum tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--entropy-threshold` | `6.0` | Entropy threshold for branching |
| `--beam-width` | `4` | Number of beams to maintain |
| `--log-every` | `1` | Log frequency (0 = quiet) |

### Examples

```bash
# Basic generation
python -m cli.main "Write a binary search function in Python"

# Quiet mode with custom model
python -m cli.main --model ./models/deepseek-coder-6.7b.gguf --log-every 0 "Implement a LRU cache"

# High exploration mode
python -m cli.main --entropy-threshold 4.0 --beam-width 8 "Write a recursive fibonacci with memoization"

# Conservative mode (less branching)
python -m cli.main --entropy-threshold 8.0 --temperature 0.3 "Parse a CSV file"
```

### Programmatic Usage

```python
from engine import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    model_path="./models/model.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,
    max_tokens=512,
    entropy_threshold=6.0,
    beam_width=4,
    temperature=0.7,
)

orch = Orchestrator(config)
output = orch.generate("Write a function to validate email addresses")
print(output)
```

---

## Core Modules

### `engine/orchestrator.py`

The main orchestrator that ties everything together. Implements entropy-gated branching:

- **High entropy** (model uncertainty) → Create branches to explore alternatives
- **Low entropy** (model confidence) → Flow forward with sampling

```python
# Entropy calculation
ent, varent = entropy_and_varentropy_from_logits(logits)

if ent >= entropy_threshold:
    # Branch: explore multiple token choices
    new_beams = search.branch_beam(beam, logits, num_branches=2)
else:
    # Flow: sample single token and continue
    token_id, logprob = model.sample(logits)
```

### `engine/search.py`

Beam search with structural checkpoint sharing.

**Key Classes:**
- `Beam`: Owns full token sequence, references checkpoints by position
- `Checkpoint`: Lightweight - stores position + KV cache ref, NOT tokens
- `BeamManager`: Manages beam lifecycle, branching, pruning

**Critical Invariant:**
```python
# MUST call before stepping ANY beam
manager.prepare_beam_step(beam)  # Restores model state
logits = model.get_logits()      # Now safe to get logits
```

### `engine/model_adapter.py`

LLM wrapper with checkpoint-based beam isolation.

**Key Features:**
- Capability probing at load time
- `CheckpointStore`: Memory pool with disk spillover (NO token storage)
- `create_checkpoint()`: Save KV cache state
- `restore_checkpoint()`: Restore state using caller's tokens (authoritative)

```python
# Checkpoint stores ONLY KV cache bytes
# Tokens come from Beam (structural sharing)
model.restore_checkpoint(checkpoint_id, beam.tokens[:position])
```

### `engine/constraints.py`

Syntactic state machine for Python code boundary detection.

**Tracks:**
- Bracket depth `([{` `}])`
- String states (single, double, triple-quoted)
- Comments (`#` to EOL)
- Escape sequences in strings

**Boundary Triggers:**
1. Newline at top level (`bracket_depth==0`, not in string/comment)
2. Decision keyword at statement start (`def`, `class`, `if`, `for`, etc.)

```python
tracker = ConstraintTracker()
boundaries = tracker.update("def foo():\n    return 1")
# boundaries = [11]  # Position after newline
```

### `engine/verify_static.py`

AST-based security analysis with comprehensive import alias tracking.

**Blocks:**
- `eval()`, `exec()`, `compile()`, `__import__()`
- `subprocess`, `shutil`, `ctypes`, `signal` imports
- `os.system()`, `os.popen()`, `os.environ` access
- `Path.write_text()`, `Path.unlink()`, dangerous Path methods
- `open()` with write modes

**Alias Tracking:**
```python
# Detects these patterns:
from subprocess import Popen as X
X("rm -rf /")  # BLOCKED: subprocess call via alias

from pathlib import Path as P
P("/etc/passwd").write_text("...")  # BLOCKED: dangerous Path method
```

---

## How It Works

### Entropy-Gated Beam Search

Traditional LLM generation picks one token at a time. SuperCoder monitors the model's uncertainty (entropy) and branches when confused:

```
Step 1: "def" → Low entropy → FLOW
Step 2: "fibonacci" → Low entropy → FLOW
Step 3: "(" → Low entropy → FLOW
Step 4: "n" → High entropy! → BRANCH
         ├── Beam 1: "n)"
         └── Beam 2: "num)"
Step 5+: Both beams continue...
Final: Best beam selected by score
```

### Structural Checkpoint Sharing

Memory-efficient design where checkpoints don't duplicate tokens:

```
Beam owns tokens: [1, 2, 3, 4, 5, 6, 7, 8]
                       ↑           ↑
Checkpoint A ─────────┘           │
  position: 3                     │
  state_ref: "cp_001"             │
                                  │
Checkpoint B ─────────────────────┘
  position: 7
  state_ref: "cp_002"
```

When branching from Checkpoint A:
1. Clone beam tokens up to position 3: `[1, 2, 3]`
2. Restore KV cache from `cp_001`
3. Each branch diverges independently

### KV Cache Management

```
┌─────────────────────────────────────┐
│           CheckpointStore           │
├─────────────────────────────────────┤
│  Memory Pool (default 512MB)        │
│  ┌─────┐ ┌─────┐ ┌─────┐           │
│  │cp_01│ │cp_02│ │cp_03│           │
│  └─────┘ └─────┘ └─────┘           │
├─────────────────────────────────────┤
│  Disk Spillover (when OOM)          │
│  ./temp/cp_04.state                 │
│  ./temp/cp_05.state                 │
└─────────────────────────────────────┘
```

---

## Configuration Guide

### Tuning Entropy Threshold

| Value | Behavior |
|-------|----------|
| `4.0` | Aggressive branching, more exploration, slower |
| `6.0` | Balanced (default) |
| `8.0` | Conservative, less branching, faster |
| `10.0+` | Almost never branches |

### Tuning Beam Width

| Value | Behavior |
|-------|----------|
| `2` | Minimal exploration, fast |
| `4` | Balanced (default) |
| `8` | More alternatives considered |
| `16` | Maximum exploration, slow |

### Tuning Temperature

| Value | Behavior |
|-------|----------|
| `0.0` | Greedy (deterministic) |
| `0.3` | Conservative sampling |
| `0.7` | Balanced (default) |
| `1.0` | Creative sampling |
| `1.5+` | Very random |

---

## Project Structure

```
SelfEngine/
├── engine/
│   ├── __init__.py           # Package exports
│   ├── constraints.py        # Syntactic boundary detection
│   ├── search.py             # Beam search with checkpoints
│   ├── model_adapter.py      # LLM wrapper with KV cache
│   ├── verify_static.py      # AST security analysis
│   └── orchestrator.py       # Main generation loop
├── cli/
│   ├── __init__.py
│   └── main.py               # CLI entry point
├── models/                   # Place GGUF models here
│   └── model.gguf
└── README.md
```

---

## Troubleshooting

### "Model capabilities insufficient"

Your model doesn't support required features. Ensure:
- Model is in GGUF format
- `llama-cpp-python` is installed correctly
- Model file isn't corrupted

### "No beam produced output"

All beams failed or were pruned. Try:
- Increase `--max-tokens`
- Lower `--entropy-threshold`
- Check if prompt is valid

### Out of Memory

KV cache checkpoints consume memory. Try:
- Reduce `--beam-width`
- Reduce `--n-ctx`
- CheckpointStore will automatically spill to disk

### Slow Generation

- Use GPU: ensure `n_gpu_layers=-1` (all layers on GPU)
- Reduce `--beam-width`
- Increase `--entropy-threshold` (less branching)
- Use smaller model

---

## Technical Details

### Why Structural Sharing?

Traditional approach (bad):
```
Checkpoint stores: tokens + KV state
→ Memory bloat: O(checkpoints × sequence_length)
```

SuperCoder approach (good):
```
Beam stores: tokens (authoritative)
Checkpoint stores: position + KV state ref
→ Memory: O(beams × sequence_length) + O(checkpoints × KV_size)
```

### Why Caller-Provided Tokens in restore_checkpoint?

```python
# BAD: Checkpoint stores tokens → can desync from beam
model.restore_checkpoint(cp_id)  # Uses stored tokens

# GOOD: Beam provides authoritative tokens
model.restore_checkpoint(cp_id, beam.tokens[:position])
```

This ensures beam is always the single source of truth for token sequences.

### Why Top-K Before Softmax?

```python
# BAD: Full softmax over vocab (128k tokens)
probs = softmax(logits)  # exp() on 128k values
top_k = get_top_k(probs)

# GOOD: Top-K first via argpartition, then softmax
top_k_indices = np.argpartition(logits, -k)[-k:]  # O(n)
top_k_probs = softmax(logits[top_k_indices])      # O(k)
```

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (coming soon)
5. Submit a pull request

---

## Acknowledgments

- Built with [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- Inspired by research on System-2 reasoning and entropy-based sampling
