# What is SelfEngine? 🛡️

## For a 5-Year-Old 👶

Imagine you have a robot that writes stories. But sometimes it writes bad words!

**The Old Way:**
The robot writes the whole story, then you read it all and say "Nope, bad story!" and throw it away. Then it tries again. That's slow and wasteful!

**SelfEngine Way:**
As the robot writes, you watch over its shoulder. When it starts writing bad words, you gently stop it right away and say "Try this instead." The robot learns and writes a better story the next time.

So: **Catch mistakes early, fix them as you go, never waste time on bad stories.**

---

## For a Senior Developer 👨‍💻

### The Problem
Traditional LLM-based code generation follows a post-hoc verification pipeline:

```
Prompt → Generate Full Sequence → Static Analysis → Accept/Reject
```

**Issues:**
- Wasted compute on paths that will be rejected
- No steering signal during generation
- No feedback loop into the decoding process
- Verification is reactive, not proactive

### The Solution: SABS (Security-Aware Beam Search)

SelfEngine integrates security verification directly into the beam search decoding loop:

```
For each beam at each step:
  → Extend with next token
  → Update syntax state (incremental parsing)
  → [At top-level boundary]
    → Analyze AST of prefix
    → Compute security penalty = Σ(severity_weights)
    → Update beam score: score = log_prob/len_penalty − λ × penalty
    → Prune CRITICAL beams, soft-penalize WARN/ERROR
  → Rescore and sort beams
```

**Key Components:**

1. **ConstraintTracker** - Maintains per-beam parse state (brackets, quotes, syntax)
2. **SecurityController** - Runs incremental AST analysis at boundaries
3. **LRU Cache** - Memoizes security analysis to avoid redundant checks
4. **Penalty Scoring** - Configurable λ allows security/quality trade-off

**Security Detection:**
- Shell execution (`os.system`, `subprocess`)
- Code injection (`eval`, `exec`, `compile`)
- Unsafe file operations (`open` with mode='w' at module level)
- Pickle/deserialization attacks
- Network/process spawning

**Algorithms:**

| Mode | Description | Use Case |
|------|-------------|----------|
| **Baseline** | Entropy threshold branching | Standard beam search |
| **SAEG** | Semantic-Adaptive Entropy Gating (novel) | Position-aware, uncertainty-aware branching |
| **SABS** | Security-Aware Beam Search | Security-critical code generation |

### Trade-offs

- **λ = 0.0**: Pure likelihood (no security constraint)
- **λ = 0.5**: Balanced (default)
- **λ = 1.0+**: Security-first (may reduce code quality)

### Results

**Measured metrics:**
- Verifier calls per generation (budget-controlled)
- Hard fails vs soft penalties
- Token overhead
- Cache hit rates
- Security violation rate reduction

**All tested:** 158 comprehensive tests covering edge cases, integration flows, and malformed input handling.

---

## TL;DR

**ELI5:** Stop bad code early instead of throwing away finished bad code.

**Dev:** Beam search + incremental AST analysis + penalty scoring = proactive security pruning with configurable trade-offs.

