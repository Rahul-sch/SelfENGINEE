# Master Study Guide: Know Your Products Cold

---

## How to Use This Guide

This document helps you answer ANY question about your products with confidence. Study each section until you can explain it without notes.

---

# PRODUCT 1: NEXUS

## What It Is (ELI5)
Imagine you ask ChatGPT something but it doesn't quite understand. Nexus is like having 4 different AI experts review your question before you ask it - one finds the confusing parts, one rewrites it better, one criticizes it, and one polishes it. You end up with a perfect question.

## What It Is (Technical)
A multi-agent prompt refinement system using LangGraph for orchestration. Four specialized agents process prompts sequentially:
1. **Clarifier**: Analyzes prompt for ambiguity, generates clarifying questions
2. **Drafter**: Produces refined prompt variants based on context
3. **Critic**: Evaluates drafts against quality criteria
4. **Finalizer**: Polishes winning variant for production use

## Architecture Deep Dive

```
User Input
    ↓
[Supabase Auth] → Validate user
    ↓
[API Route] → /api/refine/start
    ↓
[LangGraph Orchestrator]
    ↓
┌─────────────────────────────────────┐
│  Clarifier Agent                    │
│  - Parse intent                     │
│  - Identify ambiguities             │
│  - Generate clarifying questions    │
└─────────────────────────────────────┘
    ↓ (if questions → await_user)
┌─────────────────────────────────────┐
│  Drafter Agent                      │
│  - Incorporate answers              │
│  - Generate N refined variants      │
│  - Add structure and specificity    │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Critic Agent                       │
│  - Score each variant               │
│  - Identify weaknesses              │
│  - Select best candidate            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Finalizer Agent                    │
│  - Polish language                  │
│  - Ensure completeness              │
│  - Format for target LLM            │
└─────────────────────────────────────┘
    ↓
[Store in Supabase] → Refinery record
    ↓
[WebSocket] → Real-time UI update
    ↓
User sees final prompt + diff
```

## Key Technical Decisions

**Q: Why LangGraph, not just chained prompts?**
A: LangGraph provides state machines for complex agent workflows. It handles:
- Conditional branching (if user needs clarification)
- State persistence between steps
- Retry logic and error handling
- Parallel agent execution (future)

**Q: Why Supabase, not Firebase or custom backend?**
A: Supabase provides:
- Postgres (relational, good for structured refinery data)
- Row-level security (multi-tenant by design)
- Real-time subscriptions (WebSocket updates)
- Auth out of the box
- Edge functions for serverless

**Q: Why 4 agents, not 1 or 10?**
A: 4 is the sweet spot:
- 1 agent: No specialization, mediocre at everything
- 2-3 agents: Missing critical roles
- 4 agents: Each has clear responsibility
- 5+: Diminishing returns, latency increases

## Common Questions

**Q: How is this different from just iterating with ChatGPT?**
A: Three ways:
1. **Specialization**: Each agent is optimized for one task, not generalist
2. **Systematic**: Always follows the same quality process
3. **Collaborative**: Agents build on each other's work, not independent

**Q: What if I don't want clarifying questions?**
A: You can skip the clarification phase. The Clarifier will still analyze but won't block on user input.

**Q: Can I use my own API keys?**
A: Yes! The vault system stores encrypted API keys per user. Supports OpenAI, Anthropic, and custom endpoints.

---

# PRODUCT 2: VIBEGUARD

## What It Is (ELI5)
When you use AI to write code, sometimes it invents fake libraries (like imagining a tool that doesn't exist), accidentally puts passwords in the code, or makes security mistakes. VibeGuard is like a spell-checker but for these AI coding mistakes - it finds them and fixes them automatically.

## What It Is (Technical)
A CLI static analyzer specifically designed for AI-generated code. Detects three categories of issues:
1. **Supply Chain**: Hallucinated/phantom packages that don't exist in registries
2. **Secrets**: Hardcoded API keys, tokens, credentials via pattern matching
3. **Misconfigurations**: Insecure defaults (CORS *, no auth, etc.)

Auto-patching uses LLM to generate fixes with explanations.

## How Detection Works

### Supply Chain Defense
```
1. Parse imports from code (require, import, from X import)
2. Extract package names
3. Query npm/PyPI registry APIs
4. If package doesn't exist → FLAG
5. If package exists but version doesn't → FLAG
```

### Secret Detection
```
1. Regex patterns for known formats:
   - AWS keys: AKIA[0-9A-Z]{16}
   - GitHub tokens: ghp_[a-zA-Z0-9]{36}
   - Generic: (api[_-]?key|secret|password)\s*[:=]\s*['"][^'"]+['"]
2. Context analysis:
   - Is it in a string literal?
   - Is it actually assigned (not just commented)?
   - Is the value entropy high? (random-looking)
3. If matches → FLAG with severity
```

### Misconfiguration Detection
```
1. AST parsing of code
2. Pattern matching for known issues:
   - cors({ origin: '*' })
   - app.disable('x-powered-by') missing
   - process.env.NODE_ENV !== 'production' in prod code
3. Context-aware severity assignment
```

## Auto-Patching Flow

```
Issue Detected
    ↓
[Extract context] → 50 lines around issue
    ↓
[Generate prompt]
  "Fix this security issue:
   Issue: {description}
   Code: {context}
   Generate a fix with explanation."
    ↓
[LLM generates patch]
    ↓
[Validate patch]
  - Does it parse?
  - Does it break tests?
  - Is it actually different?
    ↓
[Apply or suggest]
    ↓
User sees fix + explanation
```

## Common Questions

**Q: How is this different from Snyk/Semgrep?**
A: Three key differences:
1. **AI-specific**: Catches hallucinated packages (they don't)
2. **Auto-fix with AI**: Generates fixes with explanations (they just flag)
3. **Developer-friendly**: CLI-first, not enterprise-first

**Q: Does it work for all languages?**
A: Currently optimized for JavaScript/TypeScript. Python support is partial. Other languages are on roadmap.

**Q: What about false positives?**
A: We optimize for precision over recall. Better to miss some issues than annoy developers with noise. Configurable thresholds.

**Q: How fast is it?**
A: <2s for most repos (<10K LOC). Scales linearly with codebase size.

---

# PRODUCT 3: SELFENGINE

## What It Is (ELI5)
When AI writes code, it's like a robot typing one letter at a time. Normally, we let it finish typing the whole thing, then check if it's safe. SelfEngine is different - we check for problems WHILE the robot is typing, and if it starts writing something dangerous, we make it try a different path instead.

## What It Is (Technical)
A novel algorithm called SABS (Security-Aware Beam Search) that integrates static security analysis directly into the LLM beam search decoding loop.

Instead of: `generate → verify → accept/reject`
We do: `generate_token → verify_prefix → score_beams → prune → continue`

## The SABS Algorithm

### Core Equation
```
score = (log_prob / length_penalty) − λ × security_penalty
```

Where:
- `log_prob`: Cumulative log probability of the token sequence
- `length_penalty`: Prevents favoring short sequences
- `λ`: Configurable weight (0 = ignore security, 1 = security-first)
- `security_penalty`: Sum of severity weights for detected issues

### Severity Weights
```python
SEVERITY_WEIGHTS = {
    'CRITICAL': float('inf'),  # Kill beam immediately
    'ERROR': 2.0,
    'WARNING': 0.5,
    'INFO': 0.1,
}
```

### Beam Search Integration

```
for step in range(max_tokens):
    for beam in active_beams:
        # 1. Generate next token candidates
        logits = model.forward(beam.tokens)
        top_k_tokens = get_top_k(logits, k=beam_width)

        # 2. Extend beam with each candidate
        for token in top_k_tokens:
            new_beam = beam.extend(token)

            # 3. Update constraint tracker (syntax state)
            new_beam.tracker.update(token)

            # 4. Check if at security boundary
            if new_beam.tracker.at_statement_boundary():
                # 5. Run security analysis
                code = extract_code(new_beam)
                issues = analyze_security(code)

                # 6. Compute penalty
                penalty = sum(SEVERITY_WEIGHTS[i.severity] for i in issues)
                new_beam.security_penalty += penalty

                # 7. Check for hard fail
                if any(i.severity == 'CRITICAL' for i in issues):
                    new_beam.status = FAILED
                    continue

            # 8. Compute score
            new_beam.score = (
                new_beam.log_prob / length_penalty(len(new_beam.tokens))
                - λ * new_beam.security_penalty
            )

            candidates.append(new_beam)

    # 9. Prune to top beam_width
    active_beams = sorted(candidates, key=lambda b: b.score)[:beam_width]
```

### Key Components

**ConstraintTracker**: Maintains per-beam parse state
```python
class ConstraintTracker:
    def __init__(self):
        self.bracket_stack = []  # ()[]{}
        self.in_string = False
        self.string_char = None
        self.in_comment = False

    def at_statement_boundary(self) -> bool:
        """True if current position is safe to analyze"""
        return (
            not self.in_string and
            not self.in_comment and
            len(self.bracket_stack) == 0
        )
```

**SecurityCache**: LRU cache for analysis results
```python
class SecurityCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, code_hash):
        if code_hash in self.cache:
            self.cache.move_to_end(code_hash)
            return self.cache[code_hash]
        return None
```

## Common Questions

**Q: Why not just filter bad outputs after generation?**
A: Three reasons:
1. **Efficiency**: Don't waste tokens on doomed paths
2. **Steering**: Security influences which path the model takes
3. **Early exit**: Critical issues kill beams immediately

**Q: Doesn't this slow down generation?**
A: Yes, slightly. But:
- We only analyze at statement boundaries (not every token)
- Caching prevents redundant analysis
- The overhead is ~10-20% in our benchmarks
- Trade-off: slightly slower, but more secure

**Q: How do you handle incomplete code?**
A: The ConstraintTracker knows when code is incomplete:
- Unclosed brackets/parens
- Incomplete strings
- Partial statements
We skip security analysis on incomplete code (can't parse AST anyway).

**Q: What security issues does it catch?**
A: Currently:
- `os.system`, `subprocess` (shell execution)
- `eval`, `exec`, `compile` (code injection)
- Unsafe file operations
- Pickle/deserialization attacks
- Import of known-dangerous modules

**Q: Is SABS a published algorithm?**
A: Not yet formally published, but the approach is novel. Related work exists on constrained decoding, but none integrate security verification specifically.

---

# TECH STACKS SUMMARY

## Nexus
```
Frontend: Next.js 14, TypeScript, TailwindCSS, Framer Motion
Backend: Supabase (Postgres, Auth, Real-time, Edge Functions)
AI: LangGraph, OpenAI/Anthropic APIs
Deployment: Vercel
```

## VibeGuard
```
CLI: Node.js, TypeScript, Commander.js
Analysis: AST parsing (Babel), regex patterns
Registry APIs: npm, PyPI REST APIs
Landing: Next.js, TailwindCSS
Deployment: npm registry, Vercel (landing)
```

## SelfEngine
```
Core: Python 3.8+
LLM: llama.cpp (llama-cpp-python bindings)
Analysis: Python AST module
Math: NumPy for scoring
Testing: pytest (158 tests)
```

---

# COMPETITIVE LANDSCAPE

## Nexus Competitors
| Competitor | What They Do | Why Nexus is Better |
|------------|--------------|---------------------|
| ChatGPT | Single-shot responses | No iteration, no specialization |
| PromptPerfect | Prompt optimization | Single agent, no human-in-loop |
| Custom GPTs | Pre-configured prompts | Static, not adaptive |

## VibeGuard Competitors
| Competitor | What They Do | Why VibeGuard is Better |
|------------|--------------|-------------------------|
| Snyk | Dependency scanning | Doesn't catch hallucinations |
| Semgrep | Static analysis | Not AI-aware, no auto-fix |
| Dependabot | Dependency updates | Only known vulnerabilities |

## SelfEngine Competitors
| Competitor | What They Do | Why SelfEngine is Better |
|------------|--------------|--------------------------|
| Post-hoc filtering | Generate then filter | Wastes compute on bad paths |
| Constrained decoding | Grammar constraints | No security awareness |
| Fine-tuning | Train secure model | Expensive, not adaptive |

---

# QUICK QUIZ (Test Yourself)

1. How many agents does Nexus use? What are their names?
2. What three categories of issues does VibeGuard detect?
3. What does SABS stand for?
4. What's the formula for beam scoring in SelfEngine?
5. Why do we check security at statement boundaries, not every token?
6. What's the λ parameter in SABS?
7. How does VibeGuard detect hallucinated packages?
8. What database does Nexus use and why?
9. How many tests does SelfEngine have passing?
10. What's the auto-patching flow in VibeGuard?

**Answers at bottom of document**

---

# ANSWERS

1. 4 agents: Clarifier, Drafter, Critic, Finalizer
2. Supply chain (hallucinations), Secrets (credentials), Misconfigurations (insecure defaults)
3. Security-Aware Beam Search
4. `score = log_prob / length_penalty − λ × security_penalty`
5. Because incomplete code can't be parsed for AST analysis, and checking every token would be too slow
6. The security weight - higher = more security-focused, 0 = ignore security
7. Parses imports, extracts package names, queries npm/PyPI APIs, flags if package doesn't exist
8. Supabase (Postgres) - relational for structured data, real-time subscriptions, built-in auth, row-level security
9. 158
10. Detect issue → Extract context → Generate LLM prompt → Get fix → Validate → Apply/suggest
