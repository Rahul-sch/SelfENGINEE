SelfEngine — Security-Aware Decoding for Code LLMs

SelfEngine is a research-grade code generation engine that integrates static security analysis directly into beam search decoding. Instead of generating full programs and filtering them afterward, SelfEngine prunes unsafe generation paths as they are being written.

This repository implements Security-Aware Beam Search (SABS) — a decoding algorithm that steers a language model away from insecure code by coupling probabilistic generation with deterministic AST-based verification.

This is not a prompt filter.
This is not post-hoc scanning.
This is security-guided search over program space.

What Problem Does This Solve?

Most code-generating LLMs use this pipeline:

Prompt → Generate full program → Run static analysis → Reject or accept


That approach has two major flaws:

Wasted compute
Unsafe code is often obvious early (e.g., import subprocess), but the model continues generating hundreds of tokens before being rejected.

No steering
The verifier can only say “yes” or “no” after the fact. It cannot influence what the model writes.

SelfEngine replaces this with:

Prompt → Generate token → Verify prefix → Score beam → Prune unsafe futures → Continue


The verifier becomes part of the decoding loop.

What Is Security-Aware Beam Search (SABS)?

SABS is a modification of beam search where each beam is continuously evaluated by a static security analyzer.

Each partial program is treated as a candidate program prefix. At syntactic boundaries (top-level newlines), SABS:

Reconstructs the current code prefix

Runs AST-based security analysis

Assigns a security penalty based on detected issues

Updates the beam’s score:

score = log_probability / length_penalty − λ × security_penalty


Prunes beams that become unsafe or uncompetitive

Beams that trigger critical security issues are terminated immediately.
Beams with warnings are penalized but allowed to continue.

The result is that unsafe programs simply never finish being written.

What Makes This Different from Prior Work?

All of the components are standard:

Beam search

Static code analysis

AST visitors

Syntax tracking

What is new is the integration point.

This system treats the security verifier as a controller inside the decoding loop, not as a filter after decoding. That changes the optimization objective of generation itself.

The model is no longer optimizing only for likelihood — it is optimizing for likelihood under security constraints.

How It Works (Conceptually)

Each beam carries three evolving states:

Text state
Tokens generated so far

Syntax state
A per-beam incremental parser that knows whether we are at a top-level boundary

Security state
An accumulated penalty derived from static analysis

At each step:

for each beam:
    extend beam with next token
    update syntax state

    if at boundary and budget allows:
        analyze AST of prefix
        if critical issue:
            kill beam
        else:
            add penalty

prune beams by (logprob − λ × penalty)


This produces a search over program prefixes, not strings.

What Security Means Here

Security is defined by a static analysis engine that flags patterns such as:

shell execution

eval / exec

unsafe file operations

network execution

process spawning

Each issue has a severity:

CRITICAL

ERROR

WARNING

INFO

These map to numeric penalties. CRITICAL issues terminate beams. Others reduce their score.

The system does not claim formal safety. It provides measurable reduction in unsafe code.

What You Can Measure

The engine is designed for reproducible research.

It produces:

CSV files with per-prompt results

JSONL logs with per-beam decisions

plots generated from CSV only

Key metrics:

pass@1 on HumanEval

security violation rate

beams killed early

wasted tokens

latency overhead

This allows direct trade-off analysis between security and quality via λ.

What This Project Is (and Isn’t)

This project is:

A research platform for security-guided decoding

A reproducible benchmark for “safe code generation”

A concrete algorithm (SABS) with measurable trade-offs

This project is not:

A perfect sandbox

A complete malware defense

A magical jailbreak-proof system

It is a decoding algorithm that changes what the model is allowed to write.

Why This Matters

Most AI safety for code is done at the output level.
SelfEngine moves safety into the search process.

This turns security from a filter into an optimization objective.

That shift — from filtering text to steering program search — is the real contribution.