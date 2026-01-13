"""
SuperCoder CLI entry point.

Usage:
    python -m cli.main "Write a python function to parse JSON safely"

Set model path via environment variable or use default:
    set SUPERCODER_MODEL=C:\\path\\to\\your.gguf
    python -m cli.main "..."
"""

from __future__ import annotations
import argparse
import os

from engine.orchestrator import Orchestrator, OrchestratorConfig


def main():
    parser = argparse.ArgumentParser(
        prog="supercoder",
        description="SuperCoder - System-2 Reasoning Engine for Code Generation"
    )
    parser.add_argument("prompt", type=str, help="The prompt for code generation")
    parser.add_argument("--model", type=str, default=None, help="Path to GGUF model file")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--entropy-threshold", type=float, default=6.0, help="Entropy threshold for branching")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width")
    parser.add_argument("--log-every", type=int, default=1, help="Log every N steps (0 = quiet)")
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not model_path:
        model_path = os.environ.get("SUPERCODER_MODEL", "")
    if not model_path:
        model_path = r"./models/model.gguf"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at: {model_path}")
        print("Set SUPERCODER_MODEL environment variable or use --model flag")
        return 1

    cfg = OrchestratorConfig(
        model_path=model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_batch=512,
        beam_width=args.beam_width,
        max_branches=8,
        max_tokens=args.max_tokens,
        checkpoint_interval=16,
        entropy_threshold=args.entropy_threshold,
        branch_count=2,
        temperature=args.temperature,
        top_k=40,
        top_p=0.95,
        log_every=args.log_every if args.log_every > 0 else 999999,
    )

    orch = Orchestrator(cfg)
    out = orch.generate(args.prompt)

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(out)

    return 0


if __name__ == "__main__":
    exit(main())
