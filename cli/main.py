"""
SuperCoder CLI entry point.

Usage:
    # Baseline (entropy threshold)
    python -m cli.main "Write a python function to parse JSON safely"

    # SAEG (Semantic-Adaptive Entropy Gating) - novel algorithm
    python -m cli.main --use-saeg "Write a binary search function"

    # Export decisions for analysis
    python -m cli.main --use-saeg --export-decisions ./results.json "..."

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
        description="SuperCoder - System-2 Reasoning Engine for Code Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation (baseline)
  python -m cli.main "Write a fibonacci function"

  # Using SAEG (novel algorithm)
  python -m cli.main --use-saeg "Write a binary search function"

  # Experiment mode: export decisions for analysis
  python -m cli.main --use-saeg --export-decisions ./saeg_results.json "..."

  # Quiet mode
  python -m cli.main --log-every 0 "Parse a CSV file"
"""
    )

    # Required
    parser.add_argument("prompt", type=str, help="The prompt for code generation")

    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", type=str, default=None, help="Path to GGUF model file")
    model_group.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    model_group.add_argument("--n-gpu-layers", type=int, default=-1, help="GPU layers (-1 = all)")

    # Generation configuration
    gen_group = parser.add_argument_group("Generation Configuration")
    gen_group.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    gen_group.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    gen_group.add_argument("--beam-width", type=int, default=4, help="Beam width")
    gen_group.add_argument("--entropy-threshold", type=float, default=6.0, help="Base entropy threshold")

    # SAEG configuration (novel algorithm)
    saeg_group = parser.add_argument_group("SAEG Configuration (Novel Algorithm)")
    saeg_group.add_argument("--use-saeg", action="store_true", help="Enable Semantic-Adaptive Entropy Gating")
    saeg_group.add_argument("--saeg-alpha", type=float, default=1.0, help="Entropy weight in SAEG")
    saeg_group.add_argument("--saeg-beta", type=float, default=0.5, help="Semantic uncertainty weight in SAEG")
    saeg_group.add_argument("--saeg-gamma", type=float, default=0.3, help="Position weight in SAEG")
    saeg_group.add_argument("--saeg-mu", type=float, default=0.2, help="Syntactic complexity modifier")
    saeg_group.add_argument("--saeg-lambda", type=float, default=0.1, help="Position decay rate")
    saeg_group.add_argument("--saeg-k", type=int, default=5, help="Top-k for semantic uncertainty")

    # Experiment/logging
    exp_group = parser.add_argument_group("Experiment Configuration")
    exp_group.add_argument("--log-every", type=int, default=1, help="Log every N steps (0 = quiet)")
    exp_group.add_argument("--export-decisions", type=str, default=None, help="Export SAEG decisions to JSON file")

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

        # SAEG configuration
        use_saeg=args.use_saeg,
        saeg_alpha=args.saeg_alpha,
        saeg_beta=args.saeg_beta,
        saeg_gamma=args.saeg_gamma,
        saeg_mu=args.saeg_mu,
        saeg_lambda=args.saeg_lambda,
        saeg_k=args.saeg_k,

        # Experiment
        export_decisions=args.export_decisions is not None,
        export_path=args.export_decisions,
    )

    print(f"Strategy: {'SAEG (novel)' if args.use_saeg else 'BASELINE (entropy threshold)'}")
    print(f"Model: {model_path}")
    print("-" * 60)

    orch = Orchestrator(cfg)
    out = orch.generate(args.prompt)

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(out)

    # Print experiment summary if using SAEG
    if args.use_saeg:
        results = orch.get_experiment_results()
        if 'saeg_statistics' in results:
            stats = results['saeg_statistics']
            print("\n" + "-" * 40)
            print("SAEG STATISTICS")
            print("-" * 40)
            print(f"  Decisions: {stats.get('total_decisions', 0)}")
            print(f"  Branch rate: {stats.get('branch_rate', 0):.2%}")
            print(f"  Avg entropy: {stats.get('entropy_mean', 0):.3f}")
            print(f"  Avg semantic uncertainty: {stats.get('semantic_unc_mean', 0):.3f}")
            print(f"  Branch avg entropy: {stats.get('branch_avg_entropy', 0):.3f}")
            print(f"  Flow avg entropy: {stats.get('flow_avg_entropy', 0):.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
