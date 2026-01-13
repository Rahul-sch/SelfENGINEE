"""
Experiment Script: BASELINE vs SAEG Comparison

Runs both strategies on the same prompts and collects metrics.

Usage:
    python experiments/run_comparison.py --model ./models/model.gguf

Output:
    ./experiments/results/baseline_results.json
    ./experiments/results/saeg_results.json
    ./experiments/results/comparison_summary.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.orchestrator import Orchestrator, OrchestratorConfig


# Test prompts for code generation
TEST_PROMPTS = [
    "Write a Python function to check if a number is prime",
    "Write a Python function for binary search in a sorted list",
    "Write a Python function to find the nth Fibonacci number",
    "Write a Python function to reverse a linked list",
    "Write a Python function to merge two sorted lists",
    "Write a Python function to validate a palindrome string",
    "Write a Python function for depth-first search on a graph",
    "Write a Python function to find the longest common subsequence",
    "Write a Python function to implement a stack using a list",
    "Write a Python function to parse JSON safely with error handling",
]


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    prompt: str
    strategy: str
    output: str
    generation_time: float
    num_tokens: int
    branch_rate: float = 0.0
    avg_entropy: float = 0.0
    avg_semantic_unc: float = 0.0
    completed: bool = True
    error: str = ""


def run_single_experiment(
    prompt: str,
    model_path: str,
    use_saeg: bool,
    max_tokens: int = 256,
) -> ExperimentResult:
    """Run a single generation experiment."""
    strategy = "SAEG" if use_saeg else "BASELINE"

    cfg = OrchestratorConfig(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,
        max_tokens=max_tokens,
        beam_width=4,
        entropy_threshold=6.0,
        temperature=0.7,
        log_every=0,  # Quiet mode

        # SAEG settings
        use_saeg=use_saeg,
        saeg_alpha=1.0,
        saeg_beta=0.5,
        saeg_gamma=0.3,
    )

    try:
        orch = Orchestrator(cfg)

        start_time = time.time()
        output = orch.generate(prompt)
        generation_time = time.time() - start_time

        # Get statistics
        results = orch.get_experiment_results()
        stats = results.get('saeg_statistics', {})

        return ExperimentResult(
            prompt=prompt,
            strategy=strategy,
            output=output,
            generation_time=generation_time,
            num_tokens=len(output.split()),  # Approximation
            branch_rate=stats.get('branch_rate', 0.0),
            avg_entropy=stats.get('entropy_mean', 0.0),
            avg_semantic_unc=stats.get('semantic_unc_mean', 0.0),
            completed=True,
        )

    except Exception as e:
        return ExperimentResult(
            prompt=prompt,
            strategy=strategy,
            output="",
            generation_time=0.0,
            num_tokens=0,
            completed=False,
            error=str(e),
        )


def run_comparison(
    model_path: str,
    output_dir: str,
    prompts: List[str] = None,
    max_tokens: int = 256,
) -> Dict[str, Any]:
    """Run full comparison between BASELINE and SAEG."""

    if prompts is None:
        prompts = TEST_PROMPTS

    os.makedirs(output_dir, exist_ok=True)

    baseline_results = []
    saeg_results = []

    print("=" * 60)
    print("RUNNING BASELINE vs SAEG COMPARISON")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {max_tokens}")
    print()

    for i, prompt in enumerate(prompts):
        print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")

        # Run BASELINE
        print("  BASELINE...", end=" ", flush=True)
        baseline_result = run_single_experiment(
            prompt, model_path, use_saeg=False, max_tokens=max_tokens
        )
        baseline_results.append(asdict(baseline_result))
        status = "OK" if baseline_result.completed else "FAIL"
        print(f"{status} ({baseline_result.generation_time:.2f}s)")

        # Run SAEG
        print("  SAEG...", end=" ", flush=True)
        saeg_result = run_single_experiment(
            prompt, model_path, use_saeg=True, max_tokens=max_tokens
        )
        saeg_results.append(asdict(saeg_result))
        status = "OK" if saeg_result.completed else "FAIL"
        print(f"{status} ({saeg_result.generation_time:.2f}s)")

        print()

    # Save results
    baseline_path = os.path.join(output_dir, "baseline_results.json")
    saeg_path = os.path.join(output_dir, "saeg_results.json")

    with open(baseline_path, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    print(f"Saved: {baseline_path}")

    with open(saeg_path, 'w') as f:
        json.dump(saeg_results, f, indent=2)
    print(f"Saved: {saeg_path}")

    # Compute summary
    summary = compute_summary(baseline_results, saeg_results)

    summary_path = os.path.join(output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    # Print summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"BASELINE:")
    print(f"  Success rate: {summary['baseline']['success_rate']:.2%}")
    print(f"  Avg time: {summary['baseline']['avg_time']:.2f}s")
    print()
    print(f"SAEG:")
    print(f"  Success rate: {summary['saeg']['success_rate']:.2%}")
    print(f"  Avg time: {summary['saeg']['avg_time']:.2f}s")
    print(f"  Avg branch rate: {summary['saeg']['avg_branch_rate']:.2%}")
    print()
    print(f"IMPROVEMENT:")
    print(f"  Time: {summary['improvement']['time_pct']:+.1f}%")

    return summary


def compute_summary(baseline_results: List[Dict], saeg_results: List[Dict]) -> Dict:
    """Compute summary statistics."""

    def stats_for(results: List[Dict]) -> Dict:
        completed = [r for r in results if r['completed']]
        return {
            'total': len(results),
            'completed': len(completed),
            'success_rate': len(completed) / len(results) if results else 0,
            'avg_time': sum(r['generation_time'] for r in completed) / len(completed) if completed else 0,
            'avg_tokens': sum(r['num_tokens'] for r in completed) / len(completed) if completed else 0,
        }

    baseline_stats = stats_for(baseline_results)
    saeg_stats = stats_for(saeg_results)

    # SAEG-specific stats
    saeg_completed = [r for r in saeg_results if r['completed']]
    if saeg_completed:
        saeg_stats['avg_branch_rate'] = sum(r['branch_rate'] for r in saeg_completed) / len(saeg_completed)
        saeg_stats['avg_entropy'] = sum(r['avg_entropy'] for r in saeg_completed) / len(saeg_completed)
        saeg_stats['avg_semantic_unc'] = sum(r['avg_semantic_unc'] for r in saeg_completed) / len(saeg_completed)
    else:
        saeg_stats['avg_branch_rate'] = 0
        saeg_stats['avg_entropy'] = 0
        saeg_stats['avg_semantic_unc'] = 0

    # Compute improvement
    time_improvement = 0
    if baseline_stats['avg_time'] > 0:
        time_improvement = ((saeg_stats['avg_time'] - baseline_stats['avg_time']) / baseline_stats['avg_time']) * 100

    return {
        'baseline': baseline_stats,
        'saeg': saeg_stats,
        'improvement': {
            'time_pct': time_improvement,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Run BASELINE vs SAEG comparison")
    parser.add_argument("--model", type=str, required=True, help="Path to GGUF model")
    parser.add_argument("--output-dir", type=str, default="./experiments/results", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts to run (default: all)")
    args = parser.parse_args()

    prompts = TEST_PROMPTS
    if args.num_prompts:
        prompts = prompts[:args.num_prompts]

    run_comparison(
        model_path=args.model,
        output_dir=args.output_dir,
        prompts=prompts,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
