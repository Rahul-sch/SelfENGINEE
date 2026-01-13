"""
SABS Benchmarking Suite.

Benchmarks for Security-Aware Beam Search performance:
1. Latency overhead vs baseline
2. Cache hit rates
3. Verifier call frequency
4. Memory usage
5. Lambda sensitivity

Usage:
    python -m eval.benchmark_sabs --quick
    python -m eval.benchmark_sabs --full --output results/
    python -m eval.benchmark_sabs --lambda-grid 0.0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations
import argparse
import time
import sys
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from statistics import mean, stdev
import gc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.verify_static import (
    ParseState, SecurityCache, IncrementalResult, analyze_incremental,
    SEVERITY_WEIGHTS, compute_sec_penalty
)
from engine.constraints import ConstraintTracker
from engine.security_controller import SABSConfig, SecurityController
from engine.search import Beam, BeamManager, SearchConfig, BeamStatus


# =============================================================================
# Benchmark Data Classes
# =============================================================================

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    ops_per_sec: float
    extra: Dict = None

    def to_dict(self) -> Dict:
        d = asdict(self)
        if d['extra'] is None:
            d['extra'] = {}
        return d


@dataclass
class LambdaGridResult:
    """Result of lambda grid search."""
    lambda_value: float
    avg_penalty: float
    avg_score: float
    hard_fail_rate: float
    soft_penalty_rate: float
    no_issue_rate: float


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_analyze_incremental(
    iterations: int = 1000,
    use_cache: bool = True
) -> BenchmarkResult:
    """
    Benchmark analyze_incremental() function.

    Tests the core security analysis performance with different code samples.
    """
    # Test code samples
    samples = [
        "x = 1 + 2",  # Safe, simple
        "import os\nos.system('ls')",  # Dangerous
        "def foo():\n    return 42",  # Function
        "class Bar:\n    pass",  # Class
        "import subprocess\nsubprocess.call(['echo', 'hi'])",  # subprocess
        "eval(user_input)",  # eval
        "",  # Empty
        "for i in range(10):\n    print(i)",  # Loop
    ]

    cache = SecurityCache(max_size=1000) if use_cache else None

    times = []
    for _ in range(iterations):
        sample = samples[_ % len(samples)]
        start = time.perf_counter()
        result = analyze_incremental(sample, cache)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    cache_stats = cache.stats if cache else {}

    return BenchmarkResult(
        name="analyze_incremental" + ("_cached" if use_cache else "_uncached"),
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
        extra={"cache_stats": cache_stats}
    )


def benchmark_constraint_tracker(iterations: int = 5000) -> BenchmarkResult:
    """
    Benchmark ConstraintTracker.update() performance.
    """
    # Test code chunks
    chunks = [
        "def foo(",
        "x, y, z",
        "):",
        "\n    return x + y + z\n",
        "class Bar:\n",
        "    def __init__(self):\n",
        "        pass\n",
        "# Comment\n",
        '"""Docstring"""\n',
        "if True:\n    pass\n",
    ]

    tracker = ConstraintTracker()
    times = []

    for i in range(iterations):
        chunk = chunks[i % len(chunks)]
        if i % 100 == 0:  # Reset periodically
            tracker.reset()

        start = time.perf_counter()
        tracker.update(chunk)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="constraint_tracker_update",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
    )


def benchmark_security_cache(iterations: int = 10000) -> BenchmarkResult:
    """
    Benchmark SecurityCache get/put performance.
    """
    cache = SecurityCache(max_size=1000)

    # Pre-populate with some entries
    for i in range(500):
        cache.put(f"code_{i}", IncrementalResult(ParseState.PARSE_OK, [], float(i)))

    times = []
    hits = 0
    misses = 0

    for i in range(iterations):
        key = f"code_{i % 600}"  # Mix of hits and misses

        start = time.perf_counter()
        result = cache.get(key)
        if result is None:
            cache.put(key, IncrementalResult(ParseState.PARSE_OK, [], float(i)))
            misses += 1
        else:
            hits += 1
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="security_cache_operations",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
        extra={"hits": hits, "misses": misses, "hit_rate": hits / (hits + misses)}
    )


def benchmark_beam_scoring(iterations: int = 10000) -> BenchmarkResult:
    """
    Benchmark Beam.normalized_score() calculation.
    """
    config = SearchConfig(security_lambda=0.5, length_penalty=1.0)

    # Create test beams with varying properties
    beams = []
    for i in range(100):
        beam = Beam(
            tokens=list(range(i + 10)),
            prompt_length=5,
            log_prob=-float(i) / 10,
        )
        beam.sec_penalty = float(i % 10)
        beams.append(beam)

    times = []

    for i in range(iterations):
        beam = beams[i % len(beams)]
        start = time.perf_counter()
        score = beam.normalized_score(config)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="beam_normalized_score",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
    )


def benchmark_constraint_tracker_clone(iterations: int = 5000) -> BenchmarkResult:
    """
    Benchmark ConstraintTracker.clone() performance.
    """
    # Create tracker with some state
    base_tracker = ConstraintTracker()
    base_tracker.update("def foo(x, y):\n    if True:\n        ")

    times = []

    for _ in range(iterations):
        start = time.perf_counter()
        cloned = base_tracker.clone()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return BenchmarkResult(
        name="constraint_tracker_clone",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
    )


def benchmark_security_controller_check(iterations: int = 1000) -> BenchmarkResult:
    """
    Benchmark SecurityController.check_beam() end-to-end.
    """
    from unittest.mock import Mock

    config = SABSConfig(cache_enabled=True)
    controller = SecurityController(config)

    # Create mock beam
    beam = Mock()
    beam.sec_penalty = 0.0
    beam.tokens_since_security_check = 10
    beam.beam_id = "test"
    beam.current_position = 50

    # Test code samples
    samples = [
        "x = 1 + 2",
        "import os",
        "def foo(): pass",
        "eval(x)",
        "",
    ]

    times = []

    for i in range(iterations):
        code = samples[i % len(samples)]
        start = time.perf_counter()
        decision = controller.check_beam(beam, code, "gen1", i)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    stats = controller.get_statistics()

    return BenchmarkResult(
        name="security_controller_check_beam",
        iterations=iterations,
        total_time_ms=sum(times),
        avg_time_ms=mean(times),
        std_time_ms=stdev(times) if len(times) > 1 else 0,
        min_time_ms=min(times),
        max_time_ms=max(times),
        ops_per_sec=1000 / mean(times) if mean(times) > 0 else 0,
        extra={"controller_stats": stats}
    )


def run_lambda_grid_search(
    lambda_values: List[float],
    test_codes: List[str]
) -> List[LambdaGridResult]:
    """
    Run lambda grid search to analyze penalty distribution.
    """
    results = []

    for lambda_val in lambda_values:
        config = SABSConfig(
            lambda_weight=lambda_val,
            hard_fail_on_critical=True
        )
        controller = SecurityController(config)

        search_config = SearchConfig(security_lambda=lambda_val)

        scores = []
        penalties = []
        hard_fails = 0
        soft_penalties = 0
        no_issues = 0

        from unittest.mock import Mock

        for code in test_codes:
            beam = Mock()
            beam.sec_penalty = 0.0
            beam.tokens_since_security_check = 10
            beam.beam_id = "test"
            beam.current_position = 50

            decision = controller.check_beam(beam, code, "grid", 0)

            if decision.should_terminate:
                hard_fails += 1
            elif decision.sec_penalty_delta > 0:
                soft_penalties += 1
            else:
                no_issues += 1

            penalties.append(decision.sec_penalty_delta)

            # Create a real beam for scoring
            real_beam = Beam(tokens=list(range(20)), prompt_length=5, log_prob=-5.0)
            real_beam.sec_penalty = decision.total_sec_penalty
            scores.append(real_beam.normalized_score(search_config))

            controller.reset("grid")

        total = len(test_codes)
        results.append(LambdaGridResult(
            lambda_value=lambda_val,
            avg_penalty=mean(penalties),
            avg_score=mean(scores),
            hard_fail_rate=hard_fails / total,
            soft_penalty_rate=soft_penalties / total,
            no_issue_rate=no_issues / total,
        ))

    return results


# =============================================================================
# Main Runner
# =============================================================================

def run_quick_benchmarks() -> List[BenchmarkResult]:
    """Run quick benchmark suite."""
    print("Running quick benchmarks...")
    results = []

    print("  - analyze_incremental (cached)...")
    results.append(benchmark_analyze_incremental(iterations=500, use_cache=True))

    print("  - analyze_incremental (uncached)...")
    results.append(benchmark_analyze_incremental(iterations=500, use_cache=False))

    print("  - constraint_tracker_update...")
    results.append(benchmark_constraint_tracker(iterations=2000))

    print("  - security_cache_operations...")
    results.append(benchmark_security_cache(iterations=5000))

    print("  - beam_normalized_score...")
    results.append(benchmark_beam_scoring(iterations=5000))

    print("  - constraint_tracker_clone...")
    results.append(benchmark_constraint_tracker_clone(iterations=2000))

    print("  - security_controller_check_beam...")
    results.append(benchmark_security_controller_check(iterations=500))

    return results


def run_full_benchmarks() -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    print("Running full benchmarks...")
    results = []

    print("  - analyze_incremental (cached)...")
    results.append(benchmark_analyze_incremental(iterations=5000, use_cache=True))

    print("  - analyze_incremental (uncached)...")
    results.append(benchmark_analyze_incremental(iterations=5000, use_cache=False))

    print("  - constraint_tracker_update...")
    results.append(benchmark_constraint_tracker(iterations=20000))

    print("  - security_cache_operations...")
    results.append(benchmark_security_cache(iterations=50000))

    print("  - beam_normalized_score...")
    results.append(benchmark_beam_scoring(iterations=50000))

    print("  - constraint_tracker_clone...")
    results.append(benchmark_constraint_tracker_clone(iterations=20000))

    print("  - security_controller_check_beam...")
    results.append(benchmark_security_controller_check(iterations=5000))

    return results


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Header
    print(f"{'Name':<40} {'Avg (ms)':<12} {'Std (ms)':<12} {'Ops/sec':<12}")
    print("-" * 80)

    for r in results:
        print(f"{r.name:<40} {r.avg_time_ms:<12.4f} {r.std_time_ms:<12.4f} {r.ops_per_sec:<12.1f}")

    print("-" * 80)


def print_lambda_results(results: List[LambdaGridResult]) -> None:
    """Print lambda grid search results."""
    print("\n" + "=" * 80)
    print("LAMBDA GRID SEARCH RESULTS")
    print("=" * 80)

    print(f"{'Lambda':<10} {'Avg Penalty':<15} {'Avg Score':<12} {'Hard Fail %':<12} {'Soft %':<10} {'Clean %':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r.lambda_value:<10.2f} {r.avg_penalty:<15.2f} {r.avg_score:<12.2f} "
              f"{r.hard_fail_rate * 100:<12.1f} {r.soft_penalty_rate * 100:<10.1f} "
              f"{r.no_issue_rate * 100:<10.1f}")


def save_results(results: List[BenchmarkResult], output_dir: Path) -> None:
    """Save results to JSON and CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Saved JSON to {json_path}")

    # CSV
    csv_path = output_dir / "benchmark_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "iterations", "avg_time_ms", "std_time_ms", "ops_per_sec"])
        for r in results:
            writer.writerow([r.name, r.iterations, r.avg_time_ms, r.std_time_ms, r.ops_per_sec])
    print(f"Saved CSV to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SABS Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--quick", action="store_true", help="Run quick benchmarks")
    mode_group.add_argument("--full", action="store_true", help="Run full benchmarks")

    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=None,
        help="Run lambda grid search with specified values"
    )

    args = parser.parse_args()

    # Default to quick if nothing specified
    if not args.quick and not args.full and not args.lambda_grid:
        args.quick = True

    results = []

    if args.quick:
        results = run_quick_benchmarks()
    elif args.full:
        results = run_full_benchmarks()

    if results:
        print_results(results)

        if args.output:
            save_results(results, Path(args.output))

    if args.lambda_grid:
        print("\nRunning lambda grid search...")
        test_codes = [
            "x = 1",
            "import os",
            "import subprocess",
            "os.system('ls')",
            "eval(x)",
            "exec(code)",
            "def foo(): pass",
            "open('file.txt', 'w').write('data')",
            "subprocess.call(['cmd'])",
            "pickle.load(f)",
        ]

        lambda_results = run_lambda_grid_search(args.lambda_grid, test_codes)
        print_lambda_results(lambda_results)

    print("\nBenchmarks complete.")


if __name__ == "__main__":
    main()
