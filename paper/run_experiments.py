"""
Automated Experiment Runner for SABS Paper

This script runs all experiments required to reproduce the paper results:
1. Unit tests
2. Benchmarks
3. Security evaluation
4. Figure generation

Usage:
    python paper/run_experiments.py --all
    python paper/run_experiments.py --tests
    python paper/run_experiments.py --benchmarks
    python paper/run_experiments.py --eval
    python paper/run_experiments.py --figures
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime


def print_header(title: str):
    """Print a formatted section header."""
    width = 60
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width + "\n")


def print_status(status: str, success: bool):
    """Print status with color coding."""
    symbol = "[OK]" if success else "[FAIL]"
    print(f"{symbol} {status}")


def run_command(cmd: list, description: str, timeout: int = 600) -> bool:
    """Run a command and return success status."""
    print(f"\n>>> {description}")
    print(f"    Command: {' '.join(cmd)}")
    print("-" * 40)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            timeout=timeout
        )
        elapsed = time.time() - start_time
        success = result.returncode == 0
        print(f"\n    Elapsed: {elapsed:.1f}s")
        print_status(description, success)
        return success
    except subprocess.TimeoutExpired:
        print(f"\n    TIMEOUT after {timeout}s")
        print_status(description, False)
        return False
    except Exception as e:
        print(f"\n    ERROR: {e}")
        print_status(description, False)
        return False


def run_tests():
    """Run all unit tests."""
    print_header("RUNNING UNIT TESTS")

    results = []

    # Component tests
    results.append(run_command(
        [sys.executable, "-m", "pytest", "tests/test_sabs_components.py", "-v"],
        "Unit tests (test_sabs_components.py)",
        timeout=300
    ))

    # Integration tests
    results.append(run_command(
        [sys.executable, "-m", "pytest", "tests/test_sabs_integration.py", "-v"],
        "Integration tests (test_sabs_integration.py)",
        timeout=300
    ))

    # Edge case tests
    results.append(run_command(
        [sys.executable, "-m", "pytest", "tests/test_sabs_edge_cases.py", "-v"],
        "Edge case tests (test_sabs_edge_cases.py)",
        timeout=300
    ))

    return all(results)


def run_benchmarks():
    """Run performance benchmarks."""
    print_header("RUNNING BENCHMARKS")

    return run_command(
        [sys.executable, "-m", "eval.benchmark_sabs"],
        "Performance benchmarks",
        timeout=300
    )


def run_evaluation():
    """Run security evaluation."""
    print_header("RUNNING SECURITY EVALUATION")

    return run_command(
        [sys.executable, "-m", "eval.eval_sabs"],
        "Security stress test evaluation",
        timeout=300
    )


def generate_figures():
    """Generate paper figures."""
    print_header("GENERATING FIGURES")

    return run_command(
        [sys.executable, "paper/generate_figures.py"],
        "Generate paper figures",
        timeout=120
    )


def print_summary(results: dict):
    """Print final summary of all experiments."""
    print_header("EXPERIMENT SUMMARY")

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print(f"Results: {passed}/{total} experiments succeeded\n")

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        symbol = "[OK]  " if success else "[FAIL]"
        print(f"  {symbol} {name}")

    print()

    if passed == total:
        print("All experiments completed successfully!")
        print("\nPAPER READY FOR ARXIV")
        return True
    else:
        print(f"WARNING: {total - passed} experiment(s) failed.")
        print("Please check the output above for details.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run SABS paper experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python paper/run_experiments.py --all       # Run everything
    python paper/run_experiments.py --tests     # Only tests
    python paper/run_experiments.py --figures   # Only figures
        """
    )

    parser.add_argument("--all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--tests", action="store_true",
                       help="Run unit tests only")
    parser.add_argument("--benchmarks", action="store_true",
                       help="Run benchmarks only")
    parser.add_argument("--eval", action="store_true",
                       help="Run security evaluation only")
    parser.add_argument("--figures", action="store_true",
                       help="Generate figures only")

    args = parser.parse_args()

    # Default to --all if no specific option given
    if not any([args.all, args.tests, args.benchmarks, args.eval, args.figures]):
        args.all = True

    print_header("SABS PAPER EXPERIMENT RUNNER")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Working directory: {Path.cwd()}")

    results = {}

    if args.all or args.tests:
        results["Unit Tests"] = run_tests()

    if args.all or args.benchmarks:
        results["Benchmarks"] = run_benchmarks()

    if args.all or args.eval:
        results["Security Evaluation"] = run_evaluation()

    if args.all or args.figures:
        results["Figure Generation"] = generate_figures()

    success = print_summary(results)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
