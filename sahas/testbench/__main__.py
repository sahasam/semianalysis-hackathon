"""
CLI entry point for the testbench package.

Usage:
    python -m testbench <experiment> [args...]

Examples:
    python -m testbench patterns --test headline --n-agents 3 --n-tasks 20
    python -m testbench consensus --num-agents 7 --max-rounds 15
    python -m testbench profile --serial-only --num-agents 5
    python -m testbench efficiency --num-panels 7 --mem-fraction 0.88
    python -m testbench efficiency --summarize
    python -m testbench experiment --task "Explain KV cache reuse"
"""

import sys


EXPERIMENTS = {
    "patterns":   "testbench.experiments.patterns",
    "consensus":  "testbench.experiments.consensus",
    "profile":    "testbench.experiments.profile",
    "efficiency": "testbench.experiments.efficiency",
    "experiment": "testbench.experiments.experiment",
}


def print_usage():
    print("testbench — Energy benchmarking suite for multi-agent consensus\n")
    print("Usage: python -m testbench <experiment> [args...]\n")
    print("Experiments:")
    print("  patterns    4-pattern consensus comparison (headline experiment)")
    print("  consensus   NL consensus with multi-panel orchestration")
    print("  profile     Per-hop energy profiling (serial + parallel)")
    print("  efficiency  KV cache x concurrency sweep")
    print("  experiment  Simple 2-node drafter/reviewer pipeline")
    print("\nResults are saved to: results/<experiment>/<timestamp>_<tag>/")
    print("\nPass --help after the experiment name for experiment-specific options.")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0)

    experiment = sys.argv[1]
    if experiment not in EXPERIMENTS:
        print(f"Unknown experiment: {experiment}")
        print_usage()
        sys.exit(1)

    # Remove the experiment name from argv so argparse in each module works
    sys.argv = [f"testbench-{experiment}"] + sys.argv[2:]

    import importlib
    mod = importlib.import_module(EXPERIMENTS[experiment])
    mod.main()


if __name__ == "__main__":
    main()
