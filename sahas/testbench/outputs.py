"""
Tagged output management — structured filesystem paths for experiment results.

All results land under a root results/ directory, organized as:

    results/
    ├── patterns/
    │   └── 20260315_040000_3agents_20tasks/
    │       ├── results.json
    │       └── results.csv
    ├── consensus/
    │   └── 20260315_041000_7agents_15rounds/
    │       └── results.json
    ├── profile/
    │   └── 20260315_042000_serial_5agents/
    │       └── results.json
    ├── efficiency/
    │   └── 20260315_043000_m080_3panels/
    │       └── results.json
    └── experiment/
        └── 20260315_044000_2node/
            └── results.json
"""

import json
import os
from datetime import datetime
from pathlib import Path


# Root results directory — relative to the sahas/ working directory
_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"


def get_results_root() -> Path:
    """Return the absolute path to the results root directory."""
    return _RESULTS_ROOT


def make_run_dir(experiment: str, tag: str = "") -> Path:
    """
    Create and return a timestamped, tagged output directory.

    Args:
        experiment: Experiment name (patterns, consensus, profile, efficiency, experiment).
        tag: Additional tag string appended to the directory name (e.g. '3agents_20tasks').

    Returns:
        Path to the created directory, e.g.
        results/patterns/20260315_040000_3agents_20tasks/
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    parts = [ts]
    if tag:
        parts.append(tag)
    dirname = "_".join(parts)

    run_dir = _RESULTS_ROOT / experiment / dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data, run_dir: Path, filename: str = "results.json") -> Path:
    """Save a JSON file into the run directory. Returns the written path."""
    out = run_dir / filename
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {out}")
    return out


def save_csv_rows(rows: list[dict], fieldnames: list[str], run_dir: Path,
                  filename: str = "results.csv") -> Path:
    """Save a list of dicts as CSV into the run directory."""
    import csv
    out = run_dir / filename
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  Saved: {out} ({len(rows)} rows)")
    return out
