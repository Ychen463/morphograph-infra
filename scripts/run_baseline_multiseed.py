"""Run all 25 baseline multi-seed training runs sequentially.

Trains B0, B1a, B2_best, B3, B5 each with seeds 0, 7, 42, 99, 123.
Each run calls the appropriate train_b*.py script and writes results
(including per-class metrics) to runs/<baseline>_s<seed>/.

Usage:
    # Full run (all 25):
    python scripts/run_baseline_multiseed.py --data-root data/raw

    # Subset (resume / re-run specific baselines or seeds):
    python scripts/run_baseline_multiseed.py --data-root data/raw \\
        --baselines B0 B1a --seeds 0 7 42 99 123

    # Skip runs where summary.json already has "perclass" key:
    python scripts/run_baseline_multiseed.py --data-root data/raw --skip-done

    # Dry-run to see what would run:
    python scripts/run_baseline_multiseed.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


BASELINE_CONFIGS = {
    "B0": {
        "script": "scripts/train_b0.py",
        "extra_args": [],
    },
    "B1a": {
        "script": "scripts/train_b1a.py",
        "extra_args": [],
    },
    "B2_best": {
        "script": "scripts/train_b2.py",
        "extra_args": [
            "--skel-weight", "10.0",
            "--skel-loss-type", "mse",
            "--skel-unmask",
        ],
    },
    "B3": {
        "script": "scripts/train_b345.py",
        "extra_args": ["--baseline", "B3"],
    },
    "B5": {
        "script": "scripts/train_b345.py",
        "extra_args": ["--baseline", "B5"],
    },
}


def run_one(baseline: str, seed: int, data_root: Path, runs_dir: Path,
            dry_run: bool) -> bool:
    cfg = BASELINE_CONFIGS[baseline]
    out_dir = runs_dir / f"{baseline}_s{seed}"
    cmd = [
        sys.executable, cfg["script"],
        "--data-root", str(data_root),
        "--output", str(out_dir),
        "--seed", str(seed),
    ] + cfg["extra_args"]

    if dry_run:
        print(f"  [DRY] {' '.join(cmd)}")
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    print(f"  → {out_dir.name}  (log: {log_path})", flush=True)

    t0 = time.time()
    POLL_INTERVAL = 30  # seconds between progress prints

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        last_print = t0
        while True:
            try:
                proc.wait(timeout=POLL_INTERVAL)
                break  # process finished
            except subprocess.TimeoutExpired:
                pass
            elapsed_now = time.time() - t0
            # Print last non-empty line from log as a heartbeat
            last_line = ""
            try:
                with open(log_path) as lf:
                    for line in lf:
                        line = line.rstrip()
                        if line:
                            last_line = line
            except OSError:
                pass
            print(f"    [{elapsed_now/60:.1f}m] {last_line[-120:]}", flush=True)

    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"    FAILED (exit {proc.returncode}) after {elapsed/60:.1f}m — see {log_path}")
        return False

    # Quick sanity: check summary.json was written with perclass key
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        miou = s.get("best_miou_fg", "?")
        has_pc = "perclass" in s
        pc_note = f"  iou_crack={s['perclass']['iou_crack']:.4f}" if has_pc else "  [no perclass]"
        print(f"    OK  {elapsed/60:.1f}m  miou_fg={miou:.4f}{pc_note}")
    else:
        print(f"    OK  {elapsed/60:.1f}m  [no summary.json?]")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--baselines", nargs="+",
                        default=["B0", "B1a", "B2_best", "B3", "B5"])
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[0, 7, 42, 99, 123])
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip runs where summary.json already has 'perclass' key")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    runs = [(bl, s) for bl in args.baselines for s in args.seeds]
    total = len(runs)

    if args.skip_done:
        filtered = []
        for bl, s in runs:
            p = args.runs_dir / f"{bl}_s{s}" / "summary.json"
            if p.exists() and "perclass" in json.loads(p.read_text()):
                print(f"  [SKIP] {bl}_s{s} already has perclass metrics")
            else:
                filtered.append((bl, s))
        runs = filtered

    print(f"\nBaseline multi-seed runner: {len(runs)}/{total} runs to execute")
    print(f"Baselines: {args.baselines}")
    print(f"Seeds:     {args.seeds}")
    print(f"Runs dir:  {args.runs_dir}\n")

    n_ok = n_fail = 0
    for i, (bl, s) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {bl}  seed={s}")
        ok = run_one(bl, s, args.data_root, args.runs_dir, args.dry_run)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone: {n_ok} OK, {n_fail} failed out of {len(runs)} runs.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
