"""Traceability audit: verify every paper table value against run directories.

Prints a report of which claimed values are traceable, which match, and
which are missing or contradicted. Run this before submission.

Usage:
    python scripts/audit_traceability.py [--runs-dir runs]

Exit code: 0 if all values trace cleanly, 1 if any FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    description: str
    table: str
    run_dir: str          # relative to runs-dir
    key: str              # key inside summary.json
    paper_value: float
    tolerance: float = 0.0005  # max acceptable rounding gap


def run_checks(runs_dir: Path, checks: list[Check]) -> list[dict]:
    results = []
    for c in checks:
        p = runs_dir / c.run_dir / "summary.json"
        row = {
            "description": c.description,
            "table": c.table,
            "run_dir": c.run_dir,
            "paper_value": c.paper_value,
            "actual": None,
            "delta": None,
            "status": None,
        }
        if not p.exists():
            row["status"] = "FAIL: no run directory"
        else:
            try:
                d = json.loads(p.read_text())
            except Exception as e:
                row["status"] = f"FAIL: json error ({e})"
                results.append(row)
                continue
            actual = d.get(c.key)
            if actual is None:
                row["status"] = f"FAIL: key '{c.key}' missing in summary.json"
            else:
                row["actual"] = actual
                row["delta"] = actual - c.paper_value
                if abs(row["delta"]) <= c.tolerance:
                    row["status"] = "OK"
                else:
                    row["status"] = f"FAIL: delta={row['delta']:+.4f}"
        results.append(row)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Every value that appears in a paper table, with its source run.     #
    # Add new entries here whenever a new table is added to the paper.    #
    # ------------------------------------------------------------------ #
    checks = [
        # ---- Table 1: single-run exploratory baseline ladder -----------
        Check("B0 mIoU_fg",     "Table 1 (exploratory)", "B0",            "best_miou_fg", 0.673),
        Check("B1a mIoU_fg",    "Table 1 (exploratory)", "B1a",           "best_miou_fg", 0.657),
        Check("B2_best mIoU_fg","Table 1 (exploratory)", "B2_dt_v4_w10",  "best_miou_fg", 0.683),
        Check("B3 mIoU_fg",     "Table 1 (exploratory)", "B3",            "best_miou_fg", 0.668),
        Check("B5 mIoU_fg",     "Table 1 (exploratory)", "B5",            "best_miou_fg", 0.646),

        # ---- Table 2: multi-seed baseline validation (seed columns) ----
        # Tolerance 0.001: values reported to 3dp; max rounding error is 0.0005.
        Check("B0  seed 0",   "Table 2", "B0_s0",         "best_miou_fg", 0.701, tolerance=0.001),
        Check("B0  seed 7",   "Table 2", "B0_s7",         "best_miou_fg", 0.681, tolerance=0.001),
        Check("B0  seed 42",  "Table 2", "B0_s42",        "best_miou_fg", 0.663, tolerance=0.001),
        Check("B0  seed 99",  "Table 2", "B0_s99",        "best_miou_fg", 0.701, tolerance=0.001),
        Check("B0  seed 123", "Table 2", "B0_s123",       "best_miou_fg", 0.669, tolerance=0.001),
        Check("B1a seed 0",   "Table 2", "B1a_s0",        "best_miou_fg", 0.693, tolerance=0.001),
        Check("B1a seed 7",   "Table 2", "B1a_s7",        "best_miou_fg", 0.693, tolerance=0.001),
        Check("B1a seed 42",  "Table 2", "B1a_s42",       "best_miou_fg", 0.667, tolerance=0.001),
        Check("B1a seed 99",  "Table 2", "B1a_s99",       "best_miou_fg", 0.703, tolerance=0.001),
        Check("B1a seed 123", "Table 2", "B1a_s123",      "best_miou_fg", 0.659, tolerance=0.001),
        Check("B2  seed 0",   "Table 2", "B2_best_s0",    "best_miou_fg", 0.686, tolerance=0.001),
        Check("B2  seed 7",   "Table 2", "B2_best_s7",    "best_miou_fg", 0.687, tolerance=0.001),
        Check("B2  seed 42",  "Table 2", "B2_best_s42",   "best_miou_fg", 0.672, tolerance=0.001),
        Check("B2  seed 99",  "Table 2", "B2_best_s99",   "best_miou_fg", 0.691, tolerance=0.001),
        Check("B2  seed 123", "Table 2", "B2_best_s123",  "best_miou_fg", 0.670, tolerance=0.001),
        Check("B3  seed 0",   "Table 2", "B3_s0",         "best_miou_fg", 0.688, tolerance=0.001),
        Check("B3  seed 7",   "Table 2", "B3_s7",         "best_miou_fg", 0.688, tolerance=0.001),
        Check("B3  seed 42",  "Table 2", "B3_s42",        "best_miou_fg", 0.664, tolerance=0.001),
        Check("B3  seed 99",  "Table 2", "B3_s99",        "best_miou_fg", 0.693, tolerance=0.001),
        Check("B3  seed 123", "Table 2", "B3_s123",       "best_miou_fg", 0.663, tolerance=0.001),
        Check("B5  seed 0",   "Table 2", "B5_s0",         "best_miou_fg", 0.680, tolerance=0.001),
        Check("B5  seed 7",   "Table 2", "B5_s7",         "best_miou_fg", 0.680, tolerance=0.001),
        Check("B5  seed 42",  "Table 2", "B5_s42",        "best_miou_fg", 0.664, tolerance=0.001),
        Check("B5  seed 99",  "Table 2", "B5_s99",        "best_miou_fg", 0.688, tolerance=0.001),
        Check("B5  seed 123", "Table 2", "B5_s123",       "best_miou_fg", 0.671, tolerance=0.001),

        # ---- Table (DG multi-seed) -------------------------------------
        Check("ERM  DamSeg", "Tab DG multi-seed", "D1_erm_s42",        "final_damseg_miou_fg", 0.675,  tolerance=0.010),
        Check("Mix  s2ds 42","Tab DG multi-seed", "D2c_mixstyle_s42",  "final_s2ds_miou_fg",   0.117,  tolerance=0.010),

        # ---- Tab DG exploratory (removed from paper 2026-10-08) --------
        # The single-run DG table (ERM s2ds=0.110, MixStyle s2ds=0.167) was
        # removed from the paper: no artifact was retained, so the values
        # could not be independently verified.  Checks below are commented
        # out as a permanent record of what was removed and why.
        # Check("ERM  s2ds (exploratory)", "Tab DG exploratory",
        #       "D1_erm_s42", "final_s2ds_miou_fg", 0.110),  # delta=+0.020, no artifact
        # Check("MixStyle s2ds (exploratory)", "Tab DG exploratory",
        #       "D2c_mixstyle_s42", "final_s2ds_miou_fg", 0.167),  # delta=-0.044, no artifact
    ]

    results = run_checks(args.runs_dir, checks)

    # ---- Print report ------------------------------------------------- #
    col_w = [32, 22, 7, 7, 8, 35]
    header = (f"{'Description':<{col_w[0]}} {'Table':<{col_w[1]}} "
              f"{'Paper':>{col_w[2]}} {'Actual':>{col_w[3]}} "
              f"{'Delta':>{col_w[4]}}  {'Status'}")
    print(header)
    print("-" * (sum(col_w) + 6))

    n_fail = 0
    for r in results:
        actual_s = f"{r['actual']:.4f}" if r["actual"] is not None else "N/A"
        delta_s  = f"{r['delta']:+.4f}" if r["delta"] is not None else "N/A"
        status   = r["status"]
        if status != "OK":
            n_fail += 1
        print(f"{r['description']:<{col_w[0]}} {r['table']:<{col_w[1]}} "
              f"{r['paper_value']:>{col_w[2]}.3f} {actual_s:>{col_w[3]}} "
              f"{delta_s:>{col_w[4]}}  {status}")

    print()
    n_ok = len(results) - n_fail
    print(f"Result: {n_ok}/{len(results)} OK,  {n_fail} FAIL")

    if n_fail:
        print()
        print("ACTION REQUIRED before submission:")
        print("  Every FAIL means a paper value cannot be traced to a run artifact.")
        print("  Fix by: (a) recovering original logs and re-verifying, or")
        print("          (b) correcting/removing the claim from the paper.")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
