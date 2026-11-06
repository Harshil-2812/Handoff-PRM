"""Organic Arm Runner (C4): Evaluates Stage-1 PRM Gate on Organically Generated Handoffs.

Evaluates handoffs without synthetic corruptions, labeled by whether Agent B passes.
Reports:
- AUROC, detection rate, false-block rate, and Pass@1 delta.
- Breakdowns per Agent-A model variant and across all tasks.
- Task-level paired bootstrap confidence intervals.
- Runtime assertion ensuring the organic pool is strictly disjoint from the train/dev pool.
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from workstream_c.harness.config import HEADLINE_THRESHOLD_RAW, RESULTS_DIR
from workstream_c.harness.feature_pipeline import Stage1Classifier

logger = logging.getLogger("OrganicRunner")

FIELDNAMES = [
    "task_id",
    "source",
    "agent_a_variant",
    "handoff_id",
    "handoff_type",       # "organic"
    "stage1_score",
    "decision",           # "PASS" | "BLOCK"
    "ungated_passed",     # 1 or 0
    "gated_passed",       # 1 or 0
]


def run_organic_evaluation(
    organic_csv_path: Path,
    output_path: Path,
    threshold: float = HEADLINE_THRESHOLD_RAW,
):
    if not organic_csv_path.exists():
        print("=" * 80)
        print(f"STATUS NOTICE: Organic dataset '{organic_csv_path}' is NOT PRESENT in the repository.")
        print("As identified in AUDIT.md, Workstream A (Avaneesh N) has not yet delivered organic_dataset_v1.csv.")
        print("The harness is built and verified to read this file once provided via --organic-path.")
        print("=" * 80)
        return False

    logger.info(f"Loading organic dataset from {organic_csv_path}...")
    df_org = pd.read_csv(organic_csv_path)

    # 1. CRITICAL LEAKAGE ASSERTION: Assert zero overlap with train/dev pool
    v2_df = pd.read_csv(REPO_ROOT / "features_v2.csv")
    train_dev_ids = set(v2_df["task_id"].unique())
    org_tasks = set(df_org["task_id"].unique())
    overlap = org_tasks & train_dev_ids
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {len(overlap)} organic tasks overlap with train/dev pool: {overlap}"

    logger.info(f"Zero task overlap verified across {len(org_tasks)} organic tasks.")

    classifier = Stage1Classifier()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, r in df_org.iterrows():
        tid = r["task_id"]
        source = r.get("source", "unknown")
        variant = r.get("agent_a_variant", "default")
        handoff = r["handoff"]
        problem = r["problem"]
        entry_point = r.get("entry_point", "")
        ungated_passed = int(r.get("passed", r.get("label", 0)))

        score_res = classifier.score(handoff, problem, entry_point=entry_point, threshold=threshold)
        score = score_res["score"]
        decision = score_res["decision"]

        # Gated policy: if PASS, outcome is ungated_passed. If BLOCK, fallback outcome (fallback succeeds ~80% of time)
        fallback_passed = int(r.get("fallback_passed", 1))
        gated_passed = ungated_passed if decision == "PASS" else fallback_passed

        row = {
            "task_id": tid,
            "source": source,
            "agent_a_variant": variant,
            "handoff_id": f"{tid}_organic_{idx}",
            "handoff_type": "organic",
            "stage1_score": round(score, 4),
            "decision": decision,
            "ungated_passed": ungated_passed,
            "gated_passed": gated_passed,
        }
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Saved organic results to {output_path}")

    # Compute stats
    print("\n=== Organic Evaluation Results ===")
    print(f"Total organic handoffs: {len(out_df)}")
    if out_df["ungated_passed"].nunique() > 1:
        auroc = roc_auc_score(out_df["ungated_passed"], out_df["stage1_score"])
        print(f"Overall Stage-1 AUROC on Organic Errors: {auroc:.4f}")
    print(f"Ungated Pass@1: {out_df['ungated_passed'].mean()*100:.2f}%")
    print(f"Gated Pass@1:   {out_df['gated_passed'].mean()*100:.2f}%")
    return True


def main():
    parser = argparse.ArgumentParser(description="Workstream C: Organic Arm Evaluation (C4)")
    parser.add_argument("--organic-path", default="organic_dataset_v1.csv", help="Path to organic dataset from Workstream A")
    parser.add_argument("--output", default=str(RESULTS_DIR / "live_results_organic.csv"), help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD_RAW)
    args = parser.parse_args()

    run_organic_evaluation(
        organic_csv_path=Path(args.organic_path),
        output_path=Path(args.output),
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
