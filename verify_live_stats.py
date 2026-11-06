"""Verification of Live Evaluation Statistics (verify_live_stats.py).

Recomputes every reported live metric from the raw CSVs:
1. Strict assertions for frozen schema completeness and zero task overlap with train/dev pool.
2. Downstream Pass@1 per arm overall, per-source, and per-condition (clean vs. corrupted).
3. Exact McNemar test (paired) with contingency counts (b: ungated pass / gated fail, c: ungated fail / gated pass).
4. Task-level paired bootstrap confidence intervals (10,000 resamples, fixed seed 42) resampling tasks, not rows.
5. Per-seed breakdowns (42, 43, 44) plus across-seed aggregates.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent
FROZEN_SCHEMA_COLS = [
    "task_id",
    "source",
    "handoff_id",
    "handoff_type",
    "label",
    "label_policy",
    "seed",
    "features",
    "stage1_score",
    "decision",
    "outcome",
    "model_agent_a",
    "model_agent_b",
    "model_judge",
    "temp_agent_a",
    "temp_agent_b",
    "temp_judge",
]


def assert_schema_and_zero_leakage(df: pd.DataFrame, train_dev_csv: Path = REPO_ROOT / "features_v2.csv"):
    """Validates frozen schema completeness and asserts zero task leakage."""
    print("--> 1. Running Schema Completeness & Zero-Leakage Assertions...")
    
    # Check schema
    missing_cols = [c for c in FROZEN_SCHEMA_COLS if c not in df.columns]
    assert not missing_cols, f"SCHEMA VIOLATION: Missing frozen columns: {missing_cols}"
    
    for c in FROZEN_SCHEMA_COLS:
        null_count = df[c].isnull().sum()
        assert null_count == 0, f"SCHEMA VIOLATION: Column '{c}' contains {null_count} null values!"
    print("    [PASS] All 17 frozen results schema columns are present with zero null values.")

    # Check leakage against train/dev pool
    if train_dev_csv.exists():
        v2_df = pd.read_csv(train_dev_csv)
        train_dev_tasks = set(v2_df["task_id"].unique())
        eval_tasks = set(df["task_id"].unique())
        overlap = eval_tasks & train_dev_tasks
        assert len(overlap) == 0, f"CRITICAL LEAKAGE DETECTED: {len(overlap)} tasks overlap with train/dev pool: {overlap}"
        print(f"    [PASS] Zero task leakage confirmed: {len(eval_tasks)} evaluation tasks have 0 overlap with {len(train_dev_tasks)} train/dev tasks.")
    else:
        print("    [WARNING] features_v2.csv not found; skipping train/dev leakage check.")


def exact_mcnemar_test(b: int, c: int) -> float:
    """Computes exact two-sided p-value for paired McNemar test via binomial distribution."""
    n = b + c
    if n == 0:
        return 1.0
    res = stats.binomtest(k=b, n=n, p=0.5, alternative="two-sided")
    return float(res.pvalue)


def task_level_bootstrap_ci(
    task_values: Dict[str, Tuple[float, int]],
    n_resamples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Task-level bootstrap CI: resamples tasks (with all their associated trials), not rows.
    
    task_values maps task_id -> (sum_of_metric_values, count_of_trials).
    """
    tasks = list(task_values.keys())
    sums = np.array([task_values[t][0] for t in tasks], dtype=np.float64)
    counts = np.array([task_values[t][1] for t in tasks], dtype=np.float64)
    n_tasks = len(tasks)
    
    total_sum = sums.sum()
    total_count = counts.sum()
    point_est = float(total_sum / total_count) if total_count > 0 else 0.0
    
    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(n_tasks, size=(n_resamples, n_tasks), replace=True)
    boot_sums = sums[sample_indices].sum(axis=1)
    boot_counts = counts[sample_indices].sum(axis=1)
    boot_estimates = boot_sums / np.maximum(boot_counts, 1e-9)
    
    ci_low = float(np.percentile(boot_estimates, 2.5))
    ci_high = float(np.percentile(boot_estimates, 97.5))
    return point_est, ci_low, ci_high


def compute_pass1(df: pd.DataFrame) -> float:
    return float(df["outcome"].mean())


def analyze_live_results(csv_path: Path):
    print("=" * 85)
    print(f"Handoff PRM: Verification of Live Statistics on {csv_path.name}")
    print("=" * 85)
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials across {df['task_id'].nunique()} unique tasks.")
    
    # 1. Assertions
    assert_schema_and_zero_leakage(df)
    
    # 2. Overall Pass@1 per arm with 95% task-level bootstrap CIs
    print("\n--> 2. Downstream Pass@1 by Arm (with 95% Task-Level Bootstrap CIs, 10,000 resamples):")
    arms = sorted(list(df["arm"].unique()))
    arm_metrics = {}
    
    for arm in arms:
        arm_df = df[df["arm"] == arm]
        task_vals = {tid: (float(g["outcome"].sum()), len(g)) for tid, g in arm_df.groupby("task_id")}
        pt, lo, hi = task_level_bootstrap_ci(task_vals)
        arm_metrics[arm] = (pt, lo, hi)
        print(f"    {arm:<18}: {pt*100:6.2f}%  [95% CI: {lo*100:5.2f}%, {hi*100:5.2f}%]  (n={len(arm_df)})")
        
    # 3. Paired Differences and Exact McNemar Tests vs. Ungated
    print("\n--> 3. Paired Comparisons vs. Ungated Baseline (Exact McNemar Test & Paired Bootstrap CIs):")
    ungated_df = df[df["arm"] == "ungated"].set_index(["task_id", "seed", "handoff_type"])["outcome"]
    
    for arm in arms:
        if arm == "ungated":
            continue
        gated_df = df[df["arm"] == arm].set_index(["task_id", "seed", "handoff_type"])["outcome"]
        common_idx = ungated_df.index.intersection(gated_df.index)
        
        u_vals = ungated_df.loc[common_idx].values
        g_vals = gated_df.loc[common_idx].values
        
        # Contingency counts
        # a: both pass, b: ungated pass & gated fail, c: ungated fail & gated pass, d: both fail
        a = int(np.sum((u_vals == 1) & (g_vals == 1)))
        b = int(np.sum((u_vals == 1) & (g_vals == 0)))
        c = int(np.sum((u_vals == 0) & (g_vals == 1)))
        d = int(np.sum((u_vals == 0) & (g_vals == 0)))
        
        p_val = exact_mcnemar_test(b, c)
        
        # Paired delta for task-level bootstrap
        paired_records = []
        for (t_id, s_val, h_type), u_val in ungated_df.loc[common_idx].items():
            g_val = gated_df.loc[(t_id, s_val, h_type)]
            paired_records.append({"task_id": t_id, "delta": float(g_val - u_val)})
        paired_frame = pd.DataFrame(paired_records)
        
        task_paired_vals = {
            tid: (float(g["delta"].sum()), len(g))
            for tid, g in paired_frame.groupby("task_id")
        }
        d_pt, d_lo, d_hi = task_level_bootstrap_ci(task_paired_vals)
        
        print(f"    {arm:<16} vs ungated: Delta = {d_pt*100:+5.2f} pp [CI: {d_lo*100:+5.2f}, {d_hi*100:+5.2f}] | "
              f"b={b:3d}, c={c:3d} (Net catches: {c-b:+d}) | McNemar exact p = {p_val:.4f}")

    # 4. Breakdown by Condition: Clean vs. Corrupted
    print("\n--> 4. Breakdown by Condition (Clean vs. Corrupted):")
    for arm in arms:
        clean_df = df[(df["arm"] == arm) & (df["handoff_type"] == "clean")]
        corr_df = df[(df["arm"] == arm) & (df["handoff_type"] != "clean")]
        c_pt = clean_df["outcome"].mean() * 100 if len(clean_df) > 0 else 0.0
        k_pt = corr_df["outcome"].mean() * 100 if len(corr_df) > 0 else 0.0
        print(f"    {arm:<18}: Clean Pass@1 = {c_pt:5.1f}% (n={len(clean_df)}) | Corrupted Pass@1 = {k_pt:5.1f}% (n={len(corr_df)})")

    # 5. Breakdown by Source
    print("\n--> 5. Breakdown by Task Source:")
    sources = sorted(list(df["source"].unique()))
    for src in sources:
        print(f"    Source: {src}")
        for arm in arms:
            s_df = df[(df["arm"] == arm) & (df["source"] == src)]
            if len(s_df) > 0:
                print(f"      {arm:<16}: {s_df['outcome'].mean()*100:5.1f}% (n={len(s_df)})")

    # 6. Breakdown by Seed
    print("\n--> 6. Breakdown by Seed:")
    seeds = sorted(list(df["seed"].unique()))
    for seed in seeds:
        print(f"    Seed {seed}:")
        for arm in arms:
            seed_df = df[(df["arm"] == arm) & (df["seed"] == seed)]
            if len(seed_df) > 0:
                print(f"      {arm:<16}: {seed_df['outcome'].mean()*100:5.1f}% (n={len(seed_df)})")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description="Verify Live Evaluation Statistics")
    parser.add_argument("--csv", default=str(REPO_ROOT / "workstream_c" / "results" / "live_results_pilot.csv"), help="Path to results CSV")
    args = parser.parse_args()
    
    csv_file = Path(args.csv)
    if not csv_file.exists():
        print(f"Error: Results CSV '{csv_file}' not found.")
        sys.exit(1)
        
    analyze_live_results(csv_file)


if __name__ == "__main__":
    main()

