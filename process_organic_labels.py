"""
Process Organic Labels & Generate Dual-Policy Datasets (Task A3)
=================================================================
Author: Avaneesh N (Software and Experiments)
Repository: https://github.com/Harshil-2812/Handoff-PRM

Processes results/tables/organic_dataset_raw.csv to produce:
1. results/tables/organic_dataset_v1_discard.csv
   - Discards ambiguous handoffs (where Agent B succeeded 1/3 or 2/3 times)
   - Label = 1 only if 3/3 passed; Label = 0 only if 0/3 passed
2. results/tables/dataset_v2_robust_positive.csv
   - Strict reliability policy: Label = 1 if 3/3 passed; Label = 0 otherwise
3. results/tables/dataset_v0_majority.csv
   - Standard majority vote: Label = 1 if >= 2/3 passed; Label = 0 otherwise
4. results/audit/organic_label_agreement_report.md
   - Inter-run agreement rate (3/3 agreement vs split)
   - Pairwise Cohen's kappa and Fleiss' kappa across seeds (42, 43, 44)
   - Class balance breakdown by tier and policy
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RAW_CSV = "results/tables/organic_dataset_raw.csv"
OUT_DISCARD = "results/tables/organic_dataset_v1_discard.csv"
OUT_ROBUST = "results/tables/dataset_v2_robust_positive.csv"
OUT_MAJORITY = "results/tables/dataset_v0_majority.csv"
OUT_REPORT = "results/audit/organic_label_agreement_report.md"

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/audit", exist_ok=True)


def compute_fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """
    Computes Fleiss' Kappa for inter-rater agreement across multiple raters (seeds).
    ratings_matrix: N x k array, where ratings_matrix[i, j] is the count of raters
                    who assigned category j to subject i.
    Here k = 2 categories (0=Fail, 1=Pass), and sum across categories per row = n_raters = 3.
    """
    N, k = ratings_matrix.shape
    n = ratings_matrix.sum(axis=1)[0]  # raters per subject (3)

    # Proportion of all assignments to category j
    p_j = ratings_matrix.sum(axis=0) / (N * n)

    # Extent to which raters agree on subject i
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = np.mean(P_i)

    # Mean P_e (chance agreement)
    P_e = np.sum(p_j ** 2)

    if 1.0 - P_e == 0:
        return 1.0
    kappa = (P_bar - P_e) / (1.0 - P_e)
    return float(kappa)


def process_labels():
    if not os.path.exists(RAW_CSV):
        print(f"[ERROR] Raw CSV not found: {RAW_CSV}")
        return

    df = pd.read_csv(RAW_CSV)
    if len(df) == 0:
        print("[ERROR] Raw CSV is empty.")
        return

    print("=" * 75)
    print("PROCESSING ORGANIC LABELS & COMPUTING INTER-RUN AGREEMENT (Task A3)")
    print("=" * 75)
    print(f"Total Rows in Raw Dataset: {len(df)}")
    print(f"Unique Tasks:              {df['task_id'].nunique()}")
    print("=" * 75)

    # ── 1. Inter-Run Agreement Metrics ───────────────────────────────────────
    # Boolean pass runs converted to int (1=Pass, 0=Fail)
    r1 = df["pass_run1"].astype(int).values
    r2 = df["pass_run2"].astype(int).values
    r3 = df["pass_run3"].astype(int).values

    # Pairwise Cohen's Kappa
    k_12 = cohen_kappa_score(r1, r2) if len(np.unique(r1)) > 1 or len(np.unique(r2)) > 1 else 1.0
    k_13 = cohen_kappa_score(r1, r3) if len(np.unique(r1)) > 1 or len(np.unique(r3)) > 1 else 1.0
    k_23 = cohen_kappa_score(r2, r3) if len(np.unique(r2)) > 1 or len(np.unique(r3)) > 1 else 1.0
    mean_cohen = float(np.mean([k_12, k_13, k_23]))

    # Fleiss' Kappa
    # Construct N x 2 matrix: [count_fail, count_pass] per row
    counts_pass = df["pass_count_of_3"].values
    counts_fail = 3 - counts_pass
    ratings_matrix = np.column_stack([counts_fail, counts_pass])
    fleiss_k = compute_fleiss_kappa(ratings_matrix)

    # Percentages of unanimous vs split
    n_total = len(df)
    n_unanimous_pass = int((counts_pass == 3).sum())
    n_unanimous_fail = int((counts_pass == 0).sum())
    n_unanimous = n_unanimous_pass + n_unanimous_fail
    n_split_pass = int((counts_pass == 2).sum())
    n_split_fail = int((counts_pass == 1).sum())
    n_split = n_split_pass + n_split_fail

    pct_unanimous = (n_unanimous / n_total) * 100.0
    pct_split = (n_split / n_total) * 100.0

    print(f"\nUnanimous Agreement (3/3 or 0/3): {n_unanimous}/{n_total} ({pct_unanimous:.1f}%)")
    print(f"  Unanimous Positive (3/3):        {n_unanimous_pass} ({n_unanimous_pass/n_total*100:.1f}%)")
    print(f"  Unanimous Negative (0/3):        {n_unanimous_fail} ({n_unanimous_fail/n_total*100:.1f}%)")
    print(f"Split Outcomes (2/3 or 1/3):       {n_split}/{n_total} ({pct_split:.1f}%)")
    print(f"  Majority Pass (2/3):             {n_split_pass}")
    print(f"  Majority Fail (1/3):             {n_split_fail}")
    print(f"Inter-Rater Reliability:")
    print(f"  Fleiss' Kappa (3 seeds):         {fleiss_k:.4f}")
    print(f"  Mean Pairwise Cohen's Kappa:     {mean_cohen:.4f}")

    # ── 2. Build Discard Policy Dataset (v1_discard) ──────────────────────────
    df_discard = df[df["label_discard_policy"].astype(str) != "discard"].copy()
    df_discard["label"] = df_discard["label_discard_policy"].astype(int)
    df_discard.to_csv(OUT_DISCARD, index=False)
    print(f"\n[Saved] Discard-Policy Dataset: {OUT_DISCARD}")
    print(f"  Rows retained: {len(df_discard)}/{len(df)} ({len(df_discard)/len(df)*100:.1f}%)")
    print(f"  Class balance: Positives={sum(df_discard['label']==1)}, Negatives={sum(df_discard['label']==0)}")

    # ── 3. Build Robust-Positive Dataset (v2_robust_positive) ──────────────────
    df_robust = df.copy()
    df_robust["label"] = df_robust["label_robust_positive"].astype(int)
    df_robust.to_csv(OUT_ROBUST, index=False)
    print(f"\n[Saved] Robust-Positive Dataset: {OUT_ROBUST}")
    print(f"  Rows: {len(df_robust)}")
    print(f"  Class balance: Positives={sum(df_robust['label']==1)}, Negatives={sum(df_robust['label']==0)}")

    # ── 4. Build Majority-Vote Dataset (v0_majority) ──────────────────────────
    df_majority = df.copy()
    df_majority["label"] = df_majority["label_majority"].astype(int)
    df_majority.to_csv(OUT_MAJORITY, index=False)
    print(f"\n[Saved] Majority-Vote Dataset: {OUT_MAJORITY}")
    print(f"  Rows: {len(df_majority)}")
    print(f"  Class balance: Positives={sum(df_majority['label']==1)}, Negatives={sum(df_majority['label']==0)}")

    # ── 5. Breakdown by Tier ──────────────────────────────────────────────────
    tier_summary = []
    for tier in df["tier_name"].unique():
        sub = df[df["tier_name"] == tier]
        n_sub = len(sub)
        p3 = (sub["pass_count_of_3"] == 3).sum()
        p2 = (sub["pass_count_of_3"] == 2).sum()
        p1 = (sub["pass_count_of_3"] == 1).sum()
        p0 = (sub["pass_count_of_3"] == 0).sum()
        maj_pass = (sub["label_majority"] == 1).sum()
        maj_fail = (sub["label_majority"] == 0).sum()
        fail_rate = (maj_fail / n_sub) * 100.0 if n_sub > 0 else 0.0
        tier_summary.append({
            "tier": tier,
            "n_samples": n_sub,
            "pass_3_of_3": int(p3),
            "pass_2_of_3": int(p2),
            "pass_1_of_3": int(p1),
            "pass_0_of_3": int(p0),
            "majority_pass": int(maj_pass),
            "majority_fail": int(maj_fail),
            "failure_rate_pct": fail_rate
        })

    df_tier = pd.DataFrame(tier_summary)
    print("\nTier Breakdown:")
    print(df_tier.to_string(index=False))

    # ── 6. Write Markdown Audit Report ────────────────────────────────────────
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Inter-Run Agreement and Dual Labeling Policy Audit Report\n\n")
        f.write(f"**Workstream A**: Avaneesh N (Data and Organic Failures)\n")
        f.write(f"**Evaluation Seeds**: [42, 43, 44] (Triplicate Execution)\n")
        f.write(f"**Total Evaluated Samples**: {len(df)}\n")
        f.write(f"**Unique Tasks**: {df['task_id'].nunique()}\n\n")

        f.write("## 1. Inter-Rater Reliability Metrics\n\n")
        f.write(f"- **Unanimous Agreement (3/3 or 0/3)**: {n_unanimous}/{n_total} ({pct_unanimous:.2f}%)\n")
        f.write(f"- **Split Outcome Rate (2/3 or 1/3)**: {n_split}/{n_total} ({pct_split:.2f}%)\n")
        f.write(f"- **Fleiss' Kappa ($\\kappa$)**: `{fleiss_k:.4f}`\n")
        f.write(f"- **Mean Pairwise Cohen's Kappa**: `{mean_cohen:.4f}`\n")
        f.write(f"  - Run 1 vs Run 2: `{k_12:.4f}`\n")
        f.write(f"  - Run 1 vs Run 3: `{k_13:.4f}`\n")
        f.write(f"  - Run 2 vs Run 3: `{k_23:.4f}`\n\n")

        f.write("## 2. Policy-Specific Class Distribution\n\n")
        f.write("| Policy | Total Rows | Positive (Label=1) | Negative (Label=0) | Discarded Rows | Notes |\n")
        f.write("|---|---|---|---|---|---|\n")
        f.write(f"| **Majority Vote (v0)** | {len(df_majority)} | {sum(df_majority['label']==1)} | {sum(df_majority['label']==0)} | 0 | Standard threshold (>=2 of 3) |\n")
        f.write(f"| **Discard Policy (v1)** | {len(df_discard)} | {sum(df_discard['label']==1)} | {sum(df_discard['label']==0)} | {len(df) - len(df_discard)} | Ambiguous 1/3 and 2/3 dropped |\n")
        f.write(f"| **Robust Positive (v2)** | {len(df_robust)} | {sum(df_robust['label']==1)} | {sum(df_robust['label']==0)} | 0 | Only 3/3 treated as positive |\n\n")

        f.write("## 3. Breakdown by Agent A Model Tier\n\n")
        f.write("| Tier Name | Total | 3/3 Pass | 2/3 Pass | 1/3 Pass | 0/3 Pass | Majority Fail Rate |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for t in tier_summary:
            f.write(f"| `{t['tier']}` | {t['n_samples']} | {t['pass_3_of_3']} | {t['pass_2_of_3']} | {t['pass_1_of_3']} | {t['pass_0_of_3']} | {t['failure_rate_pct']:.1f}% |\n")

    print(f"\n[Saved] Audit Report: {OUT_REPORT}")
    print("=" * 75)


if __name__ == "__main__":
    process_labels()
