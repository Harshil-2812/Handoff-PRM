"""Standalone script for Step 4: Organic C4 Follow-Up (Exploratory Analysis).

Computes:
1. Single-feature AUROCs within HumanEval (N=252) and MBPP (N=627) separately,
   with direction (Spearman rho), aligned AUROC, and 95% task-level bootstrap CIs (10,000 draws, seed 42).
2. Task-grouped 5-fold cross-validated logistic regression on the 11 features + source,
   reporting fold-wise AUROCs, pooled out-of-fold AUROC, and 95% task-level bootstrap CI.
3. Stage-1 classifier AUROC on MBPP before and after dropping function-name-mismatch failures.
Note: These analyses are strictly exploratory and need validation on Workstream A's organic dataset.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = "workstream_c/results/live_results_main.csv"
df_main = pd.read_csv(CSV_PATH)

print("=" * 80, flush=True)
print("STEP 4: ORGANIC C4 FOLLOW-UP (EXPLORATORY ANALYSIS)", flush=True)
print("=" * 80, flush=True)

# 1. Filter clean trials and map ungated clean outcome as the target label
clean_ungated = df_main[(df_main["handoff_type"] == "clean") & (df_main["arm"] == "ungated")].set_index(["task_id", "seed"])
clean_s1 = df_main[(df_main["handoff_type"] == "clean") & (df_main["arm"] == "stage1_only")].copy()

# Outcome of ungated clean trial is the ground-truth downstream label
clean_s1["ungated_outcome"] = clean_s1.apply(lambda r: clean_ungated.loc[(r["task_id"], r["seed"]), "outcome"], axis=1)
clean_s1["ungated_error"] = clean_s1.apply(lambda r: clean_ungated.loc[(r["task_id"], r["seed"]), "error_message"], axis=1)

# Extract the 11 features
feat_dicts = [json.loads(f) if isinstance(f, str) else f for f in clean_s1["features"]]
df_feats = pd.DataFrame(feat_dicts, index=clean_s1.index)
feature_cols = sorted([c for c in df_feats.columns if c != "ungated_outcome"])

for col in feature_cols:
    clean_s1[col] = df_feats[col]

print(f"Total clean trials: {len(clean_s1)} across {clean_s1['task_id'].nunique()} unique tasks.", flush=True)
print(f"HumanEval clean trials: {len(clean_s1[clean_s1['source'] == 'humaneval'])} (84 tasks)", flush=True)
print(f"MBPP clean trials:      {len(clean_s1[clean_s1['source'] == 'mbpp'])} (209 tasks)", flush=True)

def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int(np.sum(y == 1))
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(s)
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))

# Vectorized task-level bootstrap CI of AUROC
def task_level_bootstrap_auroc(df_sub: pd.DataFrame, score_col: str, label_col: str = "ungated_outcome", n_boot: int = 10000, seed: int = 42, align_pos: bool = True) -> Tuple[float, float, float]:
    y_all = df_sub[label_col].values
    s_all = df_sub[score_col].values
    
    corr, _ = stats.spearmanr(s_all, y_all)
    mult = 1.0 if (corr >= 0 or not align_pos) else -1.0
    s_eval = mult * s_all
    
    if len(np.unique(y_all)) < 2:
        return 0.5, 0.5, 0.5
        
    point_auc = fast_auc(y_all, s_eval)
    
    tasks = df_sub["task_id"].unique()
    n_tasks = len(tasks)
    task_indices = df_sub.groupby("task_id").indices
    task_idx_list = [task_indices[t] for t in tasks]
    
    rng = np.random.RandomState(seed)
    boot_aucs = np.zeros(n_boot)
    
    # Pre-generate task resamples
    sampled_task_matrix = rng.choice(n_tasks, size=(n_boot, n_tasks), replace=True)
    for b in range(n_boot):
        s_tasks = sampled_task_matrix[b]
        sampled_rows = np.concatenate([task_idx_list[i] for i in s_tasks])
        by = y_all[sampled_rows]
        bs = s_eval[sampled_rows]
        boot_aucs[b] = fast_auc(by, bs)
            
    ci_l, ci_h = np.percentile(boot_aucs, [2.5, 97.5])
    return point_auc, ci_l, ci_h

# -----------------------------------------------------------------------------
# Part 1: Single-Feature AUROCs within HumanEval and MBPP separately
# -----------------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("PART 1: SINGLE-FEATURE AUROCS BY SOURCE (TASK-LEVEL BOOTSTRAP CIS, B=10,000)", flush=True)
print("=" * 80, flush=True)

for src_name, src_key in [("HumanEval (N=252, 84 tasks)", "humaneval"), ("MBPP (N=627, 209 tasks)", "mbpp")]:
    df_src = clean_s1[clean_s1["source"] == src_key].copy().reset_index(drop=True)
    y_src = df_src["ungated_outcome"].values
    n_pos = int(np.sum(y_src == 1))
    n_neg = int(np.sum(y_src == 0))
    print(f"\n--- Benchmark: {src_name} | Passes: {n_pos} ({n_pos/len(y_src)*100:.1f}%), Failures: {n_neg} ({n_neg/len(y_src)*100:.1f}%) ---", flush=True)
    print(f"{'Feature':<26} {'Dir':<5} {'Aligned AUROC':<15} {'95% Task Bootstrap CI':<24} {'Raw AUROC':<12} {'Spearman rho'}", flush=True)
    print("-" * 92, flush=True)
    
    for feat in feature_cols:
        f_vals = df_src[feat].values
        if len(np.unique(f_vals)) <= 1:
            print(f"{feat:<26} {'0':<5} {0.5000:<15.4f} {'[0.5000, 0.5000]':<24} {0.5000:<12.4f} {'0.0000'}", flush=True)
            continue
        corr, _ = stats.spearmanr(f_vals, y_src)
        direction = "+" if corr >= 0 else "-"
        pt_auc, ci_l, ci_h = task_level_bootstrap_auroc(df_src, feat, align_pos=True)
        raw_auc = fast_auc(y_src, f_vals)
        print(f"{feat:<26} {direction:<5} {pt_auc:<15.4f} [{ci_l:.4f}, {ci_h:.4f}]          {raw_auc:<12.4f} {corr:+.4f}", flush=True)

    pt_prm, prm_l, prm_h = task_level_bootstrap_auroc(df_src, "stage1_score", align_pos=False)
    corr_prm, _ = stats.spearmanr(df_src["stage1_score"].values, y_src)
    dir_prm = "+" if corr_prm >= 0 else "-"
    print(f"{'OVERALL STAGE-1 SCORE':<26} {dir_prm:<5} {pt_prm:<15.4f} [{prm_l:.4f}, {prm_h:.4f}]          {pt_prm:<12.4f} {corr_prm:+.4f}", flush=True)

# -----------------------------------------------------------------------------
# Part 2: Task-Grouped 5-Fold Cross-Validated Logistic Regression
# -----------------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("PART 2: TASK-GROUPED 5-FOLD CV LOGISTIC REGRESSION (11 FEATURES + SOURCE)", flush=True)
print("=" * 80, flush=True)

df_all = clean_s1.copy().reset_index(drop=True)
X_num = df_all[feature_cols].copy()
source_dummies = pd.get_dummies(df_all["source"], prefix="src", drop_first=True)
X_full = pd.concat([X_num, source_dummies], axis=1)
y_all = df_all["ungated_outcome"].values
groups = df_all["task_id"].values

gkf = GroupKFold(n_splits=5)
oof_preds = np.zeros(len(df_all))
fold_aurocs = []

print(f"{'Fold':<8} {'Train Tasks':<14} {'Test Tasks':<14} {'Test Trials':<14} {'Fold AUROC'}", flush=True)
print("-" * 62, flush=True)

for fold, (train_idx, test_idx) in enumerate(gkf.split(X_full, y_all, groups)):
    X_train, y_train = X_full.iloc[train_idx], y_all[train_idx]
    X_test, y_test = X_full.iloc[test_idx], y_all[test_idx]
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
    
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    preds = clf.predict_proba(X_test_scaled)[:, 1]
    oof_preds[test_idx] = preds
    f_auc = fast_auc(y_test, preds)
    fold_aurocs.append(f_auc)
    
    n_tr_tasks = len(np.unique(groups[train_idx]))
    n_te_tasks = len(np.unique(groups[test_idx]))
    print(f"Fold {fold+1:<3} {n_tr_tasks:<14} {n_te_tasks:<14} {len(test_idx):<14} {f_auc:.4f}", flush=True)

df_all["oof_pred"] = oof_preds
pooled_auc = fast_auc(y_all, oof_preds)

# Task-level bootstrap CI on pooled out-of-fold predictions
pt_oof, oof_l, oof_h = task_level_bootstrap_auroc(df_all, "oof_pred", align_pos=False)

print("-" * 62, flush=True)
print(f"Mean Fold AUROC:          {np.mean(fold_aurocs):.4f} +/- {np.std(fold_aurocs):.4f}", flush=True)
print(f"Pooled Out-of-Fold AUROC: {pooled_auc:.4f} [95% Task-Level Bootstrap CI: {oof_l:.4f}, {oof_h:.4f}]", flush=True)

# -----------------------------------------------------------------------------
# Part 3: Classifier AUROC after Removing Function-Name Mismatches in MBPP
# -----------------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("PART 3: MBPP CLASSIFIER AUROC AFTER DROPPING FUNCTION-NAME MISMATCHES", flush=True)
print("=" * 80, flush=True)

df_mbpp = clean_s1[clean_s1["source"] == "mbpp"].copy().reset_index(drop=True)
n_mbpp_total = len(df_mbpp)
y_mbpp_orig = df_mbpp["ungated_outcome"].values

pt_mbpp_base, mbpp_base_l, mbpp_base_h = task_level_bootstrap_auroc(df_mbpp, "stage1_score", align_pos=False)
print(f"Baseline MBPP (All {n_mbpp_total} clean trials across 209 tasks):", flush=True)
print(f"  Passes:   {int(np.sum(y_mbpp_orig == 1))} ({np.mean(y_mbpp_orig == 1)*100:.2f}%)", flush=True)
print(f"  Failures: {int(np.sum(y_mbpp_orig == 0))} ({np.mean(y_mbpp_orig == 0)*100:.2f}%)", flush=True)
print(f"  AUROC:    {pt_mbpp_base:.4f} [95% Task Bootstrap CI: {mbpp_base_l:.4f}, {mbpp_base_h:.4f}]", flush=True)

fn_mismatch_mask = (df_mbpp["ungated_outcome"] == 0) & df_mbpp["ungated_error"].str.contains("not found in generated code", na=False)
n_mismatches = int(fn_mismatch_mask.sum())
n_mismatch_tasks = df_mbpp.loc[fn_mismatch_mask, "task_id"].nunique()

df_mbpp_filtered = df_mbpp[~fn_mismatch_mask].copy().reset_index(drop=True)
y_mbpp_filt = df_mbpp_filtered["ungated_outcome"].values
n_filt_total = len(df_mbpp_filtered)

pt_mbpp_filt, mbpp_filt_l, mbpp_filt_h = task_level_bootstrap_auroc(df_mbpp_filtered, "stage1_score", align_pos=False)

print(f"\nFiltered MBPP (Dropped {n_mismatches} trials with function name mismatch across {n_mismatch_tasks} tasks):", flush=True)
print(f"  Remaining trials:   {n_filt_total}", flush=True)
print(f"  Remaining passes:   {int(np.sum(y_mbpp_filt == 1))} ({np.mean(y_mbpp_filt == 1)*100:.2f}%)", flush=True)
print(f"  Remaining failures: {int(np.sum(y_mbpp_filt == 0))} ({np.mean(y_mbpp_filt == 0)*100:.2f}%)", flush=True)
print(f"  Filtered AUROC:     {pt_mbpp_filt:.4f} [95% Task Bootstrap CI: {mbpp_filt_l:.4f}, {mbpp_filt_h:.4f}]", flush=True)
print(f"  Delta in AUROC:     {pt_mbpp_filt - pt_mbpp_base:+.4f}", flush=True)

print("\n" + "=" * 80, flush=True)
print("EXPLORATORY NOTE: All C4 results above are computed offline on Workstream C's clean handoffs and are strictly exploratory; they require validation on Workstream A's organic dataset.", flush=True)
print("=" * 80, flush=True)
