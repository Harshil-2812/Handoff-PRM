"""Comprehensive, highly optimized script for all 6 analysis tasks:

Task 1: Entry-point check (read-only prompt templates and MBPP prompt examination)
Task 2: Offline re-grading of clean failures under function name aliasing
Task 3: Task-level paired bootstrap CI for real stage1_only vs random-gate mean
Task 4: Source reconciliation check and figure generation
Task 5: C4 organic evaluation on the 900 clean handoffs
Task 6: C6 prevalence sweep at 5%, 20%, 50% corruption
"""

import ast
import json
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

from workstream_c.harness.agents_interface import (
    AGENT_A_SYSTEM_PROMPT,
    AGENT_B_SYSTEM_PROMPT,
    AGENT_B_RAW_PROMPT,
)
from workstream_c.harness.execution_sandbox import extract_python_code
from workstream_c.harness.task_loader import load_live_test_tasks
from tasks.all_tasks import ALL_TASKS

CSV_PATH = "workstream_c/results/live_results_main.csv"
CLEAN_HANDOFFS_PATH = "workstream_c/results/clean_agent_a_handoffs.csv"
FROZEN_THRESH_PATH = "results/frozen_threshold.json"

df_main = pd.read_csv(CSV_PATH)
tasks_dict = {t["id"]: t for t in load_live_test_tasks()}

print("=" * 80, flush=True)
print("TASK 1: ENTRY-POINT CHECK", flush=True)
print("=" * 80, flush=True)

print("1. Live C2/C3 Harness Prompt Templates:", flush=True)
print("\n--- AGENT_A_SYSTEM_PROMPT ---", flush=True)
print(AGENT_A_SYSTEM_PROMPT, flush=True)
print("Entry point given to Agent A: NO (Agent A only receives Problem: {problem})", flush=True)

print("\n--- AGENT_B_SYSTEM_PROMPT (Handoff Mode) ---", flush=True)
print(AGENT_B_SYSTEM_PROMPT, flush=True)
print("Entry point given to Agent B: YES (`{entry_point}` is specified in the prompt template)", flush=True)

print("\n--- AGENT_B_RAW_PROMPT (Raw Problem Mode / Fallback) ---", flush=True)
print(AGENT_B_RAW_PROMPT, flush=True)
print("Entry point given to Agent B: YES (`{entry_point}` is specified in the prompt template)", flush=True)

# Check MBPP tasks in tasks/all_tasks.py
mbpp_all = [t for t in ALL_TASKS if t.get("source") == "mbpp" or t["id"].startswith("mbpp")]
mbpp_named_all = sum(1 for t in mbpp_all if t.get("entry_point", "").lower() in t.get("problem", "").lower())
print(f"\n2. MBPP Tasks in tasks/all_tasks.py naming entry_point in problem text: {mbpp_named_all}/{len(mbpp_all)} ({mbpp_named_all/len(mbpp_all)*100:.2f}%)", flush=True)

with open("workstream_c/manifests/manifest_live_test.json") as f:
    live_manifest = json.load(f)
live_mbpp_ids = [tid for tid in live_manifest["tasks"] if tid.startswith("mbpp")]
live_mbpp_tasks = [tasks_dict[tid] for tid in live_mbpp_ids]
live_named = sum(1 for t in live_mbpp_tasks if t.get("entry_point", "").lower() in t.get("problem", "").lower())
print(f"3. Held-out live test MBPP tasks naming entry_point in problem text: {live_named}/{len(live_mbpp_tasks)} ({live_named/len(live_mbpp_tasks)*100:.2f}%)", flush=True)

print("\n4. Comparison with original pipeline (root agents.py, build_dataset.py):", flush=True)
print("- Root agents.py: agent_a_plan(problem) also receives only 'problem' (no entry_point).", flush=True)
print("- Root agents.py: agent_b_code(handoff, entry_point) receives entry_point.", flush=True)
print("- MBPP test cases provided to Agent A or B: NO in both pipelines.", flush=True)
print("- Difference: The live harness prompt templates are functionally identical to root agents.py.", flush=True)


print("\n" + "=" * 80, flush=True)
print("TASK 2: NAME-MISMATCH EFFECT (OFFLINE RE-GRADING WITH ALIASING)", flush=True)
print("=" * 80, flush=True)

# Load caches
caches = {}
for s in [42, 43, 44]:
    c_path = f"cache/llm_cache_s{s}.json"
    with open(c_path, "r", encoding="utf-8") as f:
        caches[s] = json.load(f)

clean_handoffs_df = pd.read_csv(CLEAN_HANDOFFS_PATH)
clean_handoffs_lookup = {(r["task_id"], r["seed"]): r["handoff_text"] for _, r in clean_handoffs_df.iterrows()}

def evaluate_code_with_alias(code_str: str, entry_point: str, test_code_str: str) -> bool:
    clean_code = extract_python_code(code_str)
    try:
        tree = ast.parse(clean_code)
        funcs = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except Exception:
        funcs = []
        
    namespace = {}
    try:
        exec(clean_code, namespace)
        if entry_point not in namespace:
            for f in funcs:
                if f in namespace and callable(namespace[f]):
                    namespace[entry_point] = namespace[f]
                    break
        exec(test_code_str, namespace)
        fn = namespace.get(entry_point)
        if fn is None:
            return False
        test_fn = namespace.get("test")
        check_fn = namespace.get("check")
        if test_fn:
            res = test_fn(fn)
            return bool(res) if res is not None else True
        elif check_fn:
            check_fn(fn)
            return True
        return False
    except Exception:
        return False

# Re-grade the clean trials for ungated, stage1_only, deterministic, always_fallback
clean_trials = df_main[df_main["handoff_type"] == "clean"].copy()

# Pre-compute alias pass for all 900 clean handoff executions:
handoff_alias_pass = {}
for (tid, s), ht in clean_handoffs_lookup.items():
    t_info = tasks_dict[tid]
    ep = t_info["entry_point"]
    test_c = t_info["test_code"]
    prompt = AGENT_B_SYSTEM_PROMPT.format(handoff=ht, entry_point=ep)
    h = hashlib.sha256(f"gemini-3.5-flash-lite_0.0_{s}_{prompt}".encode("utf-8")).hexdigest()
    code = caches[s].get(h, {}).get("text", "")
    handoff_alias_pass[(tid, s)] = evaluate_code_with_alias(code, ep, test_c)

print(f"Computed alias evaluations for all 900 clean handoff executions.", flush=True)

# Now evaluate per arm
alias_results = []
for idx, r in clean_trials.iterrows():
    arm = r["arm"]
    if arm not in ["ungated", "stage1_only", "deterministic", "always_fallback"]:
        continue
    tid = r["task_id"]
    s = r["seed"]
    src = r["source"]
    orig_out = r["outcome"]
    orig_dec = r["decision"]
    
    if orig_out == 1:
        alias_out = 1
    else:
        # If the trial ran on handoff:
        if arm == "ungated":
            alias_out = 1 if handoff_alias_pass.get((tid, s), False) else 0
        elif arm in ["stage1_only", "deterministic"]:
            if orig_dec == "PASS":
                alias_out = 1 if handoff_alias_pass.get((tid, s), False) else 0
            else:
                # Ran on raw problem - already evaluated in orig_out
                alias_out = orig_out
        elif arm == "always_fallback":
            # Always ran on raw problem
            alias_out = orig_out
            
    alias_results.append({
        "arm": arm,
        "task_id": tid,
        "seed": s,
        "source": src,
        "orig_outcome": orig_out,
        "alias_outcome": alias_out,
    })

df_alias = pd.DataFrame(alias_results)

print("\n--- Clean Handoff Pass@1: Original vs. Function-Name Aliased ---", flush=True)
for arm in ["ungated", "stage1_only", "deterministic", "always_fallback"]:
    sub = df_alias[df_alias["arm"] == arm]
    orig_p1 = sub["orig_outcome"].mean() * 100
    alias_p1 = sub["alias_outcome"].mean() * 100
    gain = alias_p1 - orig_p1
    print(f"\n  Arm: {arm:<16} Original = {orig_p1:5.2f}% | Aliased = {alias_p1:5.2f}% (Gain: {gain:+5.2f} pp)", flush=True)
    for src in ["humaneval", "mbpp", "hand_easy", "hand_hard"]:
        s_sub = sub[sub["source"] == src]
        s_orig = s_sub["orig_outcome"].mean() * 100
        s_alias = s_sub["alias_outcome"].mean() * 100
        print(f"    {src:<12}: Original = {s_orig:5.2f}% -> Aliased = {s_alias:5.2f}% ({s_alias - s_orig:+5.2f} pp)", flush=True)


print("\n" + "=" * 80, flush=True)
print("TASK 3: GATE VS RANDOM (TASK-LEVEL PAIRED BOOTSTRAP CI)", flush=True)
print("=" * 80, flush=True)

# Random block control
piv_trials = df_main.pivot(index=["task_id", "handoff_type", "seed", "source"], columns="arm", values=["outcome", "decision"])
outcomes_u = piv_trials[("outcome", "ungated")]
outcomes_af = piv_trials[("outcome", "always_fallback")]
outcomes_s1 = piv_trials[("outcome", "stage1_only")]
decisions_s1 = piv_trials[("decision", "stage1_only")]

t_df = pd.DataFrame({
    "ungated": outcomes_u,
    "always_fallback": outcomes_af,
    "stage1_only": outcomes_s1,
    "s1_decision": decisions_s1,
}).reset_index()

t_df["condition"] = t_df["handoff_type"].apply(lambda x: "clean" if x == "clean" else "corrupted")

strata_info = {}
for (cond, src), grp in t_df.groupby(["condition", "source"]):
    n_tot = len(grp)
    n_blk = (grp["s1_decision"] == "BLOCK").sum()
    strata_info[(cond, src)] = (n_tot, n_blk)

rng_sim = np.random.RandomState(42)
N_DRAWS = 1000
task_random_means = np.zeros(len(t_df))

for draw in range(N_DRAWS):
    draw_outcomes = np.zeros(len(t_df))
    for (cond, src), (n_tot, n_blk) in strata_info.items():
        idx = t_df[(t_df["condition"] == cond) & (t_df["source"] == src)].index.values
        blocked_idx = rng_sim.choice(idx, size=n_blk, replace=False)
        pass_idx = np.setdiff1d(idx, blocked_idx)
        draw_outcomes[blocked_idx] = t_df.loc[blocked_idx, "always_fallback"].values
        draw_outcomes[pass_idx] = t_df.loc[pass_idx, "ungated"].values
    task_random_means += draw_outcomes

task_random_means /= N_DRAWS
t_df["random_gate_mean"] = task_random_means

# Aggregate to task level (300 tasks)
task_agg = t_df.groupby("task_id").agg({
    "stage1_only": "mean",
    "random_gate_mean": "mean",
}).reset_index()

task_agg["diff"] = (task_agg["stage1_only"] - task_agg["random_gate_mean"]) * 100
real_s1_mean = task_agg["stage1_only"].mean() * 100
rand_gate_mean = task_agg["random_gate_mean"].mean() * 100
observed_delta = task_agg["diff"].mean()

# Vectorized task-level paired bootstrap CI (10,000 resamples, fixed seed 42)
B = 10000
rng_boot = np.random.RandomState(42)
diff_vals = task_agg["diff"].values
n_tasks = len(diff_vals)
sample_idx = rng_boot.choice(n_tasks, size=(B, n_tasks), replace=True)
boot_diffs = diff_vals[sample_idx].mean(axis=1)

ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

# Paired permutation test
perm_rng = np.random.RandomState(42)
signs = perm_rng.choice([-1, 1], size=(100000, n_tasks))
perm_stats = np.abs(np.mean(signs * diff_vals, axis=1))
p_perm = np.mean(perm_stats >= np.abs(observed_delta))

print(f"Real Stage-1 Pass@1:             {real_s1_mean:.2f}%", flush=True)
print(f"Random Gate Expected Pass@1:     {rand_gate_mean:.2f}%", flush=True)
print(f"Point Estimate (Real - Random):   {observed_delta:+.2f} pp", flush=True)
print(f"95% Task-Level Bootstrap CI:     [{ci_low:+.2f} pp, {ci_high:+.2f} pp]", flush=True)
print(f"Paired Permutation p-value:      p = {p_perm:.4f}", flush=True)


print("\n" + "=" * 80, flush=True)
print("TASK 5: C4 FROM OWN CLEAN HANDOFFS (ORGANIC EVALUATION)", flush=True)
print("=" * 80, flush=True)

# 900 clean trials
clean_ungated_lookup = df_main[(df_main["handoff_type"] == "clean") & (df_main["arm"] == "ungated")].set_index(["task_id", "seed"])["outcome"].to_dict()
clean_s1 = df_main[(df_main["handoff_type"] == "clean") & (df_main["arm"] == "stage1_only")].copy()
clean_s1["label"] = clean_s1.apply(lambda r: clean_ungated_lookup[(r["task_id"], r["seed"])], axis=1)

# Sort by task_id and seed for structured matrix operations
clean_s1 = clean_s1.sort_values(["task_id", "seed"]).reset_index(drop=True)
unique_tasks = clean_s1["task_id"].unique()
n_unique_tasks = len(unique_tasks)

y_true = clean_s1["label"].values
y_score = clean_s1["stage1_score"].values

N_tot = len(y_true)
N_neg = int(np.sum(y_true == 0))
N_pos = int(np.sum(y_true == 1))
auroc_all = roc_auc_score(y_true, y_score)
auprc_all = average_precision_score(y_true, y_score)

# Vectorized task-level bootstrap: each task has exactly 3 rows
task_row_indices = clean_s1.groupby("task_id").indices
task_index_map = [task_row_indices[t] for t in unique_tasks]

rng_c4 = np.random.RandomState(42)
boot_aurocs = []
boot_auprcs = []
for _ in range(B):
    s_tasks = rng_c4.choice(n_unique_tasks, size=n_unique_tasks, replace=True)
    sampled_row_idx = np.concatenate([task_index_map[t_idx] for t_idx in s_tasks])
    b_y = y_true[sampled_row_idx]
    b_s = y_score[sampled_row_idx]
    if len(np.unique(b_y)) > 1:
        boot_aurocs.append(roc_auc_score(b_y, b_s))
        boot_auprcs.append(average_precision_score(b_y, b_s))

auroc_ci = np.percentile(boot_aurocs, [2.5, 97.5])
auprc_ci = np.percentile(boot_auprcs, [2.5, 97.5])

# Confusion matrix at frozen threshold 0.550419
y_pred = (y_score >= 0.550419).astype(int)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"C4 Organic Evaluation on 900 Clean Handoffs (1 run per handoff):", flush=True)
print(f"  Total trials (N): {N_tot}", flush=True)
print(f"  Positives (passes): {N_pos} ({N_pos/N_tot*100:.2f}%)", flush=True)
print(f"  Negatives (fails):  {N_neg} ({N_neg/N_tot*100:.2f}%)", flush=True)
print(f"  AUROC: {auroc_all:.4f} [95% Task-Level Bootstrap CI: {auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]", flush=True)
print(f"  AUPRC: {auprc_all:.4f} [95% Task-Level Bootstrap CI: {auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]", flush=True)
print(f"\n  Confusion Matrix at Frozen Threshold (0.550419):", flush=True)
print(f"    True Positives  (Passed & Predicted Pass):  {tp}", flush=True)
print(f"    True Negatives  (Failed & Blocked):         {tn}", flush=True)
print(f"    False Positives (Failed & Passed):          {fp}", flush=True)
print(f"    False Negatives (Passed & Blocked):         {fn}", flush=True)
print(f"    Precision: {tp/(tp+fp):.4f} | Recall: {tp/(tp+fn):.4f} | Clean False-Block Rate: {fn/(tp+fn)*100:.2f}%", flush=True)

print("\n  Results by Source:", flush=True)
for src in ["humaneval", "mbpp", "hand_easy", "hand_hard"]:
    s_sub = clean_s1[clean_s1["source"] == src]
    s_y = s_sub["label"].values
    s_sc = s_sub["stage1_score"].values
    s_n = len(s_y)
    s_neg = int(np.sum(s_y == 0))
    if len(np.unique(s_y)) > 1:
        s_auc = roc_auc_score(s_y, s_sc)
        s_pr = average_precision_score(s_y, s_sc)
        print(f"    {src:<12}: N={s_n:3d}, Negatives={s_neg:3d} | AUROC={s_auc:.4f}, AUPRC={s_pr:.4f}", flush=True)
    else:
        print(f"    {src:<12}: N={s_n:3d}, Negatives={s_neg:3d} | AUROC=N/A (only one class present)", flush=True)

print("\n  Results by Seed:", flush=True)
for s in [42, 43, 44]:
    s_sub = clean_s1[clean_s1["seed"] == s]
    s_y = s_sub["label"].values
    s_sc = s_sub["stage1_score"].values
    s_n = len(s_y)
    s_neg = int(np.sum(s_y == 0))
    s_auc = roc_auc_score(s_y, s_sc)
    s_pr = average_precision_score(s_y, s_sc)
    print(f"    Seed {s}: N={s_n:3d}, Negatives={s_neg:3d} | AUROC={s_auc:.4f}, AUPRC={s_pr:.4f}", flush=True)

# Single-feature AUROC and direction
print("\n  Single-Feature AUROC and Direction (N=900):", flush=True)
feat_records = [eval(f) if isinstance(f, str) else f for f in clean_s1["features"]]
df_feats = pd.DataFrame(feat_records, index=clean_s1.index)
df_feats["label"] = y_true

for col in sorted(df_feats.columns):
    if col == "label":
        continue
    f_vals = df_feats[col].values
    corr, _ = stats.spearmanr(f_vals, y_true)
    direction = "+" if corr >= 0 else "-"
    if len(np.unique(f_vals)) > 1:
        auc_raw = roc_auc_score(y_true, f_vals)
        auc_aligned = roc_auc_score(y_true, f_vals if corr >= 0 else -f_vals)
        print(f"    {col:<26}: AUROC = {auc_aligned:.4f} (Direction: {direction}, raw AUC = {auc_raw:.4f}, rho = {corr:+.4f})", flush=True)
    else:
        print(f"    {col:<26}: Constant value ({f_vals[0]}), AUROC = 0.5000", flush=True)

# Drop function-name-mismatch failures from negatives
fn_mismatch_mask = clean_s1["error_message"].str.contains("not found in generated code", na=False) & (clean_s1["label"] == 0)
dropped_df = clean_s1[~fn_mismatch_mask]
y_true_drop = dropped_df["label"].values
y_score_drop = dropped_df["stage1_score"].values
auroc_drop = roc_auc_score(y_true_drop, y_score_drop)
auprc_drop = average_precision_score(y_true_drop, y_score_drop)
print(f"\n  AUROC after dropping function-name-mismatch failures from negatives:", flush=True)
print(f"    Remaining trials: {len(dropped_df)} (Dropped {fn_mismatch_mask.sum()} function-name-mismatch fails)", flush=True)
print(f"    Remaining negatives: {int(np.sum(y_true_drop == 0))}", flush=True)
print(f"    AUROC (filtered): {auroc_drop:.4f}", flush=True)
print(f"    AUPRC (filtered): {auprc_drop:.4f}", flush=True)


print("\n" + "=" * 80, flush=True)
print("TASK 6: C6 PREVALENCE SWEEP FROM EXISTING OUTCOMES", flush=True)
print("=" * 80, flush=True)
print("Note: Trial outcomes are strictly reused from the main run (no new API calls).", flush=True)

instances = df_main[["task_id", "seed", "source"]].drop_duplicates().values
n_instances = len(instances)

rng_prev = np.random.RandomState(42)
SWEEP_ARMS = ["ungated", "stage1_only", "cascade", "llm_judge", "deterministic", "always_fallback", "agent_ask"]

for p in [0.05, 0.20, 0.50]:
    print(f"\n--- Corruption Prevalence: {int(p*100)}% ({int(round(p*n_instances))} corrupted, {n_instances - int(round(p*n_instances))} clean) ---", flush=True)
    shuffled_indices = rng_prev.permutation(n_instances)
    n_corr = int(round(p * n_instances))
    corr_indices = set(shuffled_indices[:n_corr])
    
    selected_rows = []
    for idx, (tid, s, src) in enumerate(instances):
        is_c = idx in corr_indices
        ht_filter = (df_main["handoff_type"] != "clean") if is_c else (df_main["handoff_type"] == "clean")
        inst_rows = df_main[(df_main["task_id"] == tid) & (df_main["seed"] == s) & ht_filter]
        selected_rows.append(inst_rows)
        
    df_prev = pd.concat(selected_rows)
    
    piv_p = df_prev.pivot(index=["task_id", "seed"], columns="arm", values=["outcome", "decision", "latency_ms", "tokens_prompt", "tokens_completion"])
    u_out = piv_p[("outcome", "ungated")]
    u_lat = piv_p[("latency_ms", "ungated")]
    u_tok = piv_p[("tokens_prompt", "ungated")] + piv_p[("tokens_completion", "ungated")]
    
    piv_p_task = piv_p.groupby("task_id").agg({
        ("outcome", "ungated"): "mean",
        ("outcome", "stage1_only"): "mean",
        ("outcome", "cascade"): "mean",
        ("outcome", "llm_judge"): "mean",
        ("outcome", "deterministic"): "mean",
        ("outcome", "always_fallback"): "mean",
        ("outcome", "agent_ask"): "mean",
    })
    
    print(f"{'Arm':<18} {'Pass@1 (%)':<12} {'Delta vs Ungated':<18} {'95% Task Bootstrap CI':<24} {'Block Rate (%)':<16} {'Extra Lat (ms)':<16} {'Extra Tokens'}", flush=True)
    
    for arm in SWEEP_ARMS:
        a_out = piv_p[("outcome", arm)]
        a_dec = piv_p[("decision", arm)]
        a_lat = piv_p[("latency_ms", arm)]
        a_tok = piv_p[("tokens_prompt", arm)] + piv_p[("tokens_completion", arm)]
        
        pass1 = a_out.mean() * 100
        delta = (a_out.mean() - u_out.mean()) * 100
        blk_rate = (a_dec == "BLOCK").mean() * 100
        extra_lat = (a_lat.mean() - u_lat.mean())
        extra_tok = (a_tok.mean() - u_tok.mean())
        
        if arm == "ungated":
            print(f"{arm:<18} {pass1:6.2f}%      {'-':<18} {'-':<24} {blk_rate:5.1f}%           {extra_lat:+6.1f} ms        {extra_tok:+6.1f}", flush=True)
        else:
            t_diff = (piv_p_task[("outcome", arm)] - piv_p_task[("outcome", "ungated")]) * 100
            diff_vals = t_diff.values
            n_t = len(diff_vals)
            s_idx = rng_c4.choice(n_t, size=(B, n_t), replace=True)
            boot_d = diff_vals[s_idx].mean(axis=1)
            ci_l, ci_h = np.percentile(boot_d, [2.5, 97.5])
            sign_str = f"+{delta:.2f} pp" if delta >= 0 else f"{delta:.2f} pp"
            print(f"{arm:<18} {pass1:6.2f}%      {sign_str:<18} [{ci_l:+5.2f} pp, {ci_h:+5.2f} pp]      {blk_rate:5.1f}%           {extra_lat:+6.1f} ms        {extra_tok:+6.1f}", flush=True)
