"""Optimized, fast runner for Step 3: Aliased Grading Sensitivity Analysis."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

import ast
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workstream_c.harness.agents_interface import (
    AGENT_B_SYSTEM_PROMPT,
    AGENT_B_RAW_PROMPT,
    AGENT_ASK_CLARIFIER_PROMPT,
    AGENT_ASK_RESOLVER_PROMPT,
)
from workstream_c.harness.execution_sandbox import extract_python_code
from workstream_c.harness.task_loader import load_live_test_tasks
from workstream_c.run_live import CORRUPTION_CHOICES

CSV_PATH = "workstream_c/results/live_results_main.csv"
CLEAN_HANDOFFS_PATH = "workstream_c/results/clean_agent_a_handoffs.csv"

df_main = pd.read_csv(CSV_PATH)
tasks_dict = {t["id"]: t for t in load_live_test_tasks()}
clean_df = pd.read_csv(CLEAN_HANDOFFS_PATH)
clean_lookup = {(r["task_id"], int(r["seed"])): r["handoff_text"] for _, r in clean_df.iterrows()}

caches = {s: json.load(open(f"cache/llm_cache_s{s}.json", "r", encoding="utf-8")) for s in [42, 43, 44]}
corr_map = dict(CORRUPTION_CHOICES)
corrupted_lookup = {}


def evaluate_code_with_alias_raw(clean_code: str, funcs: list, entry_point: str, test_code_str: str) -> bool:
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


executor_pool = ThreadPoolExecutor(max_workers=1)

def evaluate_code_with_alias(code_str: str, entry_point: str, test_code_str: str, timeout_sec: float = 1.0) -> bool:
    clean_code = extract_python_code(code_str)
    try:
        tree = ast.parse(clean_code)
        funcs = [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except Exception:
        return False

    future = executor_pool.submit(evaluate_code_with_alias_raw, clean_code, funcs, entry_point, test_code_str)
    try:
        return future.result(timeout=timeout_sec)
    except TimeoutError:
        return False
    except Exception:
        return False


def get_agent_b_code(row) -> str:
    tid = row["task_id"]
    s = int(row["seed"])
    arm = row["arm"]
    htype = row["handoff_type"]
    dec = row["decision"]
    t = tasks_dict[tid]
    ep = t["entry_point"]
    prob = t["problem"]

    if htype == "clean":
        cand_handoff = clean_lookup[(tid, s)]
    else:
        k = (tid, s)
        if k not in corrupted_lookup:
            fn = corr_map[htype]
            corrupted_lookup[k] = fn(clean_lookup[k], ep)
        cand_handoff = corrupted_lookup[k]

    if arm == "ungated":
        prompt_b = AGENT_B_SYSTEM_PROMPT.format(handoff=cand_handoff, entry_point=ep)
    elif arm == "always_fallback":
        prompt_b = AGENT_B_RAW_PROMPT.format(problem=prob, entry_point=ep)
    elif arm in ["stage1_only", "deterministic", "llm_judge", "cascade"]:
        if dec == "PASS":
            prompt_b = AGENT_B_SYSTEM_PROMPT.format(handoff=cand_handoff, entry_point=ep)
        else:
            prompt_b = AGENT_B_RAW_PROMPT.format(problem=prob, entry_point=ep)
    elif arm == "agent_ask":
        if dec == "PASS":
            prompt_b = AGENT_B_SYSTEM_PROMPT.format(handoff=cand_handoff, entry_point=ep)
        else:
            ask_prompt = AGENT_ASK_CLARIFIER_PROMPT.format(handoff=cand_handoff, problem=prob)
            ask_h = hashlib.sha256(f"gemini-3.5-flash-lite_0.0_{s}_{ask_prompt}".encode("utf-8")).hexdigest()
            ask_text = caches[s].get(ask_h, {}).get("text", "")
            res_prompt = AGENT_ASK_RESOLVER_PROMPT.format(handoff=cand_handoff, problem=prob, question=ask_text)
            res_h = hashlib.sha256(f"gemini-3.5-flash-lite_0.7_{s}_{res_prompt}".encode("utf-8")).hexdigest()
            clarified = caches[s].get(res_h, {}).get("text", cand_handoff)
            prompt_b = AGENT_B_SYSTEM_PROMPT.format(handoff=clarified, entry_point=ep)

    b_h = hashlib.sha256(f"gemini-3.5-flash-lite_0.0_{s}_{prompt_b}".encode("utf-8")).hexdigest()
    return caches[s].get(b_h, {}).get("text", "")


print("=" * 80, flush=True)
print("STEP 3: ALIASED GRADING SENSITIVITY ANALYSIS (ALL 7 ARMS & BOTH CONDITIONS)", flush=True)
print("=" * 80, flush=True)

# 1. Compute aliased outcomes
t0 = time.time()
alias_outcomes = []
cached_alias_eval = {}

fn_missing_mask = df_main["error_message"].str.contains("not found in generated code", na=False)
total_missing = fn_missing_mask.sum()
print(f"Total rows: {len(df_main)} | Failures with function name missing: {total_missing}", flush=True)

for idx, r in df_main.iterrows():
    orig_out = int(r["outcome"])
    if orig_out == 1:
        alias_outcomes.append(1)
        continue

    # Failure case
    if not fn_missing_mask.iloc[idx]:
        alias_outcomes.append(0)
        continue

    tid = r["task_id"]
    s = int(r["seed"])
    ep = tasks_dict[tid]["entry_point"]
    test_c = tasks_dict[tid]["test_code"]

    code = get_agent_b_code(r)
    if not code:
        alias_outcomes.append(0)
        continue

    eval_k = (code, ep, test_c)
    if eval_k not in cached_alias_eval:
        cached_alias_eval[eval_k] = 1 if evaluate_code_with_alias(code, ep, test_c) else 0
    alias_outcomes.append(cached_alias_eval[eval_k])

df_main["alias_outcome"] = alias_outcomes
flipped = ((df_main["outcome"] == 0) & (df_main["alias_outcome"] == 1)).sum()
print(f"Evaluation complete in {time.time() - t0:.2f}s! Total trials flipped: {flipped} / {total_missing}", flush=True)

# 2. Pass@1 Table
print("\n" + "=" * 80, flush=True)
print("ALIASED PASS@1 COMPARISON ACROSS ALL 7 ARMS (SENSITIVITY ANALYSIS)", flush=True)
print("=" * 80, flush=True)
print(f"{'Arm':<18} {'Strict Clean':<14} {'Alias Clean':<14} {'Strict Corr':<14} {'Alias Corr':<14} {'Strict All':<14} {'Alias All':<14}", flush=True)
print("-" * 80, flush=True)

arms_order = ["ungated", "stage1_only", "cascade", "llm_judge", "deterministic", "always_fallback", "agent_ask"]
for arm in arms_order:
    sub = df_main[df_main["arm"] == arm]
    sc = sub[sub["handoff_type"] == "clean"]["outcome"].mean() * 100
    ac = sub[sub["handoff_type"] == "clean"]["alias_outcome"].mean() * 100
    so = sub[sub["handoff_type"] != "clean"]["outcome"].mean() * 100
    ao = sub[sub["handoff_type"] != "clean"]["alias_outcome"].mean() * 100
    sa = sub["outcome"].mean() * 100
    aa = sub["alias_outcome"].mean() * 100
    print(f"{arm:<18} {sc:6.2f}%        {ac:6.2f}%        {so:6.2f}%        {ao:6.2f}%        {sa:6.2f}%        {aa:6.2f}%", flush=True)

# 3. Source Breakdown under Aliasing
print("\n" + "=" * 80, flush=True)
print("ALIASED PASS@1 BY SOURCE & CONDITION", flush=True)
print("=" * 80, flush=True)
for src in ["humaneval", "mbpp", "hand_easy", "hand_hard"]:
    print(f"\n--- Source: {src} ---", flush=True)
    s_sub = df_main[df_main["source"] == src]
    for arm in arms_order:
        asub = s_sub[s_sub["arm"] == arm]
        ac = asub[asub["handoff_type"] == "clean"]["alias_outcome"].mean() * 100
        ao = asub[asub["handoff_type"] != "clean"]["alias_outcome"].mean() * 100
        aa = asub["alias_outcome"].mean() * 100
        print(f"  {arm:<18}: Clean = {ac:5.2f}% | Corrupted = {ao:5.2f}% | Overall = {aa:5.2f}%", flush=True)

# 4. Task-Level Paired Bootstrap CI & Permutation Tests vs Ungated
print("\n" + "=" * 80, flush=True)
print("TASK-LEVEL PAIRED BOOTSTRAP CI & PERMUTATION TEST VS UNGATED (ALIASED)", flush=True)
print("=" * 80, flush=True)

piv_alias = df_main.pivot(index=["task_id", "handoff_type", "seed"], columns="arm", values="alias_outcome")
piv_task_overall = piv_alias.groupby("task_id").mean()
piv_task_clean = piv_alias.xs("clean", level="handoff_type").groupby("task_id").mean()
corr_mask = piv_alias.index.get_level_values("handoff_type") != "clean"
piv_task_corr = piv_alias[corr_mask].groupby("task_id").mean()

B = 10000
rng_boot = np.random.RandomState(42)
n_tasks = 300
sample_idx = rng_boot.choice(n_tasks, size=(B, n_tasks), replace=True)

perm_rng = np.random.RandomState(42)
signs = perm_rng.choice([-1, 1], size=(100000, n_tasks))

for cond_name, p_df in [("Overall (Clean + Corrupted)", piv_task_overall), ("Clean Only", piv_task_clean), ("Corrupted Only", piv_task_corr)]:
    print(f"\n--- Condition: {cond_name} ---", flush=True)
    u_vals = p_df["ungated"].values
    print(f"{'Arm':<18} {'Aliased Pass@1':<16} {'Delta vs Ungated':<18} {'95% Task Bootstrap CI':<24} {'Permutation p'}", flush=True)
    for arm in arms_order:
        if arm == "ungated":
            p1 = u_vals.mean() * 100
            print(f"{arm:<18} {p1:6.2f}%          {'-':<18} {'-':<24} {'-'}", flush=True)
            continue
        a_vals = p_df[arm].values
        delta = (a_vals.mean() - u_vals.mean()) * 100
        diff_task = (a_vals - u_vals) * 100

        boot_diffs = diff_task[sample_idx].mean(axis=1)
        ci_l, ci_h = np.percentile(boot_diffs, [2.5, 97.5])

        perm_stats = np.abs(np.mean(signs * diff_task, axis=1))
        p_perm = np.mean(perm_stats >= np.abs(delta))

        sign_str = f"+{delta:.2f} pp" if delta >= 0 else f"{delta:.2f} pp"
        print(f"{arm:<18} {a_vals.mean()*100:6.2f}%          {sign_str:<18} [{ci_l:+5.2f} pp, {ci_h:+5.2f} pp]      p = {p_perm:.4f}", flush=True)

# 5. Random Gate Control under Aliasing
print("\n" + "=" * 80, flush=True)
print("RANDOM GATE CONTROL UNDER ALIASING", flush=True)
print("=" * 80, flush=True)

piv_all_cols = df_main.pivot(index=["task_id", "handoff_type", "seed", "source"], columns="arm", values=["alias_outcome", "decision"])
u_alias = piv_all_cols[("alias_outcome", "ungated")]
af_alias = piv_all_cols[("alias_outcome", "always_fallback")]
s1_alias = piv_all_cols[("alias_outcome", "stage1_only")]
dec_s1 = piv_all_cols[("decision", "stage1_only")]

t_df = pd.DataFrame({
    "ungated": u_alias,
    "always_fallback": af_alias,
    "stage1_only": s1_alias,
    "s1_decision": dec_s1,
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

task_agg = t_df.groupby("task_id").agg({
    "stage1_only": "mean",
    "random_gate_mean": "mean",
}).reset_index()

task_agg["diff"] = (task_agg["stage1_only"] - task_agg["random_gate_mean"]) * 100
real_s1_mean = task_agg["stage1_only"].mean() * 100
rand_gate_mean = task_agg["random_gate_mean"].mean() * 100
observed_delta = task_agg["diff"].mean()

diff_vals = task_agg["diff"].values
boot_diffs = diff_vals[sample_idx].mean(axis=1)
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
perm_stats = np.abs(np.mean(signs * diff_vals, axis=1))
p_perm = np.mean(perm_stats >= np.abs(observed_delta))

print(f"Aliased Stage-1 Pass@1:          {real_s1_mean:.2f}%", flush=True)
print(f"Aliased Random Gate Expected:    {rand_gate_mean:.2f}%", flush=True)
print(f"Difference (Real - Random):      {observed_delta:+.2f} pp", flush=True)
print(f"95% Task-Level Bootstrap CI:     [{ci_low:+.2f} pp, {ci_high:+.2f} pp]", flush=True)
print(f"Paired Permutation p-value:      p = {p_perm:.4f}", flush=True)

# 6. Per-corruption interception rates under aliasing
print("\n" + "=" * 80, flush=True)
print("PER-CORRUPTION INTERCEPTION RATES & RECOVERY UNDER ALIASING", flush=True)
print("=" * 80, flush=True)
corr_df = df_main[df_main["handoff_type"] != "clean"]
print(f"{'Corruption Type':<26} {'N':<6} {'Stage1 Block %':<16} {'Cascade Block %':<18} {'Ungated Alias':<16} {'Stage1 Alias':<14} {'Gain (pp)'}", flush=True)
for ctype in sorted(corr_df["handoff_type"].unique()):
    c_sub = corr_df[corr_df["handoff_type"] == ctype]
    n_c = len(c_sub[c_sub["arm"] == "ungated"])
    s1_blk = (c_sub[c_sub["arm"] == "stage1_only"]["decision"] == "BLOCK").mean() * 100
    casc_blk = (c_sub[c_sub["arm"] == "cascade"]["decision"] == "BLOCK").mean() * 100
    u_al = c_sub[c_sub["arm"] == "ungated"]["alias_outcome"].mean() * 100
    s1_al = c_sub[c_sub["arm"] == "stage1_only"]["alias_outcome"].mean() * 100
    gain = s1_al - u_al
    print(f"{ctype:<26} {n_c:<6} {s1_blk:5.1f}%          {casc_blk:5.1f}%            {u_al:5.2f}%          {s1_al:5.2f}%        {gain:+5.2f} pp", flush=True)

executor_pool.shutdown(wait=False)
