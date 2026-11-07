"""Comprehensive analysis checks on workstream_c/results/live_results_main.csv.

Covers:
1. Task-level statistics (Pass@1, bootstrap CI, paired permutation test, McNemar per seed)
2. Block rates by condition, corruption type, source
3. Random-block control simulation & blocked vs non-blocked analysis
4. Corruption harm per corruption type (clean vs corrupted on same tasks)
5. Source reconciliation (manifest vs CSV)
6. Clean ungated failure analysis by source and failure category
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

CSV_PATH = "workstream_c/results/live_results_main.csv"
MANIFEST_PATH = "workstream_c/manifests/manifest_live_test.json"

df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("ITEM 5: SOURCE RECONCILIATION")
print("=" * 80)

with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    manifest_data = json.load(f)

print(f"Manifest 'count' field: {manifest_data.get('count')}")
print(f"Manifest 'sources' dict: {manifest_data.get('sources')}")
manifest_tasks = manifest_data.get("tasks", [])
print(f"Manifest len(tasks): {len(manifest_tasks)}")

# Count task ID prefixes in manifest
prefix_counts = {}
for t in manifest_tasks:
    p = t.split("_")[0]
    prefix_counts[p] = prefix_counts.get(p, 0) + 1
print(f"Manifest task ID prefix counts: {prefix_counts}")

# CSV unique tasks by source
csv_unique = df[["task_id", "source"]].drop_duplicates()
print("\nCSV unique tasks by 'source' column:")
print(csv_unique["source"].value_counts())
print(f"Total unique tasks in CSV: {len(csv_unique)}")

csv_prefix_counts = {}
for t in csv_unique["task_id"]:
    p = t.split("_")[0]
    csv_prefix_counts[p] = csv_prefix_counts.get(p, 0) + 1
print(f"CSV unique tasks prefix counts: {csv_prefix_counts}")

print("\nCSV total rows by 'source' column:")
print(df["source"].value_counts())
print(f"Total rows in CSV: {len(df)}")


print("\n" + "=" * 80)
print("ITEM 1: TASK-LEVEL STATISTICS")
print("=" * 80)

ARMS = [
    "ungated",
    "stage1_only",
    "cascade",
    "llm_judge",
    "always_fallback",
    "agent_ask",
    "deterministic",
]

# We compute task-level outcomes:
# For each task, condition, arm, average across the 3 seeds
# Clean condition:
clean_df = df[df["handoff_type"] == "clean"]
corr_df = df[df["handoff_type"] != "clean"]

def compute_task_stats(sub_df, condition_label):
    print(f"\n--- Condition: {condition_label} (N=300 tasks) ---")
    # Pivot: rows = task_id, columns = arm, value = mean outcome across seeds
    piv = sub_df.pivot_table(index="task_id", columns="arm", values="outcome", aggfunc="mean")
    
    rng = np.random.RandomState(42)
    B = 10000
    n_tasks = len(piv)
    
    print(f"{'Arm':<18} {'Pass@1 (%)':<12} {'Diff vs Ungated':<18} {'95% Bootstrap CI':<24} {'Paired Perm p-value'}")
    
    ungated_scores = piv["ungated"].values
    
    for arm in ARMS:
        arm_scores = piv[arm].values
        pass1 = arm_scores.mean() * 100
        diff = (arm_scores - ungated_scores) * 100
        mean_diff = diff.mean()
        
        if arm == "ungated":
            # Bootstrap CI of Pass@1 itself
            boot_means = [rng.choice(arm_scores, size=n_tasks, replace=True).mean() * 100 for _ in range(B)]
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
            print(f"{arm:<18} {pass1:6.2f}%      {'-':<18} [{ci_low:5.2f}%, {ci_high:5.2f}%]          {'-'}")
        else:
            diff_values = arm_scores - ungated_scores
            boot_diffs = [rng.choice(diff_values, size=n_tasks, replace=True).mean() * 100 for _ in range(B)]
            ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
            
            # Paired permutation test (Monte Carlo, 100,000 iterations)
            non_zero_diffs = diff_values[diff_values != 0]
            if len(non_zero_diffs) == 0:
                p_perm = 1.0
            else:
                obs_stat = np.abs(np.mean(diff_values))
                perm_rng = np.random.RandomState(42)
                signs = perm_rng.choice([-1, 1], size=(100000, len(diff_values)))
                perm_stats = np.abs(np.mean(signs * diff_values, axis=1))
                p_perm = np.mean(perm_stats >= obs_stat)
                
            sign_str = f"+{mean_diff:.2f} pp" if mean_diff >= 0 else f"{mean_diff:.2f} pp"
            print(f"{arm:<18} {pass1:6.2f}%      {sign_str:<18} [{ci_low:5.2f} pp, {ci_high:5.2f} pp]      p = {p_perm:.4f}")

# Clean condition
compute_task_stats(clean_df, "Clean")

# Corrupted condition
compute_task_stats(corr_df, "Corrupted")

# Combined condition (average of clean and corrupted for each task & seed, or all 6 trials)
print("\n--- Condition: Combined Clean + Corrupted (N=300 tasks, 6 trials/task) ---")
piv_comb = df.pivot_table(index="task_id", columns="arm", values="outcome", aggfunc="mean")
rng = np.random.RandomState(42)
B = 10000
n_tasks = len(piv_comb)
ungated_comb = piv_comb["ungated"].values

print(f"{'Arm':<18} {'Pass@1 (%)':<12} {'Diff vs Ungated':<18} {'95% Bootstrap CI':<24} {'Paired Perm p-value'}")
for arm in ARMS:
    arm_scores = piv_comb[arm].values
    pass1 = arm_scores.mean() * 100
    diff = (arm_scores - ungated_comb) * 100
    mean_diff = diff.mean()
    
    if arm == "ungated":
        boot_means = [rng.choice(arm_scores, size=n_tasks, replace=True).mean() * 100 for _ in range(B)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        print(f"{arm:<18} {pass1:6.2f}%      {'-':<18} [{ci_low:5.2f}%, {ci_high:5.2f}%]          {'-'}")
    else:
        diff_values = arm_scores - ungated_comb
        boot_diffs = [rng.choice(diff_values, size=n_tasks, replace=True).mean() * 100 for _ in range(B)]
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        
        obs_stat = np.abs(np.mean(diff_values))
        perm_rng = np.random.RandomState(42)
        signs = perm_rng.choice([-1, 1], size=(100000, len(diff_values)))
        perm_stats = np.abs(np.mean(signs * diff_values, axis=1))
        p_perm = np.mean(perm_stats >= obs_stat)
        sign_str = f"+{mean_diff:.2f} pp" if mean_diff >= 0 else f"{mean_diff:.2f} pp"
        print(f"{arm:<18} {pass1:6.2f}%      {sign_str:<18} [{ci_low:5.2f} pp, {ci_high:5.2f} pp]      p = {p_perm:.4f}")

# Per-Seed McNemar Tests (N=300 paired tasks per seed and condition)
print("\n--- Per-Seed Analysis & Exact McNemar Tests (Task-level per seed) ---")
for s in [42, 43, 44]:
    print(f"\nSeed {s}:")
    for cond_name, c_df in [("Clean", clean_df), ("Corrupted", corr_df), ("All", df)]:
        sub_s = c_df[c_df["seed"] == s]
        piv_s = sub_s.pivot_table(index="task_id", columns="arm", values="outcome", aggfunc="mean")
        print(f"  Condition: {cond_name}")
        for arm in ["stage1_only", "cascade", "llm_judge", "deterministic", "always_fallback", "agent_ask"]:
            y_u = piv_s["ungated"].values
            y_a = piv_s[arm].values
            
            # If values are binary (0 or 1)
            b = np.sum((y_u == 0) & (y_a == 1))
            c = np.sum((y_u == 1) & (y_a == 0))
            diff_pp = (y_a.mean() - y_u.mean()) * 100
            
            # Exact two-sided binomial test on discordant pairs
            if b + c > 0:
                p_val = stats.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
            else:
                p_val = 1.0
            print(f"    {arm:<16}: Pass@1={y_a.mean()*100:5.2f}% (Diff: {diff_pp:+5.2f} pp, b={b}, c={c}, exact McNemar p={p_val:.4f})")


print("\n" + "=" * 80)
print("ITEM 2: BLOCK RATES")
print("=" * 80)
GATED_ARMS = ["stage1_only", "cascade", "deterministic", "llm_judge"]

print("\n1. Overall Block Rates:")
for arm in GATED_ARMS:
    sub = df[df["arm"] == arm]
    n_tot = len(sub)
    n_blk = (sub["decision"] == "BLOCK").sum()
    print(f"  {arm:<16}: {n_blk}/{n_tot} ({n_blk/n_tot*100:5.2f}%)")

print("\n2. Block Rates by Condition (Clean vs Corrupted):")
for arm in GATED_ARMS:
    sub = df[df["arm"] == arm]
    clean_sub = sub[sub["handoff_type"] == "clean"]
    corr_sub = sub[sub["handoff_type"] != "clean"]
    
    c_blk = (clean_sub["decision"] == "BLOCK").sum()
    c_tot = len(clean_sub)
    
    k_blk = (corr_sub["decision"] == "BLOCK").sum()
    k_tot = len(corr_sub)
    
    print(f"  {arm:<16}: Clean = {c_blk}/{c_tot} ({c_blk/c_tot*100:5.2f}%), Corrupted = {k_blk}/{k_tot} ({k_blk/k_tot*100:5.2f}%)")

print("\n3. Block Rates by Corruption Type:")
corr_types = sorted([ct for ct in df["handoff_type"].unique() if ct != "clean"])
header = f"  {'Corruption Type':<26}" + "".join([f"{arm:<16}" for arm in GATED_ARMS])
print(header)
for ct in corr_types:
    row_str = f"  {ct:<26}"
    for arm in GATED_ARMS:
        sub = df[(df["arm"] == arm) & (df["handoff_type"] == ct)]
        n_blk = (sub["decision"] == "BLOCK").sum()
        n_tot = len(sub)
        row_str += f"{n_blk}/{n_tot} ({n_blk/n_tot*100:4.1f}%)  "
    print(row_str)

print("\n4. Block Rates by Source:")
sources = ["humaneval", "mbpp", "hand_easy", "hand_hard"]
header = f"  {'Source':<20}" + "".join([f"{arm:<16}" for arm in GATED_ARMS])
print(header)
for src in sources:
    row_str = f"  {src:<20}"
    for arm in GATED_ARMS:
        sub = df[(df["arm"] == arm) & (df["source"] == src)]
        n_blk = (sub["decision"] == "BLOCK").sum()
        n_tot = len(sub)
        row_str += f"{n_blk}/{n_tot} ({n_blk/n_tot*100:4.1f}%)  "
    print(row_str)

# Grouped hand-crafted
row_str = f"  {'hand_crafted (all)':<20}"
for arm in GATED_ARMS:
    sub = df[(df["arm"] == arm) & (df["source"].isin(["hand_easy", "hand_hard"]))]
    n_blk = (sub["decision"] == "BLOCK").sum()
    n_tot = len(sub)
    row_str += f"{n_blk}/{n_tot} ({n_blk/n_tot*100:4.1f}%)  "
print(row_str)


print("\n" + "=" * 80)
print("ITEM 3: RANDOM-BLOCK CONTROL")
print("=" * 80)

# Random block control for stage1_only:
# Simulate gate blocking same fraction of handoffs at random (per condition and source, 1000 draws, seed=42)
# using each task's ungated and always_fallback outcomes if blocked or not.

# Let's map each trial (task_id, handoff_type, seed) to its ungated and always_fallback outcome
piv_trials = df.pivot(index=["task_id", "handoff_type", "seed", "source"], columns="arm", values=["outcome", "decision"])
outcomes_u = piv_trials[("outcome", "ungated")]
outcomes_af = piv_trials[("outcome", "always_fallback")]
decisions_s1 = piv_trials[("decision", "stage1_only")]
outcomes_s1 = piv_trials[("outcome", "stage1_only")]

trial_df = pd.DataFrame({
    "ungated": outcomes_u,
    "always_fallback": outcomes_af,
    "stage1_only": outcomes_s1,
    "s1_decision": decisions_s1,
}).reset_index()

trial_df["condition"] = trial_df["handoff_type"].apply(lambda x: "clean" if x == "clean" else "corrupted")

# Strata: (condition, source)
strata_groups = trial_df.groupby(["condition", "source"])
print("Strata and Stage-1 block counts:")
strata_info = {}
for (cond, src), grp in strata_groups:
    n_tot = len(grp)
    n_blk = (grp["s1_decision"] == "BLOCK").sum()
    strata_info[(cond, src)] = (n_tot, n_blk)
    print(f"  Strata ({cond}, {src}): N={n_tot}, Blocked={n_blk} ({n_blk/n_tot*100:.2f}%)")

rng_sim = np.random.RandomState(42)
N_DRAWS = 1000
sim_pass1_overall = []

for draw in range(N_DRAWS):
    sim_outcomes = np.zeros(len(trial_df))
    for (cond, src), (n_tot, n_blk) in strata_info.items():
        idx = trial_df[(trial_df["condition"] == cond) & (trial_df["source"] == src)].index.values
        # randomly choose n_blk indices to block
        blocked_idx = rng_sim.choice(idx, size=n_blk, replace=False)
        pass_idx = np.setdiff1d(idx, blocked_idx)
        
        sim_outcomes[blocked_idx] = trial_df.loc[blocked_idx, "always_fallback"].values
        sim_outcomes[pass_idx] = trial_df.loc[pass_idx, "ungated"].values
        
    sim_pass1_overall.append(sim_outcomes.mean() * 100)

sim_mean = np.mean(sim_pass1_overall)
sim_ci_low, sim_ci_high = np.percentile(sim_pass1_overall, [2.5, 97.5])
real_s1_pass1 = trial_df["stage1_only"].mean() * 100
real_ungated_pass1 = trial_df["ungated"].mean() * 100
real_af_pass1 = trial_df["always_fallback"].mean() * 100

print(f"\nSimulation Results (1,000 draws, fixed seed 42):")
print(f"  Ungated Pass@1:                {real_ungated_pass1:.2f}%")
print(f"  Always Fallback Pass@1:        {real_af_pass1:.2f}%")
print(f"  Random Gate (stratified rate): {sim_mean:.2f}% [95% CI: {sim_ci_low:.2f}%, {sim_ci_high:.2f}%]")
print(f"  Real Stage-1 Gate Pass@1:      {real_s1_pass1:.2f}%")
print(f"  Advantage of Stage-1 vs Random Gate: {real_s1_pass1 - sim_mean:+.2f} pp")

# Pass@1 on blocked versus non-blocked handoffs:
print("\nStage-1 Gate Pass@1 on Blocked vs Non-Blocked Handoffs:")
blk_trials = trial_df[trial_df["s1_decision"] == "BLOCK"]
pass_trials = trial_df[trial_df["s1_decision"] == "PASS"]

print(f"  Total Blocked trials: {len(blk_trials)}")
print(f"    Ungated Pass@1 on these blocked handoffs (counterfactual):     {blk_trials['ungated'].mean()*100:.2f}% ({blk_trials['ungated'].sum()}/{len(blk_trials)})")
print(f"    Always Fallback Pass@1 on these blocked handoffs:             {blk_trials['always_fallback'].mean()*100:.2f}% ({blk_trials['always_fallback'].sum()}/{len(blk_trials)})")
print(f"    Actual Stage-1 Pass@1 on these blocked handoffs:              {blk_trials['stage1_only'].mean()*100:.2f}% ({blk_trials['stage1_only'].sum()}/{len(blk_trials)})")

print(f"\n  Total Non-Blocked (Passed) trials: {len(pass_trials)}")
print(f"    Ungated Pass@1 on passed handoffs:                            {pass_trials['ungated'].mean()*100:.2f}% ({pass_trials['ungated'].sum()}/{len(pass_trials)})")
print(f"    Always Fallback Pass@1 on passed handoffs:                    {pass_trials['always_fallback'].mean()*100:.2f}% ({pass_trials['always_fallback'].sum()}/{len(pass_trials)})")
print(f"    Actual Stage-1 Pass@1 on passed handoffs:                     {pass_trials['stage1_only'].mean()*100:.2f}% ({pass_trials['stage1_only'].sum()}/{len(pass_trials)})")

# By condition (Clean vs Corrupted)
for cond in ["clean", "corrupted"]:
    b_c = blk_trials[blk_trials["condition"] == cond]
    p_c = pass_trials[pass_trials["condition"] == cond]
    print(f"\n  Condition: {cond.upper()}")
    print(f"    Blocked (N={len(b_c)}): Ungated Pass@1 = {b_c['ungated'].mean()*100:.2f}%, Fallback Pass@1 = {b_c['always_fallback'].mean()*100:.2f}%")
    print(f"    Passed  (N={len(p_c)}): Ungated Pass@1 = {p_c['ungated'].mean()*100:.2f}%, Fallback Pass@1 = {p_c['always_fallback'].mean()*100:.2f}%")


print("\n" + "=" * 80)
print("ITEM 4: CORRUPTION HARM")
print("=" * 80)
# For each corruption type, report ungated Pass@1 on clean vs corrupted handoffs for the SAME tasks
ungated_df = df[df["arm"] == "ungated"]
corr_types = sorted([ct for ct in ungated_df["handoff_type"].unique() if ct != "clean"])

print(f"{'Corruption Type':<26} {'N (tasks*seeds)':<16} {'Clean Pass@1':<16} {'Corr Pass@1':<16} {'Harm (Clean - Corr)'}")
for ct in corr_types:
    corr_subset = ungated_df[ungated_df["handoff_type"] == ct]
    # Identify task_id and seed pairs
    pairs = set(zip(corr_subset["task_id"], corr_subset["seed"]))
    n_pairs = len(pairs)
    
    # Matching clean rows
    clean_subset = ungated_df[
        (ungated_df["handoff_type"] == "clean") & 
        ungated_df.apply(lambda r: (r["task_id"], r["seed"]) in pairs, axis=1)
    ]
    
    clean_p1 = clean_subset["outcome"].mean() * 100
    corr_p1 = corr_subset["outcome"].mean() * 100
    harm = clean_p1 - corr_p1
    
    sign_str = f"+{harm:.2f} pp" if harm >= 0 else f"{harm:.2f} pp"
    print(f"{ct:<26} {n_pairs:<16} {clean_p1:6.2f}% ({clean_subset['outcome'].sum()}/{n_pairs})   {corr_p1:6.2f}% ({corr_subset['outcome'].sum()}/{n_pairs})   {sign_str}")


print("\n" + "=" * 80)
print("ITEM 6: CLEAN BASELINE & UNGATED FAILURE REASONS")
print("=" * 80)

clean_ungated = df[(df["arm"] == "ungated") & (df["handoff_type"] == "clean")]
total_clean = len(clean_ungated)
passed_clean = (clean_ungated["outcome"] == 1).sum()
failed_clean = (clean_ungated["outcome"] == 0).sum()

print(f"Clean ungated total trials: {total_clean}")
print(f"Passed: {passed_clean} ({passed_clean/total_clean*100:.2f}%)")
print(f"Failed: {failed_clean} ({failed_clean/total_clean*100:.2f}%)")

clean_fails = clean_ungated[clean_ungated["outcome"] == 0].copy()

def categorize_error(msg):
    if not isinstance(msg, str) or not msg.strip():
        return "Unknown / Empty Error"
    if "Function" in msg and "not found in generated code" in msg:
        return "Function Name Mismatch / Missing Entry Point"
    if "Compilation/Import Error" in msg:
        return "Compilation / Syntax / Import Error"
    if "Test Setup Error" in msg:
        return "Test Setup Error"
    if "Neither 'test' nor 'check' harness found" in msg:
        return "Harness Missing Error"
    if "timed out" in msg:
        return "Execution Timeout"
    if "Assertion Failure" in msg or "Assertion check returned False" in msg:
        if "NameError" in msg:
            return "Assertion Failure: NameError / Undefined Reference"
        if "TypeError" in msg:
            return "Assertion Failure: TypeError / Signature Mismatch"
        if "IndexError" in msg:
            return "Assertion Failure: IndexError"
        if "KeyError" in msg:
            return "Assertion Failure: KeyError"
        if "ValueError" in msg:
            return "Assertion Failure: ValueError"
        if "ZeroDivisionError" in msg:
            return "Assertion Failure: ZeroDivisionError"
        if "AssertionError" in msg:
            return "Assertion Failure: Wrong Output (AssertionError)"
        return "Assertion Failure: Other Runtime Exception"
    return "Other Execution Error"

clean_fails["category"] = clean_fails["error_message"].apply(categorize_error)

print("\nFailure Categories across ALL sources (N=374 failures out of 900 clean trials):")
cat_counts = clean_fails["category"].value_counts()
for cat, count in cat_counts.items():
    print(f"  {cat:<50}: {count:3d} ({count/failed_clean*100:5.2f}% of fails, {count/total_clean*100:5.2f}% of all clean)")

print("\nFailure Categories Breakdown BY SOURCE:")
for src in ["humaneval", "mbpp", "hand_easy", "hand_hard"]:
    src_all = clean_ungated[clean_ungated["source"] == src]
    src_fails = clean_fails[clean_fails["source"] == src]
    n_src_all = len(src_all)
    n_src_fail = len(src_fails)
    print(f"\n  Source: {src} (Total clean = {n_src_all}, Failed = {n_src_fail}, Pass@1 = {src_all['outcome'].mean()*100:.2f}%):")
    src_cat_counts = src_fails["category"].value_counts()
    for cat, count in src_cat_counts.items():
        print(f"    {cat:<48}: {count:3d} ({count/n_src_fail*100:5.2f}% of fails)")

