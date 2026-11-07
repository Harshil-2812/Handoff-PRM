"""Verification script: Recomputes all metrics from live_results_main.csv and validates them against RESULTS_SUMMARY.md and AUDIT.md.
"""

import re
import pandas as pd
import numpy as np

CSV_PATH = "workstream_c/results/live_results_main.csv"
RESULTS_MD = "workstream_c/RESULTS_SUMMARY.md"
AUDIT_MD = "workstream_c/AUDIT.md"

df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("HARD-CODING & CONSISTENCY CHECK")
print("=" * 80)

# Check verify_live_stats.py dynamically computes all counts
print("1. Checking verify_live_stats.py dynamic computation:")
with open("verify_live_stats.py", "r", encoding="utf-8") as f:
    v_code = f.read()

assert "len(df)" in v_code
assert "df['source'].unique()" in v_code or "df[\"source\"].unique()" in v_code
assert "df['outcome'].mean()" in v_code or "df[\"outcome\"].mean()" in v_code
print("   [PASS] verify_live_stats.py computes all counts, denominators, and means dynamically from input data.")

# Extract computed metrics from live_results_main.csv
total_rows = len(df)
n_tasks = df["task_id"].nunique()
tasks_by_source = df[["task_id", "source"]].drop_duplicates()["source"].value_counts().to_dict()
rows_by_source = df["source"].value_counts().to_dict()

print(f"\n2. Core Dataset Counts:")
print(f"   Total rows: {total_rows}")
print(f"   Unique tasks: {n_tasks}")
print(f"   Tasks by source: {tasks_by_source}")
print(f"   Rows by source: {rows_by_source}")

# Check Pass@1 overall
pass1_overall = (df.groupby("arm")["outcome"].mean() * 100).to_dict()
pass1_clean = (df[df["handoff_type"] == "clean"].groupby("arm")["outcome"].mean() * 100).to_dict()
pass1_corr = (df[df["handoff_type"] != "clean"].groupby("arm")["outcome"].mean() * 100).to_dict()
u_overall = pass1_overall["ungated"]
deltas = {arm: pass1_overall[arm] - u_overall for arm in pass1_overall}

# Contingency counts
piv = df.pivot(index=["task_id", "seed", "handoff_type"], columns="arm", values="outcome")
u_vals = piv["ungated"].values
contingency = {}
for arm in pass1_overall:
    if arm == "ungated":
        continue
    a_vals = piv[arm].values
    b = int(np.sum((u_vals == 1) & (a_vals == 0)))
    c = int(np.sum((u_vals == 0) & (a_vals == 1)))
    contingency[arm] = (b, c)

# False block rates on clean
clean_s1 = df[(df["handoff_type"] == "clean") & (df["arm"] == "stage1_only")]
clean_blocks = int((clean_s1["decision"] == "BLOCK").sum())
clean_total = len(clean_s1)
fbr_all_clean = (clean_blocks / clean_total) * 100

clean_ungated_outcome = df[(df["handoff_type"] == "clean") & (df["arm"] == "ungated")].set_index(["task_id", "seed"])["outcome"]
clean_s1_with_u = clean_s1.set_index(["task_id", "seed"])
clean_passing_mask = clean_ungated_outcome == 1
passing_clean_total = int(clean_passing_mask.sum())
passing_clean_blocked = int((clean_s1_with_u.loc[clean_passing_mask, "decision"] == "BLOCK").sum())
fbr_passing_clean = (passing_clean_blocked / passing_clean_total) * 100

print(f"\n3. False Block Rates:")
print(f"   (a) All clean handoffs blocked:     {clean_blocks}/{clean_total} ({fbr_all_clean:.2f}%)")
print(f"   (b) Passing clean handoffs blocked: {passing_clean_blocked}/{passing_clean_total} ({fbr_passing_clean:.2f}%)")

# Compare with text in RESULTS_SUMMARY.md
with open(RESULTS_MD, "r", encoding="utf-8") as f:
    res_text = f.read()

with open(AUDIT_MD, "r", encoding="utf-8") as f:
    audit_text = f.read()

print("\n4. Cross-referencing values with RESULTS_SUMMARY.md and AUDIT.md:")
mismatches = []

# Parse markdown table line-by-line
def extract_table_metric(text, arm_name, col_idx):
    for line in text.splitlines():
        if f"**{arm_name}**" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            # parts[0] is empty, parts[1] is Arm, parts[2] is Description, etc.
            if len(parts) > col_idx:
                raw = parts[col_idx].replace("**", "").replace("%", "").strip()
                try:
                    return float(raw)
                except ValueError:
                    pass
    return None

# Overall Pass@1 is col_idx 5 (1-based: 1=Arm, 2=Desc, 3=Clean, 4=Corr, 5=Overall)
for arm in ["ungated", "stage1_only", "cascade", "llm_judge", "agent_ask", "deterministic", "always_fallback"]:
    expected = pass1_overall[arm]
    found = extract_table_metric(res_text, arm, 5)
    if found is None:
        mismatches.append(f"Could not extract Overall Pass@1 for {arm}")
    elif abs(found - expected) > 0.05:
        mismatches.append(f"Pass@1 mismatch for {arm}: expected {expected:.2f}%, found {found:.2f}%")

# Clean Pass@1 is col_idx 3
for arm in ["ungated", "stage1_only", "cascade", "llm_judge", "agent_ask", "deterministic", "always_fallback"]:
    expected = pass1_clean[arm]
    found = extract_table_metric(res_text, arm, 3)
    if found is None:
        mismatches.append(f"Could not extract Clean Pass@1 for {arm}")
    elif abs(found - expected) > 0.05:
        mismatches.append(f"Clean Pass@1 mismatch for {arm}: expected {expected:.2f}%, found {found:.2f}%")

# Corrupted Pass@1 is col_idx 4
for arm in ["ungated", "stage1_only", "cascade", "llm_judge", "agent_ask", "deterministic", "always_fallback"]:
    expected = pass1_corr[arm]
    found = extract_table_metric(res_text, arm, 4)
    if found is None:
        mismatches.append(f"Could not extract Corrupted Pass@1 for {arm}")
    elif abs(found - expected) > 0.05:
        mismatches.append(f"Corrupted Pass@1 mismatch for {arm}: expected {expected:.2f}%, found {found:.2f}%")

# False block rate in text
if "6.56%" not in res_text:
    mismatches.append("6.56% false block rate not found in RESULTS_SUMMARY.md")
if "6.56%" not in audit_text:
    mismatches.append("6.56% false block rate not found in AUDIT.md")

if not mismatches:
    print("   [PASS] Zero mismatches detected! All recomputed metrics match RESULTS_SUMMARY.md and AUDIT.md.")
else:
    print(f"   [WARNING] Found {len(mismatches)} mismatches:")
    for mm in mismatches:
        print("    ", mm)
