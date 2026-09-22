"""
always_fallback_runner_v2.py
Robust version: 1 worker, incremental save per task, longer backoff.
Runs Agent B on raw problem for all 47 PASS rows in live_pilot_results.csv.
"""
import os, sys, time, threading
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import google.genai as genai
from google.genai import types, errors as genai_errors

# ── API setup (same keys as agents.py) ──────────────────────────────────────
import ast
raw = os.getenv("GEMINI_KEYS","").strip()
KEYS = ast.literal_eval(raw) if raw else [os.getenv("GEMINI_API_KEY","")]
MODEL = "gemini-3.5-flash-lite"
GEN_CONFIG = types.GenerateContentConfig(max_output_tokens=4000)
print(f"Loaded {len(KEYS)} API keys")

# Round-robin key index (thread-safe)
_key_idx = 0
_key_lock = threading.Lock()

def next_client():
    global _key_idx
    with _key_lock:
        c = genai.Client(api_key=KEYS[_key_idx % len(KEYS)])
        _key_idx += 1
    return c

def call_model(prompt, max_attempts=8):
    """Try up to max_attempts, rotating keys and sleeping on any error."""
    for attempt in range(max_attempts):
        client = next_client()
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt, config=GEN_CONFIG)
            return (resp.text or "").strip()
        except Exception as exc:
            wait = min(2 ** attempt, 60)
            print(f"    [attempt {attempt+1}] {type(exc).__name__}: sleeping {wait}s ...")
            time.sleep(wait + 2)
    raise RuntimeError(f"Model failed after {max_attempts} attempts")

# ── Task pool ────────────────────────────────────────────────────────────────
from humaneval_tasks import HUMANEVAL_TASKS
from mbpp_tasks import MBPP_TASKS
from tasks.coding_tasks import TASKS as hand_tasks
from tasks.hard_tasks_3 import HARD_TASKS_3
from rollout_runner import grade

import re
def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

ALL_TASKS = {t["id"]: t for t in HUMANEVAL_TASKS + MBPP_TASKS + hand_tasks + HARD_TASKS_3}

PILOT_CSV  = "live_pilot_results.csv"
OUTPUT_CSV = "always_fallback_results.csv"

# ── Load what we already have ────────────────────────────────────────────────
if os.path.exists(OUTPUT_CSV):
    done_df = pd.read_csv(OUTPUT_CSV)
    done = {r["task_id"]: r.to_dict() for _, r in done_df.iterrows()
            if not str(r.get("afb_error","")).strip()}
else:
    done = {}

pilot = pd.read_csv(PILOT_CSV)
pass_rows = pilot[pilot["gate_decision"] == "PASS"]
to_run = [(i, row) for i, (_, row) in enumerate(pass_rows.iterrows())
          if row["task_id"] not in done]

print(f"Already done: {len(done)}  |  To run: {len(to_run)}  |  Total: {len(pass_rows)}")

# ── Sequential runner with incremental save ──────────────────────────────────
def run_one(task_id, ep, problem, test_code):
    print(f"  [{task_id}] calling Agent B on raw problem ...")
    prompt = f"""You are Agent B in a two-agent coding pipeline. You did NOT
see the original problem -- you only have the handoff message below from
Agent A. Write Python code based ENTIRELY on this handoff.

Handoff message from Agent A:
{problem}

Write a single Python function named `{ep}` that solves this.
Output ONLY a Python code block, nothing else."""
    try:
        text = call_model(prompt)
        code = extract_code(text)
        passed = grade(code, ep, test_code)
        print(f"    -> {'PASS' if passed else 'FAIL'}")
        return {"task_id": task_id, "afb_passed": passed, "afb_error": ""}
    except Exception as exc:
        print(f"    -> ERROR: {exc}")
        return {"task_id": task_id, "afb_passed": False, "afb_error": str(exc)[:120]}

def save_all():
    rows = list(done.values())
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

for i, (idx, row) in enumerate(to_run):
    tid = row["task_id"]
    task = ALL_TASKS.get(tid)
    if task is None:
        print(f"  [{tid}] NOT IN TASK POOL - skipping")
        done[tid] = {"task_id": tid, "afb_passed": False, "afb_error": "not_found"}
        save_all()
        continue
    
    print(f"\n[{i+1}/{len(to_run)}] {tid} ({task['entry_point']})")
    result = run_one(tid, task["entry_point"], task["problem"], task["test_code"])
    done[tid] = result
    save_all()
    print(f"  Saved. ({len(done)}/47 done)")
    # Small cooldown between tasks to avoid server error bursts
    if i < len(to_run) - 1:
        time.sleep(3)

# ── Final report ─────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  ALWAYS-FALLBACK vs GATED vs UNGATED -- FINAL RESULTS")
print("="*72)

afb_df = pd.read_csv(OUTPUT_CSV)
afb_map = dict(zip(afb_df["task_id"], afb_df["afb_passed"].astype(bool)))

def fix_bool(series):
    return series.astype(str).str.lower().map(
        lambda x: True if x in ("true","1","yes") else False)

pilot = pd.read_csv(PILOT_CSV)
pilot["is_corrupted"]   = fix_bool(pilot["is_corrupted"])
pilot["ungated_passed"] = fix_bool(pilot["ungated_passed"])
pilot["gated_passed"]   = fix_bool(pilot["gated_passed"])

def get_afb(row):
    if row["gate_decision"] == "BLOCK":
        return bool(row["gated_passed"])
    return afb_map.get(row["task_id"], False)

pilot["afb_passed"] = pilot.apply(get_afb, axis=1)
pilot.to_csv("live_pilot_results_with_afb.csv", index=False)

cln = pilot[~pilot["is_corrupted"]]
crp = pilot[pilot["is_corrupted"]]

def pp(s): return f"{s.mean()*100:.1f}% ({int(s.sum())}/{len(s)})"

W = 28
print(f"\n  {'Condition':<{W}} {'Ungated':>18} {'Always-FB':>18} {'Gated':>18}")
print(f"  {'-'*(W+56)}")
for label, sub in [("All (n=94)", pilot), ("Clean (n=47)", cln), ("Corrupted (n=47)", crp)]:
    print(f"  {label:<{W}} {pp(sub['ungated_passed']):>18} {pp(sub['afb_passed']):>18} {pp(sub['gated_passed']):>18}")

import numpy as np
from scipy.stats import binom as sp_binom

b = int(((pilot["gated_passed"]) & (~pilot["afb_passed"])).sum())
c = int(((~pilot["gated_passed"]) & (pilot["afb_passed"])).sum())
tot = b + c
if tot > 0:
    p_mc = float(2*min(sum(sp_binom.pmf(k,tot,0.5) for k in range(min(b,c)+1)), 1.0))
else:
    p_mc = 1.0
rng = np.random.default_rng(42)
delta = (pilot["gated_passed"].astype(int) - pilot["afb_passed"].astype(int)).to_numpy()
draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
ci_lo, ci_hi = float(np.percentile(draws,2.5))*100, float(np.percentile(draws,97.5))*100

print(f"\n  McNemar (gated vs always-FB): b={b}  c={c}  p={p_mc:.4f}")
print(f"  Bootstrap 95% CI (gated-AFB): [{ci_lo:+.1f}, {ci_hi:+.1f}] pp")

# PASS-only breakdown
clean_pass = cln[cln["gate_decision"]=="PASS"]
corr_pass  = crp[crp["gate_decision"]=="PASS"]
print(f"\n  PASS-only breakdown (new data):")
print(f"    Clean PASS (n={len(clean_pass)}): ungated={pp(clean_pass['ungated_passed'])}  AFB={pp(clean_pass['afb_passed'])}")
print(f"    Corr  PASS (n={len(corr_pass)}):  ungated={pp(corr_pass['ungated_passed'])}   AFB={pp(corr_pass['afb_passed'])}")
print(f"\n  Saved -> always_fallback_results.csv")
print(f"  Merged -> live_pilot_results_with_afb.csv")
print("="*72)
