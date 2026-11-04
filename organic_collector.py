"""
Organic Handoff Collection & Majority-Vote Ground Truth Engine (Workstream A: Tasks A1 & A2)
=============================================================================================
Author: Avaneesh N (Software and Experiments)
Repository: https://github.com/Harshil-2812/Handoff-PRM
Protocol: Pre-registered Week 0 Freeze (Seed: 42, 43, 44)

Scientific Mandate:
- Sample Agent A at temperature > 0 across 3 model tiers without synthetic corruptions.
- Re-run Agent B 3 times per handoff (triplicate evaluation) with seeds 42, 43, 44.
- Grade each trial live via grade() exec() against unit test suites.
- Output raw outcomes with majority-vote, discard-policy, and robust-positive labels.
- Resumable: flushes after every task, skips already completed tasks.
"""

import os
import sys
import time
import csv
import threading
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

# Load environment and setup Gemini Client
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in environment or .env")

client = genai.Client(api_key=api_key)

# ── Manifest & Output Paths ──────────────────────────────────────────────────
MANIFEST_ORGANIC = "manifests/manifest_organic_pool.txt"
OUTPUT_ORGANIC_RAW = "results/tables/organic_dataset_raw.csv"
OUTPUT_ORGANIC_DISCARD = "results/tables/organic_dataset_v1_discard.csv"
OUTPUT_ORGANIC_ROBUST = "results/tables/dataset_v2_robust_positive.csv"

os.makedirs("results/tables", exist_ok=True)

# ── 3 Agent A Tiers (Task A1: Stochasticity & Planning Ablation) ─────────────
# Tier 1 (Stochastic / Higher Error Rate): gemini-3.5-flash-lite, T=0.95
# Tier 2 (Balanced / Standard Planner):   gemini-3.5-flash-lite, T=0.70
# Tier 3 (Deterministic / Precise Plan):  gemini-3.5-flash-lite, T=0.20
AGENT_A_TIERS = [
    {"tier": "tier1_weaker",   "model": "gemini-3.5-flash-lite", "temp": 0.95},
    {"tier": "tier2_mid",      "model": "gemini-3.5-flash-lite", "temp": 0.70},
    {"tier": "tier3_stronger", "model": "gemini-3.5-flash-lite", "temp": 0.20},
]

# Agent B Coder Model (Consistent downstream executor)
MODEL_B = "gemini-3.5-flash-lite"
SEEDS_B = [42, 43, 44]  # Triplicate seeds

# ── Rate Limiter ─────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, rpm: int = 15):
        self._interval = 60.0 / rpm
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            sleep_time = self._interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            self._last_call = time.monotonic()

_LIMITERS: Dict[str, RateLimiter] = {}

def get_limiter(model: str) -> RateLimiter:
    if model not in _LIMITERS:
        # gemini-3.5-flash has stricter RPM on free tier than flash-lite
        rpm = 8 if "flash" in model and "lite" not in model else 14
        _LIMITERS[model] = RateLimiter(rpm=rpm)
    return _LIMITERS[model]

# ── Generation with Rate-Limit & Exponential Backoff ─────────────────────────
def generate_llm(prompt: str, model: str, temp: float = 0.7, seed: Optional[int] = None) -> str:
    limiter = get_limiter(model)
    max_retries = 8
    cfg = types.GenerateContentConfig(
        temperature=temp,
        seed=seed,
        max_output_tokens=3000
    )

    for attempt in range(max_retries):
        limiter.wait()
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg
            )
            return (resp.text or "").strip()
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429:
                sleep_sec = 15.0 * (1.35 ** attempt)
                print(f"    [429 Rate Limit] {model}, sleeping {sleep_sec:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
                time.sleep(sleep_sec)
            else:
                raise e
        except genai_errors.ServerError as e:
            sleep_sec = 5.0 * (1.4 ** attempt)
            print(f"    [5xx Server Error] {model}, sleeping {sleep_sec:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(sleep_sec)
        except Exception as e:
            sleep_sec = 5.0 * (1.4 ** attempt)
            print(f"    [Network/API Exception] {model} ({type(e).__name__}), sleeping {sleep_sec:.1f}s (attempt {attempt+1}/{max_retries})", flush=True)
            time.sleep(sleep_sec)

    raise RuntimeError(f"Model {model} failed after {max_retries} retries.")

# ── Prompt Templates ─────────────────────────────────────────────────────────
def agent_a_plan(problem: str, model: str, temp: float) -> str:
    prompt = f"""You are Agent A in a two-agent coding pipeline. Your job is to
read a problem statement and write a HANDOFF MESSAGE for Agent B, who will
write the actual code. Agent B will NOT see the original problem -- only
your handoff message. So your handoff must contain everything Agent B needs.

Problem statement:
{problem}

Write a handoff message that includes:
- A clear restatement of what function needs to be built
- The exact function name/signature to use
- Key edge cases to handle
- A brief plan/approach

Output ONLY the handoff message, nothing else."""
    return generate_llm(prompt, model=model, temp=temp)

def agent_b_code(handoff: str, entry_point: str, seed: int) -> str:
    prompt = f"""You are Agent B in a two-agent coding pipeline. You did NOT
see the original problem -- you only have the handoff message below from
Agent A. Write Python code based ENTIRELY on this handoff.

Handoff message from Agent A:
{handoff}

Write a single Python function named `{entry_point}` that solves this.
Output ONLY a Python code block, nothing else."""
    raw = generate_llm(prompt, model=MODEL_B, temp=0.2, seed=seed)
    return extract_code(raw)

def extract_code(text: str) -> str:
    import re
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()

# ── Isolated Unit Test Grading Harness with Safe Timeout ─────────────────────
def grade(code: str, entry_point: str, test_code: str, timeout: float = 3.0) -> tuple[bool, str]:
    def _exec():
        namespace = {}
        exec(code, namespace)
        exec(test_code, namespace)
        fn = namespace.get(entry_point)
        if fn is None:
            raise ValueError(f"Function '{entry_point}' not found in code")
        passed = bool(namespace["test"](fn))
        return passed, "Success" if passed else "AssertionFailed"

    res = [False, "Timeout"]
    def target():
        try:
            p, msg = _exec()
            res[0] = p
            res[1] = msg
        except Exception as e:
            res[0] = False
            res[1] = f"{type(e).__name__}: {str(e)[:80]}"

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return False, f"ExecutionTimeout (>{timeout}s)"
    return res[0], res[1]

# ── Checkpointing CSV Helpers ────────────────────────────────────────────────
CSV_FIELDS = [
    "task_id",
    "source",
    "entry_point",
    "tier_name",
    "agent_a_model",
    "agent_a_temp",
    "handoff_text",
    "pass_run1",
    "pass_run2",
    "pass_run3",
    "pass_count_of_3",
    "agreement_rate",
    "label_majority",
    "label_discard_policy",
    "label_robust_positive",
    "err_run1",
    "err_run2",
    "err_run3",
]

def load_completed_keys(path: str = OUTPUT_ORGANIC_RAW) -> set:
    if not os.path.exists(path):
        return set()
    completed = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed.add((r["task_id"], r["tier_name"]))
    except Exception:
        pass
    return completed

# ── Main Collection Routine ──────────────────────────────────────────────────
def collect_organic_dataset(max_tasks: Optional[int] = None):
    from tasks.all_tasks import ALL_TASKS
    task_map = {t["id"]: t for t in ALL_TASKS}

    # 1. Load Organic Pool Manifest
    if not os.path.exists(MANIFEST_ORGANIC):
        raise FileNotFoundError(f"Manifest {MANIFEST_ORGANIC} not found. Run partition_manifests.py first.")

    with open(MANIFEST_ORGANIC, "r", encoding="utf-8") as f:
        pool_ids = [line.strip() for line in f if line.strip()]

    if max_tasks:
        pool_ids = pool_ids[:max_tasks]

    completed_keys = load_completed_keys(OUTPUT_ORGANIC_RAW)
    print("=" * 75)
    print("ORGANIC HANDOFF SAMPLING & TRIPLICATE MAJORITY VOTING (Workstream A)")
    print("=" * 75)
    print(f"Target Pool Size:    {len(pool_ids)} tasks")
    print(f"Agent A Tiers:       {len(AGENT_A_TIERS)} ({[t['tier'] for t in AGENT_A_TIERS]})")
    print(f"Agent B Seeds:       {SEEDS_B} (Triplicate)")
    print(f"Completed so far:    {len(completed_keys)} (task, tier) pairs")
    print(f"Destination:         {OUTPUT_ORGANIC_RAW}")
    print("=" * 75)

    is_new = not os.path.exists(OUTPUT_ORGANIC_RAW) or os.path.getsize(OUTPUT_ORGANIC_RAW) == 0
    f_out = open(OUTPUT_ORGANIC_RAW, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if is_new:
        writer.writeheader()
        f_out.flush()

    total_negatives = 0
    total_samples = 0

    try:
        for idx, task_id in enumerate(pool_ids):
            task = task_map.get(task_id)
            if not task:
                continue

            entry_point = task["entry_point"]
            test_code = task["test_code"]
            problem = task["problem"]
            source = task.get("source", "unknown")

            for tier_info in AGENT_A_TIERS:
                tier_name = tier_info["tier"]
                a_model = tier_info["model"]
                a_temp = tier_info["temp"]

                if (task_id, tier_name) in completed_keys:
                    continue

                print(f"\n[{idx+1}/{len(pool_ids)}] Task: {task_id} ({entry_point}) | Tier: {tier_name} ({a_model}, T={a_temp})", flush=True)

                # Step 1: Agent A generates natural organic handoff (T > 0)
                try:
                    handoff = agent_a_plan(problem, model=a_model, temp=a_temp)
                    print(f"  [Agent A] Generated handoff ({len(handoff)} chars)", flush=True)
                except Exception as e:
                    print(f"  [ERROR] Agent A generation failed: {e}", flush=True)
                    continue

                # Step 2: Agent B triplicate execution (Task A2)
                passes = []
                errors = []
                for seed_b in SEEDS_B:
                    try:
                        code_b = agent_b_code(handoff, entry_point, seed=seed_b)
                        p, err = grade(code_b, entry_point, test_code)
                    except Exception as e:
                        p, err = False, f"CallFailed: {type(e).__name__}"
                    passes.append(p)
                    errors.append(err)
                    print(f"    [Agent B] Seed {seed_b}: {'PASS' if p else 'FAIL'} ({err})", flush=True)

                pass_count = sum(1 for p in passes if p)
                # Agreement rate: if 3/3 or 0/3 -> 1.0 (100% agreement); if 2/3 or 1/3 -> 0.67
                agreement = 1.0 if pass_count in (0, 3) else 0.67

                # Step 3: Dual Labeling Policies (Task A3)
                label_majority = 1 if pass_count >= 2 else 0
                label_robust_positive = 1 if pass_count == 3 else 0
                label_discard_policy = 1 if pass_count == 3 else (0 if pass_count == 0 else "discard")

                row = {
                    "task_id": task_id,
                    "source": source,
                    "entry_point": entry_point,
                    "tier_name": tier_name,
                    "agent_a_model": a_model,
                    "agent_a_temp": a_temp,
                    "handoff_text": handoff,
                    "pass_run1": passes[0],
                    "pass_run2": passes[1],
                    "pass_run3": passes[2],
                    "pass_count_of_3": pass_count,
                    "agreement_rate": agreement,
                    "label_majority": label_majority,
                    "label_discard_policy": label_discard_policy,
                    "label_robust_positive": label_robust_positive,
                    "err_run1": errors[0],
                    "err_run2": errors[1],
                    "err_run3": errors[2],
                }

                writer.writerow(row)
                f_out.flush()
                completed_keys.add((task_id, tier_name))
                total_samples += 1
                if label_majority == 0:
                    total_negatives += 1

                print(f"  Summary: Passes {pass_count}/3 | Majority: {label_majority} | Discard: {label_discard_policy} | Robust: {label_robust_positive}", flush=True)

    finally:
        f_out.close()

    print("\n" + "=" * 75)
    print("ORGANIC SAMPLING SESSION COMPLETED")
    print(f"Total Completed Rows in CSV: {len(completed_keys)}")
    print(f"Saved to: {OUTPUT_ORGANIC_RAW}")
    print("=" * 75)


def print_summary():
    if not os.path.exists(OUTPUT_ORGANIC_RAW):
        print(f"No organic raw data found at {OUTPUT_ORGANIC_RAW}")
        return
    import pandas as pd
    df = pd.read_csv(OUTPUT_ORGANIC_RAW)
    print("\n" + "=" * 75)
    print("ORGANIC DATASET CURRENT AUDIT SUMMARY")
    print("=" * 75)
    print(f"Total Rows Collected:      {len(df)}")
    print(f"Unique Tasks Evaluated:    {df['task_id'].nunique()}")
    print("\nBreakdown by Tier:")
    print(df["tier_name"].value_counts())
    print("\nPass Count Distribution (out of 3 seeds):")
    print(df["pass_count_of_3"].value_counts().sort_index())
    print("\nMajority-Vote Labels (0=Fail/Insufficient, 1=Pass/Sufficient):")
    print(df["label_majority"].value_counts())
    print("\nDiscard Policy Distribution (1=3/3 pass, 0=0/3 pass, discard=1/3 or 2/3):")
    print(df["label_discard_policy"].value_counts())
    print("\nRobust Positive Distribution (1=3/3 pass, 0=any failure):")
    print(df["label_robust_positive"].value_counts())
    print("=" * 75)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Organic Handoff Collection Engine")
    parser.add_argument("--dry-run", action="store_true", help="Run on first 2 tasks only")
    parser.add_argument("--max-tasks", type=int, default=None, help="Max number of tasks to run")
    parser.add_argument("--summary", action="store_true", help="Print audit summary of collected dataset")
    args = parser.parse_args()

    if args.summary:
        print_summary()
    elif args.dry_run:
        print("[DRY RUN MODE] Running on first 2 tasks only...")
        collect_organic_dataset(max_tasks=2)
        print_summary()
    else:
        collect_organic_dataset(max_tasks=args.max_tasks)
        print_summary()
