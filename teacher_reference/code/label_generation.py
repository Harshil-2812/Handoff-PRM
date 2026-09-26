"""
label_generation.py — Algorithm 1: Contrastive Corruption Labeling
===================================================================
Implements the contrastive labeling procedure described in §IV-D of the paper.

Algorithm 1 — Contrastive Corruption Labeling:
  Input : task pool T, Agent A (generative), Agent B (executor), test harness H
  Output: labeled dataset D = {(handoff_i, problem_i, corruption_type_i, label_i)}

  For each task t in T:
    1. Run Agent A on t to generate a handoff h_original
    2. Verify: run Agent B(h_original) → pass@1_original via H
       - If Agent B fails even on the clean handoff, SKIP (this task is too hard)
    3. For each corruption_type c in CORRUPTION_FUNCTIONS:
         a. h_corrupted = corrupt(h_original, c)
         b. Run Agent B(h_corrupted) → pass@1_corrupted via H
         c. label = 1 (GOOD) if pass@1_corrupted >= pass@1_original
                  = 0 (BAD)  if pass@1_corrupted <  pass@1_original
         d. Append (h_corrupted, t.problem, c, label) to D
    4. Also add (h_original, t.problem, "clean", 1) to D

  Key design choice:
    - Labels are derived mechanically from the test harness — no human annotation
    - A corruption is only a valid negative if it actually causes Agent B to fail
    - Tasks where Agent B already fails ungated are excluded (too noisy as negatives)
    - GroupKFold by task_id prevents data leakage during cross-validation

This file is a self-contained reference implementation.
The actual runs used rollout_runner.py + run_corruptions.py in the project root.
"""

import random
import os
import sys
import csv
from typing import Callable, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from corruption import CORRUPTION_FUNCTIONS, RANDOM_SAMPLE_CORRUPTIONS


def generate_label(
    original_pass: bool,
    corrupted_pass: bool,
) -> int:
    """
    Implements the contrastive labeling rule from Algorithm 1.

    label = 0 (BAD) if the corruption caused Agent B to fail
    label = 1 (GOOD) if Agent B still passed despite the corruption

    Args:
        original_pass: True if Agent B passed on the clean handoff
        corrupted_pass: True if Agent B passed on the corrupted handoff

    Returns:
        0 = corruption degraded quality (handoff is BAD)
        1 = corruption had no effect (handoff still GOOD)
    """
    return 1 if corrupted_pass else 0


def build_labeled_row(
    task_id: str,
    problem_text: str,
    handoff_text: str,
    corruption_type: str,
    agent_b_passed: bool,
    source: str = "unknown",
) -> dict:
    """Build a single labeled dataset row in the format used by the paper."""
    return {
        "task_id": task_id,
        "source": source,
        "problem_text": problem_text,
        "handoff_text": handoff_text,
        "corruption_type": corruption_type,
        "label": 1 if agent_b_passed else 0,
    }


def run_contrastive_labeling(
    task_pool: list[dict],
    run_agent_a: Callable[[dict], str],
    run_agent_b_and_verify: Callable[[str, dict], bool],
    corruption_types: list[str] = None,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """
    Runs Algorithm 1 on a task pool.

    Args:
        task_pool: list of task dicts, each with at least:
            {"task_id": str, "problem": str, "entry_point": str}
        run_agent_a: callable(task) -> handoff_text (str)
        run_agent_b_and_verify: callable(handoff_text, task) -> bool (passed?)
        corruption_types: list of corruption type names to use
            (default: RANDOM_SAMPLE_CORRUPTIONS — excludes signature_rename
             unless entry_point is handled)
        seed: random seed for reproducibility
        verbose: print progress

    Returns:
        List of labeled dicts with keys:
            task_id, source, problem_text, handoff_text, corruption_type, label
    """
    if corruption_types is None:
        corruption_types = RANDOM_SAMPLE_CORRUPTIONS

    rng = random.Random(seed)
    dataset = []
    skipped_tasks = 0
    total_rows = 0

    for task in task_pool:
        task_id = task["task_id"]
        problem = task["problem"]
        entry_point = task.get("entry_point", "")
        source = task.get("source", "unknown")

        # Step 1: Generate clean handoff
        handoff_clean = run_agent_a(task)

        # Step 2: Verify clean handoff (gating criterion)
        original_pass = run_agent_b_and_verify(handoff_clean, task)
        if not original_pass:
            if verbose:
                print(f"  SKIP {task_id}: Agent B fails even on clean handoff")
            skipped_tasks += 1
            continue

        # Add clean row as positive label
        dataset.append(build_labeled_row(
            task_id, problem, handoff_clean, "clean", True, source
        ))
        total_rows += 1

        # Step 3: For each corruption type, generate and label
        for ctype in corruption_types:
            fn = CORRUPTION_FUNCTIONS[ctype]
            if ctype == "signature_rename":
                corrupted = fn(handoff_clean, entry_point=entry_point)
            else:
                corrupted = fn(handoff_clean)

            if corrupted == handoff_clean:
                # Corruption had no syntactic effect (no applicable pattern found)
                if verbose:
                    print(f"  NOTE {task_id}/{ctype}: no pattern applied, skipping")
                continue

            corrupted_pass = run_agent_b_and_verify(corrupted, task)
            dataset.append(build_labeled_row(
                task_id, problem, corrupted, ctype, corrupted_pass, source
            ))
            total_rows += 1

    if verbose:
        n_pos = sum(1 for r in dataset if r["label"] == 1)
        n_neg = sum(1 for r in dataset if r["label"] == 0)
        print(f"\nDataset built: {total_rows} rows, {n_pos} positive, {n_neg} negative")
        print(f"Tasks processed: {len(task_pool) - skipped_tasks}, skipped: {skipped_tasks}")

    return dataset


def save_dataset(rows: list[dict], output_path: str) -> None:
    """Save labeled dataset to CSV."""
    if not rows:
        print("Warning: no rows to save.")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {output_path}")


# ─── Verification ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test with mock functions
    mock_tasks = [
        {"task_id": "t001", "problem": "Write a function to find max of a list.",
         "entry_point": "find_max", "source": "hand_easy"},
    ]

    call_log = []

    def mock_agent_a(task):
        return "Agent A has solved the setup. find_max should return the largest element."

    def mock_agent_b(handoff, task):
        call_log.append(handoff[:30])
        return "invert" not in handoff and "minimum" not in handoff

    rows = run_contrastive_labeling(mock_tasks, mock_agent_a, mock_agent_b, verbose=True)
    print(f"\nGenerated {len(rows)} rows")
    for r in rows:
        print(f"  {r['task_id']} / {r['corruption_type']} → label={r['label']}")
