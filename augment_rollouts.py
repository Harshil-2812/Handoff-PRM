"""
Augments rollouts.csv with new length-preserving corruption types.

Applies Category 2-4 corruptions (invert_objective, negate_edge_case,
wrong_algorithm_name, signature_rename, inject_false_constraint, fake_completion)
to all existing label=1 rows.

Label assignment:
  - signature_rename: always label=0 (test harness looks for original fn name)
  - invert_objective / negate_edge_case / wrong_algorithm_name: label=0 if
    corruption actually changed the handoff text (swap was applicable)
  - inject_false_constraint / fake_completion: label=0 if handoff changed

This produces new rows WITHOUT requiring any Agent B API calls.
Conservative: only adds rows where the corruption demonstrably changed the text.

Usage: python augment_rollouts.py
"""

import os
import re
import pandas as pd
from tasks.all_tasks import ALL_TASKS
from corruption import (
    invert_objective, negate_edge_case, wrong_algorithm_name,
    signature_rename, inject_false_constraint, fake_completion,
)

INPUT_CSV = "rollouts.csv"
OUTPUT_CSV = "rollouts.csv"  # appended in-place

# New semantic/schema corruptions to apply
NEW_CORRUPTION_FNS = [
    "invert_objective",
    "negate_edge_case",
    "wrong_algorithm_name",
    "signature_rename",
    "inject_false_constraint",
    "fake_completion",
]


def build_task_lookup():
    """Returns dict: task_id -> {entry_point, test_code, problem}."""
    lookup = {}
    for task in ALL_TASKS:
        lookup[task["id"]] = {
            "entry_point": task.get("entry_point", ""),
            "test_code":   task.get("test_code", ""),
            "problem":     task.get("problem", ""),
        }
    return lookup


def apply_corruption(handoff: str, corruption_type: str, entry_point: str = None) -> str:
    """Apply a single corruption and return the result (or original if not applicable)."""
    if corruption_type == "invert_objective":
        return invert_objective(handoff)
    elif corruption_type == "negate_edge_case":
        return negate_edge_case(handoff)
    elif corruption_type == "wrong_algorithm_name":
        return wrong_algorithm_name(handoff)
    elif corruption_type == "signature_rename":
        return signature_rename(handoff, entry_point=entry_point)
    elif corruption_type == "inject_false_constraint":
        return inject_false_constraint(handoff)
    elif corruption_type == "fake_completion":
        return fake_completion(handoff)
    return handoff


def augment():
    df = pd.read_csv(INPUT_CSV)
    task_lookup = build_task_lookup()

    # Find existing (task_id, corruption_type) pairs to avoid duplicates
    existing_pairs = set(
        zip(df["task_id"], df["corruption_type"])
    )

    positives = df[df["label"] == 1].copy()
    print(f"Positive rows available for augmentation: {len(positives)}")

    new_rows = []
    skipped_no_task = 0
    skipped_no_change = 0
    skipped_duplicate = 0

    for _, row in positives.iterrows():
        task_id = row["task_id"]
        handoff  = row["handoff"]
        problem  = row["problem"]
        source   = row.get("source", "unknown")

        task_info = task_lookup.get(task_id)
        if task_info is None:
            skipped_no_task += 1
            continue

        entry_point = task_info["entry_point"]

        for corruption_type in NEW_CORRUPTION_FNS:
            # Skip if this pair already exists
            if (task_id, corruption_type) in existing_pairs:
                skipped_duplicate += 1
                continue

            corrupted = apply_corruption(handoff, corruption_type, entry_point=entry_point)

            # Only add the row if the corruption actually changed the text
            if corrupted.strip() == handoff.strip():
                skipped_no_change += 1
                continue

            # All new corruptions are labeled 0:
            # - signature_rename: 100% reliable (NameError in test harness)
            # - others: corruption changed the handoff text → likely causes
            #   Agent B to produce code that diverges from expected behavior
            new_rows.append({
                "task_id":        task_id,
                "problem":        problem,
                "handoff":        corrupted,
                "corruption_type": corruption_type,
                "label":          0,
                "source":         source,
            })
            existing_pairs.add((task_id, corruption_type))

    if not new_rows:
        print("No new rows generated. All corruptions either unchanged or duplicate.")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(OUTPUT_CSV, index=False)

    print(f"\n=== Augmentation complete ===")
    print(f"Original rows:    {len(df)}")
    print(f"New rows added:   {len(new_rows)}")
    print(f"Total rows:       {len(combined)}")
    print(f"Skipped (no task info): {skipped_no_task}")
    print(f"Skipped (no change):    {skipped_no_change}")
    print(f"Skipped (duplicate):    {skipped_duplicate}")
    print(f"\nNew corruption type breakdown:")
    print(new_df["corruption_type"].value_counts())
    print(f"\nFull dataset label distribution:")
    print(combined["label"].value_counts())
    print(f"\nFull dataset corruption type distribution:")
    print(combined["corruption_type"].value_counts())


if __name__ == "__main__":
    augment()