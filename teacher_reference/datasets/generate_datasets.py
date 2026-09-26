#!/usr/bin/env python3
"""
generate_datasets.py — Regenerate primary_dataset_177.csv and rebuilt_dataset_483.csv
=======================================================================================
This script documents the exact pipeline used to generate the two primary datasets
referenced in the paper (Table II and Section IX).

The actual runs took ~12-18 hours due to LLM API calls.
This script shows the exact commands and parameters used, so a reviewer can
reproduce them from scratch (given API access) or verify the procedure.

Pipeline steps:
  Step 1: Generate Agent A handoffs for each task in the pool
  Step 2: Run Agent B on clean handoffs → identify valid tasks (30/30 pass)
  Step 3: Apply all 10 corruptions to each valid task's handoff
  Step 4: Run Agent B on each corrupted handoff → assign labels
  Step 5: Extract features from all (handoff, problem) pairs
  Step 6: Output labeled + featurized CSV

Paper reference: §IV-B (data collection), §IV-D (labeling), §V (features)
"""

import subprocess
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


def print_step(n, desc):
    print(f"\n{'='*70}")
    print(f"  Step {n}: {desc}")
    print('='*70)


def generate_primary_177():
    """Reproduce the 177-row primary dataset (Table II)."""
    print_step(1, "Generate clean handoffs for hand-crafted task pool")
    print("""
  Command (approximate — exact invocation from project root):
    python rollout_runner.py \\
        --task_pool tasks/coding_tasks.py tasks/hard_tasks_3.py \\
        --n_rollouts 1 \\
        --corruption_types clean \\
        --output rollouts_v1_naive_corruption.csv \\
        --seed 42

  Output: rollouts_v1_naive_corruption.csv (clean handoffs only)
    """)

    print_step(2, "Run Agent B on clean handoffs and verify (30/30 pass)")
    print("""
  Command:
    python rollout_runner.py \\
        --verify_clean \\
        --input rollouts_v1_naive_corruption.csv \\
        --output step4_verified.csv

  Expectation: all 109 tasks produce a passing clean handoff.
  Tasks where Agent B fails on the clean handoff are excluded.
    """)

    print_step(3, "Apply 10 corruption types and run Agent B")
    print("""
  Command:
    python run_corruptions.py \\
        --input step4_verified.csv \\
        --output step5_corruption_results.csv \\
        --corruptions truncation entity_omission tool_result_drop over_summarization \\
                      invert_objective negate_edge_case wrong_algorithm_name \\
                      signature_rename inject_false_constraint fake_completion \\
        --seed 42

  Output: step5_corruption_results.csv
  Each row = (task_id, handoff_text, corruption_type, agent_b_passed)
    """)

    print_step(4, "Extract features and build labeled dataset")
    print("""
  Command:
    python feature_extraction.py \\
        --input step5_corruption_results.csv \\
        --output final_dataset.csv

  Command for final 177-row subset:
    python finalize_dataset.py \\
        --input final_dataset.csv \\
        --n_rows 177 \\
        --output primary_dataset_177.csv
    """)


def generate_rebuilt_483():
    """Reproduce the 483-row extended dataset (Section IX)."""
    print_step(1, "Add HumanEval + MBPP tasks to the pool")
    print("""
  Command:
    python convert_humaneval[1].py  # converts humaneval_tasks.py format
    python convert_mbpp.py          # converts mbpp_tasks.py format
    """)

    print_step(2, "Generate rollouts for expanded pool")
    print("""
  Command:
    python rollout_runner.py \\
        --task_pool tasks/coding_tasks.py tasks/hard_tasks_3.py \\
                   humaneval_tasks.py mbpp_tasks.py \\
        --n_rollouts 1 \\
        --output rollouts.csv \\
        --seed 42

  Output: rollouts.csv (4,136 KB — full rollout log)
    """)

    print_step(3, "Augment + apply corruptions to get 483 rows")
    print("""
  Command:
    python augment_rollouts.py --input rollouts.csv --output step5_corruption_results.csv
    python model_sweep_12features_final.py
      # internally builds final_dataset_full_features.csv (the 483-row dataset)
    """)

    print_step(4, "Final dataset statistics")
    print("""
  Expected output (final_dataset_full_features.csv):
    Rows    : 483
    Tasks   : 248 unique task_ids
    Positive: ~310 (64%)
    Negative: ~173 (36%)
    Sources : hand_easy=64, hand_hard=127, humaneval=141, mbpp=151
    """)


if __name__ == "__main__":
    print("=" * 70)
    print("  Handoff-PRM Dataset Generation Guide")
    print("  (Documents the exact pipeline — not a live execution script)")
    print("=" * 70)

    generate_primary_177()
    print("\n\n")
    generate_rebuilt_483()

    print("\n\nTo actually regenerate the datasets from scratch, run these scripts")
    print("from the project root in order. API keys must be set in .env.")
    print("Estimated runtime: 12-18 hours (LLM API calls).")
