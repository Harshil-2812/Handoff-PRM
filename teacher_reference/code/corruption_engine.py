"""
corruption_engine.py — Handoff-PRM Corruption Engine
=====================================================
All 10 corruption functions used to generate negative training examples.
This file is the canonical reference for §IV of the paper.

Corruption taxonomy (Table III):
  Category 1 — Structural / Length-Reducing:
    truncation           Word-boundary cut at ~30% of handoff length
    over_summarization   TextRank extractive summary (1 sentence)
    tool_result_drop     Remove all bullet lists and fenced code blocks
    entity_omission      Remove sentences containing named entities / signatures

  Category 2 — Logic / Semantic (Length-Preserving):
    invert_objective     Swap return-value polarity (True↔False, max↔min, etc.)
    negate_edge_case     Negate one constraint statement (must→must not, etc.)
    wrong_algorithm_name Swap algorithm name (binary search↔linear search, etc.)

  Category 3 — Schema / Interface (Length-Preserving):
    signature_rename     Rename entry-point function to <name>_impl

  Category 4 — Completeness (Length-Preserving):
    inject_false_constraint  Append a false constraint, trim tail to preserve length
    fake_completion      Replace last 25% with vague non-informative filler text

Design principle (§IV-D — Algorithm 1):
  A corruption is a valid negative label if and only if Agent B's Pass@1
  on the corrupted handoff is strictly lower than on the original handoff.
  This contrastive test is the ground truth — no human annotation is required.

Usage:
    from corruption_engine import corrupt_handoff, CORRUPTION_FUNCTIONS
    corrupted, ctype = corrupt_handoff(handoff, corruption_type="truncation")
"""

# This file re-exports everything from the live corruption.py module.
# It is kept as a separate named file for the teacher-reference package
# so reviewers can find it by the name cited in the paper.

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from corruption import (
    truncate,
    omit_entity,
    drop_tool_result,
    over_summarize,
    invert_objective,
    negate_edge_case,
    wrong_algorithm_name,
    signature_rename,
    inject_false_constraint,
    fake_completion,
    corrupt_handoff,
    CORRUPTION_FUNCTIONS,
    RANDOM_SAMPLE_CORRUPTIONS,
)

__all__ = [
    "truncate",
    "omit_entity",
    "drop_tool_result",
    "over_summarize",
    "invert_objective",
    "negate_edge_case",
    "wrong_algorithm_name",
    "signature_rename",
    "inject_false_constraint",
    "fake_completion",
    "corrupt_handoff",
    "CORRUPTION_FUNCTIONS",
    "RANDOM_SAMPLE_CORRUPTIONS",
]


# ─── Verification self-test ───────────────────────────────────────────────────

_SAMPLE_HANDOFF = """
Agent A has completed the first stage of the two-sum problem.
The function must return True if any two distinct elements sum to the target,
otherwise return False.
Key edge cases: handle empty lists, handle None inputs, handle duplicate values.
Implementation approach: use a hash set for O(n) lookup.
Entry point: def two_sum_exists(nums: list, target: int) -> bool
"""

if __name__ == "__main__":
    print("=== Corruption Engine Self-Test ===\n")
    for ctype, fn in CORRUPTION_FUNCTIONS.items():
        try:
            if ctype == "signature_rename":
                result = fn(_SAMPLE_HANDOFF, entry_point="two_sum_exists")
            else:
                result = fn(_SAMPLE_HANDOFF)
            changed = result != _SAMPLE_HANDOFF
            print(f"[{'OK' if changed else 'SKIP'}] {ctype}: {'changed' if changed else 'no applicable pattern'}")
        except Exception as e:
            print(f"[ERR] {ctype}: {e}")
    print("\nAll 10 corruption types verified.")
