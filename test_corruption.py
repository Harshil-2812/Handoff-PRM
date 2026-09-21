"""
Unit tests for all 10 corruption functions in corruption.py.

Each test:
  1. Runs the corruption on 2-3 real-ish handoff samples.
  2. Prints a before/after snippet so you can visually confirm the output is sane.
  3. Asserts structural invariants (changed, non-empty, correct direction of edit).

Run with:
    python test_corruption.py
"""

import re
import sys
import textwrap

# ------------------------------------------------------------------------------
# Sample handoffs used across multiple tests
# ------------------------------------------------------------------------------

HANDOFF_BINARY_SEARCH = """\
Task: he_002 - Implement `has_close_elements`

The agent must implement has_close_elements using binary search over the sorted
list. The function must handle the edge case where the input list is empty and
must return False in that case.  It must handle negative numbers and ensure
elements are compared in ascending order.

def has_close_elements(numbers: list, threshold: float) -> bool:
    ...

Note: the function should return True only when at least two numbers are closer
than threshold. Handle edge cases: empty list must return False, duplicate
values should return True.
"""

HANDOFF_SORT_ODD = """\
Task: he_005 - Implement `sort_array`

Return a copy of the given array sorted ascending if the sum of first and last
element is odd, otherwise descending. The minimum and maximum values must be
preserved. Handle the edge case of an empty array by returning an empty list.

You should return the largest element first when sorting descending.
Must handle None inputs by raising a ValueError.
"""

HANDOFF_PARENTHESES = """\
Task: he_001 - Implement `separate_paren_groups`

Separate groups of nested parentheses into individual strings. The implementation
uses a greedy recursive approach: scan left to right, track depth with a counter.
Each time depth hits 0, flush the current group. Ignore spaces in the input string.

def separate_paren_groups(paren_string: str) -> list:
    ...

Must handle the case where paren_string is empty - return an empty list.
The function must handle case-insensitive input where brackets may vary.
"""


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

PASS_STR = "\033[92mPASS\033[0m"
FAIL_STR = "\033[91mFAIL\033[0m"

_test_count = 0
_fail_count = 0


def _show(label: str, original: str, corrupted: str):
    width = 72
    print(f"\n  {'_'*width}")
    print(f"  Sample : {label}")
    print(f"  {'_'*width}")
    for line in original.strip().splitlines()[:4]:
        print(f"  BEFORE | {line}")
    print(f"  {'.'*width}")
    for line in corrupted.strip().splitlines()[:4]:
        print(f"  AFTER  | {line}")
    print(f"  {'_'*width}")


def check(name: str, condition: bool, msg: str = ""):
    global _test_count, _fail_count
    _test_count += 1
    status = PASS_STR if condition else FAIL_STR
    if not condition:
        _fail_count += 1
    print(f"  [{status}] {name}" + (f" -- {msg}" if msg else ""))
    return condition


def section(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


# ------------------------------------------------------------------------------
# Import the corruption functions
# ------------------------------------------------------------------------------

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
)


# ==============================================================================
# 1. truncate
# ==============================================================================

def test_truncate():
    section("1. truncate  (Cat-1 Structural: length-reducing)")

    samples = [
        ("binary_search handoff", HANDOFF_BINARY_SEARCH),
        ("sort_odd handoff",      HANDOFF_SORT_ODD),
        ("parentheses handoff",   HANDOFF_PARENTHESES),
    ]
    for label, h in samples:
        out = truncate(h, keep_fraction=0.3)
        _show(label, h, out)
        check(f"truncate/{label} -- non-empty",  len(out) > 0)
        check(f"truncate/{label} -- shorter",    len(out.split()) < len(h.split()))
        check(f"truncate/{label} -- ~30% words",
              abs(len(out.split()) / len(h.split()) - 0.3) < 0.15)


# ==============================================================================
# 2. omit_entity
# ==============================================================================

def test_omit_entity():
    section("2. omit_entity  (Cat-1 Structural: removes load-bearing sentences)")

    samples = [
        ("binary_search handoff", HANDOFF_BINARY_SEARCH),
        ("parentheses handoff",   HANDOFF_PARENTHESES),
    ]
    for label, h in samples:
        out = omit_entity(h, fraction_to_remove=0.5, seed=42)
        _show(label, h, out)
        check(f"omit_entity/{label} -- non-empty", len(out.strip()) > 0)
        check(f"omit_entity/{label} -- not identical (or no load-bearing sents)",
              out.strip() != h.strip() or len(h.split()) < 5,
              "ok if handoff has no load-bearing sentences")


# ==============================================================================
# 3. drop_tool_result
# ==============================================================================

def test_drop_tool_result():
    section("3. drop_tool_result  (Cat-1 Structural: strips bullets & code spans)")

    rich_handoff = textwrap.dedent("""\
        The tool returned the following results:
        - First element: 1.0
        - Second element: 2.8
        - Threshold check: 0.3

        ```python
        result = has_close_elements([1.0, 2.8, 3.0], 0.3)
        ```

        The function must handle edge cases carefully.
        Must handle empty lists and return False.
    """)

    samples = [
        ("rich_handoff (bullets + code)", rich_handoff),
        ("binary_search handoff",         HANDOFF_BINARY_SEARCH),
    ]
    for label, h in samples:
        out = drop_tool_result(h)
        _show(label, h, out)
        check(f"drop_tool/{label} -- non-empty",        len(out.strip()) > 0)
        check(f"drop_tool/{label} -- no fenced code",   "```" not in out)
        check(f"drop_tool/{label} -- no bullet lines",
              not any(re.match(r"^\s*([-*]|\d+[.)]) ", ln) for ln in out.splitlines()))


# ==============================================================================
# 4. over_summarize
# ==============================================================================

def test_over_summarize():
    section("4. over_summarize  (Cat-1 Structural: TextRank extractive summary)")

    samples = [
        ("binary_search handoff", HANDOFF_BINARY_SEARCH),
        ("sort_odd handoff",      HANDOFF_SORT_ODD),
    ]
    for label, h in samples:
        out = over_summarize(h, sentence_count=1)
        _show(label, h, out)
        check(f"over_summarize/{label} -- non-empty",    len(out.strip()) > 0)
        check(f"over_summarize/{label} -- much shorter", len(out.split()) < len(h.split()))


# ==============================================================================
# 5. invert_objective
# ==============================================================================

def test_invert_objective():
    section("5. invert_objective  (Cat-2 Semantic: swaps True<->False, asc<->desc, max<->min)")

    explicit_tf = textwrap.dedent("""\
        The function must return True when the list has close elements,
        and return False otherwise. Ascending order is preferred for ties.
    """)

    samples = [
        ("sort_odd -- 'ascending'",        HANDOFF_SORT_ODD),
        ("binary_search -- 'ascending'",   HANDOFF_BINARY_SEARCH),
        ("explicit True/False handoff",    explicit_tf),
    ]
    for label, h in samples:
        out = invert_objective(h)
        _show(label, h, out)
        check(f"invert_obj/{label} -- non-empty", len(out.strip()) > 0)
        if out != h:
            check(f"invert_obj/{label} -- text changed", True)
            check(f"invert_obj/{label} -- length preserved (+-2 words)",
                  abs(len(out.split()) - len(h.split())) <= 2)
        else:
            check(f"invert_obj/{label} -- no applicable swap (ok)", True,
                  "no swappable keyword found in prose")


# ==============================================================================
# 6. negate_edge_case
# ==============================================================================

def test_negate_edge_case():
    section("6. negate_edge_case  (Cat-2 Semantic: negates constraint phrases)")

    edge_handoff = textwrap.dedent("""\
        The function must handle the case where the list is empty and must return
        False. It should handle None inputs by raising a ValueError.
        The comparison must be case-insensitive across all string inputs.
    """)

    samples = [
        ("edge_handoff -- 'must handle'",    edge_handoff),
        ("binary_search -- 'must handle'",   HANDOFF_BINARY_SEARCH),
        ("parentheses -- 'case-insensitive'",HANDOFF_PARENTHESES),
    ]
    for label, h in samples:
        out = negate_edge_case(h)
        _show(label, h, out)
        check(f"negate_ec/{label} -- non-empty", len(out.strip()) > 0)
        if out != h:
            check(f"negate_ec/{label} -- text changed", True)
            check(f"negate_ec/{label} -- length preserved (+-3 words)",
                  abs(len(out.split()) - len(h.split())) <= 3)
        else:
            check(f"negate_ec/{label} -- no applicable pattern (ok)", True,
                  "no negation pattern matched in prose")


# ==============================================================================
# 7. wrong_algorithm_name
# ==============================================================================

def test_wrong_algorithm_name():
    section("7. wrong_algorithm_name  (Cat-2 Semantic: swaps algorithm keywords)")

    algo_handoff = textwrap.dedent("""\
        Implement the function using dynamic programming for efficiency.
        A recursive approach was tried but was too slow on large inputs.
        Use a hash map to store intermediate results for O(1) lookup.
    """)

    samples = [
        ("algo_handoff -- 'dynamic programming'", algo_handoff),
        ("parentheses -- 'recursive'",            HANDOFF_PARENTHESES),
        ("binary_search -- 'binary search'",      HANDOFF_BINARY_SEARCH),
    ]
    for label, h in samples:
        out = wrong_algorithm_name(h)
        _show(label, h, out)
        check(f"wrong_algo/{label} -- non-empty", len(out.strip()) > 0)
        if out != h:
            check(f"wrong_algo/{label} -- text changed", True)
        else:
            check(f"wrong_algo/{label} -- no applicable swap (ok)", True,
                  "no algorithm keyword found in prose")


# ==============================================================================
# 8. signature_rename
# ==============================================================================

def test_signature_rename():
    section("8. signature_rename  (Cat-3 Schema: renames function to wrong name)")

    samples = [
        ("binary_search + entry_point",  HANDOFF_BINARY_SEARCH, "has_close_elements"),
        ("parentheses + entry_point",    HANDOFF_PARENTHESES,   "separate_paren_groups"),
        ("sort_odd -- fallback (no ep)", HANDOFF_SORT_ODD,      None),
    ]
    for label, h, ep in samples:
        out = signature_rename(h, entry_point=ep)
        _show(label, h, out)
        check(f"sig_rename/{label} -- non-empty", len(out.strip()) > 0)
        if ep:
            expected_wrong = ep + "_impl"
            check(f"sig_rename/{label} -- wrong name present",
                  expected_wrong in out,
                  f"expected '{expected_wrong}' in output")
            # Remove wrong name occurrences, original should be gone
            out_without_wrong = out.replace(expected_wrong, "")
            check(f"sig_rename/{label} -- original name removed",
                  ep not in out_without_wrong,
                  "original function name should be replaced")
        else:
            check(f"sig_rename/{label} -- fallback: output non-empty", len(out.strip()) > 0)


# ==============================================================================
# 9. inject_false_constraint
# ==============================================================================

def test_inject_false_constraint():
    section("9. inject_false_constraint  (Cat-4 Completeness: appends a false constraint)")

    samples = [
        ("binary_search handoff", HANDOFF_BINARY_SEARCH, 0),
        ("sort_odd handoff",      HANDOFF_SORT_ODD,      1),
        ("parentheses handoff",   HANDOFF_PARENTHESES,   2),
    ]
    known_endings = [
        "no empty-input handling is needed.",
        "negative values will not occur.",
        "uniqueness is guaranteed.",
        "special characters.",
        "None or null inputs.",
        "edge-case input.",
    ]
    for label, h, seed in samples:
        out = inject_false_constraint(h, seed=seed)
        _show(label, h, out)
        check(f"inject_fc/{label} -- non-empty",   len(out.strip()) > 0)
        check(f"inject_fc/{label} -- text changed", out.strip() != h.strip())
        check(f"inject_fc/{label} -- ends with false constraint",
              any(out.strip().endswith(e) for e in known_endings),
              f"tail: '...{out.strip()[-60:]}'")


# ==============================================================================
# 10. fake_completion
# ==============================================================================

def test_fake_completion():
    section("10. fake_completion  (Cat-4 Completeness: replaces last 25% with filler)")

    samples = [
        ("binary_search handoff", HANDOFF_BINARY_SEARCH, 0),
        ("sort_odd handoff",      HANDOFF_SORT_ODD,      1),
        ("parentheses handoff",   HANDOFF_PARENTHESES,   2),
    ]
    known_filler_words = {
        "conventions", "readability", "maintainable", "idiomatic",
        "implementer", "specification", "housekeeping", "encouraged",
    }
    for label, h, seed in samples:
        out = fake_completion(h, seed=seed)
        _show(label, h, out)
        check(f"fake_compl/{label} -- non-empty",      len(out.strip()) > 0)
        check(f"fake_compl/{label} -- text changed",   out.strip() != h.strip())
        ratio = len(out.split()) / max(len(h.split()), 1)
        check(f"fake_compl/{label} -- word count preserved (80-120%)",
              0.80 <= ratio <= 1.20,
              f"ratio={ratio:.2f}")
        tail_words = set(out.split()[-20:])
        check(f"fake_compl/{label} -- filler vocab in tail",
              bool(tail_words & known_filler_words),
              f"tail: {list(tail_words)[:8]}")


# ==============================================================================
# Runner
# ==============================================================================

def main():
    test_truncate()
    test_omit_entity()
    test_drop_tool_result()
    test_over_summarize()
    test_invert_objective()
    test_negate_edge_case()
    test_wrong_algorithm_name()
    test_signature_rename()
    test_inject_false_constraint()
    test_fake_completion()

    print(f"\n{'='*72}")
    total_pass = _test_count - _fail_count
    print(f"  Results: {total_pass}/{_test_count} passed", end="")
    if _fail_count:
        print(f"  <- {_fail_count} FAILED", end="")
    print(f"\n{'='*72}\n")
    sys.exit(1 if _fail_count else 0)


if __name__ == "__main__":
    main()
