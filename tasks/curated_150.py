"""
curated_150.py -- The 150-task curated evaluation set for the Handoff-PRM paper.

Selection breakdown
-------------------
  Hand-crafted easy   30  (task_001-task_030)
                          Backup pool: task_064, task_067, task_070, task_073, task_076
  Hand-crafted hard   40  (task_031-task_060 [30 from coding_tasks]
                           + task_061,062,063,065,066,068,069,071,072,074 [10 from hard_tasks_3])
                          Backup pool: task_075, task_077, task_078, task_080, task_081
  HumanEval           40  (he_000-he_043, first 30 = original paper set,
                           he_031,033-037,039,040,042,043 = 10 new, all verified in rollouts.csv)
                          Backup pool: he_044 through he_048
  MBPP (sanitised)    40  (mbpp_002-mbpp_105, first 40 non-holdout verified tasks;
                           matches original 40 verified against reference code)
                          Backup pool: mbpp_111, mbpp_113, mbpp_115, mbpp_118, mbpp_125
  -------------------------------------------------------------------------
  Total              150

Rules applied
-------------
* No holdout task IDs (holdout_task_ids.txt) appear in the active set.
* Every hand-crafted task passed a human-written reference solution before inclusion.
* HE and MBPP rows were verified (label=1, corruption_type=none) in rollouts.csv.
* BACKUP_POOL has 5 reserve tasks per category that swap in if an active
  task must be retired.

Usage
-----
    from tasks.curated_150 import CURATED_150, BACKUP_POOL
    # CURATED_150  -> list of 150 task dicts, ready for the pipeline
    # BACKUP_POOL  -> dict keyed by category, 5 reserves each
"""

from tasks.coding_tasks import TASKS as _hand_crafted
from tasks.hard_tasks_3 import HARD_TASKS_3 as _hard3
from humaneval_tasks import HUMANEVAL_TASKS as _humaneval_all
from mbpp_tasks import MBPP_TASKS as _mbpp_all

# -- build look-up maps -------------------------------------------------------
_hand_by_id  = {t["id"]: t for t in _hand_crafted}
_hard3_by_id = {t["id"]: t for t in _hard3}
_he_by_id    = {t["id"]: t for t in _humaneval_all}
_mbpp_by_id  = {t["id"]: t for t in _mbpp_all}


def _resolve(ids: list) -> list:
    """Look up IDs across all source maps; warn on any missing."""
    import warnings
    out = []
    for tid in ids:
        t = (
            _hand_by_id.get(tid)
            or _hard3_by_id.get(tid)
            or _he_by_id.get(tid)
            or _mbpp_by_id.get(tid)
        )
        if t is None:
            warnings.warn(f"curated_150: task {tid!r} not found -- skipping")
        else:
            out.append(t)
    return out


# ============================================================================
#  ACTIVE SET (150 tasks)
# ============================================================================

# -- 1. Hand-crafted EASY  (30) -----------------------------------------------
_EASY_IDS = [
    "task_001", "task_002", "task_003", "task_004", "task_005",
    "task_006", "task_007", "task_008", "task_009", "task_010",
    "task_011", "task_012", "task_013", "task_014", "task_015",
    "task_016", "task_017", "task_018", "task_019", "task_020",
    "task_021", "task_022", "task_023", "task_024", "task_025",
    "task_026", "task_027", "task_028", "task_029", "task_030",
]  # 30 tasks -- full easy pool from coding_tasks.py

# -- 2. Hand-crafted HARD  (40) -----------------------------------------------
# 30 from coding_tasks.py (task_031-060) + 10 from hard_tasks_3.py.
# hard_tasks_3 hand_hard IDs available: 061-063, 065-066, 068-069,
#                                        071-072, 074-075, 077-078, 080-081, 083-084
# Active set uses the first 10; 5 from hard_tasks_3 kept as backup.
_HARD_IDS = [
    # from coding_tasks.py (30)
    "task_031", "task_032", "task_033", "task_034", "task_035",
    "task_036", "task_037", "task_038", "task_039", "task_040",
    "task_041", "task_042", "task_043", "task_044", "task_045",
    "task_046", "task_047", "task_048", "task_049", "task_050",
    "task_051", "task_052", "task_053", "task_054", "task_055",
    "task_056", "task_057", "task_058", "task_059", "task_060",
    # from hard_tasks_3.py (10)
    "task_061", "task_062", "task_063", "task_065", "task_066",
    "task_068", "task_069", "task_071", "task_072", "task_074",
]  # 40 tasks total

# -- 3. HumanEval  (40) -------------------------------------------------------
# First 30 = original paper set (verified 30/30 against canonical solutions).
# +10 new = next contiguous non-holdout verified IDs from rollouts.csv.
_HE_IDS = [
    # original 30 (paper set)
    "he_000", "he_001", "he_002", "he_003", "he_004",
    "he_005", "he_006", "he_007", "he_008", "he_009",
    "he_011", "he_012", "he_013", "he_014", "he_015",
    "he_016", "he_017", "he_018", "he_019", "he_020",
    "he_021", "he_022", "he_023", "he_024", "he_025",
    "he_026", "he_027", "he_028", "he_029", "he_030",
    # +10 new (verified in rollouts.csv, non-holdout)
    "he_031", "he_033", "he_034", "he_035", "he_036",
    "he_037", "he_039", "he_040", "he_042", "he_043",
]  # 40 tasks total

# -- 4. MBPP sanitised  (40) --------------------------------------------------
# Original 40 verified against reference code, all in rollouts.csv.
_MBPP_IDS = [
    "mbpp_002", "mbpp_004", "mbpp_007", "mbpp_008", "mbpp_009",
    "mbpp_011", "mbpp_012", "mbpp_014", "mbpp_017", "mbpp_019",
    "mbpp_057", "mbpp_058", "mbpp_061", "mbpp_062", "mbpp_063",
    "mbpp_064", "mbpp_065", "mbpp_066", "mbpp_067", "mbpp_069",
    "mbpp_071", "mbpp_072", "mbpp_074", "mbpp_075", "mbpp_077",
    "mbpp_080", "mbpp_084", "mbpp_086", "mbpp_088", "mbpp_089",
    "mbpp_094", "mbpp_095", "mbpp_096", "mbpp_097", "mbpp_098",
    "mbpp_099", "mbpp_100", "mbpp_101", "mbpp_104", "mbpp_105",
]  # 40 tasks total


CURATED_150: list = (
    _resolve(_EASY_IDS)    # 30 hand_easy
    + _resolve(_HARD_IDS)  # 40 hand_hard
    + _resolve(_HE_IDS)    # 40 humaneval
    + _resolve(_MBPP_IDS)  # 40 mbpp
)


# ============================================================================
#  BACKUP / RESERVE POOL  (not in the active 150)
# ============================================================================

# Easy backup (5): batch-3 easy tasks not in active set
# (task_079, task_082 also available if more replacements needed)
_EASY_BACKUP_IDS = [
    "task_064", "task_067", "task_070", "task_073", "task_076",
]

# Hard backup (5): remaining hard_tasks_3 hand_hard not in active set
# (task_083, task_084 also available)
_HARD_BACKUP_IDS = [
    "task_075", "task_077", "task_078", "task_080", "task_081",
]

# HE backup (5): next verified non-holdout HE tasks after the active 40
_HE_BACKUP_IDS = [
    "he_044", "he_045", "he_046", "he_047", "he_048",
]

# MBPP backup (5): next verified non-holdout MBPP tasks after the active 40
_MBPP_BACKUP_IDS = [
    "mbpp_111", "mbpp_113", "mbpp_115", "mbpp_118", "mbpp_125",
]

BACKUP_POOL: dict = {
    "hand_easy": _resolve(_EASY_BACKUP_IDS),
    "hand_hard": _resolve(_HARD_BACKUP_IDS),
    "humaneval": _resolve(_HE_BACKUP_IDS),
    "mbpp":      _resolve(_MBPP_BACKUP_IDS),
}


# -- sanity check (run as script) ---------------------------------------------
if __name__ == "__main__":
    from collections import Counter

    counts = Counter(t["source"] for t in CURATED_150)
    ids = [t["id"] for t in CURATED_150]

    print(f"Total tasks : {len(CURATED_150)}")
    print(f"Unique IDs  : {len(set(ids))}")
    if len(ids) != len(set(ids)):
        dupes = [x for x in ids if ids.count(x) > 1]
        print(f"  WARNING -- duplicates: {set(dupes)}")
    print("By source   :")
    for src, n in sorted(counts.items()):
        print(f"  {src:<20} {n}")
    print()
    print("Backup pool :")
    for cat, tasks in BACKUP_POOL.items():
        print(f"  {cat:<20} {len(tasks)} tasks")
