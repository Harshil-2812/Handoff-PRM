"""
curated_250.py -- The 250-task curated evaluation set for the Handoff-PRM paper.

Superset of curated_150.py.  All 150 original tasks are kept (or swapped if
they failed Agent B verification); this file adds 100 more tasks across all
four categories.

Swap log (10 tasks replaced after step4_verified.csv run):
  task_020  (hand_easy, FAIL) -> task_079
  task_071  (hand_hard, FAIL) -> task_084
  mbpp_088  (mbpp, FAIL)      -> mbpp_287
  mbpp_101  (mbpp, FAIL)      -> mbpp_291
  mbpp_105  (mbpp, FAIL)      -> mbpp_292
  mbpp_125  (mbpp, FAIL)      -> mbpp_293
  mbpp_162  (mbpp, FAIL)      -> mbpp_297
  mbpp_244  (mbpp, FAIL)      -> mbpp_308
  mbpp_257  (mbpp, FAIL)      -> mbpp_309
  mbpp_266  (mbpp, FAIL)      -> mbpp_392

Selection breakdown (after swaps)
-----------------------------------
  Hand-crafted easy   35  (task_001-030 minus task_020 + task_064,067,070,073,076,079)
  Hand-crafted hard   45  (task_031-060 + task_061-074 + task_077,078,080,081,083,084
                           minus task_071)
  HumanEval           80  (original 40 + 40 new, all verified)
  MBPP (sanitised)    90  (verified 90, 10 original fails replaced with verified alts)
  -----------------------------------------------------------------------
  Total              250

Rules applied
-------------
* No holdout task IDs (holdout_task_ids.txt) appear in the active set.
* All hand-crafted tasks passed a human-written reference solution.
* All HE and MBPP rows were verified (label=1, corruption_type=none) in rollouts.csv.
* Swapped-in tasks are from the same verified pool and were NOT previously used.

Usage
-----
    from tasks.curated_250 import CURATED_250, BACKUP_POOL
    # CURATED_250  -> list of 250 task dicts, ready for the pipeline
    # BACKUP_POOL  -> dict keyed by category, reserve tasks
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
            warnings.warn(f"curated_250: task {tid!r} not found -- skipping")
        else:
            out.append(t)
    return out


# ============================================================================
#  ACTIVE SET (250 tasks)
# ============================================================================

# -- 1. Hand-crafted EASY  (35) -----------------------------------------------
# task_020 failed -> replaced with task_079 (from backup pool)
_EASY_IDS = [
    "task_001", "task_002", "task_003", "task_004", "task_005",
    "task_006", "task_007", "task_008", "task_009", "task_010",
    "task_011", "task_012", "task_013", "task_014", "task_015",
    "task_016", "task_017", "task_018", "task_019",
    # task_020 removed (FAIL) -> task_079 added below
    "task_021", "task_022", "task_023", "task_024", "task_025",
    "task_026", "task_027", "task_028", "task_029", "task_030",
    # batch-3 easy
    "task_064", "task_067", "task_070", "task_073", "task_076",
    "task_079",   # swap-in for task_020
]  # 35 tasks total

# -- 2. Hand-crafted HARD  (45) -----------------------------------------------
# task_071 failed -> replaced with task_084 (from backup pool)
_HARD_IDS = [
    # from coding_tasks.py (30)
    "task_031", "task_032", "task_033", "task_034", "task_035",
    "task_036", "task_037", "task_038", "task_039", "task_040",
    "task_041", "task_042", "task_043", "task_044", "task_045",
    "task_046", "task_047", "task_048", "task_049", "task_050",
    "task_051", "task_052", "task_053", "task_054", "task_055",
    "task_056", "task_057", "task_058", "task_059", "task_060",
    # from hard_tasks_3.py -- original 10 minus task_071
    "task_061", "task_062", "task_063", "task_065", "task_066",
    "task_068", "task_069",
    # task_071 removed (FAIL)
    "task_072", "task_074",
    # from hard_tasks_3.py -- 5 new + task_084 swap-in
    "task_077", "task_078", "task_080", "task_081", "task_083",
    "task_084",   # swap-in for task_071
]  # 45 tasks total

# -- 3. HumanEval  (80) -------------------------------------------------------
# No HE failures -- all 80 unchanged
_HE_IDS = [
    # original 30 (paper set)
    "he_000", "he_001", "he_002", "he_003", "he_004",
    "he_005", "he_006", "he_007", "he_008", "he_009",
    "he_011", "he_012", "he_013", "he_014", "he_015",
    "he_016", "he_017", "he_018", "he_019", "he_020",
    "he_021", "he_022", "he_023", "he_024", "he_025",
    "he_026", "he_027", "he_028", "he_029", "he_030",
    # +10 added in 150-task run
    "he_031", "he_033", "he_034", "he_035", "he_036",
    "he_037", "he_039", "he_040", "he_042", "he_043",
    # +40 new
    "he_044", "he_045", "he_046", "he_047", "he_048",
    "he_049", "he_051", "he_052", "he_053", "he_054",
    "he_055", "he_056", "he_057", "he_058", "he_059",
    "he_060", "he_061", "he_062", "he_063", "he_065",
    "he_066", "he_067", "he_068", "he_069", "he_070",
    "he_071", "he_072", "he_073", "he_074", "he_075",
    "he_076", "he_077", "he_079", "he_080", "he_081",
    "he_082", "he_083", "he_084", "he_085", "he_086",
]  # 80 tasks total

# -- 4. MBPP sanitised  (90) --------------------------------------------------
# 8 MBPP fails replaced with verified alternatives from the pool
_MBPP_IDS = [
    # original 40 with 5 swaps (mbpp_088,101,105 removed -> mbpp_287,291,292 in)
    "mbpp_002", "mbpp_004", "mbpp_007", "mbpp_008", "mbpp_009",
    "mbpp_011", "mbpp_012", "mbpp_014", "mbpp_017", "mbpp_019",
    "mbpp_057", "mbpp_058", "mbpp_061", "mbpp_062", "mbpp_063",
    "mbpp_064", "mbpp_065", "mbpp_066", "mbpp_067", "mbpp_069",
    "mbpp_071", "mbpp_072", "mbpp_074", "mbpp_075", "mbpp_077",
    "mbpp_080", "mbpp_084", "mbpp_086",
    # mbpp_088 removed (FAIL) -> mbpp_287
    "mbpp_089",
    "mbpp_094", "mbpp_095", "mbpp_096", "mbpp_097", "mbpp_098",
    "mbpp_099", "mbpp_100",
    # mbpp_101 removed (FAIL) -> mbpp_291
    # mbpp_105 removed (FAIL) -> mbpp_292
    "mbpp_104",
    # new 50 with 5 swaps (mbpp_125,162,244,257,266 removed -> mbpp_293,297,308,309,392 in)
    "mbpp_111", "mbpp_113", "mbpp_115", "mbpp_118",
    # mbpp_125 removed (FAIL) -> mbpp_293
    "mbpp_126", "mbpp_131", "mbpp_132", "mbpp_133", "mbpp_135",
    "mbpp_139", "mbpp_140", "mbpp_141", "mbpp_142", "mbpp_145",
    "mbpp_161",
    # mbpp_162 removed (FAIL) -> mbpp_297
    "mbpp_165", "mbpp_166", "mbpp_167",
    "mbpp_170", "mbpp_171", "mbpp_172", "mbpp_222", "mbpp_223",
    "mbpp_224", "mbpp_226", "mbpp_227", "mbpp_228", "mbpp_230",
    "mbpp_232", "mbpp_238",
    # mbpp_244 removed (FAIL) -> mbpp_308
    "mbpp_247", "mbpp_250",
    "mbpp_252", "mbpp_255", "mbpp_256",
    # mbpp_257 removed (FAIL) -> mbpp_309
    "mbpp_261",
    # mbpp_266 removed (FAIL) -> mbpp_392
    "mbpp_267", "mbpp_268", "mbpp_271", "mbpp_274",
    "mbpp_277", "mbpp_281", "mbpp_283", "mbpp_284", "mbpp_285",
    # 10 swap-ins
    "mbpp_287", "mbpp_291", "mbpp_292", "mbpp_293", "mbpp_297",
    "mbpp_308", "mbpp_309", "mbpp_392",
    # 2 extra to maintain count of 90 (replacing 8 fails + keeping 2 previously backup)
    "mbpp_394", "mbpp_397",
]  # 90 tasks total


CURATED_250: list = (
    _resolve(_EASY_IDS)    # 35 hand_easy
    + _resolve(_HARD_IDS)  # 45 hand_hard
    + _resolve(_HE_IDS)    # 80 humaneval
    + _resolve(_MBPP_IDS)  # 90 mbpp
)


# ============================================================================
#  BACKUP / RESERVE POOL  (not in the active 250)
# ============================================================================

_EASY_BACKUP_IDS  = ["task_082"]                                          # 1 left
_HARD_BACKUP_IDS  = []                                                    # pool exhausted
_HE_BACKUP_IDS    = ["he_087", "he_088", "he_089", "he_090", "he_091"]
_MBPP_BACKUP_IDS  = ["mbpp_401", "mbpp_406", "mbpp_409", "mbpp_412", "mbpp_422"]

BACKUP_POOL: dict = {
    "hand_easy": _resolve(_EASY_BACKUP_IDS),
    "hand_hard": _resolve(_HARD_BACKUP_IDS),
    "humaneval": _resolve(_HE_BACKUP_IDS),
    "mbpp":      _resolve(_MBPP_BACKUP_IDS),
}


# -- sanity check (run as script) ---------------------------------------------
if __name__ == "__main__":
    from collections import Counter

    counts = Counter(t["source"] for t in CURATED_250)
    ids = [t["id"] for t in CURATED_250]

    print(f"Total tasks : {len(CURATED_250)}")
    print(f"Unique IDs  : {len(set(ids))}")
    if len(ids) != len(set(ids)):
        dupes = sorted(set(x for x in ids if ids.count(x) > 1))
        print(f"  WARNING -- duplicates: {dupes}")
    print("By source   :")
    for src, n in sorted(counts.items()):
        print(f"  {src:<20} {n}")
    print()
    print("Backup pool :")
    for cat, tasks in BACKUP_POOL.items():
        print(f"  {cat:<20} {len(tasks)} tasks")
    print()
    from tasks.curated_150 import CURATED_150
    old_ids = set(t["id"] for t in CURATED_150)
    new_ids = [t["id"] for t in CURATED_250 if t["id"] not in old_ids]
    print(f"Tasks new vs curated_150 : {len(new_ids)}")
