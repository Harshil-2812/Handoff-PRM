"""Unified Live Evaluation Runner for Workstream C.

Supports:
- Tasks loaded from frozen live test manifest (with runtime zero-leakage assertions).
- Seeds 42, 43, 44 (paired clean vs. corrupted conditions).
- Incremental resumable logging adhering strictly to the frozen results schema.
- All 7 experimental arms (ungated, stage1_only, cascade, llm_judge, always_fallback, agent_ask, deterministic).
- C4 Organic evaluation, C5 Hidden-problem setting, C6 Prevalence sweep.
- `--dry-run` and `--limit N` flags for cost-free verification and budget forecasting.
"""

import argparse
import csv
import hashlib
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional

# Setup path and environment
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Offline HF cache flags to prevent network retries
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

from corruption import (
    drop_tool_result,
    fake_completion,
    inject_false_constraint,
    invert_objective,
    negate_edge_case,
    omit_entity,
    over_summarize,
    signature_rename,
    truncate,
    wrong_algorithm_name,
)
from workstream_c.harness.agents_interface import LLMAgentClient
from workstream_c.harness.arms import ALL_ARMS, ArmExecutor
from workstream_c.harness.config import (
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    DEFAULT_MODEL_JUDGE,
    DEFAULT_TEMP_A,
    DEFAULT_TEMP_B,
    DEFAULT_TEMP_JUDGE,
    HEADLINE_THRESHOLD_RAW,
    LIVE_SEEDS,
    RESULTS_DIR,
)
from workstream_c.harness.key_manager import KeyPool, get_key_slice
from workstream_c.harness.logger import ResultsLogger
from workstream_c.harness.task_loader import load_live_test_tasks

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RunLive")

CORRUPTION_CHOICES = [
    ("over_summarization", lambda h, ep: over_summarize(h, sentence_count=1)),
    ("tool_result_drop", lambda h, ep: drop_tool_result(h)),
    ("truncation", lambda h, ep: truncate(h, keep_fraction=0.30)),
    ("entity_omission", lambda h, ep: omit_entity(h, fraction_to_remove=0.50)),
    ("signature_rename", lambda h, ep: signature_rename(h, ep)),
    ("invert_objective", lambda h, ep: invert_objective(h)),
    ("negate_edge_case", lambda h, ep: negate_edge_case(h)),
    ("wrong_algorithm_name", lambda h, ep: wrong_algorithm_name(h)),
    ("fake_completion", lambda h, ep: fake_completion(h)),
    ("inject_false_constraint", lambda h, ep: inject_false_constraint(h)),
]


_RECORDED_CORRUPTIONS_CACHE: dict = {}


def load_recorded_corruptions(csv_path: Path) -> dict:
    """Caches and returns the earliest non-clean handoff_type recorded per (task_id, seed)."""
    csv_path = Path(csv_path).resolve()
    if csv_path in _RECORDED_CORRUPTIONS_CACHE:
        return _RECORDED_CORRUPTIONS_CACHE[csv_path]

    recorded = {}
    if csv_path.exists():
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tid = row.get("task_id", "")
                    s_str = row.get("seed", "")
                    htype = row.get("handoff_type", "")
                    if tid and s_str and htype and htype != "clean":
                        try:
                            s = int(s_str)
                            if (tid, s) not in recorded:
                                recorded[(tid, s)] = htype
                        except ValueError:
                            pass
        except Exception as e:
            logger.warning(f"Error loading recorded corruptions from {csv_path}: {e}")

    _RECORDED_CORRUPTIONS_CACHE[csv_path] = recorded
    return recorded


def pick_corruption(task_id: str, seed: int, csv_path: Optional[Path] = None):
    """Deterministically picks a corruption function based on task_id and seed.

    1. First looks up whether (task_id, seed) already has rows recorded in csv_path (the earliest recorded corruption).
    2. Otherwise uses a stable hashlib.sha256 calculation (independent of Python hash seed).
    """
    if csv_path:
        recorded = load_recorded_corruptions(Path(csv_path))
        if (task_id, seed) in recorded:
            target_name = recorded[(task_id, seed)]
            for name, fn in CORRUPTION_CHOICES:
                if name == target_name:
                    return name, fn

    # Stable hashlib fallback (never use built-in hash() which varies with PYTHONHASHSEED)
    digest = hashlib.sha256(f"{task_id}_{seed}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(CORRUPTION_CHOICES)
    return CORRUPTION_CHOICES[idx]


def run_experiment(
    arms: List[str],
    seeds: List[int],
    limit: Optional[int] = None,
    source_filter: Optional[str] = None,
    output_csv: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    key_slice: Optional[str] = None,
    dry_run: bool = False,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    model_judge: str = DEFAULT_MODEL_JUDGE,
    temp_a: float = DEFAULT_TEMP_A,
    temp_b: float = DEFAULT_TEMP_B,
    temp_judge: float = DEFAULT_TEMP_JUDGE,
    threshold: float = HEADLINE_THRESHOLD_RAW,
):
    """Executes the paired live evaluation across tasks, seeds, conditions, and arms."""
    logger.info("Initializing Workstream C Live Evaluation Pipeline...")
    output_path = output_csv or (RESULTS_DIR / "live_results_main.csv")
    results_logger = ResultsLogger(output_path)

    # 1. Load held-out tasks from manifest with zero-leakage assertion
    tasks = load_live_test_tasks(limit=limit, source_filter=source_filter)
    logger.info(f"Loaded {len(tasks)} verified held-out tasks for evaluation.")

    # 2. Initialize classifier, key pool, and client
    from workstream_c.harness.feature_pipeline import Stage1Classifier
    keys = get_key_slice(key_slice)
    key_pool = KeyPool(keys=keys)
    client = LLMAgentClient(key_pool=key_pool, cache_path=cache_path, dry_run=dry_run)
    classifier = Stage1Classifier()
    executor = ArmExecutor(classifier=classifier, client=client, threshold=threshold)

    total_combinations = len(tasks) * len(seeds) * len(arms) * 2  # 2 conditions: clean + corrupted
    completed_count = 0
    skipped_count = 0

    logger.info(
        f"Execution plan: {len(tasks)} tasks x {len(seeds)} seeds x {len(arms)} arms x 2 conditions "
        f"= {total_combinations} target trials."
    )

    t_start = time.time()

    for t_idx, task in enumerate(tasks, 1):
        task_id = task["id"]
        entry_point = task["entry_point"]
        problem = task["problem"]
        logger.info(f"[{t_idx}/{len(tasks)}] Task {task_id} ({task.get('source')}, {entry_point})")

        for seed in seeds:
            corr_name, corr_fn = pick_corruption(task_id, seed, csv_path=output_path)

            # Fast skip if both clean and corrupted conditions are already fully completed for all arms
            all_done = True
            for htype in ["clean", corr_name]:
                for arm in arms:
                    if not results_logger.is_completed(task_id, arm, seed, htype):
                        all_done = False
                        break
                if not all_done:
                    break

            if all_done:
                skipped_count += 2 * len(arms)
                continue

            # Step A: Generate or fetch Agent A's baseline clean handoff
            a_res = client.plan_agent_a(
                problem=problem,
                model=model_a,
                temperature=temp_a,
                seed=seed,
            )
            clean_handoff = a_res.get("text", "")

            # Step B: Generate paired corrupted handoff
            try:
                corrupted_handoff = corr_fn(clean_handoff, entry_point)
            except Exception as e:
                logger.warning(f"Corruption {corr_name} failed on {task_id}: {e}. Falling back to truncation.")
                corrupted_handoff = truncate(clean_handoff, 0.3)
                corr_name = "truncation"

            conditions = [
                ("clean", clean_handoff),
                (corr_name, corrupted_handoff),
            ]

            for htype, handoff_text in conditions:
                for arm in arms:
                    if results_logger.is_completed(task_id, arm, seed, htype):
                        skipped_count += 1
                        continue

                    trial_data = executor.run_arm(
                        arm=arm,
                        task=task,
                        candidate_handoff=handoff_text,
                        handoff_type=htype,
                        seed=seed,
                        model_a=model_a,
                        model_b=model_b,
                        model_judge=model_judge,
                        temp_a=temp_a,
                        temp_b=temp_b,
                        temp_judge=temp_judge,
                    )

                    results_logger.log_trial(trial_data)
                    completed_count += 1

                    if completed_count % 10 == 0 or completed_count == 1:
                        elapsed = time.time() - t_start
                        rate = completed_count / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"  Progress: {completed_count}/{total_combinations} completed "
                            f"({skipped_count} skipped) - Last: {task_id} [{arm}] {htype} -> "
                            f"Score: {trial_data['stage1_score']:.3f}, Dec: {trial_data['decision']}, "
                            f"Out: {trial_data['outcome']} ({rate:.1f} trials/s)"
                        )

    total_time = time.time() - t_start
    logger.info(
        f"Run complete. {completed_count} newly executed, {skipped_count} skipped in {total_time:.1f}s. "
        f"Saved to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Workstream C: Live End-to-End Evaluation Runner")
    parser.add_argument("--arm", choices=ALL_ARMS + ["all"], default="all", help="Arm to evaluate")
    parser.add_argument("--seeds", type=int, nargs="+", default=LIVE_SEEDS, help="Seeds to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--source", type=str, default=None, help="Filter by task source")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path")
    parser.add_argument("--cache", type=str, default=None, help="Custom cache JSON path")
    parser.add_argument("--key-slice", type=str, default=None, help="Slice of keys to use (e.g. '0:5')")
    parser.add_argument("--dry-run", action="store_true", help="Run without spending API quota")
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="Agent A model ID")
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Agent B model ID")
    parser.add_argument("--model-judge", default=DEFAULT_MODEL_JUDGE, help="LLM judge model ID")
    parser.add_argument("--temp-a", type=float, default=DEFAULT_TEMP_A, help="Agent A temperature")
    parser.add_argument("--temp-b", type=float, default=DEFAULT_TEMP_B, help="Agent B temperature")
    parser.add_argument("--temp-judge", type=float, default=DEFAULT_TEMP_JUDGE, help="Judge temperature")
    parser.add_argument("--threshold", type=float, default=HEADLINE_THRESHOLD_RAW, help="Stage 1 threshold")

    args = parser.parse_args()

    arms_to_run = ALL_ARMS if args.arm == "all" else [args.arm]
    
    if args.output:
        output_csv = Path(args.output)
    elif len(args.seeds) == 1:
        output_csv = RESULTS_DIR / f"live_results_main_s{args.seeds[0]}.csv"
    else:
        output_csv = None

    if args.cache:
        cache_path = Path(args.cache)
    elif len(args.seeds) == 1:
        cache_path = REPO_ROOT / "cache" / f"llm_cache_s{args.seeds[0]}.json"
    else:
        cache_path = None

    run_experiment(
        arms=arms_to_run,
        seeds=args.seeds,
        limit=args.limit,
        source_filter=args.source,
        output_csv=output_csv,
        cache_path=cache_path,
        key_slice=args.key_slice,
        dry_run=args.dry_run,
        model_a=args.model_a,
        model_b=args.model_b,
        model_judge=args.model_judge,
        temp_a=args.temp_a,
        temp_b=args.temp_b,
        temp_judge=args.temp_judge,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

