"""Live Prevalence Sweep (C6): Evaluates net benefit of the gate across realistic corruptions prevalences (5%, 20%, 50%).

Uses a fixed sampling seed (2026) to generate prevalence streams mixing clean (1-pi) and corrupted (pi) handoffs.
Measures:
- Net change in Pass@1 (Delta Pass@1)
- Net change in Latency and Tokens
- Precision of blocks and false-block rate under each prevalence regime
"""

import argparse
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

from workstream_c.harness.agents_interface import LLMAgentClient
from workstream_c.harness.arms import ArmExecutor
from workstream_c.harness.config import (
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    DEFAULT_MODEL_JUDGE,
    HEADLINE_THRESHOLD_RAW,
    LIVE_SEEDS,
    RESULTS_DIR,
)
from workstream_c.harness.feature_pipeline import Stage1Classifier
from workstream_c.harness.key_manager import KeyPool
from workstream_c.harness.task_loader import load_live_test_tasks
from workstream_c.run_live import pick_corruption

logger = logging.getLogger("PrevalenceSweep")

PREVALENCES = [0.05, 0.20, 0.50]
PREVALENCE_SAMPLING_SEED = 2026

FIELDNAMES = [
    "prevalence",
    "task_id",
    "source",
    "seed",
    "handoff_type",       # "clean" or corruption name
    "is_corrupted",       # 0 or 1
    "ungated_outcome",    # 1 or 0
    "stage1_outcome",     # 1 or 0
    "stage1_decision",    # "PASS" | "BLOCK"
    "stage1_score",
    "ungated_latency_ms",
    "stage1_latency_ms",
    "ungated_tokens",
    "stage1_tokens",
]


def run_prevalence_sweep(
    tasks: List[Dict],
    seeds: List[int],
    output_path: Path,
    dry_run: bool = False,
    threshold: float = HEADLINE_THRESHOLD_RAW,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
    model_judge: str = DEFAULT_MODEL_JUDGE,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = set()
    
    if output_path.exists() and os.path.getsize(output_path) > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_keys.add((float(r["prevalence"]), r["task_id"], int(r["seed"])))
        logger.info(f"Resuming: found {len(completed_keys)} completed rows in {output_path}")
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    key_pool = KeyPool()
    client = LLMAgentClient(key_pool=key_pool, dry_run=dry_run)
    classifier = Stage1Classifier()
    executor = ArmExecutor(classifier=classifier, client=client, threshold=threshold)

    rng = random.Random(PREVALENCE_SAMPLING_SEED)

    for prev in PREVALENCES:
        logger.info(f"=== Running Prevalence Sweep: {int(prev*100)}% corrupted ===")
        for t_idx, task in enumerate(tasks, 1):
            task_id = task["id"]
            source = task.get("source", "unknown")
            problem = task["problem"]
            entry_point = task["entry_point"]

            for seed in seeds:
                if (prev, task_id, seed) in completed_keys:
                    continue

                # Generate clean handoff
                a_res = client.plan_agent_a(problem=problem, model=model_a, seed=seed)
                clean_handoff = a_res.get("text", "")

                # Decide if this sample in the stream is corrupted based on prevalence
                is_corrupted = rng.random() < prev
                if is_corrupted:
                    corr_name, corr_fn = pick_corruption(task_id, seed)
                    try:
                        candidate_handoff = corr_fn(clean_handoff, entry_point)
                    except Exception:
                        candidate_handoff = clean_handoff[:len(clean_handoff)//3]
                        corr_name = "truncation"
                    htype = corr_name
                else:
                    candidate_handoff = clean_handoff
                    htype = "clean"

                # Run ungated condition
                u_res = executor.run_arm(
                    arm="ungated",
                    task=task,
                    candidate_handoff=candidate_handoff,
                    handoff_type=htype,
                    seed=seed,
                    model_a=model_a,
                    model_b=model_b,
                    model_judge=model_judge,
                )

                # Run stage1_only condition
                s1_res = executor.run_arm(
                    arm="stage1_only",
                    task=task,
                    candidate_handoff=candidate_handoff,
                    handoff_type=htype,
                    seed=seed,
                    model_a=model_a,
                    model_b=model_b,
                    model_judge=model_judge,
                )

                row = {
                    "prevalence": prev,
                    "task_id": task_id,
                    "source": source,
                    "seed": seed,
                    "handoff_type": htype,
                    "is_corrupted": 1 if is_corrupted else 0,
                    "ungated_outcome": u_res["outcome"],
                    "stage1_outcome": s1_res["outcome"],
                    "stage1_decision": s1_res["decision"],
                    "stage1_score": s1_res["stage1_score"],
                    "ungated_latency_ms": u_res["latency_ms"],
                    "stage1_latency_ms": s1_res["latency_ms"],
                    "ungated_tokens": u_res["tokens_prompt"] + u_res["tokens_completion"],
                    "stage1_tokens": s1_res["tokens_prompt"] + s1_res["tokens_completion"],
                }

                with open(output_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writerow(row)
                    f.flush()

                completed_keys.add((prev, task_id, seed))
                logger.info(f"[{int(prev*100)}%] {task_id} s{seed} ({htype}) -> Ungated: {u_res['outcome']}, Gated: {s1_res['outcome']} (Dec: {s1_res['decision']})")


def main():
    parser = argparse.ArgumentParser(description="Workstream C: Live Prevalence Sweep (C6)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--seeds", type=int, nargs="+", default=LIVE_SEEDS, help="Seeds to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Run without API calls")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "live_results_prevalence.csv"))
    args = parser.parse_args()

    tasks = load_live_test_tasks(limit=args.limit)
    run_prevalence_sweep(
        tasks=tasks,
        seeds=args.seeds,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

