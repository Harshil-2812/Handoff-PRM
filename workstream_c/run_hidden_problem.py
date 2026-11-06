"""Hidden-Problem Setting (C5): Evaluates iterative handoff refinement when Agent B cannot see the raw problem.

Compares three conditions:
1. no_gate: Agent A produces handoff once; Agent B executes directly.
2. blind_retry: If Stage-1 gate blocks (score < tau), Agent A retries up to 2 times with a generic prompt.
3. shap_hinted_retry: If Stage-1 gate blocks, Agent A retries up to 2 times receiving the worst SHAP feature hint.

Tracks:
- Pass@1
- Number of retries used (0, 1, or 2)
- Total tokens consumed
- Total end-to-end latency
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

from workstream_c.harness.agents_interface import LLMAgentClient
from workstream_c.harness.config import (
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    HEADLINE_THRESHOLD_RAW,
    LIVE_SEEDS,
    RESULTS_DIR,
)
from workstream_c.harness.execution_sandbox import grade_code
from workstream_c.harness.feature_pipeline import SHAP_FEATURE_HINTS, Stage1Classifier
from workstream_c.harness.key_manager import KeyPool
from workstream_c.harness.task_loader import load_live_test_tasks

logger = logging.getLogger("HiddenProblem")

BLIND_RETRY_TEMPLATE = """You are Agent A in a coding pipeline. Your previous handoff message was BLOCKED by an automated quality gate because it lacked sufficient clarity or completeness.

Original Problem:
{problem}

Previous Handoff:
{previous_handoff}

Please rewrite the handoff message completely, ensuring that all function signatures, constraints, algorithmic steps, and edge cases are clearly and thoroughly specified. Do not include implementation code.
"""

SHAP_HINTED_RETRY_TEMPLATE = """You are Agent A in a coding pipeline. Your previous handoff message was BLOCKED by an automated quality gate.

Original Problem:
{problem}

Previous Handoff:
{previous_handoff}

Gate Diagnostic Feedback:
{hint}

Please revise your handoff message to address this feedback, ensuring that all requirements, signatures, and edge cases are fully detailed. Do not include implementation code.
"""

FIELDNAMES = [
    "task_id",
    "source",
    "condition",          # "no_gate" | "blind_retry" | "shap_hinted_retry"
    "seed",
    "retries_used",       # 0, 1, 2
    "final_stage1_score",
    "outcome",            # 1 (passed unit tests) | 0 (failed)
    "tokens_prompt",
    "tokens_completion",
    "latency_ms",
    "error_message",
]


def run_hidden_problem_experiment(
    tasks: List[Dict],
    seeds: List[int],
    output_path: Path,
    dry_run: bool = False,
    threshold: float = HEADLINE_THRESHOLD_RAW,
    max_retries: int = 2,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = set()
    
    if output_path.exists() and os.path.getsize(output_path) > 0:
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                completed_keys.add((r["task_id"], r["condition"], int(r["seed"])))
        logger.info(f"Resuming: found {len(completed_keys)} trials in {output_path}")
    else:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

    key_pool = KeyPool()
    client = LLMAgentClient(key_pool=key_pool, dry_run=dry_run)
    classifier = Stage1Classifier()
    conditions = ["no_gate", "blind_retry", "shap_hinted_retry"]

    logger.info(f"Running Hidden-Problem Evaluation on {len(tasks)} tasks x {len(seeds)} seeds x {len(conditions)} conditions.")

    for t_idx, task in enumerate(tasks, 1):
        task_id = task["id"]
        source = task.get("source", "unknown")
        entry_point = task["entry_point"]
        problem = task["problem"]
        test_code = task["test_code"]

        for seed in seeds:
            for cond in conditions:
                if (task_id, cond, seed) in completed_keys:
                    continue

                total_p_tokens = 0
                total_c_tokens = 0
                total_latency_ms = 0.0
                retries_used = 0

                # Initial plan from Agent A
                a_res = client.plan_agent_a(problem=problem, model=model_a, seed=seed)
                total_p_tokens += a_res.get("prompt_tokens", 0)
                total_c_tokens += a_res.get("completion_tokens", 0)
                total_latency_ms += a_res.get("latency_ms", 0.0)

                current_handoff = a_res.get("text", "")
                score_res = classifier.score(current_handoff, problem, entry_point=entry_point, threshold=threshold)
                final_score = score_res["score"]

                if cond != "no_gate":
                    while retries_used < max_retries and final_score < threshold:
                        retries_used += 1
                        if cond == "blind_retry":
                            prompt = BLIND_RETRY_TEMPLATE.format(problem=problem, previous_handoff=current_handoff)
                        else:  # shap_hinted_retry
                            hint = score_res.get("hint") or "Provide more explicit constraints and edge cases."
                            prompt = SHAP_HINTED_RETRY_TEMPLATE.format(problem=problem, previous_handoff=current_handoff, hint=hint)

                        retry_res = client.generate(prompt, model=model_a, seed=seed + retries_used * 100)
                        total_p_tokens += retry_res.get("prompt_tokens", 0)
                        total_c_tokens += retry_res.get("completion_tokens", 0)
                        total_latency_ms += retry_res.get("latency_ms", 0.0)

                        current_handoff = retry_res.get("text", current_handoff)
                        score_res = classifier.score(current_handoff, problem, entry_point=entry_point, threshold=threshold)
                        final_score = score_res["score"]

                # Agent B implements code strictly from the final handoff (cannot see raw problem)
                b_res = client.code_agent_b(
                    handoff_or_problem=current_handoff,
                    entry_point=entry_point,
                    model=model_b,
                    seed=seed,
                    is_raw_problem=False,
                )
                total_p_tokens += b_res.get("prompt_tokens", 0)
                total_c_tokens += b_res.get("completion_tokens", 0)
                total_latency_ms += b_res.get("latency_ms", 0.0)

                passed, err_msg, grade_ms = grade_code(
                    code=b_res.get("text", ""),
                    entry_point=entry_point,
                    test_code=test_code,
                    timeout_seconds=5.0,
                )
                total_latency_ms += grade_ms

                row = {
                    "task_id": task_id,
                    "source": source,
                    "condition": cond,
                    "seed": seed,
                    "retries_used": retries_used,
                    "final_stage1_score": round(final_score, 4),
                    "outcome": 1 if passed else 0,
                    "tokens_prompt": total_p_tokens,
                    "tokens_completion": total_c_tokens,
                    "latency_ms": round(total_latency_ms, 2),
                    "error_message": err_msg,
                }

                with open(output_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writerow(row)
                    f.flush()

                completed_keys.add((task_id, cond, seed))
                logger.info(f"[{t_idx}/{len(tasks)}] {task_id} [{cond}] s{seed} -> Retries: {retries_used}, Score: {final_score:.3f}, Pass: {passed}")


def main():
    parser = argparse.ArgumentParser(description="Workstream C: Hidden-Problem Experiment (C5)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks")
    parser.add_argument("--seeds", type=int, nargs="+", default=LIVE_SEEDS, help="Seeds to evaluate")
    parser.add_argument("--dry-run", action="store_true", help="Run without API calls")
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "live_results_hidden_problem.csv"))
    args = parser.parse_args()

    tasks = load_live_test_tasks(limit=args.limit)
    run_hidden_problem_experiment(
        tasks=tasks,
        seeds=args.seeds,
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
