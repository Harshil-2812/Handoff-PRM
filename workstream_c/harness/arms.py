"""Experimental Arms Implementation for Workstream C.

Implements all 7 evaluation arms:
1. ungated: Agent B executes handoff directly without gating.
2. stage1_only: Gated by Stage-1 RF at frozen headline threshold (0.550419).
3. cascade: Two-stage cascade (Stage 1 RF + LLM Judge on escalated candidates).
4. llm_judge: Standalone LLM Judge using frozen B6 prompt ("with_problem").
5. always_fallback: Bypasses handoff; Agent B always receives raw problem statement.
6. agent_ask: Intercepted by AgentAsk-style clarifier checking the 4 edge error types (DG, SC, RD, CG).
7. deterministic: Simple rule-based gate checking exact signature preservation and length ratio >= 0.3.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

from workstream_c.harness.agents_interface import LLMAgentClient
from workstream_c.harness.config import HEADLINE_THRESHOLD_RAW
from workstream_c.harness.execution_sandbox import grade_code
from workstream_c.harness.feature_pipeline import Stage1Classifier

logger = logging.getLogger("Arms")

ALL_ARMS = [
    "ungated",
    "stage1_only",
    "cascade",
    "llm_judge",
    "always_fallback",
    "agent_ask",
    "deterministic",
]


class ArmExecutor:
    """Executes a task under a specific experimental arm and records outcomes."""

    def __init__(
        self,
        classifier: Stage1Classifier,
        client: LLMAgentClient,
        threshold: float = HEADLINE_THRESHOLD_RAW,
    ):
        self.classifier = classifier
        self.client = client
        self.threshold = threshold

    def run_arm(
        self,
        arm: str,
        task: Dict[str, Any],
        candidate_handoff: str,
        handoff_type: str,
        seed: int,
        model_a: str,
        model_b: str,
        model_judge: str,
        temp_a: float = 0.0,
        temp_b: float = 0.0,
        temp_judge: float = 0.0,
    ) -> Dict[str, Any]:
        """Runs a single trial for a task, handoff, and arm, returning structured trial metrics."""
        task_id = task["id"]
        source = task.get("source", "unknown")
        entry_point = task["entry_point"]
        problem = task["problem"]
        test_code = task["test_code"]
        handoff_id = f"{task_id}_{handoff_type}_s{seed}"

        # 1. Base Stage-1 scoring (measures end-to-end embedding, NER, feature extraction, and RF inference)
        import time as _t
        t0_s1 = _t.perf_counter()
        stage1_res = self.classifier.score(
            handoff=candidate_handoff,
            problem=problem,
            entry_point=entry_point,
            threshold=self.threshold,
        )
        stage1_latency_ms = (_t.perf_counter() - t0_s1) * 1e3
        stage1_score = stage1_res["score"]
        features = stage1_res["features"]

        total_latency_ms = 0.0
        total_p_tokens = 0
        total_c_tokens = 0

        decision = "PASS"
        action = "handoff"
        final_handoff_for_b = candidate_handoff
        is_raw_fallback = False

        # 2. Arm-specific gating decision and latency attribution
        if arm == "ungated":
            decision = "PASS"
            action = "handoff"
            # Ungated arm incurs zero gating latency

        elif arm == "stage1_only":
            total_latency_ms += stage1_latency_ms  # includes embedding, NER, all features, and RF scoring
            if stage1_score >= self.threshold:
                decision = "PASS"
                action = "handoff"
            else:
                decision = "BLOCK"
                action = "fallback_raw"
                is_raw_fallback = True

        elif arm == "cascade":
            total_latency_ms += stage1_latency_ms
            if stage1_score < self.threshold:
                # Confident block by Stage 1
                decision = "BLOCK"
                action = "fallback_raw_stage1"
                is_raw_fallback = True
            else:
                # Escalated to Stage 2 LLM Judge
                j_score, j_res = self.client.judge_handoff(
                    handoff=candidate_handoff,
                    problem=problem,
                    prompt_type="with_problem",
                    model=model_judge,
                    temperature=temp_judge,
                    seed=seed,
                )
                total_latency_ms += j_res.get("latency_ms", 0.0)
                total_p_tokens += j_res.get("prompt_tokens", 0)
                total_c_tokens += j_res.get("completion_tokens", 0)

                # Judge score threshold: 70/100
                if j_score is not None and j_score >= 70.0:
                    decision = "PASS"
                    action = "handoff_judge_pass"
                else:
                    decision = "BLOCK"
                    action = "fallback_raw_judge_block"
                    is_raw_fallback = True

        elif arm == "llm_judge":
            j_score, j_res = self.client.judge_handoff(
                handoff=candidate_handoff,
                problem=problem,
                prompt_type="with_problem",
                model=model_judge,
                temperature=temp_judge,
                seed=seed,
            )
            total_latency_ms += j_res.get("latency_ms", 0.0)
            total_p_tokens += j_res.get("prompt_tokens", 0)
            total_c_tokens += j_res.get("completion_tokens", 0)

            if j_score is not None and j_score >= 70.0:
                decision = "PASS"
                action = "handoff"
            else:
                decision = "BLOCK"
                action = "fallback_raw"
                is_raw_fallback = True

        elif arm == "always_fallback":
            decision = "BLOCK"
            action = "always_raw"
            is_raw_fallback = True

        elif arm == "agent_ask":
            needs_q, q_text, ask_res = self.client.agent_ask_clarify(
                handoff=candidate_handoff,
                problem=problem,
                model=model_judge,
                temperature=temp_judge,
                seed=seed,
            )
            total_latency_ms += ask_res.get("latency_ms", 0.0)
            total_p_tokens += ask_res.get("prompt_tokens", 0)
            total_c_tokens += ask_res.get("completion_tokens", 0)

            if not needs_q:
                decision = "PASS"
                action = "handoff"
            else:
                decision = "BLOCK"
                action = "clarified_retry"
                # Call Agent A to resolve clarification
                res_out = self.client.agent_ask_resolve(
                    handoff=candidate_handoff,
                    problem=problem,
                    question=q_text,
                    model=model_a,
                    temperature=temp_a,
                    seed=seed,
                )
                total_latency_ms += res_out.get("latency_ms", 0.0)
                total_p_tokens += res_out.get("prompt_tokens", 0)
                total_c_tokens += res_out.get("completion_tokens", 0)
                final_handoff_for_b = res_out.get("text", candidate_handoff)

        elif arm == "deterministic":
            # Exact signature check + length ratio >= 0.3
            has_sig = bool(features.get("function_name_preserved", 0.0) > 0.5)
            len_ok = bool(features.get("length_ratio", 0.0) >= 0.30)
            if has_sig and len_ok:
                decision = "PASS"
                action = "handoff"
            else:
                decision = "BLOCK"
                action = "fallback_raw"
                is_raw_fallback = True
        else:
            raise ValueError(f"Unknown arm: {arm}")

        # 3. Agent B Execution
        target_input = problem if is_raw_fallback else final_handoff_for_b
        b_res = self.client.code_agent_b(
            handoff_or_problem=target_input,
            entry_point=entry_point,
            model=model_b,
            temperature=temp_b,
            seed=seed,
            is_raw_problem=is_raw_fallback,
        )
        total_latency_ms += b_res.get("latency_ms", 0.0)
        total_p_tokens += b_res.get("prompt_tokens", 0)
        total_c_tokens += b_res.get("completion_tokens", 0)

        # 4. Grading generated code in sandbox
        passed, err_msg, grade_latency_ms = grade_code(
            code=b_res.get("text", ""),
            entry_point=entry_point,
            test_code=test_code,
            timeout_seconds=5.0,
        )
        total_latency_ms += grade_latency_ms
        outcome = 1 if passed else 0

        # Ground-truth label for handoff (1 if clean handoff passes without fallback, 0 otherwise)
        ground_truth_label = 1 if handoff_type == "clean" else 0

        return {
            "task_id": task_id,
            "source": source,
            "handoff_id": handoff_id,
            "handoff_type": handoff_type,
            "label": ground_truth_label,
            "label_policy": "downstream_pass1",
            "seed": seed,
            "features": features,
            "stage1_score": round(stage1_score, 4),
            "decision": decision,
            "outcome": outcome,
            "model_agent_a": model_a,
            "model_agent_b": model_b,
            "model_judge": model_judge,
            "temp_agent_a": temp_a,
            "temp_agent_b": temp_b,
            "temp_judge": temp_judge,
            "arm": arm,
            "latency_ms": round(total_latency_ms, 2),
            "tokens_prompt": total_p_tokens,
            "tokens_completion": total_c_tokens,
            "error_message": err_msg,
        }

