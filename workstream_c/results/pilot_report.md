# Workstream C: 20-Task Live Pilot Report

**Date**: November 6, 2026  
**Harness Version**: `workstream_c/run_live.py` (Commit C1)  
**Evaluated Artifact**: `workstream_c/results/live_results_pilot.csv`  
**Execution Environment**: Windows Python 3.10.7, Live Gemini `gemini-3.5-flash-lite`

---

## 1. Executive Summary

A 20-task pilot evaluation was executed to validate the reconstructed live evaluation harness, check compliance with the frozen results schema, verify zero-leakage assertions, evaluate prompt-level disk caching and key rotation, and measure empirical token counts and latencies for downstream budget forecasting.

The pilot ran **280 live trials** (20 tasks $\times$ 1 seed (42) $\times$ 7 arms $\times$ 2 conditions: clean vs. corrupted) using genuine live API calls to Google Gemini (`gemini-3.5-flash-lite`) and real Python unit test grading in an isolated execution sandbox.

---

## 2. Validation Checklist

| Item | Requirement | Observed Status | Verification Method |
|---|---|---|---|
| **Frozen Schema Compliance** | All 17 frozen columns present and non-null | **PASS (100%)** | Programmatic assertion verified zero nulls in `task_id, source, handoff_id, handoff_type, label, label_policy, seed, features, stage1_score, decision, outcome, model_agent_a, model_agent_b, model_judge, temp_agent_a, temp_agent_b, temp_judge`. |
| **Disjoint Task Pools** | Zero overlap with train/dev pool (`features_v2.csv`) | **PASS (0 overlap)** | Programmatic set intersection between 20 pilot tasks and 248 train/dev tasks returned exactly 0 tasks. |
| **Task Balance** | Include multiple sources (not mostly MBPP) | **FAIL / UNBALANCED** | The 20-task pilot was 100% HumanEval only (`he_010` to `he_099`), omitting MBPP and hand-crafted sets entirely. Furthermore, the 300-task live manifest is ~70% MBPP (209/300) and contains no LiveCodeBench tasks. |
| **Paired Design** | Every task evaluated under both clean and corrupted conditions | **PASS** | Exactly 140 clean rows and 140 corrupted rows across 10 corruption types (`inject_false_constraint`, `truncation`, `signature_rename`, `wrong_algorithm_name`, etc.). |
| **Reproducibility** | Fixed seed produces identical outcomes | **PASS** | Seed 42 produces deterministic prompt inputs, feature extractions, and cached model queries. |
| **Token & Latency Logging** | Token counts and execution latency logged per trial | **PASS** | All trials logged prompt tokens, completion tokens, and end-to-end millisecond latency. |
| **Incremental Resume** | Re-running on existing CSV skips completed trials | **PASS** | Re-run test verified that existing completed `(task_id, arm, seed, handoff_type)` trials are skipped in 0 ms without re-spending quota. |

---

## 3. Empirical Results on Pilot Set (Pass@1)

### 3.1 Downstream Pass@1 by Arm

| Arm | Total Trials | Clean Pass@1 (n=20) | Corrupted Pass@1 (n=20) | **Overall Pass@1 (n=40)** | Relative $\Delta$ vs. Ungated |
|---|---|---|---|---|---|
| **ungated** | 40 | 75.0% | 65.0% | **70.0%** | Baseline |
| **stage1_only** | 40 | 75.0% | 75.0% | **75.0%** | **+5.0 pp** |
| **cascade** | 40 | 75.0% | 75.0% | **75.0%** | **+5.0 pp** |
| **deterministic** | 40 | 80.0% | 75.0% | **77.5%** | **+7.5 pp** |
| **llm_judge** | 40 | 75.0% | 70.0% | **72.5%** | **+2.5 pp** |
| **agent_ask** | 40 | 75.0% | 70.0% | **72.5%** | **+2.5 pp** |
| **always_fallback** | 40 | 80.0% | 80.0% | **80.0%** | **+10.0 pp** |

*Pilot Observation (Sanity Check Only)*: With only $n=20$ tasks and a single seed (42), these preliminary numbers serve strictly to verify end-to-end pipeline wiring, code grading, and logging. They are not statistically significant and cannot support generalizable conclusions. In this small sample, Stage-1 gating blocked some corrupted handoffs without false-blocking clean ones, and `always_fallback` showed that providing the full raw problem directly to Agent B achieves the highest raw pass rate.

---

## 4. Latency and Token Profiling

| Arm | Mean Latency (ms) | Mean Prompt Tokens | Mean Completion Tokens | Mean Total Tokens / Trial |
|---|---|---|---|---|
| **ungated** | 2,836.7 ms | 458.2 | 129.8 | 588.0 |
| **stage1_only** | 2,846.8 ms | 405.2 | 123.8 | 529.0 |
| **deterministic** | 3,140.9 ms | 304.4 | 141.1 | 445.5 |
| **cascade** | 4,253.7 ms | 770.6 | 125.6 | 896.2 |
| **llm_judge** | 4,941.8 ms | 997.4 | 132.4 | 1,129.8 |
| **agent_ask** | 6,643.1 ms | 1,452.0 | 307.9 | 1,759.9 |
| **always_fallback**| 2,951.6 ms | 159.0 | 133.6 | 292.6 |

*Pilot Profiling Notes*:
- **Stage-1 PRM Execution**: In this initial pilot run, Stage-1 Random Forest inference was measured; the harness has now been updated so that Stage-1 latency strictly includes end-to-end MiniLM embedding, spaCy NER parsing, feature computation, and RF scoring.
- **LLM Judge Overhead**: LLM judge evaluation introduces additional API roundtrip latency (~2.1s) and approximately doubles prompt token consumption.
- **AgentAsk Overhead**: Multi-turn prompted clarification incurs the highest token consumption (~1,760 tokens/trial) and latency (~6.6s) due to the two-step interactive prompt.

