# Workstream C: Repository Audit & Status Report

**Author**: Vemisetty Harshil (Owner: Workstream C — Live End-to-End Evaluation)  
**Date**: November 6, 2026  
**Branch**: `workstream-c-live-eval` (branched from `origin/versioning-2`)  
**Target Environment**: Windows 10/11, Python 3.10.7 (`py -3.10`)  
**Source of Truth**: `experiment_workplan.md`

---

## 1. Executive Summary

This audit assesses the state of the `Handoff-PRM1` codebase prior to the implementation of **Workstream C** (Live End-to-End Evaluation). The audit cross-references the theoretical plan (`experiment_workplan.md`), the documented status of Workstream B (`results/workstream_B_results.md`), and the actual filesystem state.

### Key Audit Findings
1. **Workstream A Deliverables (Organic & Relabeled Data)**: `organic_dataset_v1.csv` and `dataset_v2_robust_positive.csv` are **MISSING** from the repository. As noted in Workstream B's report, these outputs from teammate Avaneesh N were blocked/pending. Workstream C will be architected with configurable paths and clean stubs to consume these files seamlessly once provided.
2. **Trained Stage-1 Model Serialization Incompatibility**: While model files `handoff_prm.joblib` and `results/models/prm_final.joblib` are present, both were pickled with Python 3.11/3.12 (code objects requiring 18 arguments). In Python 3.10 (the mandated target environment), loading them raises `TypeError: code expected at most 16 arguments, got 18`. However, the exact feature specification (`features_v2.json`, 11 features) and dataset (`features_v2.csv`, 483 rows) exist and train in <1 second via `common/data.py:make_rf()`.
3. **Threshold and Operating Point**: **FOUND** in `results/frozen_threshold.json` (`threshold_raw = 0.550419`, `threshold_platt = 0.475007`). This threshold was derived on out-of-fold dev predictions capping the clean false-block rate at $\le 5\%$.
4. **Task Manifests**: The explicit manifest files called for in Week 0 (`manifest_train_dev.txt`, `manifest_live_test.txt`, `manifest_organic.txt`) were never committed. However, `holdout_task_ids.txt` (94 tasks) exists and has **zero task ID overlap** with the 248 tasks in `features_v2.csv`. The repository contains 675 total tasks (427 untouched by the dev pool). LiveCodeBench tasks are not present in the repository.
5. **Results Schema Mismatch in Legacy Scripts**: Legacy pilot scripts (`live_downstream_pilot.py`, `live_pilot_cascade.py`) logged to an ad-hoc 16-column schema lacking seeds, model IDs, temperatures, and full feature maps. Workstream C will build a new harness strictly conforming to the frozen schema in `experiment_workplan.md`.
6. **Missing Script**: `verify_live_stats.py` does not exist in the repository and must be created.
7. **Dependencies**: `sumy`, `nltk`, `statsmodels`, `matplotlib`, and `joblib` were missing from `requirements.txt`. They have now been installed, verified in Python 3.10, and added to `requirements.txt`. All 33 existing unit tests pass.

---

## 2. Inventory of Repo Structure & Core Artifacts

```
Handoff-PRM1/
├── tasks/                      # Task pools (coding_tasks.py: 60, hard_tasks_3.py: 24, all_tasks.py: 675)
├── common/                     # Workstream B utilities (data.py, cv_utils.py, stats_utils.py, calib.py)
├── results/                    # Workstream B offline results, figures, tables, and frozen_threshold.json
│   └── .env                    # Contains GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3 (ignored by git)
├── tests/                      # Unit tests (test_stats_utils, test_function_name_preserved, etc.) - 33 pass
├── teacher_reference/          # Historical reference code and verification scripts
├── agents.py                   # Gemini client, rate limiting, Agent A and Agent B implementations
├── corruption.py               # 10 corruption functions across 4 categories
├── feature_extraction.py       # Structural, lexical, and semantic feature extractors
├── features_v2.csv / .json     # Frozen 483-row dev dataset and 11-feature list
├── holdout_task_ids.txt        # 94 held-out task IDs (disjoint from dev set)
├── b01_loto.py ... b09_*.py    # Workstream B analysis scripts
└── b06_llm_judge.py            # B6 LLM-judge implementation containing frozen prompt templates
```

---

## 3. Detailed Audit of Workstream A & B Deliverables

| Required Asset | Expected Location | Actual Status | Notes / Impact on Workstream C |
|---|---|---|---|
| `organic_dataset_v1.csv` (A1) | Repo root or `data/` | **MISSING** | Avaneesh has not yet delivered organic handoffs. C4 runner will expose `--organic-path` pointing to default/configured location. |
| `dataset_v2_robust_positive.csv` (A3) | Repo root or `data/` | **MISSING** | Robust-positive dataset not committed. |
| Frozen Results Schema (Week 0) | Defined in docs | **PARTIAL** | Specified in `experiment_workplan.md` (13 core concepts); past live scripts used legacy schema. C harness will strictly enforce frozen schema. |
| Task Manifests (Week 0) | Dedicated text/manifest files | **MISSING** | `holdout_task_ids.txt` exists (94 tasks). We will formalize manifest generation in `workstream_c/manifests/` with strict disjointness assertions. |
| Trained Stage-1 Classifier | `handoff_prm.joblib` | **PRESENT BUT PYTHON 3.10 INCOMPATIBLE** | Pickled with Python 3.11/3.12. Loading under Python 3.10 causes `code expected at most 16 arguments, got 18`. Retrainable deterministically from `features_v2.csv` via `common/data.py:make_rf()`. |
| Chosen Operating Threshold | `results/frozen_threshold.json` | **PRESENT** | Frozen threshold: `threshold_raw = 0.550419`, `threshold_platt = 0.475007`. Cap $\le 5\%$ false block on clean dev OOF. |
| LLM-Judge Prompts (B6) | `b06_llm_judge.py` | **PRESENT** | Prompts `generic`, `targeted`, and `with_problem` are frozen and ready for reuse in C3 arms. |
| Frozen Feature Set | `features_v2.json` | **PRESENT** | 11 surface features (NLI dropped following B8 analysis). |
| `verify_live_stats.py` | Repo root | **MISSING** | Must be implemented to compute Pass@1, exact McNemar test, and task-level paired bootstrap CIs. |

---

## 4. Verification of Environment & Dependencies

- **Python Runtime**: `Python 3.10.7` on Windows 64-bit.
- **Pip Dependencies**:
  - `requirements.txt` previously contained: `google-genai==2.20.0`, `python-dotenv>=1.0.0`, `sentence-transformers>=3.0.0`, `spacy>=3.7.0`, `xgboost>=2.0.0`, `scikit-learn>=1.4.0`, `pandas>=2.2.0`, `numpy>=1.26.0`, `shap>=0.45.0`, `tqdm>=4.66.0`.
  - Added and pinned: `sumy==0.13.0`, `nltk==3.10.4`, `statsmodels==0.15.0`, `matplotlib>=3.8.0`, `joblib>=1.3.0`.
  - Downloaded models: spaCy `en_core_web_sm` (v3.8.0), NLTK `punkt` and `punkt_tab`.
  - Tested: All core modules successfully imported.
- **Unit Tests**:
  - `pytest tests/`: **33 passed in 29.18s** (zero failures).
- **API Configuration**:
  - `results/.env` contains 3 API keys (`GEMINI_API_KEY_1`, `GEMINI_API_KEY_2`, `GEMINI_API_KEY_3`).
  - `agents.py` expects either `GEMINI_KEYS` (list) or `GEMINI_API_KEY` (string).
  - Workstream C harness will automatically read `.env` from root or `results/.env`, parse multiple key formats, and handle rotating key pools with quota backoff.

---

## 5. Task Pool Analysis for C2 Live Run

1. **Total Available Tasks**: 675 tasks across 4 files:
   - `tasks/coding_tasks.py`: 60 tasks (`hand_easy`: 30, `hand_hard`: 30)
   - `tasks/hard_tasks_3.py`: 24 tasks (`hand_hard`: 24)
   - `humaneval_tasks.py`: 164 tasks (`humaneval`)
   - `mbpp_tasks.py`: 427 tasks (`mbpp`)
2. **Train/Dev Tasks in `features_v2.csv`**: 248 tasks (`hand_easy`: 33, `hand_hard`: 44, `humaneval`: 80, `mbpp`: 91).
3. **Untouched Held-Out Tasks**: 427 tasks:
   - `mbpp`: 336
   - `humaneval`: 84
   - `hand_easy`: 4
   - `hand_hard`: 3
4. **Current Holdout Set (`holdout_task_ids.txt`)**: 94 tasks (12 HumanEval, 80 MBPP, 2 hand-crafted). Programmatic check confirmed **0 task overlap** with `features_v2.csv`.
5. **300-Task Target Feasibility**:
   - There are 427 untouched tasks available in the repo, allowing a held-out test set of 300 tasks (e.g. 84 HumanEval + 210 MBPP + 6 hand-crafted = 300 tasks) completely disjoint from the 248 train/dev tasks.
   - LiveCodeBench tasks are not present in the repository; incorporating them would require writing an external loader and verifying ground-truth execution. With 427 existing tasks available, 300 can be formed from HumanEval and MBPP.

---

## 6. Workstream C Architecture & Implementation Strategy

Workstream C will be implemented in a clean, isolated directory structure:
```
workstream_c/
├── AUDIT.md                          # This document
├── manifests/                        # Formal task manifests (train_dev, live_test, organic)
├── harness/
│   ├── __init__.py
│   ├── config.py                     # Schema definitions, model names, default paths
│   ├── key_manager.py                # Multi-key loading, rotation, quota management
│   ├── task_loader.py                # Manifest validation and task loading
│   ├── agents_interface.py           # Agent A, Agent B, and Judge calls with token/latency instrumentation
│   ├── feature_pipeline.py           # 11-feature extraction and Stage-1 RF model wrapper
│   ├── arms.py                       # Implementation of all 7 experimental arms
│   ├── execution_sandbox.py          # Secure subprocess/timeout Python test grader
│   └── logger.py                     # Incremental CSV writer adhering to frozen schema
├── run_live.py                       # Unified CLI runner for C2, C3, C4, C5, C6
├── results/                          # Pilot report, cost estimate, and experiment CSVs
└── figures/                          # Figure generation scripts and output plots
```

This architecture ensures:
1. Complete isolation from upstream code while strictly respecting frozen models, features, thresholds, and schemas.
2. Full Windows compatibility (`pathlib`, no shell-specific constructs).
3. Native support for multi-key Gemini rotation with rate-limit and daily-quota handling.
4. Resumable execution: results are written incrementally and can be safely interrupted and resumed.

---

## 7. Deliberate Protocol Choices & Experimental Settings

1. **Sampling Temperatures & Seed Variation**:
   - **Agent A (Planner)**: Set to `temperature = 0.7` across seeds 42, 43, and 44. This introduces natural semantic variation and diverse architectural plans across seeds while preserving high-quality task restatements.
   - **Agent B (Coder)**: Set to `temperature = 0.0` (greedy decoding) to isolate the causal effect of handoff quality on code generation and minimize downstream random execution noise.
   - **LLM Judge**: Set to `temperature = 0.0` (greedy decoding) for deterministic, calibrated scoring.
   - **Cache Invalidation**: The SHA-256 cache key in `agents_interface.py` hashes `model`, `temperature`, `seed`, and `prompt` (`f"{model}_{temperature}_{seed}_{prompt}"`). Therefore, pilot results generated at `temperature = 0.0` are never conflated with or reused for live production runs at `temperature = 0.7`.
2. **Task Evaluation Manifest & Composition**:
   - All 300 held-out tasks from `manifest_live_test.json` are evaluated (84 HumanEval, 210 MBPP, 6 hand-crafted), with 0 task overlap against the 248 train/dev tasks.
   - The ~70% MBPP task composition is documented as an experimental limitation in `RESULTS_SUMMARY.md`, with results reported both in aggregate and broken down by source.
3. **Partitioned API Key Pools & Concurrency**:
   - All 15 verified working keys in `.env` are partitioned into disjoint pools of 5 keys per seed (Process 1: Keys 1–5 for Seed 42; Process 2: Keys 6–10 for Seed 43; Process 3: Keys 11–15 for Seed 44).
   - Each process writes to an independent results CSV (`live_results_main_s42.csv`, `_s43.csv`, `_s44.csv`) and dedicated cache (`llm_cache_s42.json`, etc.) to eliminate concurrent rate-limit collisions and avoid file-write contention.


