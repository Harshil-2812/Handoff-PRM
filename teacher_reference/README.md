# Handoff-PRM — Teacher / Reviewer Reference Package

This directory is the **single point of truth** for a reviewer
checking the paper "Handoff-PRM: A Lightweight Process Reward Model
for Multi-Agent LLM Handoffs."

Every number, table, and claim in the paper maps to a file in this
package. The table below cross-references paper section → file.

---

## Directory Map

```
teacher_reference/
├── README.md                       ← this file
│
├── datasets/
│   ├── primary_dataset_177.csv     ← Table II (177 rows, 109 tasks)
│   ├── rebuilt_dataset_483.csv     ← Extended dataset (483 rows, 248 tasks)
│   ├── qwen3b_dataset_169.csv      ← Cross-backbone Qwen-3B dataset
│   ├── qwen7b_dataset_300.csv      ← Cross-backbone Qwen-7B dataset
│   └── task_pools/
│       ├── hand_easy_tasks.py      ← Symlink → tasks/coding_tasks.py (easy subset)
│       ├── hand_hard_tasks.py      ← Symlink → tasks/hard_tasks_3.py
│       ├── humaneval_tasks.py      ← Symlink → humaneval_tasks.py
│       └── mbpp_tasks.py           ← Symlink → mbpp_tasks.py
│
├── code/
│   ├── corruption_engine.py        ← All 10 corruption functions (§IV)
│   ├── feature_extraction.py       ← 3/7/12-feature extractors (§V)
│   ├── label_generation.py         ← Algorithm 1: contrastive labeling (§IV-D)
│   ├── train_models.py             ← LR/XGB/RF/MLP, GroupKFold (§VI)
│   ├── cascade_gate.py             ← Algorithm 2: Stage-1 + Stage-2 LLM-judge (§VII)
│   └── langgraph_pipeline/
│       ├── graph.py                ← LangGraph workflow with conditional edge
│       └── nodes.py                ← Agent A / Gate / Agent B node definitions
│
├── models/
│   ├── handoff_prm.joblib          ← Primary trained model (Gemini backbone)
│   ├── handoff_prm_gate_qwen3b.joblib  ← Qwen-3B variant
│   ├── handoff_prm_gate_qwen7b.joblib  ← Qwen-7B variant
│   └── threshold.json              ← Calibrated τ* values (§VI-D)
│
├── metrics/
│   ├── cv_results_primary.json     ← Table VI: OOF AUROC/AUPRC/Brier/logloss/F1
│   ├── ablation_results.csv        ← Table VIII: feature ablation
│   ├── anova_results.txt           ← F(3,173)=42.58 ANOVA output (§VI-C)
│   ├── confusion_matrix_primary.json  ← Table IX
│   ├── shap_values.csv             ← SHAP analyses (primary + Qwen)
│   ├── cascade_results.json        ← Section X cascade tables
│   └── bootstrap_ci.json           ← Bootstrap 95% CIs
│
└── live_gating_logs/
    ├── run_gemini_primary.log      ← Primary live run log (94 tasks)
    ├── run_qwen3b.log              ← Qwen-3B gating run
    └── run_qwen7b.log              ← Qwen-7B gating run
```

---

## Claim → File Cross-Reference

| Paper claim | Value | File |
|---|---|---|
| Pass@1 all tasks: ungated→gated | 55.3% → 67.0% (+11.7 pp) | `metrics/cv_results_primary.json`, `live_gating_logs/run_gemini_primary.log` |
| McNemar p-value | p = 0.0074 | `metrics/cascade_results.json` |
| Bootstrap 95% CI | [+4.3, +19.1] pp | `metrics/bootstrap_ci.json` |
| Primary AUROC (RF) | 0.8718 | `metrics/cv_results_primary.json` |
| Primary AUROC (XGB) | 0.8733 | `metrics/cv_results_primary.json` |
| τ* primary | 0.330 | `models/threshold.json` |
| τ* Qwen-3B | 0.695 | `models/threshold.json` |
| τ* Qwen-7B | 0.7972 | `models/threshold.json` |
| F(3,173) = 42.58 | ANOVA across corruption types | `metrics/anova_results.txt` |
| Stage-1 handles 74% of blocks | 35/47 blocks | `metrics/cascade_results.json` |
| Corrupted-task recovery | 75% (30/40 intercepted) | `live_gating_logs/run_gemini_primary.log` |
| 483-row dataset, 248 tasks | Extended (§IX) | `datasets/rebuilt_dataset_483.csv` |
| Qwen cross-backbone AUROC | See Table XIV | `datasets/qwen3b_dataset_169.csv`, `datasets/qwen7b_dataset_300.csv` |

---

## Reproducibility Checklist

- [ ] All datasets are CSVs with `task_id`, `handoff_text`, `task`, `corruption_type`, `label`
- [ ] `code/corruption_engine.py` is identical to `../corruption.py` (tracked by git)
- [ ] `code/feature_extraction.py` is identical to `../feature_extraction.py`
- [ ] `models/handoff_prm.joblib` SHA-256 matches value in `metrics/cv_results_primary.json`
- [ ] `models/threshold.json` values reproduce when running `code/train_models.py` with `seed=42`
- [ ] Live logs in `live_gating_logs/` are raw output files (not edited post-run)

---

## Quick Verification Commands

```bash
# Re-run offline evaluation (should reproduce Table VI numbers)
cd ..
python model_sweep_12features_final.py

# Regenerate ablation table (Table VIII)
python regen_ablation.py

# Re-run ANOVA
python paper_metrics.py --section anova

# Verify cascade (Section X)
python two_stage_cascade.py
```
