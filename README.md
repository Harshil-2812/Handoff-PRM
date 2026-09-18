# Handoff PRM — Process Reward Model for Inter-Agent Context Transfer

**Tagline**: *“Don’t grade the final output — grade the handoff.”*  
**Research Paper Report**: [FINAL_EXPERIMENT_REPORT.md](FINAL_EXPERIMENT_REPORT.md)  
**Master Specification**: [MASTER_RESEARCH_DOCUMENT.md](MASTER_RESEARCH_DOCUMENT.md)  

---

## 1. What the System Does

In multi-agent LLM systems (e.g., Agent A planner $\rightarrow$ Agent B coder), failure to transfer critical context during handoffs is a major source of overall system failure. Standard evaluation paradigms evaluate only the final code output after execution. 

**Handoff PRM** is an ultra-lightweight (<0.01 ms inference latency) Process Reward Model designed to grade the quality of intermediate handoff messages in real time before downstream execution.

1. **Agent A (Planner)** reads the problem statement and generates a handoff message/plan.
2. **Handoff PRM** extracts 3 fast features and computes a probability quality score $S$.
3. If $S \ge 0.330$, the handoff is **accepted** and passed to Agent B.
4. If $S < 0.330$, the handoff is **rejected**, and SHAP feature attribution returns targeted feedback hints to Agent A for closed-loop revision.
5. **Agent B (Coder)** receives *only* the handoff message (never the original problem directly) and writes Python code.

---

## 2. System Architecture

```
    Agent A (Planner)
           |
           | Handoff Message (H)
           v
    3-Feature Extractor (feature_extraction.py)
    - cosine_similarity (all-MiniLM-L6-v2)
    - entity_overlap (spaCy NER)
    - length_ratio (words_H / words_P)
           |
           v
    Handoff PRM Gate (gate.py)
          / \
         /   \
   S >= 0.330  S < 0.330
       |         |
    (PASS)    (REJECT + SHAP Feedback)
       |         |
       v         +---> Agent A Retry Loop
    Agent B (Coder)
       |
       v
    Code Execution & Unit Test Grade
```

---

## 3. Production Feature Representation

1. `cosine_similarity`: Cosine embedding distance between sentence-level embeddings of problem statement $P$ and handoff $H$ (`all-MiniLM-L6-v2`).
2. `entity_overlap`: Jaccard similarity of load-bearing identifiers (spaCy NER proper nouns, function signatures `def func(...)`, and constraint keywords).
3. `length_ratio`: Ratio of word count in handoff message to problem statement ($\text{Length}_H / \text{Length}_P$).

---

## 4. Dataset Summary

- **Total Rollouts**: **177**
- **Unique Coding Tasks**: **109**
- **Positive Examples (`label=1`)**: **109** (100% clean handoffs where Agent B passed unit tests)
- **Negative Examples (`label=0`)**: **68** (100% outcome-verified failures where corruption caused Agent B failure)
- **Discarded Weak Negatives**: **368** corrupted rollouts where Agent B passed despite corruption were explicitly filtered out.
- **Evaluation Splitting**: Evaluated under 5-Fold `GroupKFold` task-grouped cross-validation (zero task leakage across splits).

---

## 5. Model Results & Comparison

| Classifier | OOF AUROC ($\pm$ Std) | OOF AUPRC | Brier Score | Accuracy | F1 Score | Latency / Example | API Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9641 ($\pm$ 0.0453)** | **0.9715** | **0.0583** | **93.22%** | **0.9455** | **< 0.01 ms** | **$0.00** |
| **XGBoost Classifier** | 0.9535 ($\pm$ 0.0429) | 0.9455 | 0.0606 | 91.53% | 0.9321 | 0.04 ms | $0.00 |
| **Random Forest** | 0.9514 ($\pm$ 0.0411) | 0.9350 | 0.0636 | 93.22% | 0.9459 | 0.12 ms | $0.00 |
| **SVM (RBF Kernel)** | 0.9369 ($\pm$ 0.0574) | 0.9150 | 0.0553 | 93.79% | 0.9507 | 0.02 ms | $0.00 |
| *LLM-as-a-Judge Baseline* | 0.8640 | 0.8810 | N/A | 84.20% | 0.8510 | 3,933.00 ms | ~$0.0015 |

> **Note on Model Architecture & SHAP**:
> Logistic Regression is reported as the primary statistical model due to its top OOF AUROC (0.9641). The saved runtime bundle `results/models/prm_final.joblib` retains both Logistic Regression and XGBoost (0.9535 AUROC), with XGBoost powering `gate.py` to enable native `shap.TreeExplainer` feature attribution hints upon rejection.

---

## 6. Operating Threshold

- **Locked Operating Threshold**: **$\tau^* = 0.330$** (optimizes OOF F1 score to **0.9558**).
- **Rule**:
  - Score $\ge 0.330 \rightarrow$ **PASS** (Proceed to Agent B)
  - Score $< 0.330 \rightarrow$ **REJECT** (Generate SHAP feedback for Agent A revision)

---

## 7. Feature Ablation & Length Ratio Diagnostic

- **Full 3-Feature Model**: **0.9641 AUROC**
- **Without `length_ratio`**: **0.8801 AUROC** (drops by -8.40%)
- **`length_ratio` Alone**: **0.9630 AUROC**

### Diagnostic Insight & Limitation
`length_ratio` is an exceptionally strong linear indicator for coarse volumetric context compressions (ANOVA $F = 42.58$, $p = 8.66 \times 10^{-25}$, $\eta^2 = 0.4975$). On `over_summarization` and `tool_result_drop`, `length_ratio` alone achieves >0.999 AUROC. 

However, on subtle structural damage like `entity_omission`, `length_ratio` alone drops to **0.7810 AUROC**. Combining `length_ratio` with `entity_overlap` restores entity omission detection to **0.9576 AUROC**. 

*Limitation*: Length ratio is strongly predictive for benchmark corruptions, but real-world deployments where an LLM generates verbose but incorrect plans require `entity_overlap` and `cosine_similarity`.

---

## 8. Robustness Summary

- **By Corruption Category**:
  - `tool_result_drop`: 1.0000 AUROC
  - `over_summarization`: 1.0000 AUROC
  - `truncation`: 1.0000 AUROC (98.35% accuracy)
  - `entity_omission`: 0.9576 AUROC (95.73% accuracy)
- **By Task Benchmark Source**:
  - `hand_easy`: 1.0000 AUROC
  - `hand_hard`: 0.9975 AUROC
  - `humaneval`: 1.0000 AUROC
  - `mbpp`: 1.0000 AUROC

---

## 9. Live Multi-Agent Gating Experiment Results

Evaluated on 16 benchmark tasks across 3 pipeline conditions (`ungated`, `gated_blind`, `gated_shap`):

| Pipeline Condition | Task Count | Downstream Code Pass Rate | Avg PRM Score | Avg Retries Triggered | False Rejections |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ungated Baseline** | 16 | **68.75%** | 0.9038 | 0.0 | 0.0% |
| **Gated (Blind Retry)** | 16 | **68.75%** | 0.9038 | 0.0 | 0.0% |
| **Gated (SHAP Retry)** | 16 | **68.75%** | 0.9038 | 0.0 | 0.0% |

*Honest Assessment*: For clean initial generations produced from scratch by Agent A, mean PRM scores were **0.9038** (min 0.4976), cleanly exceeding $\tau^* = 0.330$. The gate registered **0 false rejections** and introduced zero token or latency overhead. On clean initial plans, the gating pilot did not change downstream pass rates because no retries were needed.

---

## 10. How to Reproduce

### 10.1 Local Reproduction (Zero Cost, No API Key Required)
Reproduce model training, calibration, diagnostics, figures, and unit test suite locally using existing outcome-grounded rollouts:

```bash
# 1. Install dependencies
pip install -r REQUIREMENTS.txt
python -m nltk.downloader punkt punkt_tab
python -m spacy download en_core_web_sm

# 2. Run length-ratio diagnostic & ablation (Phase 3)
python diagnose_length_ratio.py

# 3. Train models & compute 5-fold GroupKFold comparison (Phase 4)
python train_prm_final.py

# 4. Perform probability calibration & lock threshold (Phase 5)
python calibrate_prm.py

# 5. Run robustness analysis across corruptions & sources (Phase 9)
python analyze_robustness.py

# 6. Generate publication-ready figures (Phase 10)
python generate_figures.py

# 7. Run automated test suite (Phase 11)
python test_pipeline.py
```

### 10.2 Live LLM Rollout & Gating Execution (Requires Gemini API Key)
To run live Gemini Agent A/B rollouts or gating experiments:

```bash
# Copy placeholder env file and set your key
cp .env.example .env
# Edit .env: GEMINI_API_KEY=your_api_key_here

# Run live gating experiment
python run_gated_experiment.py
```

---

## 11. Project Directory Structure

```
handoff-prm/
├── README.md
├── REQUIREMENTS.txt
├── .env.example
├── .gitignore
├── FINAL_EXPERIMENT_REPORT.md
├── MASTER_RESEARCH_DOCUMENT.md
├── agents.py
├── corruption.py
├── feature_extraction.py
├── gate.py
├── rollout_runner.py
├── train_prm_final.py
├── calibrate_prm.py
├── diagnose_length_ratio.py
├── analyze_robustness.py
├── generate_figures.py
├── run_gated_experiment.py
├── test_pipeline.py
├── humaneval_tasks.py
├── mbpp_tasks.py
├── rollouts.csv
├── features.csv
├── tasks/
│   ├── all_tasks.py
│   └── coding_tasks.py
└── results/
    ├── calibration/
    │   ├── oof_predictions.csv
    │   └── threshold.json
    ├── figures/
    │   ├── ablation_study.png
    │   ├── calibration_curve.png
    │   ├── model_comparison_auroc.png
    │   ├── pr_curves.png
    │   ├── roc_curves.png
    │   └── shap_summary.png
    ├── gating/
    │   └── gated_experiment.csv
    ├── models/
    │   └── prm_final.joblib
    ├── robustness/
    │   ├── length_ratio_ablation.csv
    │   ├── length_ratio_diagnostic.csv
    │   ├── robustness_by_corruption.csv
    │   └── robustness_by_source.csv
    └── tables/
        ├── cv_results.csv
        └── gating_comparison.csv
```
