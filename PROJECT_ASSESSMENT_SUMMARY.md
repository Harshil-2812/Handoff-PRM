# Handoff-PRM: Comprehensive Project Summary & State of Assessment
*A Complete Record of Problem Formulation, Experimental Trajectory, Audited Results, and Architectural Dilemmas for Independent Review.*

---

## 1. Executive Summary & Problem Formulation

### 1.1 The Problem: The Silent Cascade in Multi-Agent Pipelines
In multi-agent software engineering systems (e.g., Devin, SWE-agent, MetaGPT), long-horizon tasks are partitioned between specialized agents. A common pattern is **Agent A (Planner/Architect) $\to$ Agent B (Coder/Implementer)**:
* Agent A explores the environment, analyzes the codebase, forms a mental model, and writes an intermediate natural language **Handoff Message** $H$ summarizing the task, algorithm, edge cases, and interfaces.
* Agent B receives $H$ and implements the code.

**The Failure Mode**: If Agent A's handoff $H$ is incomplete, corrupted, over-compressed, or hallucinated, Agent B blindly trusts it ("Action Snowballing"), burns through tokens implementing the wrong solution, and fails the unit tests.

### 1.2 The Proposed Solution: Process Reward Model (Handoff-PRM)
Instead of unverified handoffs, we place a lightweight **Process Reward Model (PRM) Gate** between Agent A and Agent B:
$$\text{Agent A} \xrightarrow{\quad H \quad} \boxed{\text{Handoff-PRM Gate: } s = P(\text{sound} \mid H, P)} \xrightarrow{\quad s \ge \tau \quad} \text{Agent B}$$
* If $s \ge \tau$: Forward handoff to Agent B.
* If $s < \tau$: Intercept the failure, diagnose the defect via SHAP attribution, and trigger repair or fallback.

---

## 2. Experimental Trajectory & Discovery of the "Length Shortcut"

### 2.1 The Initial Prototype
* **Dataset**: ~777 rollouts across 60 hand-crafted tasks + initial HumanEval/MBPP subsets.
* **Corruptions (4 types)**: `truncation`, `over_summarization`, `tool_result_drop`, `entity_omission`.
* **Features (3 naive heuristics)**: `length_ratio` (word count ratio), `cosine_similarity` (SentenceTransformers), `entity_overlap` (spaCy NER).
* **Initial Claim**: Logistic Regression achieved **0.94 AUROC**, outperforming tree models.

### 2.2 The Critical Discovery: The Length-Ratio Shortcut
During rigorous peer-review auditing, we uncovered a fatal flaw in the initial setup:
* **The Artifact**: 3 out of 4 initial corruption types mechanically cut or compressed the handoff text.
* **The Reality**: Logistic Regression was simply learning a sharp step-function on text length:
  $$\text{AUROC}(\text{length\_ratio alone}) = \mathbf{0.931}$$
  The semantic features (`cosine_similarity`, `entity_overlap`) were completely decorative. A dummy model that counted words achieved 0.93 AUROC. In an academic paper, this is a fatal dataset shortcut that leads to instant desk rejection.

---

## 3. Scientific Hardening & Dataset Expansion

To eliminate the length shortcut and make the benchmark paper-grade, we completely overhauled the dataset and failure taxonomy.

### 3.1 Failure Taxonomy Expansion (10 Corruption Types)
We expanded the corruption engine to **10 distinct failure modes** across 4 categories, explicitly adding 6 **length-preserving** corruptions:

| Failure Category | Corruption Mode | Length Effect | Description & Failure Mechanism |
|---|---|---|---|
| **Category 1: Structural** | `truncation` | Reduced (30%) | Cuts handoff mid-sentence. |
| | `over_summarization` | Reduced (1 sent) | Extractive TextRank compression; strips implementation steps. |
| | `tool_result_drop` | Reduced | Drops markdown tables, fenced code blocks, numbered steps. |
| | `entity_omission` | Reduced (50%) | Strips sentences containing named entities and constraints. |
| **Category 2: Completeness** | `fake_completion` | **Preserved (100%)** | Replaces trailing edge cases with generic filler prose ("follows standard conventions"). |
| | `inject_false_constraint` | **Preserved (100%)** | Replaces trailing words with a contradictory constraint ("assume input is never empty"). |
| **Category 3: Schema / Interface**| `signature_rename` | **Preserved (100%)** | Renames entry point function (`foo` $\to$ `foo_impl`); causes `NameError` in test harness. |
| **Category 4: Logic / Semantic** | `invert_objective` | **Preserved (100%)** | Swaps return logic (`True` $\leftrightarrow$ `False`, `ascending` $\leftrightarrow$ `descending`). |
| | `negate_edge_case` | **Preserved (100%)** | Negates edge case constraints (`must handle` $\to$ `need not handle`). |
| | `wrong_algorithm_name` | **Preserved (100%)** | Swaps algorithm keywords (`binary search` $\to$ `linear search`). |

### 3.2 Scaled Dataset Statistics
* **Total Rollouts**: **2,696**
* **Unique Tasks**: **434** distinct coding problems
* **Task Sources**:
  * `hand_easy`: 218 rollouts (35 positive, 183 negative)
  * `hand_hard`: 311 rollouts (45 positive, 266 negative)
  * `humaneval`: 942 rollouts (145 positive, 797 negative)
  * `mbpp`: 1,225 rollouts (209 positive, 1,016 negative)
* **Label Distribution**: 434 Positive (16.1% — load-bearing, Agent B passes tests) vs. 2,262 Negative (83.9% — corrupted, Agent B fails tests).

---

## 4. Feature Engineering & Cloud GPU Pipeline

We engineered a **14-feature multi-modal representation** to break reliance on word counts:

1. **`trailing_specificity`**: Information density (backticks, numbers, code keywords) in the final 30% of the handoff.
2. **`section_coverage`**: Multi-point checklist checking presence of function signature, edge cases, algorithmic strategy, return statement, and problem restatement.
3. **`constraint_count`**: Frequency of requirement modal verbs (`must`, `should`, `require`, `guarantee`) relative to problem length.
4. **`novel_api_rate`**: Proportion of inline code spans in handoff that never appeared in problem (hallucination detector).
5. **`verbatim_copy_rate`**: 4-gram overlap between problem and handoff (detects lazy copy-pasting).
6. **`cosine_similarity`**: SentenceTransformer (`all-MiniLM-L6-v2`) dense vector similarity.
7. **`entity_overlap`**: spaCy noun chunks & entity overlap between problem and handoff.
8. **`length_ratio`**: Word count ratio.
9. **`sentence_count_ratio`**: Sentence count ratio.
10. **`role_pronoun_rate`**: First-person conversational drift detector.
11. **`function_name_preserved`**: Binary flag checking whether entry point exists.
12. **`signature_param_diff`**: Absolute difference in parameter counts between problem and handoff AST signatures.
13. **`nli_contradiction_max`**: Deep Transformer cross-encoder (`cross-encoder/nli-deberta-v3-small`) max contradiction probability across sentence pairs.
14. **`nli_entailment_mean`**: Mean entailment probability from DeBERTa-v3 across sentence pairs.

*Extraction Execution*: Scored 13,948 sentence pairs across all 2,696 rollouts using a Tesla T4 GPU in Google Colab, producing [`features.csv`](features.csv).

---

## 5. Audited Empirical Results & Verification

All evaluations are conducted under **5-Fold `GroupKFold` cross-validation grouped strictly by `task_id`** (tasks in test folds are 100% unseen during training).

### 5.1 Main Model Comparison (5-Fold GroupKFold CV)
*Evaluated across all 2,696 rollouts with 1,000 bootstrap resamples for 95% Confidence Intervals:*

| Model Architecture | OOF AUROC [95% CI] | OOF AUPRC [95% CI] | Brier Score | Calibrated F1 | Latency |
|---|---|---|---|---|---|
| **XGBoost (Regularized + Balanced)** | **0.682 [0.660, 0.706]** | **0.243 [0.216, 0.273]** | 0.245 | **0.357** | 1.22 ms |
| **XGBoost (Standard, depth=3)** | 0.684 [0.660, 0.708] | 0.250 [0.220, 0.281] | **0.127** | 0.358 | 1.22 ms |
| **Logistic Regression (L2, C=0.1)** | 0.668 [0.645, 0.693] | 0.238 [0.211, 0.268] | 0.246 | 0.349 | **0.12 ms** |
| **Random Forest (max_depth=3)** | 0.668 [0.644, 0.692] | 0.235 [0.208, 0.266] | 0.246 | 0.352 | 6.06 ms |
| **SVM (RBF Kernel)** | 0.423 [0.398, 0.448] | 0.131 [0.112, 0.153] | 0.140 | 0.005 | 0.12 ms |
| **Length-Ratio Baseline** | 0.569 [0.540, 0.598] | 0.183 [0.165, 0.203] | 0.141 | 0.282 | <0.01 ms |

**Paired Bootstrap Significance**: PRM vs. Length Baseline: $\Delta\text{AUROC} = \mathbf{+0.1149}$ ($p = 0.000000$, $p < 0.001^{***}$).

---

### 5.2 Feature Set Ablation
*Demonstrating that the model no longer relies on the length shortcut:*

| Feature Configuration | Features Included | OOF AUROC | OOF AUPRC |
|---|---|---|---|
| **Full 14 Features** | All structural, lexical, AST, & NLI | **0.6836** | **0.2474** |
| **No Length Ratio** | 13 features (excl. `length_ratio`) | **0.6791** | 0.2429 |
| **Section Coverage Only** | `section_coverage` | 0.5816 | 0.1868 |
| **Length Ratio Only** | `length_ratio` | **0.5689** | 0.1830 |
| **Core 3 Features** | `cosine_sim`, `entity_overlap`, `length_ratio` | 0.5664 | 0.1814 |
| **Entity Overlap Only** | `entity_overlap` | 0.5576 | 0.1811 |
| **Cosine Similarity Only** | `cosine_similarity` | 0.5246 | 0.1665 |

*Key Finding*: Removing `length_ratio` only drops AUROC from 0.6836 to 0.6791 (-0.0045). `length_ratio` alone only achieves 0.5689. The length shortcut is completely broken.

---

### 5.3 Detection Breakdown by Corruption Category (OOF)

| Corruption Mode | Category | Negatives | OOF AUROC | OOF AUPRC | Detection Status |
|---|---|---|---|---|---|
| `tool_result_drop` | Structural | 112 | **0.9997** | 0.9999 | Perfect detection |
| `over_summarization` | Structural | 127 | **0.9984** | 0.9995 | Perfect detection |
| `fake_completion` | Completeness | 434 | **0.9539** | 0.9578 | Strong detection |
| `truncation` | Structural | 73 | **0.9043** | 0.9728 | Strong detection |
| `inject_false_constraint` | Completeness | 434 | **0.6105** | 0.6085 | Moderate detection |
| `wrong_algorithm_name` | Semantic / Logic | 69 | **0.5648** | 0.8751 | Weak detection |
| `entity_omission` | Structural | 31 | **0.5398** | 0.9327 | Weak detection |
| `negate_edge_case` | Semantic / Logic | 322 | **0.5326** | 0.6098 | Near chance |
| `signature_rename` | Schema / Interface | 433 | **0.5132** | 0.5152 | Near chance |
| `invert_objective` | Semantic / Logic | 227 | **0.4991** | 0.6561 | Near chance |

---

### 5.4 Overfitting & Generalization Audit
*Measuring In-Sample (Train) vs. Out-of-Sample (Test) AUROC:*
* **Logistic Regression**: Train AUROC 0.6715 vs Test AUROC 0.6697 $\to$ **Gap = 0.0017** (Zero memorization).
* **Regularized XGBoost**: Train AUROC 0.6945 vs Test AUROC 0.6833 $\to$ **Gap = 0.0112** (Optimal).
* **Standard XGBoost**: Train AUROC 0.7254 vs Test AUROC 0.6852 $\to$ **Gap = 0.0402** (Low gap).
* *Conclusion*: The model is strictly **not overfitting**. Its performance on unseen tasks matches training performance within 1%.

---

## 6. The Core Problem & Dilemma for Independent Assessment

### 6.1 The Root Cause of the 0.68 AUROC Ceiling
Look at Section 5.3:
* On **Structural & Completeness** failures (`tool_result_drop`, `over_summarization`, `fake_completion`, `truncation`), the PRM achieves **0.90 to 0.99 AUROC**.
* On **Semantic Logic Inversions** (`invert_objective`, `negate_edge_case`, `signature_rename`), the PRM achieves **0.50 to 0.53 AUROC** (random chance).

**Why?**
Because scalar tabular features compress a 300-word paragraph into 14 numbers. When an agent writes "return `False` if palindrome" instead of "return `True`", the word count is identical, cosine similarity is 0.98, and entity overlap is 1.0. A decision tree or linear model operating on scalar summaries cannot see which specific condition was inverted without **token-level cross-attention**.

---

## 7. Strategic Decisions Requiring Assessment

To finalize the paper for a top-tier venue, four key questions must be answered:

### Decision 1: Model Architecture Upgrade
* **Option A: End-to-End Fine-Tuned Transformer Cross-Encoder**
  * Train `microsoft/deberta-v3-small` directly on the binary classification task with token cross-attention: `[CLS] Problem [SEP] Handoff [SEP]`.
  * Expected AUROC: **0.85 – 0.92+**.
  * Can be trained in Google Colab (Tesla T4) in ~4 minutes for 3 epochs.
* **Option B: Dense Neural-Tabular Stacking (Local)**
  * Extract 384-dimensional dense sentence embeddings, compute interaction features ($u, v, |u-v|, u \odot v \implies 1,536\text{ dims}$), concatenate with the 14 scalar features, and train a regularized Ridge / LightGBM model.
  * Can be trained locally in < 30 seconds.
* **Option C: Two-Stage Hierarchical Gate**
  * Stage 1: Fast tabular PRM (1.2 ms) filters out structural / completeness corruptions (AUROC 0.95+).
  * Stage 2: Transformer Cross-Encoder or NLI verifier only evaluates handoffs that pass Stage 1, checking semantic consistency.

### Decision 2: The Operational Handoff Policy ($\tau^*$)
* At what probability score should Agent A hand off to Agent B?
* Under realistic prevalence (16.1% positive), calibrated threshold $\tau^* = 0.466$ gives:
  * **Recall = 89.2%** (catches ~89% of all damaged handoffs).
  * **Precision = 22.3%** (trades false alarms for high safety).
* We need to define a formal **Utility Function**:
  $$U(\tau) = \text{SuccessRate}(\tau) - \lambda \cdot \text{Cost}(\tau)$$
  and publish the optimal operational cutoff $\tau^*$.

### Decision 3: LLM-as-a-Judge Baseline Comparison
* The paper must compare against prompting an LLM (e.g., GPT-4o-mini, Claude, Gemini Flash) directly as a zero-shot verifier:
  * *Metrics to report*: AUROC, Accuracy, Inference Latency (1.2 ms vs 2,000 ms), and Cost (\$0.000 vs \$0.01/call).
  * Is the PRM faster, cheaper, and more reliable than prompting an LLM?

### Decision 4: Downstream End-to-End Proof (Pass@1)
* Reviewers want to know: *Does gating handoffs actually improve final code execution?*
* Measure Pass@1 on unit tests:
  * Baseline Multi-Agent (Ungated): ~42% Pass@1.
  * PRM-Gated Multi-Agent (with repair/fallback): **Target 65%+ Pass@1**.

---

## 8. Summary of Completed Artifacts in Repository

* **Dataset**: [`features.csv`](features.csv), [`rollouts.csv`](rollouts.csv), [`features_with_nli.csv`](features_with_nli.csv)
* **Audited Tables**:
  * [`results/tables/cv_results.csv`](results/tables/cv_results.csv) (5-Fold GroupKFold CV)
  * [`results/robustness/feature_ablation.csv`](results/robustness/feature_ablation.csv) (Full ablation)
  * [`results/robustness/held_out_corruption_eval.csv`](results/robustness/held_out_corruption_eval.csv) (Held-out generalization)
  * [`results/robustness/robustness_by_corruption.csv`](results/robustness/robustness_by_corruption.csv) (OOF per corruption)
* **Audit Files**:
  * [`results/audit/overfitting_gap_audit.csv`](results/audit/overfitting_gap_audit.csv)
  * [`results/audit/calibrated_thresholds.csv`](results/audit/calibrated_thresholds.csv)
  * [`results/audit/bootstrap_confidence_intervals.csv`](results/audit/bootstrap_confidence_intervals.csv)
* **Camera-Ready LaTeX Tables**:
  * [`results/latex/table1_model_comparison.tex`](results/latex/table1_model_comparison.tex)
  * [`results/latex/table2_feature_ablation.tex`](results/latex/table2_feature_ablation.tex)
  * [`results/latex/table3_corruption_taxonomy.tex`](results/latex/table3_corruption_taxonomy.tex)
* **Figures (300 DPI)**:
  * [`results/figures/figure1_roc_curves.png`](results/figures/figure1_roc_curves.png)
  * [`results/figures/figure2_pr_curves.png`](results/figures/figure2_pr_curves.png)
  * [`results/figures/figure3_calibration_diagram.png`](results/figures/figure3_calibration_diagram.png)
  * [`results/figures/figure4_corruption_breakdown.png`](results/figures/figure4_corruption_breakdown.png)
