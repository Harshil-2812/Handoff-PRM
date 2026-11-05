# Gate 1 Evaluation Report: PRM Performance on Organic Multi-Agent Failures

**Workstream A**: Avaneesh N (Data & Organic Communication)
**Evaluated Model**: `results/models/prm_final.joblib` (XGBoost 12-Feature Surface Scanner)
**Total Samples**: 122 organic handoffs across 41 tasks
**Headline Threshold**: $\tau^* = 0.33$

## 1. Executive Summary & Gate 1 Decision

- **Primary Organic AUROC (Majority-Vote)**: **`0.5450`**
- **Gate 1 Pre-Registered Threshold**: `0.7500`
- **Gate 1 Status**: **NEGATIVE/MEASUREMENT FRAMING RECOMMENDED**
- **Natural Communication Failure Rate**: `22.1%` (27 failures / 122 samples)

## 2. Policy-Specific Performance Breakdown

| Labeling Policy | N | Pos | Neg | AUROC | AUPRC | Accuracy | F1 Score | Specificity (Interception) |
|---|---|---|---|---|---|---|---|---|
| **Majority-Vote Policy (v0)** | 122 | 95 | 27 | `0.5450` | `0.8099` | `0.221` | `0.000` | `100.0%` |
| **Discard Policy (v1)** | 119 | 94 | 25 | `0.5553` | `0.8210` | `0.210` | `0.000` | `100.0%` |
| **Robust-Positive Policy (v2)** | 122 | 94 | 28 | `0.5475` | `0.8057` | `0.230` | `0.000` | `100.0%` |

## 3. Confusion Matrix at Headline Threshold ($\tau^* = 0.330$)

### Majority-Vote Policy (v0)
- **True Positives (TP)**: 0 (Correctly allowed sufficient handoffs)
- **True Negatives (TN)**: 27 (Correctly intercepted organic failures)
- **False Positives (FP)**: 0 (Unnecessarily blocked sufficient handoffs)
- **False Negatives (FN)**: 95 (Missed failures allowed to Agent B)
- **Precision**: `0.000` | **Recall**: `0.000` | **NPV**: `0.221`

### Discard Policy (v1)
- **True Positives (TP)**: 0 (Correctly allowed sufficient handoffs)
- **True Negatives (TN)**: 25 (Correctly intercepted organic failures)
- **False Positives (FP)**: 0 (Unnecessarily blocked sufficient handoffs)
- **False Negatives (FN)**: 94 (Missed failures allowed to Agent B)
- **Precision**: `0.000` | **Recall**: `0.000` | **NPV**: `0.210`

### Robust-Positive Policy (v2)
- **True Positives (TP)**: 0 (Correctly allowed sufficient handoffs)
- **True Negatives (TN)**: 28 (Correctly intercepted organic failures)
- **False Positives (FP)**: 0 (Unnecessarily blocked sufficient handoffs)
- **False Negatives (FN)**: 94 (Missed failures allowed to Agent B)
- **Precision**: `0.000` | **Recall**: `0.000` | **NPV**: `0.230`

## 4. Breakdown Across Agent A Model Tiers

| Tier Name | Planner Model | Temp | Samples | Pos | Neg | Fail Rate | AUROC | Interception Rate |
|---|---|---|---|---|---|---|---|---|
| `tier1_weaker` | `gemini-3.5-flash-lite` | 0.9 | 41 | 32 | 9 | `22.0%` | `0.5260` | `100.0%` |
| `tier2_mid` | `gemini-3.5-flash-lite` | 0.7 | 41 | 34 | 7 | `17.1%` | `0.5042` | `100.0%` |
| `tier3_stronger` | `gemini-3.5-flash` | 0.7 | 40 | 29 | 11 | `27.5%` | `0.6066` | `100.0%` |
