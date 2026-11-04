# Inter-Run Agreement and Dual Labeling Policy Audit Report

**Workstream A**: Avaneesh N (Data and Organic Failures)
**Evaluation Seeds**: [42, 43, 44] (Triplicate Execution)
**Total Evaluated Samples**: 105
**Unique Tasks**: 35

## 1. Inter-Rater Reliability Metrics

- **Unanimous Agreement (3/3 or 0/3)**: 105/105 (100.00%)
- **Split Outcome Rate (2/3 or 1/3)**: 0/105 (0.00%)
- **Fleiss' Kappa ($\kappa$)**: `1.0000`
- **Mean Pairwise Cohen's Kappa**: `1.0000`
  - Run 1 vs Run 2: `1.0000`
  - Run 1 vs Run 3: `1.0000`
  - Run 2 vs Run 3: `1.0000`

## 2. Policy-Specific Class Distribution

| Policy | Total Rows | Positive (Label=1) | Negative (Label=0) | Discarded Rows | Notes |
|---|---|---|---|---|---|
| **Majority Vote (v0)** | 105 | 80 | 25 | 0 | Standard threshold (>=2 of 3) |
| **Discard Policy (v1)** | 105 | 80 | 25 | 0 | Ambiguous 1/3 and 2/3 dropped |
| **Robust Positive (v2)** | 105 | 80 | 25 | 0 | Only 3/3 treated as positive |

## 3. Breakdown by Agent A Model Tier

| Tier Name | Total | 3/3 Pass | 2/3 Pass | 1/3 Pass | 0/3 Pass | Majority Fail Rate |
|---|---|---|---|---|---|---|
| `tier1_weaker` | 35 | 27 | 0 | 0 | 8 | 22.9% |
| `tier2_mid` | 35 | 28 | 0 | 0 | 7 | 20.0% |
| `tier3_stronger` | 35 | 25 | 0 | 0 | 10 | 28.6% |
