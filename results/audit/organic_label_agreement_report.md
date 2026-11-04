# Inter-Run Agreement and Dual Labeling Policy Audit Report

**Workstream A**: Avaneesh N (Data and Organic Failures)
**Evaluation Seeds**: [42, 43, 44] (Triplicate Execution)
**Total Evaluated Samples**: 46
**Unique Tasks**: 20

## 1. Inter-Rater Reliability Metrics

- **Unanimous Agreement (3/3 or 0/3)**: 46/46 (100.00%)
- **Split Outcome Rate (2/3 or 1/3)**: 0/46 (0.00%)
- **Fleiss' Kappa ($\kappa$)**: `1.0000`
- **Mean Pairwise Cohen's Kappa**: `1.0000`
  - Run 1 vs Run 2: `1.0000`
  - Run 1 vs Run 3: `1.0000`
  - Run 2 vs Run 3: `1.0000`

## 2. Policy-Specific Class Distribution

| Policy | Total Rows | Positive (Label=1) | Negative (Label=0) | Discarded Rows | Notes |
|---|---|---|---|---|---|
| **Majority Vote (v0)** | 46 | 37 | 9 | 0 | Standard threshold (>=2 of 3) |
| **Discard Policy (v1)** | 46 | 37 | 9 | 0 | Ambiguous 1/3 and 2/3 dropped |
| **Robust Positive (v2)** | 46 | 37 | 9 | 0 | Only 3/3 treated as positive |

## 3. Breakdown by Agent A Model Tier

| Tier Name | Total | 3/3 Pass | 2/3 Pass | 1/3 Pass | 0/3 Pass | Majority Fail Rate |
|---|---|---|---|---|---|---|
| `tier1_weaker` | 20 | 16 | 0 | 0 | 4 | 20.0% |
| `tier2_mid` | 20 | 16 | 0 | 0 | 4 | 20.0% |
| `tier3_stronger` | 6 | 5 | 0 | 0 | 1 | 16.7% |
