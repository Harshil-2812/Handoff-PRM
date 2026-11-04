# Inter-Run Agreement and Dual Labeling Policy Audit Report

**Workstream A**: Avaneesh N (Data and Organic Failures)
**Evaluation Seeds**: [42, 43, 44] (Triplicate Execution)
**Total Evaluated Samples**: 122
**Unique Tasks**: 41

## 1. Inter-Rater Reliability Metrics

- **Unanimous Agreement (3/3 or 0/3)**: 119/122 (97.54%)
- **Split Outcome Rate (2/3 or 1/3)**: 3/122 (2.46%)
- **Fleiss' Kappa ($\kappa$)**: `0.9520`
- **Mean Pairwise Cohen's Kappa**: `0.9520`
  - Run 1 vs Run 2: `0.9759`
  - Run 1 vs Run 3: `0.9524`
  - Run 2 vs Run 3: `0.9277`

## 2. Policy-Specific Class Distribution

| Policy | Total Rows | Positive (Label=1) | Negative (Label=0) | Discarded Rows | Notes |
|---|---|---|---|---|---|
| **Majority Vote (v0)** | 122 | 95 | 27 | 0 | Standard threshold (>=2 of 3) |
| **Discard Policy (v1)** | 119 | 94 | 25 | 3 | Ambiguous 1/3 and 2/3 dropped |
| **Robust Positive (v2)** | 122 | 94 | 28 | 0 | Only 3/3 treated as positive |

## 3. Breakdown by Agent A Model Tier

| Tier Name | Total | 3/3 Pass | 2/3 Pass | 1/3 Pass | 0/3 Pass | Majority Fail Rate |
|---|---|---|---|---|---|---|
| `tier1_weaker` | 41 | 31 | 1 | 1 | 8 | 22.0% |
| `tier2_mid` | 41 | 34 | 0 | 0 | 7 | 17.1% |
| `tier3_stronger` | 40 | 29 | 0 | 1 | 10 | 27.5% |
