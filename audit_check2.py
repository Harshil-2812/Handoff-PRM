import os, pandas as pd

# Check all flagged audit/robustness files
print('=== Checking flagged files ===')
flagged = [
    'results/audit/overfitting_gap_audit.csv',
    'results/audit/bootstrap_confidence_intervals.csv',
    'results/robustness/held_out_corruption_eval.csv',
    'results/robustness/robustness_by_source.csv',
    'results/robustness/feature_ablation.csv',
    'results/tables/downstream_pass1_results.csv',
]
for f in flagged:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f'EXISTS: {f}  ({len(df)} rows, cols: {list(df.columns)[:5]})')
    else:
        print(f'NOT FOUND: {f}')
