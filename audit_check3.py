import pandas as pd

print('=== overfitting_gap_audit.csv ===')
df = pd.read_csv('results/audit/overfitting_gap_audit.csv')
print(df.to_string())

print()
print('=== bootstrap_confidence_intervals.csv ===')
df = pd.read_csv('results/audit/bootstrap_confidence_intervals.csv')
print(df.to_string())

print()
print('=== held_out_corruption_eval.csv ===')
df = pd.read_csv('results/robustness/held_out_corruption_eval.csv')
print(df[['held_out_corruption','train_size','n_pos_test','auroc']].to_string())

print()
print('=== robustness_by_source.csv ===')
df = pd.read_csv('results/robustness/robustness_by_source.csv')
print(df.to_string())

print()
print('=== downstream_pass1_results.csv ===')
df = pd.read_csv('results/tables/downstream_pass1_results.csv')
print(df.to_string())
