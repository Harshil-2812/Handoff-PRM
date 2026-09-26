import pandas as pd

# Check cv_results.csv vs model_comparison_results_12feat.csv
print('=== results/tables/cv_results.csv (FLAGGED) ===')
cv = pd.read_csv('results/tables/cv_results.csv')
print(cv[['model','oof_auroc']].to_string())

print()
print('=== model_comparison_results_12feat.csv (CORRECT - 12 features) ===')
mc = pd.read_csv('model_comparison_results_12feat.csv')
print(mc[['Model','OOF_AUROC_overall']].to_string())

print()
print('=== final_dataset_full_features.csv (ground truth) ===')
df = pd.read_csv('final_dataset_full_features.csv')
print('Rows:', len(df))
print('Label distribution:', df['label'].value_counts().to_dict())
print('Columns:', list(df.columns))
