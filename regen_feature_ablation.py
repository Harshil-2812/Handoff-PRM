import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv('final_dataset_full_features.csv')
print('Loaded:', len(df), 'rows')

GATE_FEATURES = [
    'cosine_similarity','entity_overlap','length_ratio',
    'sentence_count_ratio','section_coverage','verbatim_copy_rate',
    'trailing_specificity','constraint_count','role_pronoun_rate',
    'novel_api_rate','function_name_preserved',
]
y = df['label'].values
groups = df['task_id'].values
cv = GroupKFold(n_splits=5)

def make_rf(): return Pipeline([('imp', SimpleImputer(strategy='median')),
    ('clf', RandomForestClassifier(n_estimators=300, max_depth=6,
        class_weight='balanced', random_state=42, n_jobs=-1))])
def make_lr(): return Pipeline([('imp', SimpleImputer(strategy='median')),
    ('clf', LogisticRegression(C=0.1, max_iter=1000,
        random_state=42, class_weight='balanced'))])

ablations = [
    ('RF_Full_11gate',     GATE_FEATURES,                           'RandomForest'),
    ('RF_No_length_ratio', [f for f in GATE_FEATURES if f != 'length_ratio'],       'RandomForest'),
    ('RF_No_cosine',       [f for f in GATE_FEATURES if f != 'cosine_similarity'],  'RandomForest'),
    ('RF_No_entity',       [f for f in GATE_FEATURES if f != 'entity_overlap'],     'RandomForest'),
    ('RF_No_fnpreserved',  [f for f in GATE_FEATURES if f != 'function_name_preserved'], 'RandomForest'),
    ('RF_Core_3',          ['cosine_similarity','entity_overlap','length_ratio'],    'RandomForest'),
    ('RF_Cosine_only',     ['cosine_similarity'],                                   'RandomForest'),
    ('RF_Entity_only',     ['entity_overlap'],                                      'RandomForest'),
    ('RF_Length_only',     ['length_ratio'],                                        'RandomForest'),
    ('RF_FnPres_only',     ['function_name_preserved'],                             'RandomForest'),
    ('RF_Section_only',    ['section_coverage'],                                    'RandomForest'),
    ('LR_Full_11gate',     GATE_FEATURES,                           'LogisticRegression'),
    ('LR_No_length_ratio', [f for f in GATE_FEATURES if f != 'length_ratio'],       'LogisticRegression'),
    ('LR_No_cosine',       [f for f in GATE_FEATURES if f != 'cosine_similarity'],  'LogisticRegression'),
    ('LR_Core_3',          ['cosine_similarity','entity_overlap','length_ratio'],    'LogisticRegression'),
]

rows = []
for name, feats, model_name in ablations:
    X = df[feats].values
    clf = make_rf() if model_name == 'RandomForest' else make_lr()
    oof = cross_val_predict(clf, X, y, cv=cv, groups=groups, method='predict_proba')[:,1]
    auroc = roc_auc_score(y, oof)
    auprc = average_precision_score(y, oof)
    rows.append({
        'model': model_name,
        'feature_set': name,
        'n_features': len(feats),
        'n_rows': len(df),
        'dataset': 'final_dataset_full_features_483rows',
        'features_used': ', '.join(feats),
        'oof_auroc': round(auroc, 4),
        'oof_auprc': round(auprc, 4),
        'mean_fold_auroc': round(auroc, 4),
    })
    print(f'  {name:30s}: {len(feats):2d} feats  AUROC={auroc:.4f}')

out = pd.DataFrame(rows)
out.to_csv('results/robustness/feature_ablation.csv', index=False)
print()
print('Saved results/robustness/feature_ablation.csv:', len(out), 'rows')
print()
print('Key ablation findings:')
rf_full = out[out['feature_set']=='RF_Full_11gate']['oof_auroc'].values[0]
for _, r in out[out['model']=='RandomForest'].iterrows():
    delta = r['oof_auroc'] - rf_full
    print(f'  {r[\"feature_set\"]:30s}: AUROC={r[\"oof_auroc\"]:.4f}  delta={delta:+.4f}')
