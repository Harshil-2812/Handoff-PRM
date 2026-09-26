import pandas as pd, os

pilot = pd.read_csv('live_pilot_results.csv')
print('=== live_pilot_results.csv ===')
print('Rows:', len(pilot))
print('Sources:', pilot['source'].value_counts().to_dict())
print('is_corrupted:', pilot['is_corrupted'].value_counts().to_dict())
print('Unique task_ids:', pilot['task_id'].nunique())
print()

log = 'live_pilot_run6.log'
size = os.path.getsize(log)
print('=== ' + log + ' ===')
print('Size bytes:', size)
with open(log, encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
print('Lines:', len(lines))
print('First 8 lines:')
for l in lines[:8]: print(' ', l.rstrip())
print('...')
print('Last 8 lines:')
for l in lines[-8:]: print(' ', l.rstrip())
