import pandas as pd
from humaneval_tasks import HUMANEVAL_TASKS
from mbpp_tasks import MBPP_TASKS
from tasks.coding_tasks import TASKS as hand_tasks
from tasks.hard_tasks_3 import HARD_TASKS_3

ALL_TASKS = {t['id']: t for t in HUMANEVAL_TASKS + MBPP_TASKS + hand_tasks + HARD_TASKS_3}

pilot = pd.read_csv('live_pilot_results.csv')
pass_rows = pilot[pilot['gate_decision'] == 'PASS']
print('PASS rows:', len(pass_rows))

missing = [tid for tid in pass_rows['task_id'] if tid not in ALL_TASKS]
found   = [tid for tid in pass_rows['task_id'] if tid in ALL_TASKS]
print('Found in task pool:', len(found))
print('Missing from task pool:', len(missing))
if missing:
    print('Missing:', missing)
else:
    print('All PASS tasks found - safe to run!')
    for i, (_, row) in enumerate(pass_rows.iterrows()):
        t = ALL_TASKS[row['task_id']]
        print(str(i+1).zfill(2), row['task_id'], 'corrupted=' + str(row['is_corrupted']), 'ungated=' + str(row['ungated_passed']))
