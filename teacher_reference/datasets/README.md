# Datasets — Handoff-PRM

This directory contains all datasets used in the paper.

## File Descriptions

### primary_dataset_177.csv
- **Paper**: Table II (original 177-row dataset)
- **Rows**: 177 (109 tasks × ~1.6 rollouts/task average)
- **Tasks**: 109 unique task_ids
- **Source**: Gemini-1.5-flash backbone, hand-crafted task pool (easy + hard)
- **Columns**:
  - `task_id`: unique task identifier
  - `source`: task pool origin (`hand_easy`, `hand_hard`)
  - `handoff_text`: the handoff message passed from Agent A to Agent B
  - `task`: the original problem statement
  - `corruption_type`: corruption applied (`clean`, `truncation`, `invert_objective`, etc.)
  - `label`: 1 = good handoff (Agent B passed), 0 = bad handoff (Agent B failed)
- **Label distribution**: ~60% positive (clean/harmless corruptions), 40% negative
- **Note**: This is the *original* dataset before cross-source expansion. Used in early ablation runs.

### rebuilt_dataset_483.csv
- **Paper**: Section IX, Table XIV (extended 483-row dataset)
- **Rows**: 483
- **Tasks**: 248 unique task_ids
- **Source**: Gemini-1.5-flash backbone, expanded to include HumanEval + MBPP conversions
- **Columns**: same as primary_dataset_177.csv + all 11 gate features pre-computed
- **Additional columns** (features):
  `cosine_similarity`, `entity_overlap`, `length_ratio`, `sentence_count_ratio`,
  `section_coverage`, `verbatim_copy_rate`, `trailing_specificity`, `constraint_count`,
  `role_pronoun_rate`, `novel_api_rate`, `function_name_preserved`
- **This is the dataset used for all main results** (Table VI, Table VIII, Table IX)

### qwen3b_dataset_169.csv
- **Paper**: Section IX (Cross-Backbone Evaluation)
- **Rows**: 169
- **Backbone**: Qwen/Qwen2.5-Coder-3B-Instruct
- **Used for**: cross-backbone transfer evaluation (Table XIV)
- **τ\***: 0.695 (higher than primary due to different score distribution)

### qwen7b_dataset_300.csv
- **Paper**: Section IX (Cross-Backbone Evaluation)
- **Rows**: 300
- **Backbone**: Qwen/Qwen2.5-Coder-7B-Instruct
- **Used for**: cross-backbone transfer evaluation (Table XIV)
- **τ\***: 0.7972

## task_pools/

Contains the raw task definitions used to generate rollouts.
Each file provides task dicts with:
- `task_id`: string identifier
- `problem`: full problem statement
- `entry_point`: function name expected by the test harness
- `test_cases`: list of test cases used for pass@1 verification

| File | Pool | n_tasks | Source |
|------|------|---------|--------|
| `hand_easy_tasks.py` | Hand-crafted easy | ~70 | Domain expert authored |
| `hand_hard_tasks.py` | Hand-crafted hard | ~40 | Domain expert authored, multi-constraint |
| `humaneval_tasks.py` | HumanEval converted | 164 | Chen et al. 2021, converted via `convert_humaneval.py` |
| `mbpp_tasks.py` | MBPP converted | 374 | Austin et al. 2021, converted via `convert_mbpp.py` |

### Verification counts (referenced in §IV-C)
- hand_easy: 30/30 tasks passed Agent B verification (100%)
- hand_hard: 30/30 tasks passed Agent B verification on clean handoffs (100%)
- humaneval_subset: 140/164 tasks used (24 excluded: trivial/ambiguous)
- mbpp_subset: 148/374 tasks used (226 excluded: verification failures or ambiguity)

## Column Schema

```
task_id          : str   — unique task identifier (e.g., "he_078", "mbpp_018")
source           : str   — pool origin: hand_easy | hand_hard | humaneval | mbpp
handoff_text     : str   — full handoff message from Agent A
task             : str   — original problem statement given to Agent A
corruption_type  : str   — clean | truncation | over_summarization | ... (10 types)
label            : int   — 1 (GOOD: Agent B passed) | 0 (BAD: Agent B failed)
```
