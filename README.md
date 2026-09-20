# Handoff PRM

Handoff PRM is a small research pipeline for measuring whether an AI planner's
handoff gives a coding agent enough information to solve a task. It uses two
Gemini agents:

1. Agent A reads the original problem and writes a handoff/specification.
2. Agent B reads only that handoff and generates the implementation.
3. The implementation is automatically tested.
4. Passing handoffs are corrupted in several ways and sent to Agent B again.
5. Original handoffs are labeled `1`; corrupted handoffs that cause failure are
   labeled `0`.
6. Structural and semantic features are extracted and used to train an XGBoost
   classifier, which can score future handoffs before they reach Agent B.

## Requirements

- Python 3.10 or newer
- A Gemini API key with access to the configured models in `agents.py`
- Internet access for Gemini, Sentence Transformers, and spaCy model downloads

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a local `.env` file in the project root:

```text
GEMINI_API_KEY=your-gemini-api-key
```

`agents.py` loads this value with `python-dotenv` and `os.getenv`. The `.env`
file is ignored by Git. Never commit API keys. Because a key was previously
present in source history, rotate it before using this project publicly.

## Run The Pipeline

Run these commands from the project root, in order:

```powershell
# Generate and label rollouts
python rollout_runner.py

# Convert rollouts.csv into numeric features.csv
python build_dataset.py

# Train and save the classifier as handoff_prm.joblib
python train_classifier.py
```

The first command can take a long time because requests are rate-limited and
retried. It is resumable: completed task IDs are read from `rollouts.csv`, and
those tasks are skipped on later runs. If daily quota is exhausted, the run
stops without writing a partial task.

## Check Model Access

Before a long run, verify that the configured Gemini model IDs are available to
the API key:

```powershell
python -c "from agents import check_models; check_models()"
```

The two model names and their requests-per-minute limits are configured near
the top of `agents.py` as `MODEL_A`, `MODEL_B`, and `RPM_LIMITS`.

## Use The Gate

After `train_classifier.py` creates `handoff_prm.joblib`, score a handoff:

```python
from gate import HandoffGate

gate = HandoffGate(threshold=0.7)
result = gate.check(handoff_text, problem_text)

print(result["score"])
print(result["pass"])
if not result["pass"]:
    print(result["reason"])
```

The result contains the probability that the handoff is acceptable, the pass
decision, all extracted features, and, for blocked handoffs, the feature that
contributed most negatively to the score.

## Files

| File | Purpose |
|---|---|
| `agents.py` | Gemini client, rate limiting, retries, Agent A, and Agent B |
| `corruption.py` | Handoff corruption functions and corruption registry |
| `tasks/coding_tasks.py` | Coding tasks, entry-point names, and automated tests |
| `rollout_runner.py` | Runs original/corrupted rollouts and writes `rollouts.csv` |
| `rollouts.csv` | Resumable raw rollout dataset |
| `feature_extraction.py` | Computes cosine similarity, entity overlap, and length ratio |
| `build_dataset.py` | Converts `rollouts.csv` into `features.csv` |
| `features.csv` | Numeric labeled dataset used for training |
| `train_classifier.py` | Trains and evaluates the XGBoost classifier |
| `handoff_prm.joblib` | Saved model, SHAP explainer, and feature-column metadata |
| `gate.py` | Scores live handoffs and creates failure explanations |
| `.env` | Local Gemini API key; ignored by Git |
| `.gitignore` | Prevents local secrets and Python caches from being tracked |

## Dataset And Labels

Each task first needs a passing original rollout. Failed original rollouts are
discarded because a later corrupted failure cannot be attributed to the
corruption. For a passing original, each corruption is run independently:

- Original handoff: label `1`.
- Corrupted handoff that still passes: discarded as a weak negative.
- Corrupted handoff that fails: label `0`.

This means the CSV may contain fewer rows than the number of tasks multiplied
by five. It also means missing task IDs can indicate original failures or
incomplete runs, not necessarily a CSV-format problem.

## Features

The classifier currently uses three features:

- `cosine_similarity`: semantic similarity between the handoff and problem.
- `entity_overlap`: fraction of reference noun chunks and named entities found
  in the handoff.
- `length_ratio`: handoff word count divided by problem word count.

The Sentence Transformers model `all-MiniLM-L6-v2` is downloaded on first use.
The spaCy model `en_core_web_sm` must be installed during setup.

## Current Limitations

- The included dataset starts with only ten coding tasks.
- Corruption functions are heuristic and should be reviewed against real
  outputs.
- Training currently uses one stratified train/test split; cross-validation is
  preferable for a larger study.
- The gate is a standalone component. It is not yet wired into a LangGraph
  conditional workflow.
- The current feature set is structural and lightweight; it is not a complete
  semantic correctness model.

## Security

Keep `.env` local and do not commit it. If an API key is exposed, revoke or
rotate it immediately and replace the value in `.env`.
