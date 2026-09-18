# Handoff PRM — Master Research Document

**Purpose of this file:** a complete, self-contained record of everything
done on this project so far — motivation, related work, methodology,
architecture, every dataset and every result — sufficient for someone with
zero prior context to draft a research paper up to this point without
needing to ask what was done or why. All numbers below are real, measured
results from actual runs, not projections or placeholders, unless
explicitly marked "not yet run."

---

## 1. Title and One-Line Pitch

**Handoff PRM: A Lightweight Process Reward Model for Scoring Inter-Agent
Context Transfer in Multi-Agent LLM Systems**

*Don't grade the final output — grade the handoff.*

Field: Multi-Agent Systems, Reward Modeling, Trustworthy AI.

---

## 2. The Problem

Multi-agent LLM pipelines (Agent A → Agent B → Result) fail often. Per
**MAST** (Cemri et al., 2025, arXiv 2503.13657), roughly 37-44% of
multi-agent system failures are due to inter-agent misalignment — not weak
underlying models. Agent A completes part of a task and hands information
to Agent B; this handoff is frequently incomplete or poorly structured.

Two specific failure patterns motivate the project:
- **Full context dump**: Agent A forwards everything it knows; important
  information gets buried and Agent B suffers "lost in the middle" attention
  degradation over long context.
- **Over-summarization**: Agent A compresses too aggressively, silently
  dropping reasoning, variables, tool results, or evidence Agent B needs.

**The gap:** there is currently no real-time signal that tells a running
pipeline "this handoff was insufficient" before the downstream agent wastes
a step acting on it. Existing tooling (LangSmith, observability platforms)
only allows post-hoc log inspection after a failure has already occurred.

---

## 3. Proposed Solution

A small, fast classifier — the **Handoff PRM** — sits between Agent A and
Agent B. It scores whether a handoff message contains sufficient
information for Agent B to succeed. If the score falls below a threshold,
the pipeline blocks in real time and Agent A is asked to retry with
additional context, instead of letting Agent B proceed and fail.

Training labels are generated **automatically, with zero human
annotation**, via **contrastive corruption**: take a real handoff that led
to task success, deliberately and deterministically damage it, and check
whether Agent B now fails on the same task using only the damaged handoff.

---

## 4. Related Work and Precise Positioning

| Paper | Core contribution | The gap it leaves open |
|---|---|---|
| **MAST** (Cemri et al. 2025, arXiv 2503.13657) | First large-scale empirically-grounded taxonomy of MAS failures (14 modes across 3 clusters: Specification, Inter-Agent Misalignment, Verification). Introduces **MAST-Data** (large trace dataset, 7 MAS architectures × 4 model families) and **MAST-Data-human** (human-annotated subset with inter-annotator agreement study). Uses an **LLM-as-judge annotation pipeline** (OpenAI's **o1** model, calibrated against human labels at ~92-94% accuracy / Cohen's κ ≈ 0.77-0.88) to scale labeling of already-collected failed traces against the fixed 14-category taxonomy. Purely diagnostic/retrospective. | No real-time prevention mechanism. Diagnoses failures after they happen; does not predict or intervene before a specific handoff causes a specific failure. |
| **AgentPRM** (arXiv 2502.10325, same paper as InversePRM) | Process reward modeling for a *single* agent's own step-by-step decisions, via actor-critic framework with Monte Carlo rollout groups (multiple sampled continuations per state, averaged) to estimate Q(s,a). | Never evaluates what's passed *between* different agents — only evaluates one agent's own internal reasoning/decision quality. |
| **InversePRM** (same paper as AgentPRM) | Learns process rewards **from expert demonstrations, explicitly WITHOUT outcome supervision** — closer to inverse RL (infers the reward function that would explain expert behavior). | Methodologically opposite to our approach: we rely entirely on outcome supervision (pass/fail after corruption). Cite as contrast, not as support for outcome-based labeling — this was a real correction made during this project (an earlier informal analysis of the project mischaracterized InversePRM as supporting outcome-based labeling; it does the opposite). |
| **MASPRM** (arXiv 2510.24803) | Scores **routed inter-agent transcripts** — i.e., handoff-level scoring already exists in prior work; do not claim novelty on "no one scores handoffs." Trained via **MCTS rollouts**: many sampled branching continuations per state, final outcome backed up to score intermediate nodes. Used as a real-time inference-time controller (beam search / MCTS pruning). Tested on Qwen2.5-1.5B-Instruct, one fixed architecture, benchmarks GSM8K/MATH/MMLU/LOGIQA. | Needs expensive MCTS rollout supervision (requires a resimulate-able environment, many rollouts per data point). Heavier transcript-level representations at inference. Single fixed architecture and model tested. |
| **CARA** (arXiv 2605.20548, "Category-Aware Recovery Augmentation") | Manually discovers 5 semantic content categories in agent messages via open-coding on 400 human-sampled traces (Answer, Reasoning, Verification, Reference, Unchanged), scaled to 41,330 total traces via GPT-4o LLM-as-judge annotation (92% accuracy, κ=0.88 vs. human labels). Uses **leave-one-out occlusion** (replacing removed spans with `[MASK]` to preserve structure) across 5 MAS architectures (Seq-U, Seq-R, Debate, CR-MC, CR-SV) × 2 LLM backbones (Qwen2.5-Instruct, Qwen2.5-Coder) × 6 benchmarks to find which categories are load-bearing. Recovers failures via **CARA**: prompt augmentation requiring specific categories + **rule-based keyword-presence checking** (looks for literal labels like "Reasoning:", "Verification:") with up to 3 regeneration retries. Reports up to 86.2% recovery of TF (originally-succeeded-then-failed-after-occlusion) cases in best-case configurations; also measures execution time/token cost overhead of the intervention. | Needs a hand-built, fixed taxonomy of content categories decided in advance. Recovery mechanism is **binary rule-based keyword presence**, not a learned, continuous score. Cannot generalize to information-sufficiency signals outside its 5 predefined categories. |
| **Agent-Radar** (arXiv 2605.30136) | **Training-free**: steers agent attention toward relevant context using semantic relevance + spatial (agent-graph distance) + temporal decay, directly targeting the "lost in the middle" problem in MAS. | Doesn't score or block anything — re-weights attention only. No real-time gating decision, no explicit sufficiency judgment. Useful as a training-free baseline to compare our gated approach against. |

**Exact novelty statement (use this near-verbatim in the paper's
introduction/related work close):**

> "MASPRM proved handoff-level scoring is possible but needs expensive MCTS
> rollout supervision and heavier transcript-level representations. CARA
> proved that missing content categories causally drive failure, but
> recovers them via a fixed, hand-built taxonomy and rule-based keyword
> checking. We show a lightweight, continuously-scored alternative — a
> classifier trained via cheap, deterministic contrastive corruption
> labeling (not MCTS, not a manual taxonomy), using structural features
> (embedding similarity, entity overlap, length ratio) cheap enough for
> real-time production gating, with SHAP-based failure explanations fed
> back to the upstream agent."

**Explicit instruction: do NOT claim "no prior work studies agent
handoffs"** — this is false (MASPRM and CARA both do) and will be caught
immediately by any reviewer familiar with the space. The novelty is
specifically: cheap + deterministic + continuous + feature-based + no
fixed taxonomy needed, versus MASPRM's expensive/MCTS-based and CARA's
fixed-taxonomy/binary-recovery approaches.

**A third, less central comparison:** Math-Shepherd (2023) — "Verify and
Reinforce LLMs Step-by-step without Human Annotations" — established the
general principle of automated rollout-based labeling for intermediate
process steps (not agent-to-agent communication specifically). Useful as
a precedent for "automated outcome-based labeling is a legitimate
methodology," not a direct competitor.

---

## 5. Architecture

### 5.1 Real-time inference/gating flow (PlantUML)
```plantuml
@startuml
title Handoff PRM - Real-Time Gating Architecture
skinparam backgroundColor white
skinparam shadowing false
skinparam componentStyle rectangle

rectangle "Agent A\n(Upstream)" as A
rectangle "Handoff Message" as H
rectangle "Feature Extraction\n(Cosine Sim | Entity Overlap | Length Ratio)" as F
rectangle "Handoff PRM\n(XGBoost)" as PRM
diamond "Score >=\nThreshold?" as D
rectangle "Agent B\n(Downstream)" as B
rectangle "Re-request\nContext + SHAP reason" as R

A --> H
H --> F
F --> PRM
PRM --> D
D --> B : YES
D --> R : NO
R --> A
@enduml
```

### 5.2 Offline training / label generation flow (PlantUML)
```plantuml
@startuml
title Handoff PRM - Model Training Pipeline (Network View)
skinparam backgroundColor white
skinparam shadowing false
left to right direction
skinparam node { BackgroundColor #F5F5F5  BorderColor #333333 }
skinparam database { BackgroundColor #EAF2FF  BorderColor #333333 }
skinparam package { BackgroundColor #FFFFFF  BorderColor #888888 }

cloud "Benchmark\nTasks" as TASK
package "1. Rollout Generation" {
  node "Agent A" as A
  node "Agent B" as B
  database "Successful\nHandoffs" as SUCC
}
package "2. Contrastive Labeling" {
  node "Corruption\nEngine" as CORR
  node "Re-run\nAgent B" as RERUN
  database "Label\n0 / 1" as LBL
}
package "3. Train Model" {
  node "Feature\nExtractor" as FE
  node "XGBoost\nTraining" as TRAIN
  node "Handoff PRM\n(trained)" as MODEL
}
TASK --> A
A --> B : handoff
B --> SUCC : pass
SUCC --> CORR
CORR --> RERUN
RERUN --> LBL : fail = 0
SUCC --> LBL : original = 1
LBL --> FE
FE --> TRAIN
TRAIN --> MODEL
@enduml
```

### 5.3 Known gap between architecture diagram and actual running code
**The diagrams describe the target architecture. The code as actually run
does NOT yet use LangGraph** — the pipeline currently executes as plain,
sequential Python function calls (`agent_a_plan()` then `agent_b_code()`
called directly), not as a LangGraph graph with a conditional edge for
gating. Wiring the gate into an actual LangGraph conditional edge remains
unfinished work (see Section 12, Remaining Work).

---

## 6. Methodology

### 6.1 Automatic label generation procedure
1. Run Agent A → Agent B on a task. Grade with an objective, automatic
   signal — unit tests for code tasks (exact-match against ground truth
   for a planned but not-yet-built SQL secondary task).
2. If the ORIGINAL (uncorrupted) handoff already fails, **discard the task
   entirely.** A failure with nothing to compare against cannot be
   attributed to the handoff — this is a deliberate methodological choice,
   not an oversight.
3. If the original succeeds, label that handoff `1` (sufficient). Apply
   corruption to it.
4. Re-run Agent B on the corrupted handoff. If it now fails, label the
   corrupted version `0` (insufficient). If it still passes, **discard**
   (a "weak negative" — the corruption didn't damage anything that
   mattered for this task).

This produces automatically-labeled (handoff, sufficiency) pairs with zero
human or LLM judgment involved anywhere in the labeling process — a
stronger "no manual annotation" claim than even MAST's own LLM-as-judge
approach, since MAST still requires an LLM's judgment call as the actual
source of each label, whereas this project's labels come purely from
re-executing the downstream agent and checking an objective, automatic
outcome.

### 6.2 Why corruption is deterministic, NOT LLM-generated
An earlier design direction considered generating corrupted handoffs by
prompting an LLM to compress/rewrite them (to mimic real-world context
compression more faithfully). **This was deliberately rejected** for the
following reasons, arrived at through explicit reasoning during this
project:
1. **Confound avoidance**: an LLM-rewritten corruption conflates "missing
   information" with "a second LLM's rewriting noise/errors" — breaking
   the clean causal story that a corruption's effect can be attributed
   solely to what was removed.
2. **Reproducibility**: LLM generation is not guaranteed bit-identical
   across runs, even at low temperature; deterministic functions guarantee
   the same corrupted output for the same input every time.
3. **Precedent**: CARA itself uses deterministic, structure-preserving
   ablation (`[MASK]` token replacement) for exactly this reason, rather
   than model-generated rewriting — this project follows that precedent
   explicitly.
4. **Cost/speed**: no extra API/model calls per corruption, which also
   avoids compounding rate-limit pressure during data generation.

### 6.3 The four corruption types — design, diagnosis, and fixes

| Type | Mechanism | History |
|---|---|---|
| **Truncation** | Word-boundary cutoff, keeping the first `keep_fraction` of words | Originally `keep_fraction=0.5`, cutting at sentence boundaries. **Diagnosed as too gentle** by inspecting real examples: Agent A's handoff format consistently front-loads the function signature and problem restatement, so a 50% cutoff preserved essentially all load-bearing content and only trimmed a disposable "proposed approach" section. **Fixed**: lowered to `keep_fraction=0.3`, cutoff changed to word-boundary (not sentence-respecting) to more faithfully mirror real context-window truncation, which cuts mid-sentence in production systems. |
| **Entity omission** | Removes sentences flagged as containing important content | Originally used spaCy's `like_num` (flags any sentence containing a number) plus a paren-detection heuristic. **Diagnosed as non-selective** by inspecting real examples: on math/algorithm tasks, nearly every sentence contains a number (e.g., `n <= 1`, step numbers), so the filter approximated random deletion rather than targeting genuinely critical content. **Fixed**: now targets (a) real spaCy named entities excluding bare CARDINAL/ORDINAL/QUANTITY numbers, (b) sentences matching a function-signature regex pattern, (c) sentences containing edge-case language ("must", "should", "require", "edge case"). |
| **Structured-section removal** (internally named `tool_result_drop` in code — **flagged as a misleading name, should be renamed in the paper**) | Strips numbered/bulleted lines and fenced/inline code spans | Confirmed via direct inspection to be highly effective (34-55% break rates) — but investigation revealed it isn't actually simulating "dropped tool results" (the current task domain has no real external tool calls). It works because Agent A consistently formats the function signature and edge-case list as numbered/bulleted sections, so this corruption strips the signature and edge cases wholesale. **Recommended reframing for the paper**: call this "structured-section removal," and note that a genuine tool-result-dropping corruption would require a task domain with real tool calls (a stretch-goal future work item, not yet built). |
| **Over-summarization** | TextRank extractive summarization down to 1 sentence (deterministic graph-based algorithm, not neural/LLM-based) | Originally kept only the literal first sentence. **Fixed** to TextRank (via the `sumy` library) — picks the most "central" sentence(s) by graph centrality rather than always the first, more faithfully mirroring real automatic conversation-summarization nodes used in production agent memory systems (still deterministic, no LLM call). |

### 6.4 Verified before/after impact of the fixes (real measured data)

Using the hard/information-dense task batch as the comparison point (most
sensitive to corruption, so where a fix should show the clearest effect):

| Corruption type | Before fix | After fix |
|---|---|---|
| Truncation | ~4% break rate | ~13-17% break rate |
| Entity omission | ~6% break rate | ~10-14% break rate |
| Structured-section removal | 34-55% (unchanged, already effective) | unchanged |
| Over-summarization | 42-69% (unchanged, already effective) | unchanged |

Both fixes were validated as genuinely working, not just theoretically
better — this before/after measurement is itself a citable methodological
result, demonstrating the corruption engine was empirically tuned rather
than assumed correct.

### 6.5 Known remaining methodological limitation (state honestly in the paper)
**Single-sample-per-datapoint**: unlike MASPRM/AgentPRM, which average
outcomes over multiple sampled rollouts per state (via MCTS or Monte Carlo
groups), this project's labels come from a single execution per
(task, corruption) pair. This means a label could in principle reflect
one unlucky/lucky generation rather than a stable underlying tendency.
A cheap partial mitigation (re-running each corrupted variant 2-3 times
and majority-voting the label) was identified but **not yet implemented**.

---

## 7. Task Design and Data Sources

### 7.1 Hand-crafted tasks (`source` tag: `hand_easy`, `hand_hard`)
- **30 "easy" tasks**: general-purpose coding problems (string/list/dict
  operations, e.g. palindrome check, list flattening). Every task's test
  was independently verified against a hand-written reference solution
  before inclusion (this verification process caught zero bugs in this
  batch).
- **30 "hard" / information-dense tasks**: deliberately designed with
  multiple precise constraints (e.g., specific tie-breaking rules for
  rounding, exact edge-case behaviors for budget-splitting algorithms,
  precise rate-limiter refill timing) such that a damaged handoff should
  plausibly cause failure. Every task's test was independently verified
  against a hand-written reference solution — **this process caught and
  fixed 3 real bugs** in the originally-written tests (task_037's
  window-based dedup logic, task_054's longest-increasing-run expected
  value, task_055's currency-rounding tie-breaking behavior), before any
  API calls were spent generating data against them.

### 7.2 Real benchmark tasks (`source` tag: `humaneval`, `mbpp`)
- **30 HumanEval problems** (OpenAI, `openai/openai_humaneval` /
  raw GitHub source): converted from HumanEval's code-completion-style
  prompt format into a natural-language problem statement suitable for
  Agent A to restate/plan from. Every converted task verified against
  HumanEval's own canonical solution before inclusion (30/30 passed
  verification).
- **40 MBPP (sanitized) problems** (Google Research,
  `google-research/google-research` repo,
  `mbpp/sanitized-mbpp.json`): converted the same way. Every converted
  task verified against MBPP's own reference code before inclusion
  (40/40 passed verification). Note: the sanitized MBPP task_id sequence
  is genuinely non-contiguous (skips e.g. IDs 5, 10, 13, 15) — this is a
  property of the source dataset, not a conversion bug.

### 7.3 Rationale for this specific combination
No standard benchmark (HumanEval, MBPP) is purpose-built to test
handoff-sufficiency sensitivity — they weren't designed with this project
in mind. The chosen design uses **real, externally-validated benchmarks
for general-purpose difficulty** (addressing "why should I trust your
easy/general task set" — a legitimate concern about self-authored tasks)
**and a deliberately hand-crafted, information-dense batch as a targeted
probe** for the specific phenomenon under study (which no existing
benchmark tests). This is presented as a considered research design
choice, not an ad hoc mixture.

### 7.4 Deliberate data-volume decision
An initial instinct was to use the FULL available pools (164 HumanEval +
427 MBPP problems, plus all hand-crafted tasks). This was **explicitly
decided against**: with only 3 simple numeric features feeding the
classifier, a dataset in the low hundreds of rows is sufficient, and
generating from the full pools (600+ tasks, ~4,000+ API/model calls) would
consume disproportionate time for a UG timeline with limited returns.
The target subset size was set at ~120-130 tasks total.

### 7.5 Final dataset composition (Gemini backbone, actual measured result)
**177 total labeled rows, 109 unique tasks:**

| Source | Rows | Successful originals contributed |
|---|---|---|
| hand_hard | 70 | (majority of the 70 rows) |
| hand_easy | 38 | — |
| humaneval | 38 | — |
| mbpp | 31 | — |

Label balance: **109 positive (label 1) / 68 negative (label 0)**, roughly
a 62/38 split — mild imbalance, judged acceptable without resampling,
mitigated at training time by using `class_weight`-aware or otherwise
imbalance-tolerant model configurations where relevant.

Data integrity confirmed clean: zero duplicate (task_id, corruption_type)
pairs, zero tasks with more than one "original" row, zero missing
`source` values (after backfilling 95 pre-existing rows from before the
`source` column was added, by deriving source from task_id prefix
patterns), zero null values in key columns **except** 3 rows in a
subsequent LOCAL-MODEL pilot run (see Section 9) where
`structured-section removal` corruption stripped a handoff down to a
completely empty string — a known degenerate edge case, filtered out
before use, with a suggested code fix (fallback to keep at least one line)
not yet implemented.

---

## 8. Models / Backbones

| Backbone | Role | Status |
|---|---|---|
| **Gemini Flash** (Google AI Studio API, free tier) | Primary bulk data generation backbone | 177 rows generated, this is the main reported dataset |
| **Claude** (Anthropic API, Haiku 4.5 recommended for cost) | Available, used for early pipeline development | Not used for bulk generation in the final dataset |
| **Qwen2.5-Coder-3B-Instruct** (local, via Ollama) | Small-model / cross-scale generalization check | Pilot-validated (see Section 9), not yet scaled to full volume |

**Why multiple backbones**: tests generalization across both model
*family* (Claude vs. Gemini vs. Qwen) and model *scale* (frontier API
models vs. a 3B local model) — directly follows CARA's own justification
for testing two backbones (Qwen2.5-Instruct + Qwen2.5-Coder) to show
findings aren't an artifact of one specific model.

**Hardware constraint informing the 3B choice (not 7B as originally
considered)**: local development machine has no dedicated GPU (Intel
integrated graphics only — the "128MB" figure Windows reports is a
reserved allocation, not real usable VRAM; Ollama's GPU acceleration
targets NVIDIA/AMD/Apple, not Intel iGPUs). All local inference runs on
CPU with 16GB RAM. A 3B model was judged the practical ceiling for usable
throughput on this hardware; 7B was originally suggested but revised down
after considering the hardware specifically.

---

## 9. Local Qwen 3B Pilot Results (preliminary, smaller sample)

Two pilot rounds run, sampling evenly across all 4 sources (not a blind
slice of the task list, which would have only hit `hand_easy` tasks first
due to list concatenation order):

**Round 1** (3 tasks per source, 12 attempted): 9 successful originals.
Notable: all 3 attempted `hand_hard` tasks broke on **all 4** corruption
types (100% break rate at this small n) — a striking early signal.

**Round 2** (10 tasks per source, 40 attempted): 27 successful originals
(69 usable rows after removing 3 degenerate empty-handoff rows).

| Source | n_orig | Overall break rate (Qwen 3B) |
|---|---|---|
| hand_hard | 5 | **90.0%** |
| mbpp | 6 | 37.5% |
| hand_easy | 7 | 32.1% |
| humaneval | 9 | 25.0% |

**Cross-backbone comparison (same hand_hard task design):** Gemini showed
~40-55% break rate on hand_hard tasks; Qwen 3B showed **90%** on the same
task design — nearly double. This is a real, citable directional finding
(consistent across both pilot rounds): **a weaker downstream agent appears
substantially more sensitive to handoff corruption than a stronger one.**
Framed honestly as preliminary (small n per source, 5-9), not yet at a
sample size to report with full statistical confidence, but the
consistency across two independent pilot rounds strengthens it beyond pure
noise.

---

## 10. Classifier Results — Including a Critical Bug Found and Fixed

### 10.1 Features
Three lightweight structural features, computed WITHOUT any LLM call at
inference time (the core enabler of real-time, cheap gating):
- **Cosine similarity**: MiniLM sentence embeddings (`all-MiniLM-L6-v2`),
  handoff vs. original problem statement.
- **Entity overlap**: spaCy-based — fraction of the problem's noun
  chunks/named entities still present in the handoff.
- **Length ratio**: handoff word count / original problem word count.

### 10.2 CRITICAL BUG FOUND: task-level data leakage in the first training run
The first classifier run (row-level `train_test_split`, not grouped by
task) produced suspiciously perfect metrics:

**Leaked (WRONG, do not report these numbers) — XGBoost:**
Precision 1.000, Recall 0.929, AUROC 1.000.

**Root cause, diagnosed and confirmed**: with 177 rows across only 109
unique tasks, many tasks contribute multiple rows (an original handoff
plus its corrupted siblings, which share nearly all their text). A plain
row-level split can place a task's original in the training set and its
corrupted sibling in the test set (or vice versa) — letting the model
partially recognize near-duplicate text it has effectively already seen,
rather than learning genuine handoff-sufficiency signal. This was
confirmed structurally: 177 rows / 109 unique task_ids, i.e., ~1.6 rows
per task on average, meaning row-level splitting was virtually guaranteed
to leak some task's sibling rows across the split boundary.

**Fix**: switched to `GroupShuffleSplit` (and later `GroupKFold`), grouping
by `task_id`, guaranteeing all rows belonging to any one task fall entirely
within EITHER train OR test, never split across both. A runtime assertion
verifying zero task_id overlap between train and test was added and
confirmed to pass on every run.

### 10.3 Corrected single-split results (XGBoost, group-aware split)
Train: 127 rows / 81 tasks. Test: 50 rows / 28 tasks. Confirmed zero task
overlap.

**Precision: 0.800, Recall: 1.000, AUROC: 0.904.**

Reproduced independently on two different machines (this session's sandbox
and the collaborator's own machine) with identical results — confirms the
fix is deterministic and not an artifact of one environment.

### 10.4 Model comparison sweep (group-aware split, same train/test partition)

| Model | Precision | Recall | AUROC | Latency (ms/example) |
|---|---|---|---|---|
| Logistic Regression | 0.844 | 0.964 | **0.943** | ~0.003-0.02 |
| Small MLP | 0.818 | 0.964 | **0.946** | ~0.003-0.004 |
| Random Forest | 0.800 | 1.000 | 0.912 | ~0.12-0.16 |
| XGBoost | 0.800 | 1.000 | 0.904 | ~0.04-0.13 |

**Notable finding, worth foregrounding rather than downplaying**:
Logistic Regression and a small MLP slightly OUTPERFORM XGBoost on AUROC
here, while being computationally cheaper (Logistic Regression's latency
is roughly an order of magnitude lower than XGBoost's). This is a stronger
version of the project's core "lightweight" thesis than originally
planned: with well-chosen features, classifier complexity barely matters
— even the simplest model performs comparably or better. **Suggested
framing for the paper: consider presenting Logistic Regression as the
primary reported model** ("simplest model that works"), with
RF/MLP/XGBoost as the comparison sweep, rather than defaulting to XGBoost
as originally planned in the proposal stage.

**LLM-as-judge baseline**: script written and tested (calls a live model
to directly score handoff sufficiency as the "expensive" comparison
point), but **not yet confirmed to have actually run** in the results
above — needs verification that the `rollouts.csv` + API key + correct
`agents.py` function name were all available when `model_comparison.py`
was executed. This is the single most important baseline still to
confirm, since it's the direct evidence for "cheap features beat expensive
LLM judgment."

### 10.5 5-fold GroupKFold cross-validated results (most robust reported number)
Every task appears in exactly one test fold across 5 folds; zero task
overlap confirmed in every fold.

| Metric | Mean ± Std |
|---|---|
| Precision | 0.924 ± 0.060 |
| Recall | 0.935 ± 0.079 |
| F1 | 0.926 ± 0.038 |
| Accuracy | 0.910 ± 0.047 |
| AUROC | **0.953 ± 0.052** |

Per-fold AUROC ranged from 0.896 to 1.000 across the 5 folds. Notably,
the earlier single-split result (0.904) landed on one of the harder folds
— i.e., it was if anything a slightly conservative estimate, not an
inflated one. **This mean ± std result is the recommended headline
classifier performance number for the paper**, since it demonstrates
stability across different held-out task sets rather than resting on one
particular split.

**Independently reproduced a third time** (this session's sandbox, the
collaborator's first machine, and a subsequent run) with identical
per-fold and mean values — strong confirmation the result is deterministic
and not an artifact of any one environment.

The final deployed/gate-ready model is trained on ALL 177 rows (not just
one fold's training partition) — the cross-validation exists purely to
produce an honest performance estimate, and this distinction should be
stated explicitly in the methods section.

### 10.6 SHAP feature importance and an honest caveat
Mean |SHAP value| per feature (single-split run): `length_ratio` ≈ 3.47,
`entity_overlap` ≈ 0.79, `cosine_similarity` ≈ 0.21. **`length_ratio`
dominates by a wide margin (4-17x the other features).**

**Caveat identified and not yet resolved — include in Limitations**:
the two most damaging corruption types (structured-section removal,
over-summarization) also happen to drastically shorten the handoff as a
side effect of what they remove. `length_ratio` may therefore be acting
partly as a *proxy for which corruption type was applied*, rather than a
fully independent signal of "how much meaning survived." Recommended
honest framing: *"length_ratio dominates SHAP importance, though this may
partly reflect correlation with corruption type rather than a purely
independent sufficiency signal — a correlation analysis between
length_ratio and corruption_type is a natural follow-up check."* This
follow-up check has NOT yet been performed.

---

## 11. Real-Time Gating and Explainability (built, not yet stress-tested at scale)

When a handoff is blocked (score below threshold), **SHAP** is used to
identify which feature contributed most negatively to that specific
instance's score, and a plain-language hint template (keyed to whichever
feature was most responsible) is returned to Agent A for its retry —
turning a blind "insufficient, try again" into an explained retry (e.g.,
"key details from the task are missing from your handoff"). This
mechanism is implemented and unit-verified but not yet tested against the
final trained model at scale, and not yet compared (blind-retry vs.
explained-retry recovery rate) as a controlled experiment — this
comparison is identified as a natural additional experiment but not yet
run.

---

## 12. Remaining Work (as of this document)

1. **Confirm the LLM-as-judge baseline actually ran** and get real
   numbers for it (Section 10.4) — highest priority remaining gap.
2. **Correlation check between `length_ratio` and `corruption_type`**
   (Section 10.6) — needed to honestly qualify the SHAP finding.
3. **Scale the local Qwen 3B pilot** if time allows, to firm up the
   cross-backbone comparison (Section 9) beyond n=5-9 per source.
4. **Wire the gate into an actual LangGraph conditional edge** (Section
   5.3) — currently plain sequential function calls, not what the
   architecture diagrams depict.
5. **Run the full gated-vs-ungated end-to-end system comparison**
   (task success rate, latency overhead, cost overhead), and ideally
   compare against Agent-Radar as a training-free baseline.
6. **Blind-retry vs. explained-retry (SHAP-hinted) recovery rate
   comparison** (Section 11) — not yet run.
7. **Single-sample-per-datapoint limitation** (Section 6.5) — could
   partially address via repeated corrupted reruns + majority vote, if
   time allows; otherwise state plainly as a limitation.
8. Optional/stretch: a real tool-execution-result task domain, to make
   "structured-section removal" a genuine simulation of dropped tool
   results rather than its current repurposed meaning (Section 6.3).
9. Optional/stretch: SQL secondary task domain (Spider/WikiSQL) for a
   second domain beyond code generation, to test generalization beyond
   Python coding tasks specifically.

---

## 13. Full File Inventory

| File | Purpose | Status |
|---|---|---|
| `tasks/coding_tasks.py`, `additional_tasks.py` | 30 hand-crafted easy tasks | Done, verified, tagged `source=hand_easy` |
| `hard_tasks.py`, `hard_tasks_2.py` | 30 hand-crafted hard/information-dense tasks | Done, verified (3 test bugs found+fixed), tagged `source=hand_hard` |
| `convert_humaneval.py` → `humaneval_tasks.py` | 30 HumanEval problems, converted + verified | Done |
| `convert_mbpp.py` → `mbpp_tasks.py` | 40 MBPP (sanitized) problems, converted + verified | Done |
| `tasks/all_tasks.py` | Merges all 4 sources into one list, source-tagged | Done |
| `agents.py` / `agents_multi_backbone.py` / `agents_with_local.py` | Agent A/B via Claude, Gemini, and/or local Ollama | Done |
| `agents_local.py` | Standalone local-only agent implementation for Qwen pilot | Done |
| `corruption.py` | 4 deterministic corruption functions, post-fix versions | Done, validated with real before/after data |
| `rollout_runner.py` | Orchestrates rollout generation + labeling, resume-safe, carries `source` column | Done, survived a real `KeyboardInterrupt` cleanly with no duplication |
| `run_local_pilot.py` | Small-scale pilot runner for local Qwen, source-balanced sampling | Done |
| `feature_extraction.py` | 3 structural features | Done |
| `build_dataset.py` | rollouts.csv → features.csv, carries `source` | Done |
| `train_classifier.py` | XGBoost, group-aware single split (leakage-fixed) | Done |
| `train_classifier_kfold.py` | XGBoost, 5-fold GroupKFold, mean±std reporting | Done — **this is the version whose results should be reported** |
| `model_comparison.py` | LogReg/RF/MLP/XGBoost/LLM-judge sweep, group-aware split | Done, LLM-judge baseline unconfirmed as actually executed |
| `gate.py` | Real-time scoring + SHAP failure-reason feedback | Written, not stress-tested against the final model at scale |
| `analyze_by_source.py` | Break-rate breakdown sliced by data source | Written |

---

## 14. Key Numbers Cheat-Sheet (for quick reference while writing)

- Total labeled rows: **177**, unique tasks: **109**
- Label balance: **109 positive / 68 negative**
- Sources: hand_hard 70, hand_easy 38, humaneval 38, mbpp 31
- Corrected single-split XGBoost: **Precision 0.800 / Recall 1.000 / AUROC 0.904**
- **5-fold CV (report this one): Precision 0.924±0.060 / Recall 0.935±0.079 / AUROC 0.953±0.052**
- Best single model by AUROC: **Small MLP (0.946)**, then Logistic Regression (0.943)
- SHAP importance ranking: length_ratio ≫ entity_overlap > cosine_similarity
- Easy batch overall corruption break rate: **13.0%**
- Hard batch overall corruption break rate: **39.7-46.7%** (stable across two sample sizes)
- Post-fix per-type break rates (hard batch): structured-section removal & over-summarization ~55%, truncation ~17%, entity omission ~14%
- Qwen 3B hand_hard break rate: **90%**, vs. Gemini's ~40-55% on identical task design
