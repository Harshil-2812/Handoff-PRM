# Handoff PRM: Experiment Work Split

Goal: remove the main reasons a reviewer would reject the paper (synthetic-only corruptions, thin live evidence, missing baselines, unclear statistics) by running the experiments below.

Honest note: nothing can guarantee "cannot be rejected." This plan covers the objections I found; reviewers can still raise others. A pre-registered protocol and honest reporting of negative results keep the paper defensible even if some results are weak.

Authors and roles:
- **Avaneesh N**: Workstream A (data, organic errors, relabeling)
- **Aravind G**: Workstream B (offline validity, baselines, statistics)
- **Vemisetty Harshil**: Workstream C (live evaluation, hidden-problem setting)
- **Dr. Suhasini S**: sign-off and review (no experiment workload)

---

## Week 0 (everyone, about 3 days): freeze the protocol first

Do this before anyone runs anything. It is what makes the later results trustworthy.

- [ ] One shared repo, one results schema (CSV columns: task_id, source, handoff_id, handoff_type [clean / corruption name / organic], label, label_policy, seed, features, stage1_score, decision, outcome per policy, model IDs, temperature).
- [ ] Task manifests: **train/dev pool**, **live test set** (disjoint IDs, never used for tuning), **organic-error pool** (disjoint from both). Commit the manifests.
- [ ] Seeds: 42, 43, 44 for every live run. Log exact model IDs and temperatures for Agent A, Agent B, and the LLM judge.
- [ ] Declare up front: primary metric (downstream Pass@1), tests (exact McNemar, task-level paired bootstrap), and how thresholds are chosen (**dev-set out-of-fold predictions only**, one threshold for the headline result).
- [ ] Declare up front that negative results will be reported, not dropped.

---

## Workstream A: Data and labels (Avaneesh N)

Builds the evidence that the gate works on realistic errors, not just injected ones.

- [ ] **A1. Organic handoff dataset.** Sample Agent A at temperature > 0, with at least 3 Agent-A variants (a weaker model, a mid model, a stronger model). Do NOT inject corruptions. Label by whether Agent B passes. Clean handoffs mostly succeed (about 95% earlier), so negatives are rare: sample enough handoffs to get **at least ~100 organic negatives** (roughly 1,000+ handoffs at a 10% failure rate; weaker Agent A raises the rate).
- [ ] **A2. Majority-vote labels.** Re-run Agent B 3 times per handoff; label by majority. Report agreement rate between runs.
- [ ] **A3. Both labeling policies.** Build the discard-policy and the robust-positive versions of the 483-row dataset (raw outcomes already logged) and of the organic dataset.
- [ ] **A4. Verify tasks.** Check every new task against a reference solution before spending API calls (as done for the original tasks).
- [ ] **A5. Fill the metadata.** Pull the exact Gemini model ID, judge model ID, temperature, and how τ=0.45 was chosen from the run logs (the four `[FILL]` items).
- [ ] **A6 (optional, if time).** A small second domain (e.g., SQL on Spider) to show the method is not Python-only.

Deliverables: `organic_dataset_v1.csv`, `dataset_v2_robust_positive.csv`, label-agreement report. Size: large.

---

## Workstream B: Offline validity, baselines, statistics (Aravind G)

Can start immediately on the existing 483 rows; the organic evaluation waits for A1.

- [ ] **B1. Redo leave-one-corruption-type-out** with strictly task-grouped folds, and explain why Table XXVIII and Table XXXII disagree (invert objective 0.448 vs 0.731, wrong algorithm name 0.427 vs 0.793). Assert zero task overlap in code.
- [ ] **B2. `function_name_preserved` diagnosis.** Compute its single-feature AUROC on the 14 signature_rename rows. Bug, or explain why not.
- [ ] **B3. Threshold protocol.** Choose each operating threshold on dev OOF predictions only; report one headline threshold. Recompute the confusion matrix.
- [ ] **B4. Task-level bootstrap CIs** for every reported AUROC/AUPRC (resample tasks, not rows). Add CIs to the per-corruption-type table.
- [ ] **B5. Calibration.** ECE and a reliability plot for the deployed classifier.
- [ ] **B6. Baselines on the 483-row and organic datasets:** length threshold; exact-signature/entry-point match; embedding-only logistic regression; a fine-tuned small encoder (e.g., MiniLM/DeBERTa-small); an LLM judge with a good prompt (with and without the raw problem). Report **end-to-end latency** (embedding, NER, any NLI), not just classifier predict time.
- [ ] **B7. Multivariate analysis** that controls for corruption type in one model (type fixed effects), to settle the length-ratio confound.
- [ ] **B8. NLI feature decision.** The class means run the wrong way (0.126 vs 0.105); drop it or justify it.
- [ ] **B9. Prevalence sweep (offline).** Precision/NPV and net benefit at 5%, 20%, and 50% corrupted handoffs.
- [ ] **B10. Organic evaluation** (after A1): AUROC with CIs on organic errors; report per Agent-A variant.

Deliverables: one notebook/script per item plus a single results table set. Size: large.

---

## Workstream C: Live end-to-end evaluation (Vemisetty Harshil)

- [ ] **C1. Rebuild the live harness** around the frozen schema; run a 20-task pilot to check logging.
- [ ] **C2. Main live run.** At least **300 held-out tasks**, balanced across sources (not mostly MBPP; add HumanEval and a harder set such as LiveCodeBench), **3 seeds**. Use a **paired design**: every task appears in both the clean and the corrupted condition, so cross-arm comparisons are not confounded by task difficulty.
- [ ] **C3. Arms.** Ungated, Stage-1 only, two-stage cascade, **LLM-judge only**, always-fallback, an **AgentAsk-style prompted clarifier** (use AgentAsk's published prompt; describe it honestly as an approximation, not their trained model), and a simple deterministic check (signature/length). Report Stage-1-only vs ungated with its own McNemar test.
- [ ] **C4. Organic arm.** Run the gate on organically generated handoffs (from A1), with no injected corruptions. This is the most realistic result in the paper.
- [ ] **C5. Hidden-problem setting.** Agent B cannot see the raw problem; the gate (which can) blocks, and Agent A retries. Compare **no gate, blind retry, SHAP-hinted retry**, max 2 retries. Report Pass@1, retries, tokens, latency.
- [ ] **C6. Prevalence sweep (live).** Mix clean and corrupted handoffs at 5%, 20%, and 50% corrupted; report whether the gate nets positive at each.

Deliverables: `live_results_*.csv`, `verify_live_stats.py` updated, figures. Size: large.

---

## Supervisor (Dr. Suhasini S)

- [ ] Sign off the Week 0 protocol before any runs.
- [ ] Decide at the two gates below.
- [ ] Read the final draft cold.
- [ ] Confirm the target venue and its preprint policy.

---

## Timeline

| Weeks | A (Avaneesh) | B (Aravind) | C (Harshil) |
|-------|--------------|-------------|-------------|
| 0 | Protocol freeze | Protocol freeze | Protocol freeze |
| 1-2 | A3, A4, A5; start A1 | B1-B5, B7, B8 | C1 pilot; build arms |
| 3 | A1 + A2 delivered (**Gate 1**) | B6, B10 on organic | C2 main run starts |
| 4-5 | A6 (optional) | B9; finalize tables | C2 finish, C3, C4 |
| 5-6 | support reruns | support reruns | C5, C6 (**Gate 2**) |

## Gates (the thresholds are my rough suggestions; agree your own at Week 0)

- **Gate 1 (end of week 3): organic-error detection.** If AUROC on organic errors is about 0.75 or higher, keep the "detection gate" framing. If it is near 0.5 to 0.65, switch to the measurement/negative-result framing (what cheap features can and cannot detect). That is still publishable but changes the venue and the title.
- **Gate 2 (end of week 6): live evidence.** If the gate helps on organic or realistic-prevalence arms, report it. If it helps only on injected corruptions and hurts on organic, present the work as a stress test, not a deployment recommendation.

## Small non-experiment blockers (do these alongside; they still count)

- [ ] Verify refs [6]-[20] (Harshil, while runs execute).
- [ ] Put code and data on GitHub + Zenodo (Avaneesh, at the end of weeks 5-6).
- [ ] Remove all `[FILL]` text (Avaneesh, from A5).

## Risks

- API budget and rate limits (the original data used a free tier). Estimate token cost for C2 (300 tasks x 3 seeds x arms) before starting.
- Few organic negatives. Mitigation: weaker Agent A models and larger sampling.
- A lone live run can still be noisy; the 3 seeds and paired design address this.
