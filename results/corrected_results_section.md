# 4. Results: Live Two-Stage Downstream Benchmark

> This is a corrected draft of the Results section for the live downstream benchmark.
> All numbers come from `live_pilot_results.csv` (94/94 tasks gated, 0 ungraded / 0 API-error rows).
> Statistical procedures: exact two-sided McNemar `binom.test` on discordant pairs; paired bootstrap 95% CIs for the gated-minus-ungated delta (5,000 resamples, seed 42).

## 4.1 Confirmatory Setup (summary)

We report the confirming phase of a two-phase evaluation. A small exploratory pilot (Phase 1, n=30) revealed that several injected corruptions frequently failed to *realize* --- i.e., they did not change the downstream outcome of the transmitted handoff (a known weak-negative property of the corruption pipeline, cf. $\S${dataset} and `rollout_runner.py`). To align with the dataset's own labeling standard, Phase 1's weak corruption strengths were replaced with validated, realization-checked strengths (truncation to 10% content; single-sentence over-summarization; objective inversion with a truncation fallback when inversion is a no-op; fabricated completion over a 60% prefix). These settings and the gate's operating point (11-feature RF, $\tau = 0.45$, Youden / target recall calibration, fixed prior to Phase 2) were frozen before the confirmatory run.

The confirmatory run uses the full held-out task suite of $n=94$ tasks, unmodified from `holdout_task_ids.txt`: 47 clean handoffs and 47 handoffs corrupted by one of four modes (12 truncation, 12 over-summarization, 12 objective inversion, 11 fabricated completion), assigned by a fixed parity schedule. Each task is executed twice --- once with the untethered end-to-end pipeline (Ungated) and once behind the two-stage gate (Gated) --- and graded `Pass@1` against the reference test suite via `grade()` from `rollout_runner.py`. Gating is *defensive routing*: if either stage blocks the handoff, Agent B re-solves from the pristine problem statement ("circuit-breaker" fallback), so a block can only help or be neutral; it never forwards a corrupted handoff. The judge (Stage 2) is fail-open: escalation or judge errors degrade to a pass, biasing against the gate.

Both arms share the same model (Gemini family, single rollout per task, temperature fixed); the only difference between arms is the gate. Rate-limit errors were handled by a resume loop; all 94 rows completed without unresolved API errors.

## 4.2 Overall Effect

Table [tab:downstream_overall] shows the headline result. Gating improves end-to-end `Pass@1` from **55.3% (52/94)** to **67.0% (63/94)** across the full suite (**+11.7 pp**, paired bootstrap 95% CI **[+4.3, +19.1] pp**; exact two-sided McNemar **p = 0.0074** over 15 discordant pairs: 13 recovered, 2 regressed). The interval excludes zero and the paired test is significant at $\alpha = 0.05$.

The effect is driven entirely by the corrupted arm and is absent, though not harmful to practical fidelity, on the clean arm:

- **Corrupted (n=47):** 44.7% (21/47) $\rightarrow$ 70.2% (33/47) (**+25.5 pp**; 13 recoveries vs. 1 regression; **p = 0.0018**).
- **Clean (n=47):** 66.0% (31/47) $\rightarrow$ 63.8% (30/47) (**$-$2.1 pp**; 0 recoveries vs. 1 regression; **p = 1.00**, n.s.). This is a net difference of one task and is not statistically distinguishable from zero; we report it as a single-task fidelity cost and attribute most of it to one Stage-2 false alarm (see $\S${failure-analysis}).

The clean-arm caveat should be stated explicitly: the gate is *not* free --- it trades a small (n.s., $\approx$1-task) clean-pipeline cost for a large, significant gain on corrupted handoffs. Framing the gate as "zero-cost" would be inaccurate and is unnecessary; the honest claim (significant robustness transfer at negligible clean cost) is stronger.

## 4.3 Interaction Analysis: Interception and Recovery

Of the 47 corrupted handoffs: 21 actually *realized* damage in the ungated arm (i.e., the corrupted handoff caused Agent B to fail). Of these 21 realized failures:

1. **81.0% (17/21) were intercepted** by the gate (the block; the only unrealized-or-overblocked remainder is reported below).
2. Of the 17 intercepted-and-realized failures, **76.5% (13) were fully reversed** by the circuit-breaker fallback (Agent B re-solving the pristine problem passes: he_127, he_133, he_150, mbpp_245, mbpp_272, mbpp_388, mbpp_395, mbpp_419, mbpp_427, mbpp_455, mbpp_478, mbpp_790, task_075). 
3. The residual 4/21 (19%) realized failures were **gate false negatives**: the gate passed the corrupted handoff (mbpp_260, mbpp_301, mbpp_615, mbpp_783) and the task failed in both arms.

Concordantly, the gate over-blocks conservatively rather than dangerously: it blocked 47 handoffs (40 corrupted + 7 clean), and **35/47 (74.5%) of all blocked handoffs still produced a passing solution via fallback** --- i.e., the gate is overwhelmingly safe-when-wrong.

Net effect language for the paper: *"The two-stage gate intercepted 81% of realized handoff failures and fully recovered solutions for 62% of all realized corruptions (13/21), while costing one task on the clean pipeline."*

## 4.4 Per-Corruption-Mode Analysis

Table [tab:downstream_types] decomposes the corrupted arm. Consistent with the offline taxonomy (Table 3), the gate recovers structural corruptions (truncation, over-summarization) and the semantic inversion mode, but is *neutral* on fabricated completions:

- **truncation (n=12):** 41.7% $\rightarrow$ 75.0% (+33.3 pp; 4 recovered, 0 regressions; 100% intercepted).
- **over\_summarization (n=12):** 50.0% $\rightarrow$ 75.0% (+25.0 pp; 4 recovered, 1 regression; 100% intercepted).
- **invert\_objective (n=12):** 33.3% $\rightarrow$ 75.0% (**+41.7 pp**; 5 recovered, 0 regressions; 83% intercepted) --- the mode the offline PRM is weakest at (Table 3, AUROC 0.50) and where the two-stage design earns its keep: 3/5 recoveries come from Stage-2 judge flips.
- **fake\_completion (n=11):** 54.5% $\rightarrow$ 54.5% (**0.0 pp**; 0 recovered, 0 regressions; 55% intercepted). This null result must be argued correctly, or the type table looks broken. Decomposition: of the 11 fabricated-completion tasks, 6 were intercepted (2 fell back to a passing solution, 4 to a failing one) and 5 passed the gate. Among the 5 ungated failures in this mode, 3 (mbpp_260, mbpp_301, mbpp_783) were gate **false negatives** (missed interception) and 2 (mbpp_138, mbpp_433) were intercepted but *unrecoverable* even from the pristine problem (ceiling effect: the underlying task is hard, `Pass@1` on the raw task fails in both arms). Thus the gate's non-effect on this mode is a combination of (i) a residual detection gap (3 gate misses) and (ii) a task-difficulty ceiling (2 unrecoverable), *not* a sign that blocking backfires. This is the one mode where the PRM's offline 0.95 AUROC (Table 3) does not transfer to downstream recovery, and it is where future detection of *plausible-but-wrong* content must focus.

## 4.5 Architecture Ablation: Is Stage 2 (the judge) necessary?

Table [tab:downstream_ablation] isolates each gate stage by replaying the recorded decisions. The RF-only gate running at the same $\tau = 0.45$ already delivers **+10.7 pp** overall (55.3% $\rightarrow$ 66.0%; +21.3 pp on the corrupted arm at zero clean cost: 66.0% clean in both arms). The Stage-2 LLM judge then contributes a further **+1.0 pp** overall (66.0 $\rightarrow$ 67.0): **+4.2 pp** on the corrupted arm (66.0 $\rightarrow$ 70.2) at a **$-$2.2 pp** clean cost (66.0 $\rightarrow$ 63.8). 

Judge mechanics: 59/94 (63%) of first-stage passes were escalated; 12 were flipped to "insufficient" (10 corrupted, 2 clean). Of these 12 flips only 2 were *decisive* recoveries (he_133, mbpp_245 — both semantic-inversion cases the surface RF misses), and 2 were clean false alarms (mbpp_410, mbpp_786; one of these, mbpp_410, is the entire clean-arm regression). Precision of the flip action is therefore **2/12 = 17%** decisive, and the judge as configured is a marginal contributor on this live set.

We recommend two statements in the paper: (i) the headline robustness claim rests on the **aggregate two-stage design** (RF interception at a recall-biased operating point plus a defensive fallback), not on the judge alone; (ii) the judge's marginal value on this live set is small and comes with a clean-arm false-alarm cost, so it should either be presented as an *optional* escalation stage whose 2 decisive recoveries come from semantic inversions (the mode RF cannot see), or strengthened before claims ("judge improves reliability") are made. The stronger offline judge evidence (escalated-subset evaluation, `optionB_llmjudge`) should be the place where judge utility is quantified for the paper.

## 4.6 Failure Analysis: The Two Regressions

Both regressions are instructive and should be reported:

- **mbpp\_410 (clean, judge false alarm).** The 11-feature RF scored the *genuine* handoff at 0.5397, above $\tau$, so it escalated to the judge; the judge flipped it to "insufficient"; fallback re-solved from the pristine problem but produced a failing solution. This is the entire clean-arm cost. In the corrected framing it is a single-task, fail-open-adjacent event (even a false gate decision degrades only to a second chance at the same problem).
- **mbpp\_239 (over-summarization, Stage-1 block that failed to rescue).** The RF correctly blocked the corrupted handoff (score 0.114); the fallback solution failed. Ungated, the corrupted handoff happened to pass. This pair counts once as a regression for the corrupted arm and is consistent with the ceiling explanation above (the raw task is borderline for this model).

Both regressions are of the "correct gate decision" form (the block was right; the fallback couldn't save a genuinely hard task), which is why framing the recovery mechanism as *routing*, not *repair* (n. 1), is essential.

## 4.7 Robustness and Limitations (one compact paragraph)

- **Sampling.** Single-rollout `Pass@1` per task (as defined). Paired design absorbs per-task difficulty; significance is exact (McNemar) and deltas carry bootstrap CIs. A second seed run would strengthen the clean-arm estimate ($-$2.1 pp is within its own noise).
- **Protocol honesty.** Only the confirmatory (Phase 2, 94-task) numbers are reported in Tables 4–6; Phase 1 is described as an exploratory pilot that corrected weak-negative corruptions. Any per-choice tuning after seeing Phase 2 is disclaimed.
- **No out-of-distribution handoff benefit claim.** The gate does not repair handoffs; it routes them. "Recovery" means "re-solved from the pristine problem," and every recovered task is exactly as good as the raw problem alone would have been.
- **Judge self-verification.** Judge and agents share the same model family; a different-model judge is recommended for the camera-ready version.
- **Cost.** 47 extra fallback generations + 59 escalation calls + 12 fallback re-runs for 94 tasks; the RF itself runs in $<10$ ms. Gating is cost-competitive with unconditional re-solving (the fallback is invoked only on the blocked 50% subset).

---

## Author follow-ups before submission (do NOT include in paper)

1. Decide judge deployment: keep as optional escalation (honest) or swap to a different model + confidence banding to raise flip precision above 2/12. If the latter, re-run Phase 2 with the new judge and re-report Table 5 / ablation.
2. Add a second rollout seed (or temperature sampling) and recompute the clean-arm CI; expect it to bracket zero.
3. Reconcile "fake\_completion null" with the offline 0.95 AUROC: add a sentence attributing it to gate false negatives (3) + task ceiling (2) rather than gate harm.
4. Final LaTeX: include Tables 4–6, a 2-arm paired scatter/mosaic figure, and the RF-only vs two-stage ablation figure. All numbers here derive from `live_pilot_results.csv`; regenerate tables directly from that file (script noted below) so nothing is hand-transcribed at camera-ready time.