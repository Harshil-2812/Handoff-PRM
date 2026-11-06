# Workstream C: Comprehensive Token, Quota & Cost Estimate (C2–C6)

**Date**: November 6, 2026  
**Author**: Vemisetty Harshil (Owner: Workstream C)  
**Evaluated Scope**: C2 (Main Run), C3 (Arms), C4 (Organic Arm), C5 (Hidden Problem), C6 (Prevalence Sweep)  
**Model Config**: `gemini-3.5-flash-lite` for Agent A, Agent B, and LLM Judge  
**Pricing / Quota Basis**: Google AI Studio Gemini API Free Tier & Tier 1 Commercial Rates  

---

## 1. Verified Free-Tier Limits for `gemini-3.5-flash-lite`

Based on Google AI Studio's enforced quota parameters for `gemini-3.5-flash-lite`:
- **Requests Per Minute (RPM)**: 15 RPM per project/model.
  - *Harness Setting*: We enforce **12 RPM per key** (leaving a 20% safety buffer to avoid transient 429 rate-limit spikes).
  - *Pool Throughput*: Across **3 distinct keys/accounts**, maximum aggregate throughput is **36 RPM** (~2,160 requests/hour).
- **Requests Per Day (RPD)**: 1,500 RPD per project, resetting daily at **midnight Pacific Time (PT)**.
  - Across 3 distinct keys/accounts: $3 \times 1,500 = \mathbf{4,500\text{ RPD total}}$.
- **Tokens Per Minute (TPM)**: 1,000,000 TPM per project (our average prompt length of ~650 tokens means TPM is never the binding constraint).

---

## 2. Granular Call Breakdown Across Experiments (C2–C6)

Because the harness implements persistent, prompt-level disk caching on `hash(model, prompt, temperature, seed)`:

### C2 & C3: Main Live Run & 7 Experimental Arms
- **Matrix**: 300 tasks $\times$ 3 seeds (42, 43, 44) $\times$ 7 arms $\times$ 2 conditions (clean vs. corrupted) = **12,600 trial rows**.
- **Unique API Calls Required**:
  1. *Agent A (clean plan)*: 300 tasks $\times$ 3 seeds = **900 calls**.
  2. *Corrupted Handoff Generation*: 100% heuristic Python string transformations (`corruption.py`) = **0 API calls**.
  3. *Stage-1 PRM Scoring*: 100% local CPU Random Forest inference (`stage1_rf.joblib`) = **0 API calls**.
  4. *Agent B on Clean Handoffs*: Evaluated on `ungated` arm; identical prompt cached and reused by passing arms = **900 calls**.
  5. *Agent B on Corrupted Handoffs*: Evaluated on `ungated` corrupted arm; cached for other passing arms = **900 calls**.
  6. *Agent B on Raw Problem Fallbacks*: Bypasses handoff; at temperature 0, identical across seeds = 300 tasks $\times$ 1 = **300 calls**.
  7. *LLM Judge Arm*: Evaluates 300 tasks $\times$ 3 seeds $\times$ 2 conditions = **1,800 calls** (cached and shared with `cascade` escalated trials).
  8. *AgentAsk Clarifier Arm*: Intercepts 300 tasks $\times$ 3 seeds $\times$ 2 conditions = **1,800 calls** (plus ~10% requiring Agent A refinement = **+180 calls**).
  9. *Deterministic Arm*: Rule-based signature and length check = **0 API calls**.
- **C2/C3 Total Calls**: **6,780 unique calls**.

### C4: Organic Arm
- **Task Scope**: 127 organic pool tasks (`manifest_organic.json`).
- If evaluated once Workstream A delivers `organic_dataset_v1.csv` across 3 Agent-A variants:
  - Stage-1 PRM gate runs locally: **0 API calls**.
  - Fallback Agent B calls triggered only on blocked handoffs (~20% block rate): $127 \times 3 \times 0.20 \approx \mathbf{76\text{ calls}}$.
- **C4 Total Calls**: **~76 calls** (or 0 additional calls if evaluating offline labels).

### C5: Hidden-Problem Setting
- **Matrix**: 100 held-out tasks $\times$ 3 seeds across 3 conditions (`no_gate`, `blind_retry`, `shap_hinted_retry`).
- *Calls Required*:
  - `no_gate`: 300 Agent A + 300 Agent B (cached from C2) = **0 new calls**.
  - `blind_retry`: ~30% blocked $\to$ ~90 retries from Agent A + 90 Agent B calls = **180 calls**.
  - `shap_hinted_retry`: ~30% blocked $\to$ ~90 retries from Agent A + 90 Agent B calls = **180 calls**.
- **C5 Total Calls**: **360 new calls**.

### C6: Live Prevalence Sweep
- **Matrix**: 3 prevalences (5%, 20%, 50%) across stream of clean and corrupted handoffs.
- Because C6 samples from the handoffs and agent outputs generated in C2, nearly all prompts hit local disk cache.
- **C6 Total Calls**: **0 to ~50 calls**.

---

## 3. Aggregate Workstream C Budget & Feasibility

| Experiment | Target Rows | Unique API Calls | Est. Prompt Tokens | Est. Completion Tokens | Free-Tier Feasibility |
|---|---|---|---|---|---|
| **C2/C3: Main Run & Arms** | 12,600 | 6,780 | 4.41 M | 1.09 M | 1.51 daily cycles (~2 days) |
| **C4: Organic Arm** | 381 | 76 | 0.05 M | 0.01 M | < 2% of 1 daily cycle |
| **C5: Hidden Problem** | 900 | 360 | 0.25 M | 0.06 M | ~8% of 1 daily cycle |
| **C6: Prevalence Sweep** | 900 | 50 | 0.03 M | 0.01 M | < 1% of 1 daily cycle |
| **TOTAL (C2–C6)** | **14,781** | **~7,266** | **~4.74 M** | **~1.17 M** | **1.61 daily cycles (~2 days)** |

---

## 4. Financial Cost if Run on Paid Tier (`gemini-3.5-flash-lite`)

Google Cloud Vertex / AI Studio Commercial Pricing:
- Input / Prompt Tokens: **\$0.075 per 1,000,000 tokens**
- Output / Completion Tokens: **\$0.300 per 1,000,000 tokens**

$$\text{Input Cost} = 4.74\text{ M} \times \$0.075 = \$0.356$$
$$\text{Output Cost} = 1.17\text{ M} \times \$0.300 = \$0.351$$
$$\mathbf{\text{Total Estimated Cost (USD)}} = \mathbf{\$0.71\text{ USD}}$$

The entire Workstream C experimental campaign (C2 through C6) will cost **under \$1.00 USD total**.

---

## 5. Execution & Batching Proposal for User Approval

Given that the total requirement is ~7,266 calls and our 3 keys provide 4,500 requests/day:

1. **Option 1 (Full 300-Task Main Run across 2 Days)**:
   - **Day 1**: Run Tasks 1–150 ($\approx$ 3,600 calls, ~80% of daily quota).
   - **Day 2**: Run Tasks 151–300 plus C4, C5, C6 ($\approx$ 3,666 calls).
   - The harness automatically stops and saves progress if daily quota is reached, and resumes seamlessly on Day 2.
2. **Option 2 (Immediate 100-Task Single-Day Run)**:
   - Run 100 held-out tasks $\times$ 3 seeds $\times$ 7 arms ($\approx$ 2,420 calls).
   - Runs fully within today's remaining quota in ~1.2 hours.
3. **Option 3 (Dry-Run / Simulated Verification First)**:
   - Execute all C2–C6 runners with `--dry-run`. Validates all stats, zero task overlap, exact McNemar tests, bootstrap CIs, and figures instantaneously with 0 API calls before launching live execution.
