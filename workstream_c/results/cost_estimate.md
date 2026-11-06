# Workstream C: Token, Quota & Cost Estimate for C2 Main Run

**Date**: November 6, 2026  
**Empirical Baseline**: 20-task live pilot (`workstream_c/results/pilot_report.md`)  
**Target Matrix**: 300 held-out tasks $\times$ 3 seeds (42, 43, 44) $\times$ 7 arms $\times$ 2 conditions (clean vs. corrupted)  
**Total Target Trials**: 12,600 trials  
**Model Config**: `gemini-3.5-flash-lite` across Agent A, Agent B, and Judge  
**Quota Context**: Google Gemini API Free Tier with 3 configured accounts/keys

---

## 1. Call Volume & Prompt-Level Caching Analysis

In a naive un-cached evaluation, 12,600 trials would imply $>20,000$ API calls. However, because our harness implements deterministic disk caching on `(model, prompt, temperature, seed)`:
1. **Agent A Planning**: Exactly 1 call per task per seed = $300 \times 3 = \mathbf{900\text{ calls}}$.
2. **Agent B on Clean Handoffs**: Generated on the `ungated` arm and reused by all subsequent passing arms = $\mathbf{900\text{ calls}}$.
3. **Agent B on Corrupted Handoffs**: Generated on `ungated` corrupted condition and reused by passing arms = $\mathbf{900\text{ calls}}$.
4. **Agent B on Raw Problem Fallbacks**: The raw problem code generation is identical across seeds for temperature 0 = $300 \times 1 = \mathbf{300\text{ calls}}$.
5. **Stage 1 PRM Scoring**: 100% local CPU Random Forest inference = $\mathbf{0\text{ API calls}}$.
6. **Deterministic Arm**: Local signature and word count check = $\mathbf{0\text{ API calls}}$.
7. **LLM Judge**: Evaluates unique clean and corrupted handoffs = $300 \times 3 \times 2 = \mathbf{1,800\text{ calls}}$.
8. **AgentAsk Clarifier**: Intercepts unique candidate handoffs = $\mathbf{1,800\text{ calls}}$ (with ~10% requiring Agent A refinement = $+180\text{ calls}$).

### Summary of Unique API Calls
$$\text{Total Unique API Calls} = 900 + 900 + 900 + 300 + 1,800 + 1,980 \approx \mathbf{6,780\text{ calls}}$$

---

## 2. Token Volume & Financial Cost Estimate

Based on empirical token measurements from the pilot:
- **Mean Prompt Tokens per Call**: ~650 tokens
- **Mean Completion Tokens per Call**: ~160 tokens
- **Total Prompt Tokens**: $6,780 \times 650 \approx \mathbf{4,407,000\text{ tokens}}$ (~4.41 M tokens)
- **Total Completion Tokens**: $6,780 \times 160 \approx \mathbf{1,085,000\text{ tokens}}$ (~1.09 M tokens)

### Commercial API Cost Calculation (`gemini-3.5-flash-lite` rates)
- Prompt Tokens: $4.41\text{ M} \times \$0.075 / \text{M tokens} = \$0.331$
- Completion Tokens: $1.09\text{ M} \times \$0.300 / \text{M tokens} = \$0.327$
- **Total Estimated Dollar Cost**: **\$0.66 USD** (under \$1.00 USD total).

---

## 3. Free-Tier Quota Feasibility & Batching Strategy

### Google Gemini Free-Tier Constraints
- **Rate Limit**: 15 Requests Per Minute (RPM) per model per project.
  - Harness limits requests to **12 RPM per key** (leaving a 20% safety margin).
  - With **3 keys**, aggregate throughput is **36 RPM** (approx. 2,160 calls/hour).
- **Daily Quota Limit**: 1,500 requests per day per project.
  - Across 3 distinct accounts/keys: $3 \times 1,500 = \mathbf{4,500\text{ requests / day}}$.

### Feasibility Assessment
$$\frac{6,780\text{ unique calls}}{4,500\text{ requests / day}} \approx \mathbf{1.51\text{ daily quota cycles}}$$

The full 300-task, 3-seed evaluation exceeds a single 24-hour free-tier quota window by ~2,280 calls.

### Execution Options
1. **Option A (2-Day Free Tier Batching — Zero Dollar Cost)**:
   - Day 1: Run 150 tasks $\times$ 3 seeds (~3,400 calls). Hits ~75% of daily quota.
   - Day 2: Run remaining 150 tasks $\times$ 3 seeds. The harness will automatically resume exactly where Day 1 ended without re-spending quota.
2. **Option B (Immediate 100-Task Main Run — 1-Day Free Tier)**:
   - 100 tasks $\times$ 3 seeds $\approx$ 2,260 calls.
   - Completes fully within ~1.2 hours in a single daily quota window.
3. **Option C (Dry-Run / Simulated Verification)**:
   - Validates all statistics, McNemar tests, bootstrap CIs, and figures instantaneously with 0 API calls.
