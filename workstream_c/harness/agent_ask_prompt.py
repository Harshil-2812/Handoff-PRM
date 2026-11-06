"""AgentAsk-style Prompted Clarifier Approximation.

NOTE ON PROVENANCE & METHODOLOGICAL INTEGRITY:
This implementation approximates the edge-level clarifier proposed in the paper:
  "AgentAsk: Multi-Agent Systems Need to Ask" (Lin et al., arXiv:2510.07593 / ACL).
The authors of AgentAsk train a specialized student model using Supervised Transfer
and E-GRPO reinforcement learning over failure traces. Because their trained weights
are not bundled in this pipeline, we use a zero-shot LLM prompt reflecting their
exact published edge-level taxonomy (Appendix D/F):
  - DG (Data Gap): missing required specifications, inputs, or edge constraints.
  - SC (Signal Corruption): distorted, truncated, or syntactically broken structures.
  - RD (Referential Drift): shifted names, alias confusion, or signature mismatches.
  - CG (Capability Gap): tasks or requirements outside the recipient's domain.

This is explicitly documented as an *approximation of AgentAsk's prompting architecture*,
not their trained model weights.
"""

AGENT_ASK_CLARIFIER_PROMPT = """You are AgentAsk, a lightweight, edge-level clarification module positioned between Agent A (a planner) and Agent B (a coder).
Your job is to inspect the HANDOFF message produced by Agent A from the PROBLEM statement before Agent B receives it.

Agent B will only see the handoff and must implement a complete, working solution with correct function name and logic.

Inspect the handoff for communication breakdown according to this taxonomy:
1. DG (Data Gap): Missing critical problem constraints, input/output types, or edge conditions.
2. SC (Signal Corruption): Inverted logic, contradictory requirements, or severely degraded text.
3. RD (Referential Drift): Renamed function signature, wrong entry-point name, or parameter drift.
4. CG (Capability Gap): Requesting impossible actions or hallucinated dependencies.

TASK:
Decide whether Agent B needs clarification to succeed.
If the handoff is complete and sufficient: output "STATUS: SUFFICIENT".
If clarification is needed:
output:
STATUS: NEEDS_CLARIFICATION
ERROR_TYPE: [DG | SC | RD | CG]
QUESTION: [One precise, minimal question directed to Agent A asking for the missing or corrected detail]

PROBLEM:
{problem}

CANDIDATE HANDOFF:
{handoff}
"""

AGENT_ASK_RESOLVER_PROMPT = """You are Agent A. AgentAsk has identified an issue in your handoff message and asked for clarification.

PROBLEM:
{problem}

ORIGINAL HANDOFF:
{handoff}

CLARIFICATION QUESTION FROM AGENTASK:
{question}

Provide the updated, fully complete handoff message for Agent B resolving the issue:
"""

