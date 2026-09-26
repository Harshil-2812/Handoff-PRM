"""
graph.py — LangGraph Workflow for Handoff-PRM
==============================================
Defines the LangGraph StateGraph with three nodes:
  agent_a  → generates the handoff message
  gate     → CascadeGate decides PASS or BLOCK
  agent_b  → executes the task (only reached on PASS)

Conditional edge from gate:
  PASS  → agent_b  (proceed with handoff as-is)
  BLOCK → agent_a  (re-request with gate's reason, up to max_retries)
           OR → fallback_agent_b (raw task, break-glass mode)

Paper section: §VIII (System Integration)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print("WARNING: langgraph not installed. Graph definition is illustrative only.")

from typing import TypedDict, Annotated, Optional
from cascade_gate import CascadeGate, make_gate_node


# ─── State Schema ─────────────────────────────────────────────────────────────

class HandoffState(TypedDict):
    """State passed between graph nodes."""
    # Task context (immutable)
    task_id: str
    problem: str
    entry_point: str

    # Agent A output
    handoff: str
    retry_count: int
    max_retries: int

    # Gate output
    gate_decision: Optional[str]   # "PASS" or "BLOCK"
    gate_score: Optional[float]
    gate_reason: Optional[str]
    gate_stage: Optional[int]

    # Agent B output
    agent_b_output: Optional[str]
    agent_b_passed: Optional[bool]

    # Metadata
    messages: Annotated[list, add_messages] if LANGGRAPH_AVAILABLE else list


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph(
    agent_a_fn,
    agent_b_fn,
    gate: CascadeGate,
    max_retries: int = 2,
):
    """
    Build and compile the LangGraph workflow.

    Args:
        agent_a_fn: callable(state) -> state with 'handoff' filled
        agent_b_fn: callable(state) -> state with 'agent_b_output', 'agent_b_passed' filled
        gate: CascadeGate instance
        max_retries: max times Agent A is re-invoked before fallback

    Returns:
        Compiled StateGraph (or None if LangGraph not available)
    """
    if not LANGGRAPH_AVAILABLE:
        print("LangGraph not available — cannot compile graph.")
        return None

    gate_node = make_gate_node(gate)

    def agent_a_node(state: HandoffState) -> HandoffState:
        if state.get("gate_reason"):
            # Retry with feedback from gate
            state = agent_a_fn(state, feedback=state["gate_reason"])
        else:
            state = agent_a_fn(state)
        state["retry_count"] = state.get("retry_count", 0)
        return state

    def gate_router(state: HandoffState) -> str:
        """Conditional edge: decide next node after gate."""
        if state["gate_decision"] == "PASS":
            return "agent_b"
        if state.get("retry_count", 0) >= state.get("max_retries", max_retries):
            return "fallback_agent_b"
        state["retry_count"] = state.get("retry_count", 0) + 1
        return "agent_a"

    def fallback_agent_b_node(state: HandoffState) -> HandoffState:
        """Break-glass: run Agent B directly on the raw problem (Always-Fallback policy)."""
        state = agent_b_fn(state, use_raw_problem=True)
        return state

    graph = StateGraph(HandoffState)
    graph.add_node("agent_a", agent_a_node)
    graph.add_node("gate", gate_node)
    graph.add_node("agent_b", agent_b_fn)
    graph.add_node("fallback_agent_b", fallback_agent_b_node)

    graph.set_entry_point("agent_a")
    graph.add_edge("agent_a", "gate")
    graph.add_conditional_edges(
        "gate",
        gate_router,
        {
            "agent_b": "agent_b",
            "agent_a": "agent_a",
            "fallback_agent_b": "fallback_agent_b",
        }
    )
    graph.add_edge("agent_b", END)
    graph.add_edge("fallback_agent_b", END)

    return graph.compile()


# ─── Example / Smoke Test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LangGraph Handoff-PRM graph definition.")
    print("Nodes: agent_a → gate → [agent_b | fallback_agent_b | agent_a (retry)]")
    print()
    print("To instantiate:")
    print("  gate = CascadeGate(model_path='../../handoff_prm.joblib', backbone='primary')")
    print("  compiled = build_graph(my_agent_a, my_agent_b, gate)")
    print("  result = compiled.invoke({'task_id': 't001', 'problem': '...', 'entry_point': 'foo'})")
