"""
nodes.py — LangGraph Node Implementations (Handoff-PRM)
========================================================
Defines the actual functions that run at each graph node.
These are the concrete implementations of the stubs in graph.py.

Paper section: §VIII (System Integration), §III-B (Architecture)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from typing import Optional
import google.generativeai as genai


# ─── Agent A Node ─────────────────────────────────────────────────────────────

AGENT_A_SYSTEM_PROMPT = """You are a senior software engineer completing the
analysis phase of a coding task. Your job is to produce a detailed handoff
message for a colleague who will implement the solution.

Your handoff MUST include:
1. The exact function signature (entry_point name, parameters, return type)
2. The core algorithm or approach to use
3. All edge cases and constraints that must be handled
4. Any relevant data structures or helper patterns to apply
5. Expected behavior for typical inputs and boundary conditions

Be specific and technical. Do NOT write any implementation code — only the
plan/specification that your colleague needs to implement correctly.
"""

def agent_a_node(state: dict, feedback: Optional[str] = None) -> dict:
    """
    Agent A: generates a handoff message from the problem statement.

    If `feedback` is provided (from a previous gate BLOCK), the agent is
    instructed to address the specific weakness identified by the gate.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    problem = state["problem"]
    entry_point = state.get("entry_point", "")

    if feedback:
        user_message = (
            f"Problem:\n{problem}\n\n"
            f"Previous handoff was rejected. Reviewer feedback: {feedback}\n\n"
            f"Please generate an improved handoff that addresses this feedback."
        )
    else:
        user_message = f"Problem:\n{problem}\n\nEntry point: {entry_point}"

    chat = model.start_chat(history=[])
    response = chat.send_message(
        [{"role": "user", "parts": [AGENT_A_SYSTEM_PROMPT + "\n\n" + user_message]}]
    )
    state["handoff"] = response.text
    return state


# ─── Agent B Node ─────────────────────────────────────────────────────────────

AGENT_B_SYSTEM_PROMPT = """You are a software engineer implementing a function
based on a detailed handoff specification. Implement ONLY the specified function.
Return ONLY valid Python code, no explanation.
"""

def agent_b_node(state: dict, use_raw_problem: bool = False) -> dict:
    """
    Agent B: implements the function based on the handoff (or raw problem).

    If `use_raw_problem` is True (break-glass / always-fallback mode),
    Agent B receives the original problem directly, bypassing the handoff.
    This is the "always-fallback" baseline from the paper.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    if use_raw_problem:
        # Break-glass: ignore the handoff, use raw problem statement
        prompt = (
            AGENT_B_SYSTEM_PROMPT + "\n\n"
            f"Problem:\n{state['problem']}\n\n"
            f"Entry point: {state.get('entry_point', '')}\n\n"
            "Implement the function directly from this specification."
        )
    else:
        prompt = (
            AGENT_B_SYSTEM_PROMPT + "\n\n"
            f"Handoff specification:\n{state['handoff']}\n\n"
            f"Entry point: {state.get('entry_point', '')}\n\n"
            "Implement the function according to the handoff specification."
        )

    chat = model.start_chat(history=[])
    response = chat.send_message([{"role": "user", "parts": [prompt]}])
    state["agent_b_output"] = response.text
    # Note: actual pass@1 evaluation requires running the test harness
    # state["agent_b_passed"] is set by the test harness after execution
    return state


# ─── Fallback Agent B Node ────────────────────────────────────────────────────

def fallback_agent_b_node(state: dict) -> dict:
    """Break-glass fallback: Agent B on raw problem (skips handoff entirely)."""
    return agent_b_node(state, use_raw_problem=True)
