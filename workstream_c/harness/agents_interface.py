"""LLM Agent Interface: Agent A, Agent B, Judge, Clarifiers with multi-key rotation, caching, and token logging."""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types
from google.genai import errors as genai_errors

from workstream_c.harness.agent_ask_prompt import (
    AGENT_ASK_CLARIFIER_PROMPT,
    AGENT_ASK_RESOLVER_PROMPT,
)
from workstream_c.harness.config import CACHE_DIR, REPO_ROOT
from workstream_c.harness.key_manager import KeyPool, QuotaExhaustedError

logger = logging.getLogger("AgentsInterface")

# Frozen prompts from Workstream B (b06_llm_judge.py)
JUDGE_PROMPTS = {
    "generic": (
        "You are reviewing a handoff message written by a planning agent for a coding agent.\n"
        "Rate from 0 to 100 how sufficient the HANDOFF is for the coding agent to implement a correct solution "
        "(100 = fully sufficient). Reply with ONLY the integer.\n\nHANDOFF:\n{handoff}"
    ),
    "targeted": (
        "You review handoffs for a coding agent. Check for: (1) dropped constraints or edge cases, (2) inverted or "
        "changed objective, (3) wrong algorithm named, (4) wrong function name/signature. Rate 0-100 how sufficient "
        "and correct the HANDOFF is (100 = no defect). Reply with ONLY the integer.\n\nHANDOFF:\n{handoff}"
    ),
    "with_problem": (
        "Original problem:\n{problem}\n\nHandoff written from it:\n{handoff}\n\nRate 0-100 how faithfully and "
        "sufficiently the handoff conveys the problem for a coding agent (100 = fully). Reply with ONLY the integer."
    ),
}

AGENT_A_SYSTEM_PROMPT = """You are Agent A in a two-agent coding pipeline. Your job is to
read a problem statement and write a HANDOFF MESSAGE for Agent B, who will
write the actual code. Agent B will NOT see the original problem -- only
your handoff message. So your handoff must contain everything Agent B needs.

Your handoff message should include:
- A clear restatement of what the code must do
- The exact function signature / entry point name
- The approach and algorithm
- Any critical edge cases and constraints

Write ONLY the handoff message. Do not write the full implementation code."""

AGENT_B_SYSTEM_PROMPT = """You are Agent B in a two-agent coding pipeline. You did NOT
see the original problem -- you only have the handoff message below from
Agent A. Write Python code based ENTIRELY on this handoff.

Handoff message from Agent A:
{handoff}

Write a single Python function named `{entry_point}` that solves this.
Output ONLY a Python code block, nothing else."""

AGENT_B_RAW_PROMPT = """You are Agent B in a coding pipeline. Write a single Python function named `{entry_point}` that solves the problem below.
Output ONLY a Python code block, nothing else.

Problem:
{problem}"""


class LLMAgentClient:
    """Handles LLM calls with persistent caching, token accounting, and key rotation."""

    def __init__(
        self,
        key_pool: Optional[KeyPool] = None,
        cache_path: Optional[Path] = None,
        dry_run: bool = False,
    ):
        self.key_pool = key_pool or KeyPool()
        self.dry_run = dry_run
        self.cache_path = cache_path or (CACHE_DIR / "llm_cache.json")
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.exists():
            try:
                with open(self.cache_path, encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached LLM responses.")
            except Exception as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}")

    def _save_cache_entry(self, cache_key: str, data: Dict[str, Any]):
        self.cache[cache_key] = data
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=1)
        except Exception as e:
            logger.warning(f"Failed to save cache entry: {e}")

    def _cache_key(self, model: str, prompt: str, temperature: float, seed: int) -> str:
        h = hashlib.sha256()
        h.update(f"{model}_{temperature}_{seed}_{prompt}".encode("utf-8"))
        return h.hexdigest()

    def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
        max_output_tokens: int = 4000,
    ) -> Dict[str, Any]:
        """Generates text from Gemini model with caching, key rotation, and metrics."""
        cache_k = self._cache_key(model, prompt, temperature, seed)
        if cache_k in self.cache:
            entry = dict(self.cache[cache_k])
            entry["from_cache"] = True
            return entry

        # Dry run simulation mode
        if self.dry_run:
            simulated_text = self._mock_response(prompt)
            data = {
                "text": simulated_text,
                "latency_ms": 10.0,
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(simulated_text.split()),
                "total_tokens": len(prompt.split()) + len(simulated_text.split()),
                "from_cache": False,
            }
            self._save_cache_entry(cache_k, data)
            return data

        # Live API call with key rotation
        max_attempts = 10
        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )

        for attempt in range(max_attempts):
            key_idx, api_key = self.key_pool.wait_and_acquire()
            client = genai.Client(api_key=api_key)
            t0 = time.perf_counter()
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config,
                )
                ms = (time.perf_counter() - t0) * 1e3
                text = (resp.text or "").strip()

                p_tokens = getattr(resp.usage_metadata, "prompt_token_count", len(prompt.split())) if hasattr(resp, "usage_metadata") and resp.usage_metadata else len(prompt.split())
                c_tokens = getattr(resp.usage_metadata, "candidates_token_count", len(text.split())) if hasattr(resp, "usage_metadata") and resp.usage_metadata else len(text.split())

                data = {
                    "text": text,
                    "latency_ms": ms,
                    "prompt_tokens": p_tokens or len(prompt.split()),
                    "completion_tokens": c_tokens or len(text.split()),
                    "total_tokens": (p_tokens or 0) + (c_tokens or 0),
                    "from_cache": False,
                }
                self._save_cache_entry(cache_k, data)
                return data

            except genai_errors.ClientError as err:
                status_code = getattr(err, "code", None)
                err_msg = str(err)
                if status_code == 429 or "RESOURCE_EXHAUSTED" in err_msg or "PerDay" in err_msg:
                    logger.warning(f"Key #{key_idx + 1} hit quota error: {err_msg[:120]}")
                    self.key_pool.rotate_on_quota(key_idx, reason=f"HTTP 429: {err_msg[:80]}")
                    continue
                else:
                    # Other client error (e.g. invalid arg)
                    logger.error(f"ClientError: {err}")
                    raise err

            except Exception as exc:
                wait_time = min(2 ** attempt, 30)
                logger.warning(f"Attempt {attempt + 1}/{max_attempts} error: {exc}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

        raise RuntimeError(f"Failed to generate content after {max_attempts} attempts.")

    def _mock_response(self, prompt: str) -> str:
        """Deterministic dry-run mock response generator."""
        if "Agent A in a two-agent coding pipeline" in prompt:
            return "Handoff Specification:\n- Implement the function as requested.\n- Ensure all edge cases and constraints are satisfied.\n- Function signature must match exactly."
        elif "You review handoffs for a coding agent" in prompt or "Rate 0-100" in prompt:
            return "85"
        elif "AgentAsk" in prompt:
            return "STATUS: SUFFICIENT"
        elif "Write Python code" in prompt:
            m = re.search(r"named `(\w+)`", prompt)
            ep = m.group(1) if m else "solution"
            return f"```python\ndef {ep}(*args, **kwargs):\n    return None\n```"
        return "OK"

    # High-level wrappers
    def plan_agent_a(self, problem: str, model: str, temperature: float = 0.0, seed: int = 42) -> Dict[str, Any]:
        prompt = f"{AGENT_A_SYSTEM_PROMPT}\n\nProblem:\n{problem}"
        return self.generate(prompt, model=model, temperature=temperature, seed=seed)

    def code_agent_b(
        self,
        handoff_or_problem: str,
        entry_point: str,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
        is_raw_problem: bool = False,
    ) -> Dict[str, Any]:
        if is_raw_problem:
            prompt = AGENT_B_RAW_PROMPT.format(problem=handoff_or_problem, entry_point=entry_point)
        else:
            prompt = AGENT_B_SYSTEM_PROMPT.format(handoff=handoff_or_problem, entry_point=entry_point)
        return self.generate(prompt, model=model, temperature=temperature, seed=seed)

    def judge_handoff(
        self,
        handoff: str,
        problem: str,
        prompt_type: str = "with_problem",
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        seed: int = 42,
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        template = JUDGE_PROMPTS.get(prompt_type, JUDGE_PROMPTS["with_problem"])
        prompt = template.format(handoff=handoff, problem=problem)
        res = self.generate(prompt, model=model, temperature=temperature, seed=seed)
        text = res.get("text", "")
        m = re.search(r"\b(\d{1,3})\b", text)
        score = min(100.0, float(m.group(1))) if m else None
        return score, res

    def agent_ask_clarify(
        self,
        handoff: str,
        problem: str,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        prompt = AGENT_ASK_CLARIFIER_PROMPT.format(handoff=handoff, problem=problem)
        res = self.generate(prompt, model=model, temperature=temperature, seed=seed)
        text = res.get("text", "")
        needs_clarification = "NEEDS_CLARIFICATION" in text
        question = ""
        m = re.search(r"QUESTION:\s*(.+)", text, re.DOTALL)
        if m:
            question = m.group(1).strip()
        return needs_clarification, question, res

    def agent_ask_resolve(
        self,
        handoff: str,
        problem: str,
        question: str,
        model: str,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> Dict[str, Any]:
        prompt = AGENT_ASK_RESOLVER_PROMPT.format(handoff=handoff, problem=problem, question=question)
        return self.generate(prompt, model=model, temperature=temperature, seed=seed)

