"""
Agent A (planner) and Agent B (coder).

Agent A reads the problem statement and produces a plan / spec -- this plan
IS the "handoff message" your whole project is about.

Agent B reads ONLY Agent A's handoff (never the original problem statement
directly) and writes the code. This is important: if Agent B could always
see the original problem, a bad handoff wouldn't matter as much. Forcing
Agent B to depend entirely on the handoff is what makes handoff quality
causally matter.

API key is hardcoded below for now -- rotate it before this repo goes anywhere.
"""

import os
import re
import threading
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY is not set in the environment or .env")

client = genai.Client(api_key=api_key)

# --- Model split -------------------------------------------------------------
# Free-tier quota is per-project-PER-MODEL, so putting the two agents on
# different models gives each its own daily bucket instead of sharing one.
# Agent A runs 1x per task, Agent B runs 5x (1 original + 4 corruptions),
# so B is the one that needs headroom.
#
# Verify these strings against client.models.list() -- see check_models() below.
MODEL_A = "gemini-3.5-flash-lite"   # ~15 RPM / 500 RPD
MODEL_B = "gemini-3.1-flash-lite"   # ~15 RPM / 500 RPD, separate bucket

# Requests-per-minute ceiling for each model. Keep these at or below the
# dashboard values -- overrunning RPM produces 429s long before the daily cap.
RPM_LIMITS = {
    MODEL_A: 15,
    MODEL_B: 15,
}

MAX_RETRIES = 5  # attempts per call when the API returns a retryable error

# No thinking_config: Gemini 3.x rejects the numeric thinking_budget parameter,
# so we leave thinking on and give it headroom.
GEN_CONFIG = types.GenerateContentConfig(max_output_tokens=4000)


class QuotaExhausted(RuntimeError):
    """Raised when a model's quota is gone and retrying within this run is futile.

    Distinct from a normal generation failure: the runner must NOT record this
    as a task outcome, because no outcome was actually observed.
    """


class _RateLimiter:
    """Spaces calls to one model so we stay under its requests-per-minute cap."""

    def __init__(self, rpm: int):
        self._min_interval = 60.0 / rpm
        self._last_call = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


_LIMITERS = {model: _RateLimiter(rpm) for model, rpm in RPM_LIMITS.items()}


def _retry_delay_seconds(err: genai_errors.ClientError) -> float | None:
    """Pulls Google's suggested retryDelay ('32s') out of a 429 payload."""
    details = (getattr(err, "details", None) or {})
    if not isinstance(details, dict):
        return None
    for detail in details.get("error", {}).get("details", []):
        if detail.get("@type", "").endswith("RetryInfo"):
            raw = str(detail.get("retryDelay", "")).rstrip("s")
            try:
                return float(raw)
            except ValueError:
                return None
    return None


def _is_daily_quota(err: genai_errors.ClientError) -> bool:
    """True if the 429 is the per-day cap rather than the per-minute one.

    A per-minute overrun is worth sleeping through. A daily cap is not -- it
    resets on Google's clock, not in the next 60 seconds.
    """
    return "PerDay" in str(getattr(err, "message", "")) or "PerDay" in str(err)


def _generate(prompt: str, model: str) -> str:
    """Single-turn call to Gemini, rate-limited and retried on 429/5xx."""
    limiter = _LIMITERS.get(model)

    for attempt in range(MAX_RETRIES):
        if limiter:
            limiter.wait()
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=GEN_CONFIG,
            )
            return (response.text or "").strip()

        except genai_errors.ClientError as err:
            if getattr(err, "code", None) != 429:
                raise  # 400/404 etc. are our bug, not a transient condition
            if _is_daily_quota(err):
                raise QuotaExhausted(f"Daily quota exhausted for {model}") from err
            delay = _retry_delay_seconds(err) or (2.0 ** attempt)
            print(f"    rate limited on {model}, sleeping {delay:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(delay + 1.0)

        except genai_errors.ServerError:
            delay = 2.0 ** attempt
            print(f"    server error on {model}, retrying in {delay:.0f}s")
            time.sleep(delay)

    raise QuotaExhausted(f"{model} still rate limited after {MAX_RETRIES} attempts")


def check_models() -> None:
    """Prints the model IDs this key can reach. Run before a long job."""
    available = {m.name.removeprefix("models/") for m in client.models.list()}
    for label, model in (("MODEL_A", MODEL_A), ("MODEL_B", MODEL_B)):
        mark = "OK " if model in available else "NOT FOUND"
        print(f"{mark} {label} = {model}")
    print(f"\n{len(available)} models visible to this key:")
    for name in sorted(available):
        print(f"  {name}")


def agent_a_plan(problem: str) -> str:
    """
    Agent A: reads the problem, produces a handoff message for Agent B.
    This handoff should contain: a restated problem, an approach/plan,
    key edge cases, and the required function signature.
    """
    prompt = f"""You are Agent A in a two-agent coding pipeline. Your job is to
read a problem statement and write a HANDOFF MESSAGE for Agent B, who will
write the actual code. Agent B will NOT see the original problem -- only
your handoff message. So your handoff must contain everything Agent B needs.

Problem statement:
{problem}

Write a handoff message that includes:
- A clear restatement of what function needs to be built
- The exact function name/signature to use
- Key edge cases to handle
- A brief plan/approach

Output ONLY the handoff message, nothing else."""
    handoff = _generate(prompt, MODEL_A)
    if not handoff:
        raise RuntimeError("Agent A returned an empty handoff")
    return handoff


def agent_b_code(handoff_message: str, entry_point: str) -> str:
    """
    Agent B: reads ONLY the handoff message, writes Python code.
    Returns the extracted code (str). Does not see the original problem.
    """
    prompt = f"""You are Agent B in a two-agent coding pipeline. You did NOT
see the original problem -- you only have the handoff message below from
Agent A. Write Python code based ENTIRELY on this handoff.

Handoff message from Agent A:
{handoff_message}

Write a single Python function named `{entry_point}` that solves this.
Output ONLY a Python code block, nothing else."""

    text = _generate(prompt, MODEL_B)
    if not text:
        raise RuntimeError("Agent B returned an empty response")
    return _extract_code(text)


def _extract_code(text: str) -> str:
    """Pulls code out of a ```python ... ``` block, or returns raw text if no block found."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
