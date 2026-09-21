"""
Agent A (planner) and Agent B (coder).

Agent A reads the problem statement and produces a plan / spec -- this plan
IS the "handoff message" your whole project is about.

Agent B reads ONLY Agent A's handoff (never the original problem statement
directly) and writes the code. This is important: if Agent B could always
see the original problem, a bad handoff wouldn't matter as much. Forcing
Agent B to depend entirely on the handoff is what makes handoff quality
causally matter.

Multi-key parallel support: set GEMINI_KEYS in .env as a Python list literal,
e.g. GEMINI_KEYS=['key1','key2','key3']. Each key gets its own client and
rate-limiter; a ThreadPoolExecutor with N workers runs N tasks simultaneously,
one per key.
"""

import ast
import os
import queue
import re
import threading
import time
from contextlib import contextmanager

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv()

# ─── Key loading ──────────────────────────────────────────────────────────────
# Prefer GEMINI_KEYS (list) for parallel runs; fall back to single GEMINI_API_KEY.

def _load_keys() -> list[str]:
    raw = os.getenv("GEMINI_KEYS", "").strip()
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and parsed:
                return [str(k).strip() for k in parsed if str(k).strip()]
        except (ValueError, SyntaxError):
            pass
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        return [single]
    raise RuntimeError(
        "No API keys found. Set GEMINI_KEYS=['key1','key2',...] "
        "or GEMINI_API_KEY=key in your .env"
    )

GEMINI_KEYS = _load_keys()
N_KEYS = len(GEMINI_KEYS)

# ─── Model names ──────────────────────────────────────────────────────────────
# Free-tier quota is per-project-PER-MODEL, so putting the two agents on
# different models gives each its own daily bucket instead of sharing one.
MODEL_A = "gemini-3.5-flash-lite"
MODEL_B = "gemini-3.5-flash-lite"

# Per-key RPM cap — set below the 15 RPM hard limit to leave a safety buffer.
RPM_PER_KEY = 10

MAX_RETRIES = 5
GEN_CONFIG = types.GenerateContentConfig(max_output_tokens=4000)


# ─── Public exceptions ────────────────────────────────────────────────────────

class QuotaExhausted(RuntimeError):
    """Raised when a key's daily quota is gone — not worth retrying this run."""


# ─── Rate limiter (one per key, not shared) ───────────────────────────────────

class _RateLimiter:
    """Enforces a minimum interval between calls on one key."""

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


# ─── Key pool ─────────────────────────────────────────────────────────────────
# A Queue of (client, limiter) pairs. Each worker thread grabs one, uses it,
# then returns it — UNLESS the key hit its daily quota, in which case the key
# is retired (removed permanently) and the thread retries with a fresh key.

class _KeyPool:
    def __init__(self, keys: list[str], rpm: int):
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._active: set = set()          # id(entry) of keys still alive
        self._entries = []
        for key in keys:
            c = genai.Client(api_key=key)
            lim = _RateLimiter(rpm)
            entry = (c, lim)
            self._entries.append(entry)
            self._active.add(id(entry))
            self._q.put(entry)

    @contextmanager
    def acquire(self):
        """Block until a (client, limiter) pair is free.

        Yields (entry, retire_callback).  Call retire_callback() inside the
        with-block to permanently remove the key from the pool instead of
        returning it to the queue.
        """
        item = self._q.get()
        retired = [False]

        def retire():
            """Remove this key from the pool permanently."""
            retired[0] = True
            with self._lock:
                self._active.discard(id(item))
                try:
                    self._entries.remove(item)
                except ValueError:
                    pass
            print(f"    [key pool] key retired (daily quota) — "
                  f"{self.size()} key(s) still active")

        try:
            yield item, retire
        finally:
            if not retired[0]:
                self._q.put(item)

    def size(self) -> int:
        """Number of keys still active (not yet retired)."""
        with self._lock:
            return len(self._active)

    def all_clients(self):
        with self._lock:
            return [c for c, _ in self._entries]


# One shared pool for the whole process (thread-safe).
_POOL = _KeyPool(GEMINI_KEYS, RPM_PER_KEY)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _retry_delay_seconds(err: genai_errors.ClientError) -> float | None:
    details = getattr(err, "details", None) or {}
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
    return "PerDay" in str(getattr(err, "message", "")) or "PerDay" in str(err)


# ─── Core generation (acquires a key from the pool) ───────────────────────────

def _generate(prompt: str, model: str) -> str:
    """Grabs a free key from the pool, calls Gemini, returns it. Thread-safe.

    When a key hits its DAILY quota it is retired from the pool and this
    function automatically retries with the next available key.  Only when
    the pool is fully empty does QuotaExhausted propagate to the caller.
    """
    while True:
        if _POOL.size() == 0:
            raise QuotaExhausted("All API keys have exhausted their daily quota")

        with _POOL.acquire() as ((client, limiter), retire):
            for attempt in range(MAX_RETRIES):
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
                        raise
                    if _is_daily_quota(err):
                        # Retire this key and let the outer while loop
                        # pick up a fresh one from the pool.
                        retire()
                        break   # exits the for-loop; outer while retries
                    delay = _retry_delay_seconds(err) or (2.0 ** attempt)
                    print(f"    [key pool] rate limited, sleeping {delay:.0f}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay + 1.0)

                except genai_errors.ServerError:
                    delay = 2.0 ** attempt
                    print(f"    [key pool] server error, retrying in {delay:.0f}s")
                    time.sleep(delay)
            else:
                # for-loop exhausted all MAX_RETRIES without a daily-quota hit
                raise QuotaExhausted(f"{model} still rate limited after {MAX_RETRIES} attempts")
            # If we broke out (key retired), outer while loop continues.



# ─── Public API (unchanged signatures) ───────────────────────────────────────

def check_models() -> None:
    """Prints models reachable by the first key. Run before a long job."""
    first_client = _POOL.all_clients()[0]
    available = {m.name.removeprefix("models/") for m in first_client.models.list()}
    for label, model in (("MODEL_A", MODEL_A), ("MODEL_B", MODEL_B)):
        mark = "OK " if model in available else "NOT FOUND"
        print(f"{mark} {label} = {model}")
    print(f"\n{len(available)} models visible to key[0]:")
    for name in sorted(available):
        print(f"  {name}")
    print(f"\nKey pool size: {N_KEYS} key(s) loaded")


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
    """Pulls code out of a ```python ... ``` block, or returns raw text."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
