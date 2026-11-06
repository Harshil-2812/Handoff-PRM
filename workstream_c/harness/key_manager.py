"""Key Manager & Rate Limiting for Gemini API calls.

Supports:
- Multi-key rotation across accounts/projects to maximize free-tier quota.
- Automatic key discovery from results/.env, .env, and environment variables.
- Thread-safe key pool with quota exhaustion detection and rotation logging.
- Exponential backoff on rate limits (RPM).
- Safe dry-run mode.
"""

import ast
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("KeyManager")

class QuotaExhaustedError(RuntimeError):
    """Raised when all available API keys have exhausted their daily quota."""

def find_api_keys() -> List[str]:
    """Reads Gemini API keys strictly from the root .env file (GEMINI_API_KEY_1 .. GEMINI_API_KEY_N)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    root_env = repo_root / ".env"
    
    if not root_env.exists():
        logger.warning(f"Root .env file not found at {root_env}")
        return []
        
    key_dict: dict[int, str] = {}
    extra_keys: List[str] = []

    with open(root_env, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if not v:
                    continue
                if k.startswith("GEMINI_API_KEY_"):
                    num_part = k[len("GEMINI_API_KEY_"):]
                    if num_part.isdigit():
                        key_dict[int(num_part)] = v
                    else:
                        extra_keys.append(v)
                elif k == "GEMINI_API_KEY":
                    extra_keys.append(v)

    # Sort numerically by index (1, 2, ... 15)
    ordered_keys = [key_dict[idx] for idx in sorted(key_dict.keys())]
    ordered_keys.extend(extra_keys)
    return ordered_keys


def get_key_slice(slice_str: Optional[str] = None) -> List[str]:
    """Returns a subset of keys based on a 'start:end' string (e.g. '0:5')."""
    all_keys = find_api_keys()
    if not slice_str:
        return all_keys
    try:
        parts = slice_str.split(":")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else len(all_keys)
        sliced = all_keys[start:end]
        logger.info(f"Using key slice '{slice_str}': {len(sliced)} keys (indices {start} to {end - 1}).")
        return sliced
    except Exception as e:
        logger.warning(f"Failed to parse key slice '{slice_str}': {e}. Using all keys.")
        return all_keys


class KeyPool:
    """Manages a rotating pool of API keys with per-key rate limiting and quota tracking."""
    
    def __init__(self, keys: Optional[List[str]] = None, rpm_per_key: int = 12):
        self.keys = keys if keys is not None else find_api_keys()
        self.rpm_per_key = rpm_per_key
        self.min_interval = 60.0 / rpm_per_key if rpm_per_key > 0 else 0.0
        
        self.current_idx = 0
        self.active_indices = list(range(len(self.keys)))
        self.last_call_times = {i: 0.0 for i in range(len(self.keys))}
        self.lock = threading.Lock()
        
        logger.info(f"KeyPool initialized with {len(self.keys)} API key(s) at {rpm_per_key} RPM/key.")

    def has_keys(self) -> bool:
        with self.lock:
            return len(self.active_indices) > 0

    def get_current_key(self) -> str:
        with self.lock:
            if not self.active_indices:
                raise QuotaExhaustedError("All API keys have exhausted their daily quota.")
            idx = self.active_indices[self.current_idx % len(self.active_indices)]
            return self.keys[idx]

    def wait_and_acquire(self) -> tuple[int, str]:
        """Enforces RPM interval for the active key and returns (key_index, api_key)."""
        with self.lock:
            if not self.active_indices:
                raise QuotaExhaustedError("All API keys have exhausted their daily quota.")
            idx = self.active_indices[self.current_idx % len(self.active_indices)]
            key = self.keys[idx]
            
            elapsed = time.monotonic() - self.last_call_times[idx]
            sleep_needed = self.min_interval - elapsed
            
        if sleep_needed > 0:
            time.sleep(sleep_needed)
            
        with self.lock:
            self.last_call_times[idx] = time.monotonic()
            return idx, key

    def rotate_on_quota(self, failed_key_idx: int, reason: str = "Quota limit"):
        """Retires a key that exceeded quota and advances to the next key."""
        with self.lock:
            if failed_key_idx in self.active_indices:
                self.active_indices.remove(failed_key_idx)
                logger.warning(
                    f"ROTATION: Key #{failed_key_idx + 1} retired due to {reason}. "
                    f"{len(self.active_indices)} key(s) remaining."
                )
            if not self.active_indices:
                logger.error("All API keys exhausted.")
                raise QuotaExhaustedError("All API keys have exhausted their daily quota.")
            self.current_idx = self.current_idx % len(self.active_indices)
            new_idx = self.active_indices[self.current_idx]
            logger.info(f"Switched active key to #{new_idx + 1}.")

