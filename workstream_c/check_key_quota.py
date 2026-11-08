"""Connectivity test and quota check for all available API keys.
Does NOT print any key values.
"""

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workstream_c.harness.key_manager import find_api_keys
from google import genai

keys = find_api_keys()
print(f"Total keys found in .env: {len(keys)}")

print(f"{'Key Index':<12} {'Status':<16} {'Error Details'}")
print("-" * 50)

working_keys = []
exhausted_keys = []
invalid_keys = []

for idx, key in enumerate(keys, 1):
    try:
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model="models/gemini-3.5-flash-lite",
            contents="hi",
            config={"max_output_tokens": 5}
        )
        print(f"Key {idx:<8} WORKING")
        working_keys.append(idx)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            print(f"Key {idx:<8} EXHAUSTED        Quota limit reached")
            exhausted_keys.append(idx)
        else:
            err_summary = err_str.split("\n")[0][:40]
            print(f"Key {idx:<8} FAILED           {err_summary}")
            invalid_keys.append(idx)

print("-" * 50)
print(f"Summary: {len(working_keys)} working, {len(exhausted_keys)} quota-exhausted, {len(invalid_keys)} failed/other.")
