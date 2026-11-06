"""Execution Sandbox: Grades generated code against automated unit tests with strict isolation and timeouts."""

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

PYTHON_EXE = sys.executable

RUNNER_TEMPLATE = '''
import sys
import json

code_str = {code_repr}
entry_point = {entry_point_repr}
test_code_str = {test_code_repr}

namespace = {{}}
try:
    exec(code_str, namespace)
except Exception as e:
    print(json.dumps({{"passed": False, "error": f"Compilation/Import Error: {{type(e).__name__}}: {{str(e)}}"}}))
    sys.exit(0)

try:
    exec(test_code_str, namespace)
except Exception as e:
    print(json.dumps({{"passed": False, "error": f"Test Setup Error: {{type(e).__name__}}: {{str(e)}}"}}))
    sys.exit(0)

fn = namespace.get(entry_point)
if fn is None:
    print(json.dumps({{"passed": False, "error": f"Function '{{entry_point}}' not found in generated code."}}))
    sys.exit(0)

test_fn = namespace.get("test")
if test_fn is None:
    # Some HumanEval tests use candidate/check rather than test(fn)
    check_fn = namespace.get("check")
    if check_fn is not None:
        try:
            check_fn(fn)
            print(json.dumps({{"passed": True, "error": ""}}))
            sys.exit(0)
        except Exception as e:
            print(json.dumps({{"passed": False, "error": f"Assertion Failure: {{str(e)[:200]}}"}}))
            sys.exit(0)
    print(json.dumps({{"passed": False, "error": "Neither 'test' nor 'check' harness found in test code."}}))
    sys.exit(0)

try:
    res = test_fn(fn)
    passed = bool(res) if res is not None else True
    print(json.dumps({{"passed": passed, "error": "" if passed else "Assertion check returned False."}}))
except Exception as e:
    print(json.dumps({{"passed": False, "error": f"Assertion Failure: {{type(e).__name__}}: {{str(e)[:200]}}"}}))
'''


def extract_python_code(raw_text: str) -> str:
    """Extracts Python code from Markdown code fences or returns raw text."""
    if not raw_text:
        return ""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw_text.strip()


def grade_code(
    code: str,
    entry_point: str,
    test_code: str,
    timeout_seconds: float = 5.0,
) -> Tuple[bool, str, float]:
    """Executes code + test in a separate subprocess with timeout.
    
    Returns:
        (passed: bool, message: str, latency_ms: float)
    """
    clean_code = extract_python_code(code)
    
    runner_script = RUNNER_TEMPLATE.format(
        code_repr=repr(clean_code),
        entry_point_repr=repr(entry_point),
        test_code_repr=repr(test_code),
    )
    
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
        tmp.write(runner_script)
        tmp_path = Path(tmp.name)
        
    try:
        proc = subprocess.run(
            [PYTHON_EXE, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        
        stdout = proc.stdout.strip()
        if not stdout:
            stderr = proc.stderr.strip()[:300]
            return False, f"Process failed with no output: {stderr}", elapsed_ms
            
        import json
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                data = json.loads(line)
                return bool(data.get("passed", False)), data.get("error", ""), elapsed_ms
                
        return False, f"Invalid test runner output: {stdout[:200]}", elapsed_ms

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return False, f"Execution timed out after {timeout_seconds} seconds.", elapsed_ms
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1e3
        return False, f"Execution Sandbox Error: {type(e).__name__}: {str(e)[:200]}", elapsed_ms
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

