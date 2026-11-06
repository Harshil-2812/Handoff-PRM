"""Workstream C Harness: Configuration & Frozen Protocol Schema."""

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSTREAM_C_DIR = REPO_ROOT / "workstream_c"
MANIFEST_DIR = WORKSTREAM_C_DIR / "manifests"
RESULTS_DIR = WORKSTREAM_C_DIR / "results"
CACHE_DIR = REPO_ROOT / "cache"

# Frozen schema: primary CSV columns specified by experiment_workplan.md
FROZEN_SCHEMA_COLUMNS = [
    "task_id",
    "source",
    "handoff_id",
    "handoff_type",       # "clean" | corruption name (e.g. "truncation") | "organic"
    "label",              # ground-truth downstream pass (1 = passed, 0 = failed)
    "label_policy",       # "downstream_pass1"
    "seed",               # 42, 43, 44
    "features",           # JSON string of extracted 11 features
    "stage1_score",       # float probability from Stage 1 PRM
    "decision",           # "PASS" | "BLOCK"
    "outcome",            # Downstream execution outcome under arm's policy (1 = pass, 0 = fail)
    "model_agent_a",      # Model ID for Agent A
    "model_agent_b",      # Model ID for Agent B
    "model_judge",        # Model ID for LLM Judge
    "temp_agent_a",       # Temperature for Agent A
    "temp_agent_b",       # Temperature for Agent B
    "temp_judge",         # Temperature for Judge
]

# Additive instrumentation columns (documented extension for latency, cost, and debugging)
EXTENDED_SCHEMA_COLUMNS = FROZEN_SCHEMA_COLUMNS + [
    "arm",                # "ungated" | "stage1_only" | "cascade" | "llm_judge" | "always_fallback" | "agent_ask" | "deterministic"
    "latency_ms",         # Total end-to-end latency in milliseconds
    "tokens_prompt",      # Total prompt tokens
    "tokens_completion",  # Total completion tokens
    "error_message",      # Error message if code execution failed
]

# Models & Defaults
DEFAULT_MODEL_A = "gemini-3.5-flash-lite"
DEFAULT_MODEL_B = "gemini-3.5-flash-lite"
DEFAULT_MODEL_JUDGE = "gemini-3.5-flash-lite"

# Headline threshold frozen from Workstream B (results/frozen_threshold.json)
# Rule: largest t with clean false-block rate <= 0.05 on dev OOF
HEADLINE_THRESHOLD_RAW = 0.5504194147823097
HEADLINE_THRESHOLD_PLATT = 0.47500690369880694

# Seeds for evaluation
LIVE_SEEDS = [42, 43, 44]

