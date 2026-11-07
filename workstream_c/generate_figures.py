"""Figure Generation for Workstream C (generate_figures.py).

Produces publication-quality IEEE-styled figures (PNG at 300 DPI and vector PDF):
1. figure_pass1_by_arm: Downstream Pass@1 across all arms with 95% task-level bootstrap CIs.
2. figure_paired_clean_vs_corrupted: Paired comparison of Clean vs. Corrupted Pass@1 by arm.
3. figure_cost_latency_tradeoff: Pass@1 vs. End-to-End Latency / Prompt Tokens across arms.
4. figure_source_breakdown: Pass@1 per source (HumanEval vs. MBPP) across arms.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "workstream_c" / "figures"

# Clean publication style
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

ARM_DISPLAY_NAMES = {
    "ungated": "Ungated Baseline",
    "stage1_only": "Stage-1 PRM Only",
    "cascade": "Stage 1+2 Cascade",
    "llm_judge": "LLM Judge (Full)",
    "agent_ask": "AgentAsk Clarifier",
    "always_fallback": "Always Fallback",
    "deterministic": "Deterministic Rules",
}

ARM_COLORS = {
    "ungated": "#7f7f7f",         # Gray
    "stage1_only": "#1f77b4",     # Primary Blue
    "cascade": "#2ca02c",         # Green (Highlight)
    "llm_judge": "#ff7f0e",       # Orange
    "agent_ask": "#9467bd",       # Purple
    "always_fallback": "#d62728", # Red
    "deterministic": "#8c564b",   # Brown
}


def task_bootstrap_ci(
    task_values: Dict[str, Tuple[float, int]],
    n_resamples: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes point estimate and 95% task-level bootstrap confidence intervals."""
    tasks = list(task_values.keys())
    sums = np.array([task_values[t][0] for t in tasks], dtype=np.float64)
    counts = np.array([task_values[t][1] for t in tasks], dtype=np.float64)
    n_tasks = len(tasks)

    total_sum = sums.sum()
    total_count = counts.sum()
    point_est = float(total_sum / total_count) if total_count > 0 else 0.0

    rng = np.random.RandomState(seed)
    sample_indices = rng.choice(n_tasks, size=(n_resamples, n_tasks), replace=True)
    boot_sums = sums[sample_indices].sum(axis=1)
    boot_counts = counts[sample_indices].sum(axis=1)
    boot_estimates = boot_sums / np.maximum(boot_counts, 1e-9)

    ci_low = float(np.percentile(boot_estimates, 2.5))
    ci_high = float(np.percentile(boot_estimates, 97.5))
    return point_est, ci_low, ci_high


def plot_pass1_by_arm(df: pd.DataFrame, output_dir: Path):
    """Figure 1: Downstream Pass@1 across all arms with 95% task-bootstrap error bars."""
    arms = [a for a in ["ungated", "deterministic", "stage1_only", "cascade", "llm_judge", "agent_ask", "always_fallback"] if a in df["arm"].unique()]
    
    means = []
    yerr_lower = []
    yerr_upper = []
    labels = []
    colors = []

    for arm in arms:
        arm_df = df[df["arm"] == arm]
        tvals = {tid: (float(g["outcome"].sum()), len(g)) for tid, g in arm_df.groupby("task_id")}
        pt, lo, hi = task_bootstrap_ci(tvals)
        means.append(pt * 100)
        yerr_lower.append((pt - lo) * 100)
        yerr_upper.append((hi - pt) * 100)
        labels.append(ARM_DISPLAY_NAMES.get(arm, arm))
        colors.append(ARM_COLORS.get(arm, "#333333"))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(arms))
    yerr = [yerr_lower, yerr_upper]

    bars = ax.bar(x, means, yerr=yerr, capsize=4, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    
    # Baseline dashed line
    if "ungated" in arms:
        u_idx = arms.index("ungated")
        u_mean = means[u_idx]
        ax.axhline(u_mean, color="#7f7f7f", linestyle="--", alpha=0.7, label=f"Ungated Baseline ({u_mean:.1f}%)")

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 2.5, f"{m:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Downstream Pass@1 (%)")
    ax.set_ylim(0, min(100, max(means) + 15))
    ax.set_title("Downstream Pass@1 Across Evaluation Arms (with 95% Task-Level Bootstrap CIs)")
    ax.legend(loc="upper left")
    plt.tight_layout()

    fig.savefig(output_dir / "figure_pass1_by_arm.png", dpi=300)
    fig.savefig(output_dir / "figure_pass1_by_arm.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'figure_pass1_by_arm.png'}")


def plot_clean_vs_corrupted(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Paired Clean vs. Corrupted Downstream Pass@1 across arms."""
    arms = [a for a in ["ungated", "deterministic", "stage1_only", "cascade", "llm_judge", "agent_ask", "always_fallback"] if a in df["arm"].unique()]
    
    clean_means = []
    corr_means = []
    labels = []

    for arm in arms:
        c_df = df[(df["arm"] == arm) & (df["handoff_type"] == "clean")]
        k_df = df[(df["arm"] == arm) & (df["handoff_type"] != "clean")]
        c_mean = c_df["outcome"].mean() * 100 if len(c_df) > 0 else 0.0
        k_mean = k_df["outcome"].mean() * 100 if len(k_df) > 0 else 0.0
        clean_means.append(c_mean)
        corr_means.append(k_mean)
        labels.append(ARM_DISPLAY_NAMES.get(arm, arm))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(arms))
    width = 0.35

    bars1 = ax.bar(x - width/2, clean_means, width, label="Clean Handoff", color="#1f77b4", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x + width/2, corr_means, width, label="Corrupted Handoff", color="#d62728", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Paired Performance: Clean vs. Corrupted Context Handoffs")
    ax.legend(loc="upper right")
    plt.tight_layout()

    fig.savefig(output_dir / "figure_paired_clean_vs_corrupted.png", dpi=300)
    fig.savefig(output_dir / "figure_paired_clean_vs_corrupted.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'figure_paired_clean_vs_corrupted.png'}")


def plot_cost_latency_tradeoff(df: pd.DataFrame, output_dir: Path):
    """Figure 3: Pass@1 vs. End-to-End Latency Pareto Tradeoff."""
    arms = [a for a in ["ungated", "stage1_only", "cascade", "llm_judge", "agent_ask"] if a in df["arm"].unique()]
    
    means = []
    latencies = []
    labels = []
    colors = []

    for arm in arms:
        arm_df = df[df["arm"] == arm]
        p_mean = arm_df["outcome"].mean() * 100
        lat_sec = arm_df["latency_ms"].mean() / 1000.0 if "latency_ms" in arm_df.columns else 1.0
        means.append(p_mean)
        latencies.append(lat_sec)
        labels.append(ARM_DISPLAY_NAMES.get(arm, arm))
        colors.append(ARM_COLORS.get(arm, "#333333"))

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for x, y, lab, c in zip(latencies, means, labels, colors):
        ax.scatter(x, y, s=140, color=c, edgecolor="black", label=lab, zorder=3)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)

    ax.set_xlabel("Mean Latency per Task (seconds)")
    ax.set_ylabel("Downstream Pass@1 (%)")
    ax.set_title("Pareto Tradeoff: Downstream Accuracy vs. Verification Latency")
    ax.set_ylim(min(means) - 5, min(100, max(means) + 8))
    ax.set_xlim(max(0, min(latencies) - 0.5), max(latencies) + 1.5)
    plt.tight_layout()

    fig.savefig(output_dir / "figure_cost_latency_tradeoff.png", dpi=300)
    fig.savefig(output_dir / "figure_cost_latency_tradeoff.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'figure_cost_latency_tradeoff.png'}")


def plot_source_breakdown(df: pd.DataFrame, output_dir: Path):
    """Figure 4: Pass@1 by Source (HumanEval vs MBPP) across evaluation arms."""
    arms = [a for a in ["ungated", "deterministic", "stage1_only", "cascade", "llm_judge", "agent_ask", "always_fallback"] if a in df["arm"].unique()]
    
    he_means = []
    mbpp_means = []
    labels = []

    for arm in arms:
        he_df = df[(df["arm"] == arm) & (df["source"] == "humaneval")]
        mbpp_df = df[(df["arm"] == arm) & (df["source"] == "mbpp")]
        he_means.append(he_df["outcome"].mean() * 100 if len(he_df) > 0 else 0.0)
        mbpp_means.append(mbpp_df["outcome"].mean() * 100 if len(mbpp_df) > 0 else 0.0)
        labels.append(ARM_DISPLAY_NAMES.get(arm, arm))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(arms))
    width = 0.35

    bars1 = ax.bar(x - width/2, he_means, width, label="HumanEval (84 tasks, n=504)", color="#2ca02c", alpha=0.85, edgecolor="black")
    bars2 = ax.bar(x + width/2, mbpp_means, width, label="MBPP (209 tasks, n=1254)", color="#1f77b4", alpha=0.85, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Pass@1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Downstream Pass@1 by Benchmark Source (HumanEval vs. MBPP)")
    ax.legend(loc="upper left")
    plt.tight_layout()

    fig.savefig(output_dir / "figure_source_breakdown.png", dpi=300)
    fig.savefig(output_dir / "figure_source_breakdown.pdf")
    plt.close(fig)
    print(f"Saved: {output_dir / 'figure_source_breakdown.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Figures for Workstream C")
    parser.add_argument("--csv", default=str(REPO_ROOT / "workstream_c" / "results" / "live_results_main.csv"), help="Results CSV")
    parser.add_argument("--output-dir", default=str(FIGURES_DIR), help="Output directory for plots")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} trials from {csv_path.name}. Generating figures...")

    plot_pass1_by_arm(df, output_dir)
    plot_clean_vs_corrupted(df, output_dir)
    plot_cost_latency_tradeoff(df, output_dir)
    plot_source_breakdown(df, output_dir)
    print("All figures successfully generated.")


if __name__ == "__main__":
    main()
