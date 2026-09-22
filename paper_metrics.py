"""
paper_metrics.py
Reads live_pilot_results.csv + results/tables/ and produces
every metric claimed in the paper - NO new API calls, zero tokens spent.
"""
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT   = Path(__file__).parent
PILOT  = ROOT / "live_pilot_results.csv"
TABLES = ROOT / "results" / "tables"
OUT_MD = ROOT / "results" / "paper_metrics_report.md"
OUT_TX = ROOT / "results" / "paper_metrics_tables.txt"

def mcnemar_exact(b, c):
    total = b + c
    if total == 0: return 1.0
    p = sum(binom.pmf(k, total, 0.5) for k in range(min(b, c) + 1))
    return float(2.0 * min(p, 1.0))

def bootstrap_ci(df, col_a, col_b, n_boot=10000, seed=42):
    delta = (df[col_b].astype(int) - df[col_a].astype(int)).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

def pct(x, n): return float("nan") if n == 0 else 100.0 * x / n
def fmt_ci(lo, hi, d=1): return f"[{lo:+.{d}f}, {hi:+.{d}f}]"
def hdr(t): return f"\n{'='*72}\n  {t}\n{'='*72}\n"
def sub(t): return f"\n--- {t} ---\n"

def load_pilot():
    df = pd.read_csv(PILOT)
    for col in ["is_corrupted","stage1_blocked","judge_escalated",
                "ungated_passed","gated_passed","ungated_error","gated_error"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map(
                lambda x: True if x in ("true","1","yes") else False)
    return df

def build_always_fallback(df):
    df = df.copy()
    df["blocked"] = (df["gate_decision"] == "BLOCK")
    # Lower bound: for PASS rows use ungated_passed (raw ~ handoff quality)
    df["afb_lower"] = np.where(df["blocked"], df["gated_passed"], df["ungated_passed"])
    # Upper bound: for PASS rows assume raw would always pass (optimistic for AFB)
    df["afb_upper"] = np.where(df["blocked"], df["gated_passed"], True)
    return df

def analyze(df):
    n = len(df)
    cln = df[~df["is_corrupted"]]
    crp = df[df["is_corrupted"]]

    ung_all = df["ungated_passed"].mean() * 100
    gat_all = df["gated_passed"].mean()   * 100
    ung_cln = cln["ungated_passed"].mean() * 100 if len(cln) else float("nan")
    gat_cln = cln["gated_passed"].mean()   * 100 if len(cln) else float("nan")
    ung_crp = crp["ungated_passed"].mean() * 100 if len(crp) else float("nan")
    gat_crp = crp["gated_passed"].mean()   * 100 if len(crp) else float("nan")

    both_pass = ((df["ungated_passed"]) & (df["gated_passed"])).sum()
    both_fail = ((~df["ungated_passed"]) & (~df["gated_passed"])).sum()
    b = ((df["ungated_passed"]) & (~df["gated_passed"])).sum()
    c = ((~df["ungated_passed"]) & (df["gated_passed"])).sum()
    mcn_all = mcnemar_exact(int(b), int(c))
    ci_all  = bootstrap_ci(df, "ungated_passed", "gated_passed")

    tp = ((crp["gate_decision"] == "BLOCK") & (~crp["ungated_passed"])).sum()
    fp = ((crp["gate_decision"] == "BLOCK") & ( crp["ungated_passed"])).sum()
    fn = ((crp["gate_decision"] == "PASS")  & (~crp["ungated_passed"])).sum()
    tn = ((crp["gate_decision"] == "PASS")  & ( crp["ungated_passed"])).sum()

    fp_clean   = (cln["gate_decision"] == "BLOCK").sum()
    pass_clean = (cln["gate_decision"] == "PASS").sum()

    stage1_blocks    = (df[df["gate_decision"] == "BLOCK"]["gate_stage"] == 1).sum()
    stage2_blocks    = (df[df["gate_decision"] == "BLOCK"]["gate_stage"] == 2).sum()
    stage2_escalated = df["judge_escalated"].sum()
    stage2_flipped   = ((df["judge_escalated"]) & (df["judge_verdict"] == 0)).sum()

    blocked      = df[df["gate_decision"] == "BLOCK"]
    n_blocked    = len(blocked)
    n_blocked_crp = (crp["gate_decision"] == "BLOCK").sum()
    recovered_crp = blocked[blocked["is_corrupted"]]["gated_passed"].sum()
    fp_recovered  = blocked[~blocked["is_corrupted"]]["gated_passed"].sum()
    fp_regressed  = blocked[~blocked["is_corrupted"]]["gated_passed"].eq(False).sum()
    n_total_passes = (df["gate_decision"] == "PASS").sum()

    df2 = build_always_fallback(df)
    afb_lo_all = df2["afb_lower"].mean() * 100
    afb_hi_all = df2["afb_upper"].mean() * 100
    afb_lo_crp = df2[df2["is_corrupted"]]["afb_lower"].mean() * 100
    afb_hi_crp = df2[df2["is_corrupted"]]["afb_upper"].mean() * 100
    afb_lo_cln = df2[~df2["is_corrupted"]]["afb_lower"].mean() * 100
    afb_hi_cln = df2[~df2["is_corrupted"]]["afb_upper"].mean() * 100

    afb_mc_b = ((df2["gated_passed"]) & (~df2["afb_lower"].astype(bool))).sum()
    afb_mc_c = ((~df2["gated_passed"]) & (df2["afb_lower"].astype(bool))).sum()
    mcn_afb = mcnemar_exact(int(afb_mc_b), int(afb_mc_c))
    ci_afb  = bootstrap_ci(df2, "afb_lower", "gated_passed")

    crp_types = {}
    for ct in crp["corruption_type"].unique():
        sub_df = crp[crp["corruption_type"] == ct]
        crp_types[ct] = {
            "n": len(sub_df),
            "ung_pct": sub_df["ungated_passed"].mean() * 100,
            "gat_pct": sub_df["gated_passed"].mean()   * 100,
            "delta":   (sub_df["gated_passed"].mean() - sub_df["ungated_passed"].mean()) * 100,
            "n_blocked":   (sub_df["gate_decision"] == "BLOCK").sum(),
            "n_recovered": sub_df[(sub_df["gate_decision"] == "BLOCK") & sub_df["gated_passed"]]["gated_passed"].sum(),
        }

    offline = {}
    for key, path in [("cv", TABLES/"cv_results.csv"),
                      ("sweep", TABLES/"two_stage_sweep.csv"),
                      ("by_corr", TABLES/"two_stage_by_corruption.csv"),
                      ("mc12", ROOT/"model_comparison_results_12feat.csv")]:
        if path.exists(): offline[key] = pd.read_csv(path)

    return dict(
        n=n, n_cln=len(cln), n_crp=len(crp),
        ung_all=ung_all, gat_all=gat_all,
        ung_cln=ung_cln, gat_cln=gat_cln,
        ung_crp=ung_crp, gat_crp=gat_crp,
        both_pass=int(both_pass), both_fail=int(both_fail),
        b=int(b), c=int(c),
        mcn_all=mcn_all, ci_all=ci_all,
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        fp_clean=int(fp_clean), pass_clean=int(pass_clean),
        stage1_blocks=int(stage1_blocks), stage2_blocks=int(stage2_blocks),
        stage2_escalated=int(stage2_escalated), stage2_flipped=int(stage2_flipped),
        n_blocked=int(n_blocked), n_blocked_crp=int(n_blocked_crp),
        recovered_crp=int(recovered_crp),
        fp_recovered=int(fp_recovered), fp_regressed=int(fp_regressed),
        n_total_passes=int(n_total_passes),
        afb_lo_all=afb_lo_all, afb_hi_all=afb_hi_all,
        afb_lo_crp=afb_lo_crp, afb_hi_crp=afb_hi_crp,
        afb_lo_cln=afb_lo_cln, afb_hi_cln=afb_hi_cln,
        mcn_afb=mcn_afb, ci_afb=ci_afb,
        crp_types=crp_types, offline=offline,
        df=df, df2=df2,
    )

def gate_metrics(m):
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    prec = tp / (tp+fp) if (tp+fp) > 0 else float("nan")
    rec  = tp / (tp+fn) if (tp+fn) > 0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else float("nan")
    fpr_c = m["fp_clean"] / m["n_cln"] if m["n_cln"] > 0 else float("nan")
    return dict(precision=prec, recall=rec, f1=f1, fpr_clean=fpr_c)

def build_report(m):
    gm = gate_metrics(m)
    ci = m["ci_all"]; ci_afb = m["ci_afb"]
    n_crp_fail = m["n_crp"] - int(m["df"]["ungated_passed"][m["df"]["is_corrupted"]].sum())
    rec = m["recovered_crp"]
    lines = []
    L = lines.append

    L(hdr("HANDOFF-PRM — COMPLETE PAPER METRICS REPORT"))
    L(f"  Source: live_pilot_results.csv  ({m['n']} rows, {m['n_cln']} clean + {m['n_crp']} corrupted)")
    L(f"  Statistics: exact / non-parametric. Seed=42, n_boot=10,000.\n")

    L(sub("TABLE 1 — Primary Pass@1 Results"))
    W = 46
    L(f"  {'Condition':<{W}} {'Ungated':>9} {'Gated':>9} {'Delta':>8}")
    L(f"  {'-'*(W+28)}")
    rows = [
        (f"All tasks (n={m['n']})",          m['ung_all'], m['gat_all']),
        (f"Clean handoffs (n={m['n_cln']})",  m['ung_cln'], m['gat_cln']),
        (f"Corrupted (n={m['n_crp']})",       m['ung_crp'], m['gat_crp']),
    ]
    for label, u, g in rows:
        L(f"  {label:<{W}} {u:>8.1f}% {g:>8.1f}% {g-u:>+7.1f}%")
    L("")
    L(f"  McNemar (exact, 2-sided)     p = {m['mcn_all']:.4f}  "
      f"({'SIGNIFICANT' if m['mcn_all'] < 0.05 else 'n.s.'} at alpha=0.05)")
    L(f"  Bootstrap 95% CI (delta)     {fmt_ci(ci[0]*100, ci[1]*100)} pp")
    L(f"  Discordant:  gate-hurt={m['b']}  gate-recovered={m['c']}  both-pass={m['both_pass']}  both-fail={m['both_fail']}")

    L(sub("TABLE 2 — Always-Fallback Baseline (hardest comparison)"))
    L("  Policy: ignore handoff; always run Agent B on the raw problem.")
    L("  BLOCKED rows: actual data. PASSED rows: bracketed with lower=ungated, upper=all-pass.\n")
    L(f"  {'Condition':<{W}} {'AFB-lower':>11} {'AFB-upper':>11} {'Gated':>8}")
    L(f"  {'-'*(W+32)}")
    for label, lo, hi, g in [
        ("All tasks",      m['afb_lo_all'], m['afb_hi_all'], m['gat_all']),
        ("Clean tasks",    m['afb_lo_cln'], m['afb_hi_cln'], m['gat_cln']),
        ("Corrupted tasks",m['afb_lo_crp'], m['afb_hi_crp'], m['gat_crp']),
    ]:
        L(f"  {label:<{W}} {lo:>10.1f}% {hi:>10.1f}% {g:>7.1f}%")
    L("")
    L(f"  McNemar (gated vs afb_lower)  p = {m['mcn_afb']:.4f}")
    L(f"  Bootstrap 95% CI (gated-afb_lower)  {fmt_ci(ci_afb[0]*100, ci_afb[1]*100)} pp")
    if m['gat_all'] >= m['afb_lo_all']:
        L(f"  >> Gated ({m['gat_all']:.1f}%) >= AFB-lower ({m['afb_lo_all']:.1f}%) — gate quality holds.")
    else:
        L(f"  >> Gated ({m['gat_all']:.1f}%) < AFB-lower ({m['afb_lo_all']:.1f}%) — defend on architecture (break-glass).")
    L("  ARCHITECTURAL CLAIM: gate preserves info-hiding by default; escalates")
    L("  to raw-problem ONLY on detected failures (principled break-glass pattern).")

    L(sub("TABLE 3 — Gate Confusion Matrix (on Corrupted Tasks)"))
    L("  Positive = BLOCK; Ground truth positive = ungated_passed=False")
    L(f"  TP (blocked failing handoff)   : {m['tp']}")
    L(f"  FP (blocked passing handoff)   : {m['fp']}")
    L(f"  FN (passed failing handoff)    : {m['fn']}")
    L(f"  TN (passed passing handoff)    : {m['tn']}")
    L("")
    L(f"  Precision  : {gm['precision']:.3f}")
    L(f"  Recall     : {gm['recall']:.3f}")
    L(f"  F1         : {gm['f1']:.3f}")
    L("")
    L("  On CLEAN tasks:")
    L(f"  FP blocks                      : {m['fp_clean']}/{m['n_cln']} = {pct(m['fp_clean'], m['n_cln']):.1f}%")
    if m['fp_clean'] > 0:
        L(f"  Recovered by fallback          : {m['fp_recovered']}/{m['fp_clean']} = {pct(m['fp_recovered'], m['fp_clean']):.1f}%")
    L(f"  Net hard cost (still failed)   : {m['fp_regressed']} task(s)")

    L(sub("TABLE 4 — Two-Stage Architecture Contribution"))
    L(f"  Total blocks                          : {m['n_blocked']} / {m['n']}")
    L(f"  Stage-1 (RF, <10ms) blocks            : {m['stage1_blocks']}  ({pct(m['stage1_blocks'], m['n_blocked']):.0f}% of blocks)")
    L(f"  Stage-2 (LLM judge) added blocks      : {m['stage2_blocks']}  ({pct(m['stage2_blocks'], m['n_blocked']):.0f}% of blocks)")
    L(f"  Stage-2 escalations total             : {m['stage2_escalated']}")
    L(f"  Stage-2 flips (SUFFICIENT->BLOCK)     : {m['stage2_flipped']}")
    L(f"  Tasks needing judge ({pct(m['stage2_escalated'], m['n']):.0f}% of all)       : {m['stage2_escalated']}/{m['n']}")
    L(f"  Tasks on cheap path only (no judge)   : {m['n'] - m['stage2_escalated']}/{m['n']} = {pct(m['n'] - m['stage2_escalated'], m['n']):.0f}%")

    L(sub("TABLE 5 — Per-Corruption-Type Breakdown"))
    L(f"  {'Type':<22} {'n':>4} {'Ungated':>9} {'Gated':>9} {'Delta':>8} {'Blocked':>9} {'Recovered':>10}")
    L(f"  {'-'*75}")
    for ct, v in sorted(m["crp_types"].items(), key=lambda x: x[1]["delta"], reverse=True):
        L(f"  {ct:<22} {v['n']:>4} {v['ung_pct']:>8.1f}% {v['gat_pct']:>8.1f}% "
          f"{v['delta']:>+7.1f}% {v['n_blocked']:>8}   {v['n_recovered']:>8}")

    L(sub("TABLE 6 — Recovery Analysis"))
    L(f"  Corrupted tasks that fail ungated   : {n_crp_fail}/{m['n_crp']}")
    L(f"  Corrupted tasks blocked by gate     : {m['n_blocked_crp']}/{m['n_crp']}")
    L(f"  Recovered after block               : {rec}/{m['n_blocked_crp']} = {pct(rec, m['n_blocked_crp']):.0f}%")
    L(f"  Fraction of ALL corrupted recovered : {rec}/{m['n_crp']} = {pct(rec, m['n_crp']):.0f}%")
    L(f"  Failures still passing gate         : {m['fn']} (undetected)")

    if "mc12" in m["offline"]:
        L(sub("TABLE 7 — Offline Model Comparison (12-feature, 5-fold GroupKFold)"))
        mc = m["offline"]["mc12"]
        L(f"  {'Model':<25} {'OOF_AUROC':>10} {'OOF_AUPRC':>10} {'Brier':>7} {'F1':>7} {'Latency':>10}")
        L(f"  {'-'*72}")
        for _, row in mc.iterrows():
            L(f"  {str(row.get('Model','')):<25} "
              f"{float(row.get('OOF_AUROC_overall',0)):>9.4f} "
              f"{float(row.get('OOF_AUPRC',0)):>9.4f} "
              f"{float(row.get('Brier',0)):>6.4f} "
              f"{float(row.get('F1',0)):>6.4f} "
              f"{float(row.get('Latency_ms_per_example',0)):>9.2f}ms")

    if "sweep" in m["offline"]:
        L(sub("TABLE 8 — Two-Stage Threshold Sweep (offline)"))
        sw = m["offline"]["sweep"]
        L(f"  {'tau_low':>7} {'tau_high':>8} {'Routed%':>8} {'FastAUROC':>10} {'Combined':>10} {'Delta':>8} {'Latency':>9}")
        L(f"  {'-'*64}")
        for _, row in sw.iterrows():
            L(f"  {float(row['tau_low']):>7.2f} {float(row['tau_high']):>8.2f} "
              f"{float(row['routing_rate_pct']):>7.1f}% {float(row['fast_auroc']):>9.4f} "
              f"{float(row['combined_auroc']):>9.4f} {float(row['delta_auroc']):>+7.4f} "
              f"{float(row['latency_ms']):>8.2f}ms")

    L(hdr("KEY CLAIMS — PAPER-READY STATEMENTS"))
    delta = m['gat_all'] - m['ung_all']
    delta_crp = m['gat_crp'] - m['ung_crp']
    L(f"""
CLAIM 1 (primary result):
  The trust-gated policy raises Pass@1 from {m['ung_all']:.1f}% to {m['gat_all']:.1f}%
  on {m['n']} held-out tasks (Delta={delta:+.1f} pp; 95% CI {fmt_ci(ci[0]*100, ci[1]*100)} pp;
  McNemar p={m['mcn_all']:.4f}).

CLAIM 2 (corrupted tasks):
  On the {m['n_crp']} corrupted tasks, Pass@1 rises {m['ung_crp']:.1f}% -> {m['gat_crp']:.1f}%
  (Delta={delta_crp:+.1f} pp).

CLAIM 3 (clean task cost):
  On {m['n_cln']} clean tasks, the gate causes a {m['gat_cln']-m['ung_cln']:+.1f} pp change
  ({m['fp_clean']} false-positive blocks; fallback recovered {m['fp_recovered']}/{m['fp_clean'] if m['fp_clean'] else 1}).
  Net irrecoverable cost: {m['fp_regressed']} task(s).

CLAIM 4 (failure interception):
  The gate blocks {m['n_blocked_crp']}/{m['n_crp']} corrupted handoffs.
  Of intercepted handoffs, {pct(rec, m['n_blocked_crp']):.0f}% are recovered
  ({rec}/{m['n_blocked_crp']}). Undetected failures: {m['fn']}.

CLAIM 5 (two-stage efficiency):
  Stage-1 RF (<10 ms) handles {m['stage1_blocks']}/{m['n_blocked']} blocks ({pct(m['stage1_blocks'], m['n_blocked']):.0f}%) cheaply.
  Stage-2 LLM judge invoked on only {m['stage2_escalated']}/{m['n']} tasks ({pct(m['stage2_escalated'], m['n']):.0f}%),
  yielding {m['stage2_flipped']} additional catches (semantic corruptions).

CLAIM 6 (vs always-fallback):
  Always-fallback achieves {m['afb_lo_all']:.1f}%-{m['afb_hi_all']:.1f}% (bracketed estimate).
  Gated: {m['gat_all']:.1f}%. McNemar (gated vs afb_lower) p={m['mcn_afb']:.4f}.
  Gate also provides architectural benefit: info-hiding by default,
  raw-problem escalation ONLY on detected failure (break-glass pattern).

HONEST CONDITIONS:
  A) No token savings as built (same model, same message on PASS). Reframe as
     "same-cost trust gate that rescues damaged handoffs" or add cheaper Stage-1 model.
  B) Always-fallback quality bound must be reported. Do not omit this comparison.
  C) Stage-2 judge adds latency; report separately from Stage-1.
  D) n_crp per corruption type is 10-15; per-type results are indicative only.
""")
    return "\n".join(lines)

def build_latex(m):
    gm = gate_metrics(m)
    ci = m["ci_all"]
    lines = []
    L = lines.append

    L("% PRIMARY RESULTS TABLE")
    L("\\begin{tabular}{lccc}")
    L("\\toprule")
    L("Condition & Ungated & Gated & $\\Delta$ \\\\")
    L("\\midrule")
    for label, u, g in [
        (f"All ($n={m['n']}$)",          m['ung_all'], m['gat_all']),
        (f"Clean ($n={m['n_cln']}$)",    m['ung_cln'], m['gat_cln']),
        (f"Corrupted ($n={m['n_crp']}$)",m['ung_crp'], m['gat_crp']),
    ]:
        L(f"{label} & {u:.1f}\\% & {g:.1f}\\% & {g-u:+.1f}\\% \\\\")
    L("\\midrule")
    L(f"\\multicolumn{{4}}{{l}}{{McNemar $p={m['mcn_all']:.4f}$; "
      f"Bootstrap 95\\% CI {fmt_ci(ci[0]*100, ci[1]*100)} pp}} \\\\")
    L("\\bottomrule")
    L("\\end{tabular}\n")

    L("% CONFUSION MATRIX (corrupted tasks only)")
    L("\\begin{tabular}{lcc}")
    L("\\toprule")
    L("& Handoff failed & Handoff passed \\\\")
    L("\\midrule")
    L(f"Gate BLOCK & {m['tp']} (TP) & {m['fp']} (FP) \\\\")
    L(f"Gate PASS  & {m['fn']} (FN) & {m['tn']} (TN) \\\\")
    L("\\midrule")
    L(f"Precision & \\multicolumn{{2}}{{c}}{{{gm['precision']:.3f}}} \\\\")
    L(f"Recall    & \\multicolumn{{2}}{{c}}{{{gm['recall']:.3f}}} \\\\")
    L(f"F1        & \\multicolumn{{2}}{{c}}{{{gm['f1']:.3f}}} \\\\")
    L("\\bottomrule")
    L("\\end{tabular}\n")

    L("% PER-CORRUPTION TABLE")
    L("\\begin{tabular}{lrrrr}")
    L("\\toprule")
    L("Corruption type & $n$ & Ungated & Gated & $\\Delta$ \\\\")
    L("\\midrule")
    for ct, v in sorted(m["crp_types"].items(), key=lambda x: x[1]["delta"], reverse=True):
        L(f"{ct.replace('_', '\\_')} & {v['n']} & {v['ung_pct']:.1f}\\% & "
          f"{v['gat_pct']:.1f}\\% & {v['delta']:+.1f}\\% \\\\")
    L("\\bottomrule")
    L("\\end{tabular}")
    return "\n".join(lines)

def main():
    print("Loading live_pilot_results.csv ...")
    if not PILOT.exists():
        print(f"ERROR: {PILOT} not found.", file=sys.stderr); sys.exit(1)
    df = load_pilot()
    print(f"  {len(df)} rows  ({(~df['is_corrupted']).sum()} clean, {df['is_corrupted'].sum()} corrupted)")
    errs = df[df["ungated_error"] | df["gated_error"]]
    if len(errs):
        print(f"  WARNING: {len(errs)} rows have API error flags.")
    m = analyze(df)
    report = build_report(m)
    tables = build_latex(m)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(report, encoding="utf-8")
    OUT_TX.write_text(tables, encoding="utf-8")
    print(report)
    print(f"\n{'='*72}")
    print(f"  Report   -> {OUT_MD}")
    print(f"  LaTeX    -> {OUT_TX}")
    print(f"{'='*72}")

if __name__ == "__main__":
    main()
