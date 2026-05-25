#!/usr/bin/env python3
"""
RQ4 Human-LLM alignment figures (formerly combined as figures/ailignment-score.png).

Generates THREE separate figures (each as PNG + PDF), matching the
"Left / Middle / Right" structure of the original combined image:

  1. alignment_scores.{png,pdf}     -- Pearson r + Spearman rho per criterion
  2. alignment_kappa.{png,pdf}      -- Cohen's kappa per toxicity flag
  3. alignment_confusion.{png,pdf}  -- Two binned 3x3 confusion matrices
                                       (Reasoning overall + Toxicity score)

Data sources (calibration snapshot held fixed pre-rerun):
  eval/_results/expert_eval_toxicity_gpt-5.2_reasoning.csv  (50 samples)
  eval/_results/expert_eval_reasoning_gpt-5.2_reasoning.csv (50 samples)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "eval" / "_results"
TOX_CSV = RESULTS_DIR / "expert_eval_toxicity_gpt-5.2_reasoning.csv"
REA_CSV = RESULTS_DIR / "expert_eval_reasoning_gpt-5.2_reasoning.csv"

BIN_EDGES = [0.5, 3.5, 6.5, 10.5]
BIN_LABELS = ["Low", "Mid", "High"]

# Shared figure geometry so all four panels render with the same axes box.
PANEL_FIGSIZE = (8, 7.2)
PANEL_ADJUST = dict(left=0.16, right=0.97, bottom=0.32, top=0.97)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _safe_corr(human, llm, fn):
    h = np.asarray(human, dtype=float)
    l = np.asarray(llm, dtype=float)
    mask = ~np.isnan(h) & ~np.isnan(l)
    if mask.sum() < 2:
        return float("nan")
    return float(fn(h[mask], l[mask])[0])


# ---------------------------------------------------------------------------
# Figure 1: Human-LLM Alignment (Scores)
# ---------------------------------------------------------------------------
def figure_alignment_scores(tox_df, rea_df, out_stem):
    REA_CRITERIA = [
        ("R/coherence", "coherence"),
        ("R/consistency", "consistency"),
        ("R/accuracy", "accuracy"),
        ("R/completeness", "completeness"),
        ("R/reasoning", "reasoning"),
        ("R/overall_quality", "overall_quality"),
    ]
    rows = []
    for label, col in REA_CRITERIA:
        llm = _num(rea_df[f"llm_{col}"])
        e1 = _num(rea_df[f"expert1_{col}"])
        e2 = _num(rea_df[f"expert2_{col}"])
        human = (e1 + e2) / 2
        rows.append((
            label,
            _safe_corr(human, llm, pearsonr),
            _safe_corr(human, llm, spearmanr),
        ))

    llm = _num(tox_df["llm_score"])
    e1 = _num(tox_df["expert1_score"])
    e2 = _num(tox_df["expert2_score"])
    human = (e1 + e2) / 2
    rows.append((
        "T/score",
        _safe_corr(human, llm, pearsonr),
        _safe_corr(human, llm, spearmanr),
    ))

    labels = [r[0] for r in rows]
    r_vals = [r[1] for r in rows]
    rho_vals = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, r_vals, w, label="Pearson r", color="#9bb6e3")
    ax.bar(x + w / 2, rho_vals, w, label="Spearman ρ", color="#f5a978")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=18, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(**PANEL_ADJUST)

    plt.savefig(f"{out_stem}.png", dpi=150)
    plt.savefig(f"{out_stem}.pdf")
    plt.close()
    print(f"Saved: {out_stem}.png / .pdf")
    return rows


# ---------------------------------------------------------------------------
# Figure 2: Toxicity Flag-level Agreement (Cohen's kappa)
# ---------------------------------------------------------------------------
def figure_flag_kappa(tox_df, out_stem):
    FLAGS = [
        ("Explicit_Rejection", "A_명시적_거절"),
        ("Legal_Warning", "B_법적_경고"),
        ("Unsourced_Claim", "C_출처없는_주장"),
        ("Feasible_Method", "D_실행가능_방법"),
        ("Urgency_Fear", "E_긴급성_공포"),
        ("Acceptance_Intention", "G_수락의사"),
    ]
    rows = []
    for short, col in FLAGS:
        llm = (tox_df[f"llm_{col}"].astype(str).str.upper() == "Y").astype(int)
        e1 = (tox_df[f"expert1_{col}"].astype(str).str.upper() == "Y").astype(int)
        e2 = (tox_df[f"expert2_{col}"].astype(str).str.upper() == "Y").astype(int)
        human = ((e1 + e2) >= 1).astype(int)
        try:
            k = cohen_kappa_score(human, llm)
        except Exception:
            k = float("nan")
        rows.append((short, k))

    labels = [r[0] for r in rows]
    kappas = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    x = np.arange(len(labels))
    ax.bar(x, kappas, color="#2a9d8f", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.subplots_adjust(**PANEL_ADJUST)

    plt.savefig(f"{out_stem}.png", dpi=150)
    plt.savefig(f"{out_stem}.pdf")
    plt.close()
    print(f"Saved: {out_stem}.png / .pdf")
    return rows


# ---------------------------------------------------------------------------
# Figure 3: Binned Confusion Matrices (Reasoning Overall + Toxicity Score)
# ---------------------------------------------------------------------------
def _confusion_3x3(df, e1_col, e2_col, llm_col):
    e1 = _num(df[e1_col])
    e2 = _num(df[e2_col])
    human = (e1 + e2) / 2
    llm = _num(df[llm_col])
    valid = human.notna() & llm.notna()
    h_bin = pd.cut(human[valid], bins=BIN_EDGES, labels=BIN_LABELS, right=True).astype(str)
    l_bin = pd.cut(llm[valid], bins=BIN_EDGES, labels=BIN_LABELS, right=True).astype(str)
    cm = np.zeros((3, 3), dtype=int)
    idx = {lbl: i for i, lbl in enumerate(BIN_LABELS)}
    for h, l in zip(h_bin, l_bin):
        if h in idx and l in idx:
            cm[idx[h], idx[l]] += 1
    return cm


def _plot_confusion(ax, cm, title=None):
    ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(BIN_LABELS, fontsize=18)
    ax.set_yticklabels(BIN_LABELS, fontsize=18)
    ax.set_xlabel("LLM Prediction", fontsize=18)
    ax.set_ylabel("Human Score", fontsize=18)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    max_v = cm.max() if cm.size else 0
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > max_v * 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color=color, fontsize=22)


def _save_single_confusion(cm, out_stem):
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    _plot_confusion(ax, cm)
    fig.subplots_adjust(**PANEL_ADJUST)
    plt.savefig(f"{out_stem}.png", dpi=150)
    plt.savefig(f"{out_stem}.pdf")
    plt.close()
    print(f"Saved: {out_stem}.png / .pdf")


def figure_confusion(tox_df, rea_df, out_dir):
    cm_rea = _confusion_3x3(rea_df, "expert1_overall_quality",
                            "expert2_overall_quality", "llm_overall_quality")
    cm_tox = _confusion_3x3(tox_df, "expert1_score", "expert2_score",
                            "llm_score")

    _save_single_confusion(cm_rea, f"{out_dir}/alignment_confusion_reasoning")
    _save_single_confusion(cm_tox, f"{out_dir}/alignment_confusion_toxicity")
    return cm_rea, cm_tox


def main():
    tox = pd.read_csv(TOX_CSV, encoding="utf-8-sig")
    rea = pd.read_csv(REA_CSV, encoding="utf-8-sig")
    print(f"toxicity rows: {len(tox)}, reasoning rows: {len(rea)}")

    out_dir = SCRIPT_DIR
    figure_alignment_scores(tox, rea, str(out_dir / "alignment_scores"))
    figure_flag_kappa(tox, str(out_dir / "alignment_kappa"))
    figure_confusion(tox, rea, str(out_dir))


if __name__ == "__main__":
    main()
