"""
Legacy results-analysis / plotting helpers (not used by the current paper experiments).

Produces:
  1. Privacy–utility frontier (the headline figure — method 5 vs 6)
  2. Communication savings bar chart
  3. MIA-AUC vs ε plot
  4. Non-IID robustness (α sweep)
  5. Ablation tables (a–e from Section 7)
  6. Per-dataset summary tables (LaTeX-ready)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # headless / CPU rendering
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False
    log.warning("matplotlib not found; figures will not be generated.")


# ─────────────────────────────────────────────────────────────────────────────
# Save helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Results saved → %s", path)


def load_results(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Headline figure: privacy–utility frontier (method 5 vs 6)
# ─────────────────────────────────────────────────────────────────────────────

def make_privacy_utility_figure(
    results: list,
    out_dir: Path,
    datasets: list = None,
) -> None:
    """
    Plot F1 vs ε for FedAvg+DP (no dedup) and with deduplication,
    one subplot per dataset.  Error bands = ±1 std over seeds.
    """
    if not _MPL:
        return

    datasets = datasets or ["skab", "tep", "iiot"]
    n_ds     = len(datasets)

    fig, axes = plt.subplots(1, n_ds, figsize=(5 * n_ds, 4.5), squeeze=False)
    fig.suptitle("Privacy-Utility Frontier: dedup vs no-dedup",
                 fontsize=13, fontweight="bold")

    CMAP = {"fedavg_dp_nodedup": ("#d62728", "o", "FedAvg+DP (no dedup)"),
            "fedavg_dp_dedup":   ("#1f77b4", "s", "with deduplication")}

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        ds_rows = [r for r in results if r.get("dataset") == ds]

        for cond, (color, marker, label) in CMAP.items():
            cond_rows = [r for r in ds_rows if r.get("condition") == cond]
            if not cond_rows:
                continue

            eps_vals = sorted(set(r.get("epsilon_target", r.get("epsilon", float("inf"))) for r in cond_rows if r.get("epsilon_target", r.get("epsilon")) not in (float("inf"), None)))
            means, stds = [], []
            for eps in eps_vals:
                f1s = [r["f1"] for r in cond_rows if r.get("epsilon_target", r.get("epsilon")) == eps]
                means.append(np.mean(f1s))
                stds.append(np.std(f1s))

            means = np.array(means)
            stds  = np.array(stds)

            ax.plot(eps_vals, means, marker=marker, color=color, label=label, linewidth=2)
            ax.fill_between(eps_vals, means - stds, means + stds,
                            alpha=0.15, color=color)

        ax.set_xlabel("Privacy budget ε", fontsize=11)
        ax.set_ylabel("F1 score", fontsize=11)
        ax.set_title(ds.upper(), fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([0.5, 1, 2, 4, 8])
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "privacy_utility_frontier.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved figure → %s", out_path)


def make_communication_figure(results: list, out_dir: Path) -> None:
    """Bar chart: communication bytes (no dedup vs dedup) per dataset."""
    if not _MPL:
        return

    datasets = ["skab", "tep", "iiot"]
    fig, ax  = plt.subplots(figsize=(8, 4))

    bar_w = 0.35
    x     = np.arange(len(datasets))
    conds = ["fedavg_dp_nodedup", "fedavg_dp_dedup"]
    colors = ["#d62728", "#1f77b4"]
    labels = ["DP no-dedup", "deduplication"]

    for i, (cond, color, label) in enumerate(zip(conds, colors, labels)):
        vals = []
        for ds in datasets:
            rows = [r for r in results
                    if r.get("dataset") == ds and r.get("condition") == cond]
            vals.append(np.mean([r.get("comm_bytes", 0) for r in rows]) / 1e6 if rows else 0)
        ax.bar(x + i * bar_w, vals, bar_w, label=label, color=color, alpha=0.8)

    ax.set_xticks(x + bar_w / 2)
    ax.set_xticklabels([d.upper() if d != "iiot" else "IIoT" for d in datasets])
    ax.set_ylabel("Communication (MB)")
    ax.set_title("Total Communication Cost per Condition")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "communication_cost.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info("Saved figure → %s", out_path)


def make_mia_figure(results: list, out_dir: Path) -> None:
    """MIA-AUC vs ε for both DP conditions."""
    if not _MPL:
        return

    datasets = ["skab", "tep", "iiot"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), squeeze=False)
    fig.suptitle("MIA-AUC vs ε (lower = more private)", fontsize=12)

    CMAP = {"fedavg_dp_nodedup": ("#d62728", "o", "DP no-dedup"),
            "fedavg_dp_dedup":   ("#1f77b4", "s", "deduplication")}

    for col, ds in enumerate(datasets):
        ax = axes[0][col]
        ax.axhline(0.5, linestyle="--", color="gray", linewidth=1, label="Random (AUC=0.5)")
        ds_rows = [r for r in results if r.get("dataset") == ds]

        for cond, (color, marker, label) in CMAP.items():
            cond_rows = [r for r in ds_rows if r.get("condition") == cond
                         and not np.isnan(r.get("mia_auc", float("nan")))]
            if not cond_rows:
                continue
            eps_vals = sorted(set(r.get("epsilon_target", r.get("epsilon", float("inf"))) for r in cond_rows if r.get("epsilon_target", r.get("epsilon")) not in (float("inf"), None)))
            means = [np.mean([r["mia_auc"] for r in cond_rows if r.get("epsilon_target", r.get("epsilon")) == e])
                     for e in eps_vals]
            ax.plot(eps_vals, means, marker=marker, color=color, label=label, linewidth=2)

        ax.set_xlabel("ε"); ax.set_ylabel("MIA-AUC")
        ax.set_title(ds.upper()); ax.set_xscale("log")
        ax.set_ylim(0.45, 1.0)
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_dir / "mia_auc_vs_epsilon.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_latex_table(results: list, dataset: str, out_dir: Path) -> str:
    """Generate LaTeX table for one dataset: conditions × metrics (mean±std over seeds)."""
    from collections import defaultdict

    rows_by_cond = defaultdict(list)
    for r in results:
        if r.get("dataset") == dataset:
            rows_by_cond[r["condition"]].append(r)

    COND_ORDER = [
        "centralized", "local_only",
        "fedavg_nodp_nodedup", "fedprox_nodp_nodedup",
        "fedavg_dp_nodedup",   "fedavg_dp_dedup",
    ]
    COND_NAMES = {
        "centralized":          "Centralized (upper bound)",
        "local_only":           "Local-only (lower bound)",
        "fedavg_nodp_nodedup":  "FedAvg (no DP, no dedup)",
        "fedprox_nodp_nodedup": "FedProx (no DP, no dedup)",
        "fedavg_dp_nodedup":    "FedAvg+DP (no dedup)",
        "fedavg_dp_dedup":      r"with deduplication",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Results on " + dataset.upper() + r" (mean±std over 5 seeds, ε=1.0)}",
        r"\label{tab:" + dataset + r"}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & F1 & Recall & AUPRC & MIA-AUC \\",
        r"\midrule",
    ]

    for cond in COND_ORDER:
        rows = [r for r in rows_by_cond.get(cond, [])
                if r.get("epsilon_target", r.get("epsilon", float("inf"))) in [1.0, float("inf")]]
        if not rows:
            continue
        name   = COND_NAMES.get(cond, cond)
        f1s    = [r["f1"]     for r in rows]
        rec    = [r["recall"] for r in rows]
        auprc  = [r["auprc"]  for r in rows]
        mia    = [r["mia_auc"] for r in rows if not np.isnan(r.get("mia_auc", float("nan")))]

        def fmt(vals):
            if not vals:
                return "—"
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.3f}±{s:.3f}"

        lines.append(f"{name} & {fmt(f1s)} & {fmt(rec)} & {fmt(auprc)} & {fmt(mia)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    latex = "\n".join(lines)

    out_path = out_dir / f"table_{dataset}.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex)
    log.info("LaTeX table → %s", out_path)
    return latex


def generate_all_figures_and_tables(results_path: Path, out_dir: Path) -> None:
    """One-shot: load results and produce all paper-ready outputs."""
    results = load_results(results_path)
    make_privacy_utility_figure(results, out_dir / "figures")
    make_communication_figure(results,  out_dir / "figures")
    make_mia_figure(results,            out_dir / "figures")
    for ds in ["skab", "tep", "iiot"]:
        make_latex_table(results, ds, out_dir / "tables")
    log.info("All figures and tables written to %s", out_dir)


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/all_results.json")
    generate_all_figures_and_tables(path, path.parent)
