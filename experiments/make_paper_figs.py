r"""Generate paper figures from artifact JSONs. Saves PDF+EPS to paper/figs/.
Fig1 = leakage-controlled vs uncontrolled pipeline schematic (the paper's signature).
Fig2 = selection-invariance frontier (F1 vs eps, +/-1 std) SWaT+SKAB.
Fig3 = MIA weak vs LiRA bars.
NEVER fabricate: Fig2/Fig3 read from characterization.json / lira_mia.json.

Publication style: Okabe-Ito colourblind-safe palette, serif + Computer-Modern math
to match the Springer sn-jnl body text, despined axes, vector PDF+EPS output.
"""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGS = ROOT / "paper" / "figs"; FIGS.mkdir(parents=True, exist_ok=True)

# --- publication rcParams (serif body, Computer-Modern math, thin clean axes) ---
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.5,
    "figure.dpi": 150,
    "savefig.dpi": 600,
})

# Okabe-Ito colourblind-safe palette
OI = {
    "black":   "#000000",
    "blue":    "#0072B2",   # controlled / safe
    "orange":  "#E69F00",
    "vermil":  "#D55E00",   # uncontrolled / artifact
    "green":   "#009E73",
    "skyblue": "#56B4E9",
}
C_UNCTRL, C_CTRL = OI["vermil"], OI["blue"]     # warm=uncontrolled, cool=controlled (CVD-safe)


def save(fig, name):
    for ext in ("pdf", "eps"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig); print("wrote", name)


def fig1_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    def pipeline(ax, title, gate, det, out, good):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
        edge = C_CTRL if good else C_UNCTRL
        face_hi = "#DCE9F4" if good else "#F7E0CF"     # light tint of edge colour
        face_lo = "#BFD8EF" if good else "#F0C6A6"     # stronger for the outcome box
        # centres spread over more vertical range with clear gaps for arrows
        boxes = [(0.5, 0.88, "Sensor stream"), (0.5, 0.615, gate),
                 (0.5, 0.35, det), (0.5, 0.085, out)]
        face = ["#EAEAEA", face_hi, face_hi, face_lo]
        bh = 0.135                                        # half-height of each box
        for k, ((x, y, txt), fc) in enumerate(zip(boxes, face)):
            ec = "#555" if k == 0 else edge                # stream box neutral; pipeline boxes coloured
            lw = 1.0 if k == 0 else 1.8
            b = FancyBboxPatch((x-0.43, y-bh/2), 0.86, bh, boxstyle="round,pad=0.012",
                               linewidth=lw, edgecolor=ec, facecolor=fc)
            ax.add_patch(b)
            weight = "bold" if k == 3 else "normal"
            ax.text(x, y, txt, ha="center", va="center", fontsize=7.9, weight=weight)
        # arrows span the full gap between consecutive boxes (more prominent)
        centres = [0.88, 0.615, 0.35, 0.085]
        for y_top, y_bot in zip(centres[:-1], centres[1:]):
            ax.add_patch(FancyArrowPatch((0.5, y_top-bh/2-0.005), (0.5, y_bot+bh/2+0.005),
                         arrowstyle="-|>", mutation_scale=13, linewidth=1.3,
                         color=edge, shrinkA=0, shrinkB=0))
    pipeline(axes[0], "(a) Uncontrolled pipeline (artifact)",
             "Gate fit on PRIVATE\n+anomaly data", "Detector on\nCONTAMINATED data",
             "Spurious gain\n+0.05 to +0.16 F1", good=False)
    pipeline(axes[1], "(b) Leakage-controlled protocol",
             "Gate on PUBLIC\nnormal-only data", "Detector on\nNORMAL-only data",
             "Gain disappears\n0.204 vs 0.208", good=True)
    fig.subplots_adjust(wspace=0.10)
    save(fig, "fig_schematic")


def fig2_frontier():
    c = json.load(open(ROOT / "results" / "characterization.json"))
    eps = [0.5, 1.0, 2.0, 4.0]; modes = ["full", "random", "tdedup", "fps"]
    lab = {"full": "full", "random": "random", "tdedup": "temporal", "fps": "diversity"}
    mk  = {"full": "o-", "random": "s--", "tdedup": "^:", "fps": "d-."}
    col = {"full": OI["black"], "random": OI["blue"], "tdedup": OI["green"], "fps": OI["vermil"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    handles = None
    for ax, ds in zip(axes, ["swat", "skab"]):
        for m in modes:
            mu = [np.mean([x["f1"] for x in c if x["dataset"]==ds and x["mode"]==m and x["epsilon"]==e]) for e in eps]
            sd = [np.std([x["f1"] for x in c if x["dataset"]==ds and x["mode"]==m and x["epsilon"]==e]) for e in eps]
            mu = np.array(mu); sd = np.array(sd)
            # light band underneath; line + markers clearly on top (high zorder)
            ax.fill_between(eps, mu-sd, mu+sd, alpha=0.09, color=col[m], linewidth=0, zorder=1)
            ax.plot(eps, mu, mk[m], label=lab[m], color=col[m], markersize=4.5,
                    linewidth=1.4, markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.set_title({"swat":"SWaT","skab":"SKAB"}.get(ds, ds.upper()))
        ax.set_xlabel(r"privacy budget $\varepsilon$")
        from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(eps))
        ax.xaxis.set_major_formatter(FixedFormatter([("%g" % e) for e in eps]))
        ax.xaxis.set_minor_locator(NullLocator())          # kill log minor-tick sci-notation artifact
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.4)
        if handles is None:
            handles, labels_ = ax.get_legend_handles_labels()
    axes[0].set_ylabel(r"F1 (mean $\pm$ 1 std)")            # left panel only; per-panel scales differ
    # single shared legend ABOVE both panels (no overlap with data)
    fig.legend(handles, labels_, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), handlelength=2.0, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.24, top=0.82)
    save(fig, "fig_frontier")


def fig3_mia():
    disp = {"skab": "SKAB", "swat": "SWaT"}
    naive  = {(r["dataset"], r["epsilon"]): r for r in json.load(open(ROOT / "results" / "mia_artifact_summary.json"))}
    honest = {(r["dataset"], r["epsilon"]): r for r in json.load(open(ROOT / "results" / "lira_mia_summary.json"))}
    settings = [("skab", 0.5), ("skab", 2.0), ("swat", 0.5), ("swat", 2.0)]
    labels = [f"{disp[d]}\n$\\varepsilon$={e}" for d, e in settings]
    nv = [naive[s]["lira_mean"] for s in settings]; ne = [naive[s]["lira_std"] for s in settings]
    hv = [honest[s]["lira_mean"] for s in settings]; he = [honest[s]["lira_std"] for s in settings]
    x = np.arange(4); w = 0.38
    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    ekw = dict(capsize=2.5, ecolor="#333", elinewidth=0.8, capthick=0.8)
    ax.bar(x-w/2, nv, w, yerr=ne, label="contiguous split", color=C_UNCTRL,
           edgecolor="white", linewidth=0.4, error_kw=ekw)
    ax.bar(x+w/2, hv, w, yerr=he, label="randomized split", color=C_CTRL,
           edgecolor="white", linewidth=0.4, error_kw=ekw)
    for xi, v, e in zip(x-w/2, nv, ne): ax.text(xi, v+e+0.012, f"{v:.2f}", ha="center", fontsize=6.5)
    for xi, v, e in zip(x+w/2, hv, he): ax.text(xi, v+e+0.012, f"{v:.2f}", ha="center", fontsize=6.5)
    ax.axhline(0.5, color="#444", linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(3.52, 0.505, "chance", fontsize=6.5, color="#444", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("LiRA MIA AUC")
    ax.set_ylim(0.45, 0.82)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    save(fig, "fig_mia")
    print("naive:", nv, "honest:", hv)


if __name__ == "__main__":
    fig1_schematic(); fig2_frontier(); fig3_mia()
    print("DONE figs in", FIGS)
