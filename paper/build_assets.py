#!/usr/bin/env python
"""Build every figure and data table of the paper from ``results/``.

    python paper/build_assets.py

Reads only files already in the repository (``results/all_results.json``,
``results/tables/results_aggregate.json``, ``results/significance.json``,
``results/ensemble_sweep.json``, ``results/figures/figure_data.json`` and the
RL logs under ``RL/runs/tetris-rl/``) and writes

    paper/figures/*.pdf   vector figures sized for the IEEE two-column layout
    paper/tables/*.tex    colour-styled tables, \\input by main.tex

so no number in the paper is typed by hand. Styling macros used in the tables
(\\hdr, \\best, \\gooda, ...) are defined in main.tex.
"""

import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

COLW, TEXTW = 3.5, 7.16  # IEEE column / text width, inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "axes.edgecolor": "#4A4F55",
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": "#E4E7EB", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

# One colour per run, used identically in every figure and in main.tex.
C = {
    "base_model": "#8D96A0",
    "nimble_recipe": "#EE8A2E",
    "ce_only": "#8C62C4",
    "pact_full": "#0E8C86",
}
LABEL = {
    "base_model": "Base", "nimble_recipe": "Nimble", "ce_only": "CE-only",
    "pact_full": "PACT",
    "no_cf": r"$-$CF", "no_pcr": r"$-$PC", "no_nr": r"$-$NR", "no_emd": r"$-$EMD",
    "no_perm_views": r"$-$PermViews",
}
TEXLABEL = {
    "base_model": "Base model", "nimble_recipe": "Nimble recipe",
    "ce_only": "CE-only", "pact_full": "\\textbf{PACT (ours)}",
    "no_cf": "$-\\mathcal{L}_{\\mathrm{CF}}$", "no_pcr": "$-\\mathcal{L}_{\\mathrm{PC}}$",
    "no_nr": "$-\\mathcal{L}_{\\mathrm{NR}}$", "no_emd": "$-\\mathcal{L}_{\\mathrm{EMD}}$",
    "no_perm_views": "$-$permuted views",
}
POS, NEG, INK = "#0E8C86", "#E0584C", "#2B2F33"
SEEDS = (17, 18, 19)
MAIN = ("nimble_recipe", "ce_only", "pact_full")
ABL = ("no_cf", "no_pcr", "no_nr", "no_emd", "no_perm_views")
DOMAINS = ("public_services", "commerce", "supply_chain", "travel", "education", "media")
KINDS = ("choice", "noul", "score")
KINDNAME = {"choice": "Choice", "noul": "Boolean", "score": "Score"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


RUNS = {(r["run"], r["seed"]): r for r in load(RES / "all_results.json")}
AGG = load(RES / "tables" / "results_aggregate.json")
SIG = load(RES / "significance.json")
SWEEP = load(RES / "ensemble_sweep.json")
FIGDATA = load(RES / "figures" / "figure_data.json")
BASE_KIND = {"choice": 0.5959, "noul": 0.8421, "score": 0.5000}  # results_tables.md, base row


def m(run, seed, key, mode="single", kind="all"):
    return RUNS[(run, seed)]["holdout"][mode]["summary"][kind][key]


def seeds_of(run):
    return [s for s in SEEDS if (run, s) in RUNS]


def stat(run, key, mode="single", kind="all"):
    values = [m(run, s, key, mode, kind) for s in seeds_of(run)]
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0), values


def base(key, mode="single"):
    entry = AGG[mode]["base_model"].get(key)
    return None if entry is None else entry["mean"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("figure", name)


def panel_tag(ax, text, x=-0.02, y=1.04):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold",
            ha="left", va="bottom", color=INK)


# --------------------------------------------------------------------- figures
def fig_main():
    """Four headline metrics, multi-seed arms with every seed drawn."""
    runs = ("base_model",) + MAIN
    panels = [
        ("accuracy", "single", "Accuracy (%)", True, (55, 95)),
        ("pair_accuracy", "single", "Pair accuracy (%)", True, (25, 85)),
        ("ece", "single", r"ECE (%) $\downarrow$", False, (0, 27)),
        ("permutation_tv", "ensemble", r"Permutation TV (%) $\downarrow$", False, (0, 13.5)),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 1.72))
    rng = np.random.default_rng(3)
    for ax, (key, mode, title, _, ylim), tag in zip(axes, panels, "abcd"):
        for i, run in enumerate(runs):
            color = C[run]
            if run == "base_model":
                mean, sd, vals = base(key, mode) * 100, 0.0, []
            else:
                mean, sd, vals = stat(run, key, mode)
                mean, sd, vals = mean * 100, sd * 100, [v * 100 for v in vals]
            ax.bar(i, mean, width=0.66, color=color, alpha=0.28, edgecolor=color,
                   linewidth=1.0, zorder=2)
            if sd:
                ax.errorbar(i, mean, yerr=sd, color=INK, capsize=2.2, lw=0.8,
                            capthick=0.8, zorder=4)
            for v in vals:
                ax.scatter(i + rng.uniform(-0.16, 0.16), v, s=9, color=color,
                           edgecolor="white", linewidth=0.4, zorder=5)
            ax.text(i, mean + (sd if sd else 0) + (ylim[1] - ylim[0]) * 0.025,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=6.3, color=INK)
            if key == "ece" and run != "base_model":
                cal, _, _ = stat(run, "ece", "single_calibrated")
                ax.scatter(i, cal * 100, marker="D", s=13, color="white",
                           edgecolor=color, linewidth=1.1, zorder=6)
        ax.set_xticks(range(len(runs)), [LABEL[r] for r in runs], rotation=0)
        ax.tick_params(axis="x", length=0, labelsize=6.6)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=3)
        ax.grid(axis="x", visible=False)
        panel_tag(ax, f"({tag})", x=-0.2)
    handles = [Line2D([], [], marker="o", ls="", color="#666", markersize=3.2,
                      markeredgecolor="white", label="individual seed"),
               Line2D([], [], color=INK, lw=0.8, label=r"mean $\pm$ s.d."),
               Line2D([], [], marker="D", ls="", markerfacecolor="white",
                      markeredgecolor="#666", markersize=3.5, label=r"after $T(x)$")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.09))
    fig.tight_layout(w_pad=1.0)
    save(fig, "main_results")


def fig_calibration():
    fd = FIGDATA
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.05),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    # (a) reliability of the shipped model, before and after T(x)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls=(0, (3, 2)), color="#9AA0A6", lw=0.8, zorder=1)
    for key, color, label, marker in (
            ("reliability_uncalibrated", "#E0584C", "uncalibrated", "o"),
            ("reliability_calibrated", C["pact_full"], r"with $T(x)$", "s")):
        bins = [b for b in fd[key] if b["count"] > 0]
        conf = np.array([b["confidence"] for b in bins])
        acc = np.array([b["accuracy"] for b in bins])
        size = np.array([b["count"] for b in bins])
        ax.plot(conf, acc, color=color, lw=0.8, alpha=0.45, zorder=2)
        ax.scatter(conf, acc, s=4 + np.sqrt(size) * 3.2, color=color, edgecolor="white",
                   linewidth=0.5, zorder=3, marker=marker, label=label)
    ax.set_xlim(0.25, 1.01)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title("Reliability (shipped model)", pad=3)
    ax.legend(loc="upper left", handletextpad=0.2, markerscale=0.7)
    ax.text(0.97, 0.05, "marker area $\\propto$ bin size", transform=ax.transAxes,
            ha="right", fontsize=6, color="#6B7178")
    panel_tag(ax, "(a)", x=-0.24)
    # (b) risk-coverage
    ax = axes[1]
    for key, color, label in (("risk_coverage_uncalibrated", "#E0584C", "uncalibrated"),
                              ("risk_coverage_calibrated", C["pact_full"], r"with $T(x)$")):
        curve = fd[key]["curve"]
        cov = [p["coverage"] for p in curve]
        risk = [p["risk"] * 100 for p in curve]
        ax.plot(cov, risk, color=color, lw=1.2,
                label=f"{label} (AURC {fd[key]['aurc'] * 100:.1f})")
    full = (1 - m("pact_full", 18, "accuracy")) * 100
    ax.axhline(full, color="#9AA0A6", lw=0.7, ls=(0, (3, 2)))
    ax.text(0.03, full + 0.7, "error at full coverage", fontsize=6, color="#6B7178")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 30)
    ax.set_xlabel("coverage (answered fraction)")
    ax.set_ylabel("selective risk (%)")
    ax.set_title("Risk\u2013coverage (shipped model)", pad=3)
    ax.legend(loc="upper left", handlelength=1.4)
    panel_tag(ax, "(b)", x=-0.2)
    # (c) holdout ECE before -> after, every multi-seed run
    ax = axes[2]
    y, ticks = 0, []
    for run in MAIN:
        for s in SEEDS:
            before = m(run, s, "ece") * 100
            after = m(run, s, "ece", "single_calibrated") * 100
            ax.plot([before, after], [y, y], color=C[run], lw=1.4, alpha=0.55,
                    solid_capstyle="round", zorder=2)
            ax.scatter(before, y, s=14, facecolor="white", edgecolor=C[run], lw=1.0, zorder=3)
            ax.scatter(after, y, s=16, color=C[run], zorder=4)
            ticks.append((y, f"{LABEL[run]} s{s}"))
            y += 1
        y += 0.6
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("holdout ECE (%)")
    ax.set_xlim(0, 22)
    ax.grid(axis="y", visible=False)
    ax.set_title(r"Holdout ECE before $\rightarrow$ after $T(x)$", pad=3)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#666",
               markersize=3.6, label="uncalibrated"),
        Line2D([], [], marker="o", ls="", color="#666", markersize=3.6, label=r"with $T(x)$")],
        loc="lower right", fontsize=6.2, handletextpad=0.1)
    panel_tag(ax, "(c)", x=-0.36)
    fig.tight_layout(w_pad=1.2)
    save(fig, "calibration")


def fig_position_bias():
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 1.8))
    ax = axes[0]
    tv = np.array(FIGDATA["permutation_tv"]) * 100
    bins = np.linspace(0, 52, 27)
    ax.hist(tv, bins=bins, color=C["pact_full"], alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.axvline(tv.mean(), color="#E0584C", lw=1.0, ls=(0, (3, 2)))
    ax.text(tv.mean() + 1, ax.get_ylim()[1] * 0.85, f"mean {tv.mean():.1f}",
            color="#E0584C", fontsize=6.5)
    ax.text(0.97, 0.55, f"median {np.median(tv):.1f}\n{(tv < 1).mean() * 100:.0f}% of items < 1",
            transform=ax.transAxes, ha="right", fontsize=6.3, color="#555")
    ax.set_yscale("log")
    ax.set_xlabel("per-item permutation TV (%)")
    ax.set_ylabel("items")
    ax.set_title("Residual position bias (shipped, $K{=}4$)", pad=3)
    panel_tag(ax, "(a)", x=-0.22)
    for ax, key, title, tag in ((axes[1], "accuracy", "Accuracy vs. ensemble size", "(b)"),
                                (axes[2], "permutation_tv", "Position bias vs. ensemble size", "(c)")):
        for run, label in (("pact_full_seed18", "PACT (shipped)"), ("base_model", "Base")):
            color = C["pact_full"] if run != "base_model" else C["base_model"]
            pts = [p for p in SWEEP[run] if key == "accuracy" or p["k"] > 1]
            ks = [p["k"] for p in pts]
            vals = [p[key] * 100 for p in pts]
            ax.plot(ks, vals, color=color, lw=1.4, marker="o", markersize=3.6,
                    markeredgecolor="white", markeredgewidth=0.5, label=label)
            for k, v in zip(ks, vals):
                ax.text(k, v + (0.8 if key == "accuracy" else 0.45), f"{v:.1f}",
                        ha="center", fontsize=5.8, color=color)
        ax.set_xticks([1, 2, 3, 4] if key == "accuracy" else [2, 3, 4])
        ax.set_xlabel("permutations $K$")
        ax.set_ylabel("accuracy (%)" if key == "accuracy" else "mean pairwise TV (%)")
        if key == "accuracy":
            ax.set_ylim(60, 86)
        else:
            ax.set_ylim(0, 21)
            ax.set_xlim(1.7, 4.3)
        ax.set_title(title, pad=3)
        ax.legend(loc="center right" if key == "accuracy" else "center right")
        panel_tag(ax, tag, x=-0.2)
    fig.tight_layout(w_pad=1.1)
    save(fig, "position_bias")


def ema(values, alpha=0.25):
    out, acc = [], None
    for v in values:
        acc = v if acc is None else alpha * v + (1 - alpha) * acc
        out.append(acc)
    return out


def fig_training():
    log = FIGDATA["train_log"]
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.85))
    ax = axes[0]
    terms = (("ce", r"$\mathcal{L}_{\mathrm{CE}}$", "#4A4F55"),
             ("cf", r"$\mathcal{L}_{\mathrm{CF}}$", "#3B7DD8"),
             ("pcr", r"$\mathcal{L}_{\mathrm{PC}}$", "#EE8A2E"),
             ("nr", r"$\mathcal{L}_{\mathrm{NR}}$", "#E0584C"),
             ("emd", r"$\mathcal{L}_{\mathrm{EMD}}$", "#8C62C4"))
    for key, label, color in terms:
        pts = [(r["step"], r[key]) for r in log if r.get(key) is not None]
        steps = [p[0] for p in pts]
        vals = ema([max(p[1], 1e-4) for p in pts])
        ax.plot(steps, vals, color=color, lw=1.0, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("loss (EMA)")
    ax.set_title("Loss terms", pad=2)
    ax.set_ylim(5e-4, 200)
    ax.legend(ncol=3, fontsize=5.8, loc="upper center", handlelength=1.0, columnspacing=0.5,
              handletextpad=0.3, borderaxespad=0.1)
    panel_tag(ax, "(a)", x=-0.3)
    ax = axes[1]
    steps = [r["step"] for r in log]
    ax.plot(steps, ema([r["cf_delta"] for r in log]), color="#3B7DD8", lw=1.2,
            label=r"pair margin $d$")
    ax.axhline(2.0, color="#3B7DD8", lw=0.7, ls=(0, (3, 2)))
    ax.text(590, 2.5, "$m=2$", fontsize=6, color="#3B7DD8", ha="right")
    ax.plot(steps, ema([r["necessity_gap"] for r in log]), color="#E0584C", lw=1.2,
            label=r"gap on $x^{\circ}$")
    ax.set_xlabel("optimiser step")
    ax.set_ylabel("logit units (nats)")
    ax.set_title("Pair margin and necessity gap", pad=2)
    ax.set_ylim(-1, 19)
    ax.legend(loc="upper left", fontsize=6, borderaxespad=0.1)
    panel_tag(ax, "(b)", x=-0.3)
    fig.tight_layout(w_pad=0.6)
    save(fig, "training")


def fig_domains():
    """PACT minus Nimble recipe per field kind and per domain, all three seeds."""
    comps = {c["seed"]: c for c in SIG["comparisons"] if c["run_b"] == "nimble_recipe"}
    rows = [("kind", k) for k in KINDS] + [("domain", d) for d in DOMAINS]
    fig, ax = plt.subplots(figsize=(COLW, 2.25))
    markers = {17: "o", 18: "s", 19: "^"}
    labels = []
    for i, (group, name) in enumerate(rows):
        table = "by_kind" if group == "kind" else "by_domain"
        deltas = [comps[s][table][name]["delta"] * 100 for s in SEEDS]
        n = comps[17][table][name]["n"]
        mean = st.mean(deltas)
        color = POS if mean >= 0 else NEG
        ax.barh(i, mean, height=0.62, color=color, alpha=0.25, edgecolor=color, lw=0.9)
        for s, v in zip(SEEDS, deltas):
            ax.scatter(v, i, marker=markers[s], s=12, color=color, edgecolor="white",
                       linewidth=0.4, zorder=4)
        same = all(v > 0 for v in deltas) or all(v < 0 for v in deltas)
        tag = KINDNAME.get(name, name.replace("_", " "))
        labels.append(f"{tag} ($n$={n})" + (" *" if same else ""))
    ax.axhline(2.5, color="#B8BEC5", lw=0.6)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(range(len(rows)), labels, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta$ accuracy, PACT $-$ Nimble recipe (pp)")
    ax.set_xlim(-21, 18)
    ax.grid(axis="y", visible=False)
    ax.text(-20.5, 1.0, "field\nkind", fontsize=6.2, color="#6B7178", va="center")
    ax.text(-20.5, 5.5, "domain", fontsize=6.2, color="#6B7178", va="center")
    handles = [Line2D([], [], marker=markers[s], ls="", color="#666", markersize=3.4,
                      markeredgecolor="white", label=f"seed {s}") for s in SEEDS]
    ax.legend(handles=handles, loc="lower right", fontsize=6, handletextpad=0.1)
    fig.tight_layout()
    save(fig, "domains")


def fig_rl():
    rl = ROOT / "RL" / "runs" / "tetris-rl"
    train = [json.loads(line) for line in (rl / "train_log.jsonl").read_text().splitlines() if line.strip()]
    evals = [json.loads(line) for line in (rl / "eval_log.jsonl").read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.6))
    ax = axes[0]
    steps = np.array([r["step"] for r in train])
    rate = np.array([r["top_pick_rate"] for r in train]) * 100
    ax.plot(steps, rate, color=C["pact_full"], lw=0.35, alpha=0.3)
    win = 50
    smooth = np.convolve(rate, np.ones(win) / win, mode="valid")
    ax.plot(steps[win - 1:], smooth, color=C["pact_full"], lw=1.3, label="50-step mean")
    ax.set_ylim(0, 104)
    ax.set_xlabel("REINFORCE step")
    ax.set_ylabel("top-pick rate (%)")
    ax.set_title("Top-pick rate (sampled)", pad=2)
    ax.legend(loc="lower right", fontsize=6)
    panel_tag(ax, "(a)", x=-0.34)
    ax = axes[1]
    es = [e["step"] for e in evals]
    ax.fill_between(es, [e["mean_score"] / 1000 for e in evals],
                    [e["max_score"] / 1000 for e in evals], color=C["nimble_recipe"], alpha=0.2, lw=0)
    ax.plot(es, [e["mean_score"] / 1000 for e in evals], color=C["nimble_recipe"], lw=1.2,
            marker="o", markersize=2.2, label="mean of 5")
    ax.plot(es, [e["max_score"] / 1000 for e in evals], color=C["nimble_recipe"], lw=0.6,
            ls=(0, (2, 1.5)), label="best of 5")
    ax.set_ylim(15, 20)
    ax.set_xlabel("REINFORCE step")
    ax.set_ylabel("score (thousands)")
    ax.set_title("Greedy play, 400-piece cap", pad=2)
    ax.legend(loc="lower right", fontsize=6)
    panel_tag(ax, "(b)", x=-0.34)
    fig.tight_layout(w_pad=0.6)
    save(fig, "rl_tetris")


# ---------------------------------------------------------------------- tables
def pct(v, d=1):
    return f"{v * 100:.{d}f}"


def fmt_ms(mean, sd, scale=100, d=1):
    return f"{mean * scale:.{d}f}\\,{{\\scriptsize$\\pm$\\,{sd * scale:.{d}f}}}"


def write(name, text):
    (TAB / f"{name}.tex").write_text(text, encoding="utf-8")
    print("table", name)


def best_index(values, lower):
    clean = [(v, i) for i, v in enumerate(values) if v is not None]
    return (min if lower else max)(clean)[1]


def tab_main():
    cols = [  # key, mode, scale, digits, lower-is-better
        ("accuracy", "single", 100, 1, False),
        ("pair_accuracy", "single", 100, 1, False),
        ("nll", "single", 1, 3, True),
        ("brier", "single", 1, 3, True),
        ("ece", "single", 100, 1, True),
        ("ece", "single_calibrated", 100, 1, True),
        ("aurc", "single", 100, 2, True),
        ("expected_score_mae", "single", 1, 3, True),
    ]
    runs = ("base_model",) + MAIN
    cells = {r: [] for r in runs}
    means = [[] for _ in cols]
    for j, (key, mode, scale, d, lower) in enumerate(cols):
        for r in runs:
            if r == "base_model":
                v = base(key, mode) if mode == "single" else None
                cells[r].append(None if v is None else f"{v * scale:.{d}f}")
                means[j].append(v)
            else:
                mean, sd, _ = stat(r, key, mode)
                cells[r].append(fmt_ms(mean, sd, scale, d))
                means[j].append(mean)
    lines = []
    for r in runs:
        row = []
        for j, (key, mode, scale, d, lower) in enumerate(cols):
            text = cells[r][j] if cells[r][j] is not None else "--"
            if cells[r][j] is not None and runs.index(r) == best_index(means[j], lower):
                text = f"\\best{{{text}}}"
            row.append(text)
        prefix = "\\rowcolor{pactrow}" if r == "pact_full" else ""
        lines.append(prefix + TEXLABEL[r] + " & " + " & ".join(row) + " \\\\")
    body = "\n".join(lines)
    write("main", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Holdout results, single-pass decoding ($K{{=}}1$), mean\\,$\\pm$\\,s.d.\\ over seeds 17/18/19
(324 items, 162 pairs). Accuracy, pair accuracy, ECE and AURC in \\%. ECE$_T$ is after the contextual
temperature $T(x)$ (not fitted for the base model). \\best{{Shaded}}: best mean per column.}}
\\label{{tab:main}}
\\setlength{{\\tabcolsep}}{{5.2pt}}
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l cc ccccc c}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Run}} & \\hd{{Acc.\\,$\\uparrow$}} & \\hd{{Pair acc.\\,$\\uparrow$}} & \\hd{{NLL\\,$\\downarrow$}} & \\hd{{Brier\\,$\\downarrow$}} &
\\hd{{ECE\\,$\\downarrow$}} & \\hd{{ECE$_T$\\,$\\downarrow$}} & \\hd{{AURC\\,$\\downarrow$}} & \\hd{{Score MAE\\,$\\downarrow$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def tab_ensemble():
    runs = ("base_model",) + MAIN
    cols = [("accuracy", False), ("pair_accuracy", False), ("ece", True),
            ("permutation_tv", True), ("permutation_flip_rate", True)]
    means = {k: [] for k, _ in cols}
    cells = {r: [] for r in runs}
    for key, lower in cols:
        for r in runs:
            if r == "base_model":
                v = base(key, "ensemble")
                cells[r].append(pct(v))
                means[key].append(v)
            else:
                mean, sd, _ = stat(r, key, "ensemble")
                cells[r].append(fmt_ms(mean, sd))
                means[key].append(mean)
    lines = []
    for i, r in enumerate(runs):
        row = []
        for j, (key, lower) in enumerate(cols):
            text = cells[r][j]
            if i == best_index(means[key], lower):
                text = f"\\best{{{text}}}"
            row.append(text)
        if r == "base_model":
            gain = base("accuracy", "ensemble") - base("accuracy", "single")
        else:
            gain = stat(r, "accuracy", "ensemble")[0] - stat(r, "accuracy")[0]
        gtext = f"{gain * 100:+.1f}".replace("-", "$-$")
        row.append(f"\\gainc{{{gtext}}}")
        prefix = "\\rowcolor{pactrow}" if r == "pact_full" else ""
        lines.append(prefix + TEXLABEL[r] + " & " + " & ".join(row) + " \\\\")
    body = "\n".join(lines)
    write("ensemble", f"""\\begin{{table}}[t]
\\centering
\\caption{{Permutation-ensembled decoding ($K{{=}}4$), \\%. TV: mean pairwise total variation
between the $K$ canonical distributions; Flip: share of items whose argmax changes across
orderings. $\\Delta_K$: accuracy gain of $K{{=}}4$ over $K{{=}}1$ (pp); a position-invariant
model should gain nothing.}}
\\label{{tab:ensemble}}
\\setlength{{\\tabcolsep}}{{2.6pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l ccccc c}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Run}} & \\hd{{Acc.}} & \\hd{{Pair}} & \\hd{{ECE}} & \\hd{{TV\\,$\\downarrow$}} & \\hd{{Flip\\,$\\downarrow$}} & \\hd{{$\\Delta_K$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def delta_cell(value, ref, scale, d, lower, strong):
    """Colour a seed-matched ablation cell by whether removing the term hurt."""
    delta = (value - ref) * scale
    hurt = delta > 0 if lower else delta < 0
    size = abs(delta)
    text = f"{value * scale:.{d}f}"
    if size < strong * 0.25:
        return text
    shade = "hurtB" if hurt and size >= strong else "hurtA" if hurt \
        else "helpB" if size >= strong else "helpA"
    return f"\\cellcolor{{{shade}}}{text}"


def tab_ablation():
    cols = [  # key, mode, scale, digits, lower, "strong" threshold in display units
        ("accuracy", "single", 100, 1, False, 2.0),
        ("pair_accuracy", "single", 100, 1, False, 2.0),
        ("nll", "single", 1, 3, True, 0.08),
        ("ece", "single", 100, 1, True, 2.0),
        ("aurc", "single", 100, 2, True, 1.0),
        ("expected_score_mae", "single", 1, 3, True, 0.05),
        ("permutation_tv", "ensemble", 100, 1, True, 1.5),
        ("permutation_flip_rate", "ensemble", 100, 1, True, 2.0),
    ]
    ref = {c[:2]: m("pact_full", 17, c[0], c[1]) for c in cols}
    lines = []
    for r in ("pact_full",) + ABL + ("ce_only",):
        row = []
        for key, mode, scale, d, lower, strong in cols:
            v = m(r, 17, key, mode)
            row.append(f"{v * scale:.{d}f}" if r == "pact_full"
                       else delta_cell(v, ref[(key, mode)], scale, d, lower, strong))
        label = TEXLABEL[r] if r != "ce_only" else "CE-only (all four off)"
        prefix = "\\rowcolor{pactrow}" if r == "pact_full" else ""
        lines.append(prefix + label + " & " + " & ".join(row) + " \\\\")
        if r == "pact_full" or r == ABL[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    write("ablation", f"""\\begin{{table*}}[t]
\\centering
\\caption{{Seed-matched leave-one-out ablations (seed 17 for every row; single-pass except the
last two columns, which use $K{{=}}4$). Cells are coloured by the change relative to the full
method in the first row (seed 17):
\swatch{{hurtB}}\,removing the term clearly hurts, \swatch{{hurtA}}\,slightly hurts,
\swatch{{helpA}}\,slightly helps, \swatch{{helpB}}\,clearly helps; uncoloured: change below a
quarter of the ``clear'' threshold (2\,pp for accuracy-type metrics). One seed per row: read colours as
directions, not as significance.}}
\\label{{tab:ablation}}
\\setlength{{\\tabcolsep}}{{5.6pt}}
\\begin{{tabular}}{{l cccccc cc}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Variant (seed 17)}} & \\hd{{Acc.\\,$\\uparrow$}} & \\hd{{Pair acc.\\,$\\uparrow$}} & \\hd{{NLL\\,$\\downarrow$}} &
\\hd{{ECE\\,$\\downarrow$}} & \\hd{{AURC\\,$\\downarrow$}} & \\hd{{Score MAE\\,$\\downarrow$}} &
\\hd{{TV$_{{K=4}}$\\,$\\downarrow$}} & \\hd{{Flip$_{{K=4}}$\\,$\\downarrow$}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")


def tab_significance():
    lines, current = [], None
    order = {"nimble_recipe": 0, "ce_only": 1, "base_model": 2}
    comps = sorted(SIG["comparisons"], key=lambda c: (order[c["run_b"]], c["seed"]))
    for c in comps:
        if c["run_b"] != current:
            if current is not None:
                lines.append("\\midrule")
            current = c["run_b"]
            name = {"nimble_recipe": "vs.\\ Nimble recipe", "ce_only": "vs.\\ CE-only",
                    "base_model": "vs.\\ Base model$^\\dagger$"}[current]
            lines.append(f"\\multicolumn{{7}}{{l}}{{\\cellcolor{{subhdr}}\\textit{{PACT {name}}}}} \\\\")
        p = c["mcnemar"]["exact_p"]
        lo, hi = c["bootstrap"]["ci95"]
        sig = p < 0.05
        ptext = "$<$0.001" if p < 0.001 else f"{p:.3f}"
        ptext = f"\\sigc{{{ptext}}}" if sig else ptext
        ci = f"[{lo * 100:+.1f}, {hi * 100:+.1f}]".replace("-", "$-$")
        diff = (c["accuracy_a"] - c["accuracy_b"]) * 100
        dtext = "0.0" if abs(diff) < 0.05 else f"{diff:+.1f}".replace("-", "$-$")
        lines.append(f"{c['seed']} & {pct(c['accuracy_a'])} & {pct(c['accuracy_b'])} & {dtext} & "
                     f"{c['mcnemar']['b']}/{c['mcnemar']['c']} & {ptext} & {ci} \\\\")
    body = "\n".join(lines)
    write("significance", f"""\\begin{{table}}[t]
\\centering
\\caption{{Paired tests on holdout accuracy (\\%), same 324 items. $b/c$: items only PACT / only
the control gets right; exact two-sided McNemar $p$; 95\\% CI of the accuracy difference from a
10{{,}}000-resample paired bootstrap (pp). \\sigc{{Highlighted}}: $p<0.05$.
$^\\dagger$Base-model predictions are from a local 4-bit re-evaluation (acc.\\ 63.9\\%).}}
\\label{{tab:significance}}
\\setlength{{\\tabcolsep}}{{3.1pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{c ccc c c c}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Seed}} & \\hd{{PACT}} & \\hd{{Ctrl.}} & \\hd{{$\\Delta$}} & \\hd{{$b/c$}} & \\hd{{$p$}} & \\hd{{95\\% CI}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_kind():
    runs_rows = [("base_model", None)] + [(r, s) for r in MAIN for s in SEEDS] + [(r, 17) for r in ABL]
    lines = []
    for r, s in runs_rows:
        if r == "base_model":
            vals = [f"{BASE_KIND[k] * 100:.1f}" for k in KINDS] + [f"{base('expected_score_mae'):.3f}"]
            seed = "--"
        else:
            vals = [pct(m(r, s, "accuracy", kind=k)) for k in KINDS] + \
                [f"{m(r, s, 'expected_score_mae'):.3f}"]
            seed = str(s)
        prefix = "\\rowcolor{pactrow}" if r == "pact_full" else ""
        lines.append(prefix + TEXLABEL[r].replace("\\textbf{PACT (ours)}", "\\textbf{PACT}") +
                     f" & {seed} & " + " & ".join(vals) + " \\\\")
        if r == "base_model" or (s == 19) or False:
            lines.append("\\midrule")
    body = "\n".join(lines)
    write("kind", f"""\\begin{{table}}[t]
\\centering
\\caption{{Holdout accuracy (\\%) by field kind, single-pass decoding (Choice $n{{=}}146$,
Boolean $n{{=}}114$, Score $n{{=}}64$), and expected-level MAE on Score fields.}}
\\label{{tab:kind}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l c ccc c}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Run}} & \\hd{{Seed}} & \\hd{{Choice}} & \\hd{{Boolean}} & \\hd{{Score}} & \\hd{{Score MAE}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_calibration():
    lines = []
    for r in MAIN:
        for s in SEEDS:
            cal = RUNS[(r, s)]["calibration"]
            prefix = "\\rowcolor{pactrow}" if r == "pact_full" else ""
            seed = f"{s}" + ("$^\\star$" if (r, s) == ("pact_full", 18) else "")
            inner_after = pct(cal["ece_after"])
            if cal["ece_after"] > cal["ece_before"]:
                inner_after = f"\\worse{{{inner_after}}}"
            lines.append(prefix + TEXLABEL[r].replace("\\textbf{PACT (ours)}", "\\textbf{PACT}") +
                         f" & {seed} & {cal['temperature_mean']:.2f} "
                         f"[{cal['temperature_min']:.2f}, {cal['temperature_max']:.2f}] & "
                         f"{pct(cal['ece_before'])} & {inner_after} & "
                         f"{pct(m(r, s, 'ece'))} & {pct(m(r, s, 'ece', 'single_calibrated'))} \\\\")
        if r != MAIN[-1]:
            lines.append("\\midrule")
    body = "\n".join(lines)
    write("calibration", f"""\\begin{{table}}[t]
\\centering
\\caption{{Contextual temperature per run: mean [min, max] of $T(x)$ on the calibration split,
and ECE (\\%) before/after on the calibration split (fit) and on the holdout (test).
$^\\star$Shipped model. \\worse{{Red}}: $T(x)$ raised ECE.}}
\\label{{tab:calibration}}
\\setlength{{\\tabcolsep}}{{3pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l c c cc cc}}
\\toprule
\\rowcolor{{hdr}}
 & & & \\multicolumn{{2}}{{c}}{{\\hd{{Calib.\\ split}}}} & \\multicolumn{{2}}{{c}}{{\\hd{{Holdout}}}} \\\\
\\rowcolor{{hdr}}
\\hd{{Run}} & \\hd{{Seed}} & \\hd{{$T(x)$}} & \\hd{{pre}} & \\hd{{post}} & \\hd{{pre}} & \\hd{{post}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def dev_history(run, seed):
    rows = load(RES / "runs" / run / f"seed-{seed}" / "dev_history.json")
    seen, out = set(), []
    for r in rows:
        if r["step"] not in seen:
            seen.add(r["step"])
            out.append((r["step"], r["summary"]["all"]["nll"], r["summary"]["all"]["accuracy"]))
    return out


def fig_selection():
    """Selection-split trajectories: how noisy checkpoint selection is."""
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.75))
    styles = {17: "-", 18: (0, (4, 1.5)), 19: (0, (1, 1.2))}
    for run in MAIN:
        for s in SEEDS:
            hist = dev_history(run, s)
            best = RUNS[(run, s)]  # selected step is stored in results.json
            chosen = load(RES / "runs" / run / f"seed-{s}" / "results.json")["train"]["best"]["step"]
            steps = [h[0] for h in hist]
            for ax, idx, scale in ((axes[0], 1, 1), (axes[1], 2, 100)):
                vals = [h[idx] * scale for h in hist]
                ax.plot(steps, vals, color=C[run], lw=1.0, ls=styles[s], alpha=0.9,
                        marker="o", markersize=2.0)
                j = steps.index(chosen)
                ax.scatter(steps[j], vals[j], marker="*", s=42, color=C[run],
                           edgecolor="white", linewidth=0.4, zorder=5)
            del best
    axes[0].set_ylabel("selection-split NLL")
    axes[1].set_ylabel("selection-split acc. (%)")
    for ax, tag in zip(axes, "ab"):
        ax.set_xlabel("optimiser step")
        ax.set_xticks([0, 146, 292, 438, 584])
        ax.tick_params(axis="x", labelsize=6)
        panel_tag(ax, f"({tag})", x=-0.3)
    axes[0].set_title("Checkpoint-selection NLL", pad=2)
    axes[1].set_title("Checkpoint-selection accuracy", pad=2)
    handles = [Line2D([], [], color=C[r], lw=1.4, label=LABEL[r]) for r in MAIN] + \
        [Line2D([], [], marker="*", ls="", color="#555", markersize=6, label="selected")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.1),
               fontsize=6.4, handlelength=1.4, columnspacing=0.9)
    fig.tight_layout(w_pad=0.5)
    save(fig, "selection")


def temperature(weights, n_choices, tokens):
    value = weights["bias"][0] + weights["log_choices"] * np.log(n_choices) \
        + weights["log_tokens"] * np.log(tokens / 1000)
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0)


def fig_temperature():
    fig, axes = plt.subplots(1, 2, figsize=(COLW, 1.75))
    ax = axes[0]
    shipped = load(RES / "model" / "calibration.json")["weights"]
    tokens = np.linspace(300, 1100, 60)
    ramp = plt.get_cmap("viridis")
    for i, n in enumerate((2, 3, 4, 5, 6)):
        ax.plot(tokens, temperature(shipped, n, tokens), color=ramp(0.12 + i * 0.19), lw=1.3,
                label=f"$C$={n}")
    ax.axhline(1, color="#9AA0A6", lw=0.7, ls=(0, (3, 2)))
    ax.text(1090, 1.02, "$T=1$", fontsize=6, color="#6B7178", ha="right", va="bottom")
    ax.set_xlabel("prompt length $L$ (tokens)")
    ax.set_ylabel("temperature $T(x)$")
    ax.set_title("Shipped model", pad=2)
    ax.legend(fontsize=5.8, loc="upper right", handlelength=1.1, ncol=1, borderaxespad=0.1)
    panel_tag(ax, "(a)", x=-0.3)
    ax = axes[1]
    grid = np.arange(2, 7)
    for run in MAIN:
        for s in SEEDS:
            w = load(RES / "runs" / run / f"seed-{s}" / "calibration.json")["weights"]
            ax.plot(grid, temperature(w, grid, 700.0), color=C[run], lw=1.0, marker="o",
                    markersize=2.3, alpha=0.9)
    ax.axhline(1, color="#9AA0A6", lw=0.7, ls=(0, (3, 2)))
    ax.set_ylim(0, 3.1)
    ax.set_xticks(grid)
    ax.set_xlabel("allowed answers $C$ ($L$=700)")
    ax.legend(handles=[Line2D([], [], color=C[r], lw=1.3, label=LABEL[r]) for r in MAIN],
              fontsize=5.8, loc="lower center", bbox_to_anchor=(0.5, 0.98), ncol=3,
              handlelength=1.0, columnspacing=0.6, borderaxespad=0.1)
    panel_tag(ax, "(b)", x=-0.3)
    fig.tight_layout(w_pad=0.5)
    save(fig, "temperature")


def shipped_rows():
    return load(RES / "holdout_rows.json")["rows"]


def pair_outcomes(rows):
    fams = {}
    for r in rows:
        fams.setdefault(r["family"], {})[r["variant"]] = r
    out = {}
    for pair in fams.values():
        b, c = pair["base"], pair["counterfactual"]
        key = ("both" if b["correct"] and c["correct"] else "base" if b["correct"]
               else "cf" if c["correct"] else "neither")
        for kind in ("all", b["kind"]):
            out.setdefault(kind, {"both": 0, "base": 0, "cf": 0, "neither": 0, "same": 0})
            out[kind][key] += 1
            out[kind]["same"] += int(b["prediction"] == c["prediction"])
    return out


def fig_errors():
    rows = shipped_rows()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTW, 2.1),
                             gridspec_kw={"width_ratios": [1, 1.15, 1.25]})
    # (a) confidence of right and wrong answers
    ax = axes[0]
    bins = np.linspace(0.2, 1.0, 17)
    right = [r["confidence"] for r in rows if r["correct"]]
    wrong = [r["confidence"] for r in rows if not r["correct"]]
    ax.hist(right, bins=bins, color=C["pact_full"], alpha=0.55, label=f"correct ($n$={len(right)})",
            edgecolor="white", linewidth=0.4)
    ax.hist(wrong, bins=bins, color=NEG, alpha=0.8, label=f"wrong ($n$={len(wrong)})",
            edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xlabel("stated confidence (uncalibrated)")
    ax.set_ylabel("items")
    ax.set_title("Confidence of right vs. wrong", pad=2)
    ax.legend(loc="upper left", fontsize=6.2)
    panel_tag(ax, "(a)", x=-0.24)
    # (b) pair outcomes by field kind
    ax = axes[1]
    outcomes = pair_outcomes(rows)
    order = ("all", "choice", "noul", "score")
    names = {"all": "All", **KINDNAME}
    parts = (("both", "both correct", C["pact_full"]),
             ("base", "only base correct", "#F2B880"),
             ("cf", "only counterfactual correct", "#9DB8E6"),
             ("neither", "both wrong", NEG))
    for i, kind in enumerate(order):
        total = sum(outcomes[kind][k] for k, _, _ in parts)
        left = 0
        for key, label, color in parts:
            share = outcomes[kind][key] / total * 100
            ax.barh(i, share, left=left, color=color, height=0.62, edgecolor="white",
                    linewidth=0.5, label=label if i == 0 else None)
            if share >= 7:
                ax.text(left + share / 2, i, f"{share:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if key in ("both", "neither") else INK)
            left += share
        same = outcomes[kind]["same"] / total * 100
        ax.text(101.5, i, f"{same:.0f}%", va="center", fontsize=6.2, color="#555")
    ax.text(101.5, -0.72, "same ans.", fontsize=5.6, color="#6B7178", va="center")
    ax.set_yticks(range(len(order)), [f"{names[k]} ({sum(outcomes[k][p] for p, _, _ in parts)})"
                                      for k in order], fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of pairs (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Pair outcomes (shipped model)", pad=2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.36), ncol=2, fontsize=5.8,
              handlelength=0.9, columnspacing=0.8)
    panel_tag(ax, "(b)", x=-0.3)
    # (c) ECE by field kind, before/after T(x), mean over seeds
    ax = axes[2]
    width = 0.26
    for j, run in enumerate(MAIN):
        for i, kind in enumerate(KINDS):
            before = stat(run, "ece", "single", kind)[0] * 100
            after = stat(run, "ece", "single_calibrated", kind)[0] * 100
            x = i + (j - 1) * width
            ax.bar(x, before, width=width * 0.92, color=C[run], alpha=0.3, edgecolor=C[run], lw=0.8)
            ax.bar(x, after, width=width * 0.5, color=C[run])
    ax.set_xticks(range(3), [KINDNAME[k] for k in KINDS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("holdout ECE (%)")
    ax.set_title(r"ECE by field kind (mean of 3 seeds)", pad=2)
    handles = [Patch(facecolor=C[r], label=LABEL[r]) for r in MAIN] + \
        [Patch(facecolor="#BBBBBB", alpha=0.35, edgecolor="#888", label="uncalibrated"),
         Patch(facecolor="#777777", label=r"with $T(x)$")]
    ax.legend(handles=handles, fontsize=5.8, ncol=3, loc="upper left", handlelength=1.0,
              columnspacing=0.7)
    ax.set_ylim(0, 40)
    panel_tag(ax, "(c)", x=-0.2)
    fig.tight_layout(w_pad=1.0)
    save(fig, "errors")


def tab_data():
    views = load(RES / "runs" / "pact_full" / "seed-17" / "results.json")["views"]["primary_kinds"]
    split = {"train": "Training", "dev_select": "Selection", "dev_calib": "Calibration"}
    holdout = {"choice": 146, "noul": 114, "score": 64}
    rows = []
    for key, name in split.items():
        kinds = {k: views[f"{key}/{k}"] for k in KINDS}
        rows.append((name, kinds))
    rows.append(("Holdout (frozen)", holdout))
    families = {"Training": 30, "Selection": 2, "Calibration": 2, "Holdout (frozen)": 6}
    use = {"Training": "train", "Selection": "select",
           "Calibration": "fit $T$", "Holdout (frozen)": "test only"}
    lines = []
    for name, kinds in rows:
        total = sum(kinds.values())
        prefix = "\\rowcolor{pactrow}" if name.startswith("Holdout") else ""
        lines.append(prefix + f"{name} & {families[name]} & {total:,} & {total // 2:,} & "
                     f"{kinds['choice']} & {kinds['noul']} & {kinds['score']} & {use[name]} \\\\"
                     .replace(",", "{,}"))
    body = "\n".join(lines)
    write("data", f"""\\begin{{table}}[t]
\\centering
\\caption{{Data splits. Splits are disjoint at the level of source families; every record belongs
to a two-record contrastive pair. The holdout is never used for selection or calibration.}}
\\label{{tab:data}}
\\setlength{{\\tabcolsep}}{{2.6pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l c r r ccc l}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Split}} & \\hd{{Fam.}} & \\hd{{Records}} & \\hd{{Pairs}} & \\hd{{Choice}} & \\hd{{Bool.}} & \\hd{{Score}} & \\hd{{Role}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_cost():
    runs = [("nimble_recipe", SEEDS), ("ce_only", SEEDS), ("pact_full", SEEDS)] + [(r, (17,)) for r in ABL]
    lines = []
    for run, seeds in runs:
        res = [load(RES / "runs" / run / f"seed-{s}" / "results.json") for s in seeds]
        steps = res[0]["train"]["steps"]
        v = res[0]["views"]["views"]
        per_pair = (v.get("train/primary", 0) + v.get("train/perm1", 0) + v.get("train/necessity", 0)) \
            / (v["train/primary"] / 2)
        hours = st.mean(r["train"]["wall_seconds"] for r in res) / 3600
        peak = max(r["train"]["peak_gpu_gib"] for r in res)
        acc = st.mean(m(run, s, "accuracy") for s in seeds) * 100
        label = TEXLABEL[run].replace("\\textbf{PACT (ours)}", "\\textbf{PACT}")
        prefix = "\\rowcolor{pactrow}" if run == "pact_full" else ""
        lines.append(prefix + f"{label} & {steps} & {per_pair:.1f} & {hours:.2f} & {peak:.1f} & {acc:.1f} \\\\")
        if run == "pact_full":
            lines.append("\\midrule")
    body = "\n".join(lines)
    write("cost", f"""\\begin{{table}}[t]
\\centering
\\caption{{Training cost per run (one 48\\,GB GPU per job): optimiser steps, training views per pair,
mean wall-clock time, peak GPU memory, and mean holdout accuracy (\\%). Serving cost is identical
for all runs.}}
\\label{{tab:cost}}
\\setlength{{\\tabcolsep}}{{3.2pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l c c c c c}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Run}} & \\hd{{Steps}} & \\hd{{Views/pair}} & \\hd{{Hours}} & \\hd{{Peak GiB}} & \\hd{{Acc.}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def tab_hparams():
    full = load(RES / "runs" / "pact_full" / "seed-17" / "results.json")
    cfg = full["config"]
    mo, lo, op, da, ev = cfg["model"], cfg["loss"], cfg["optim"], cfg["data"], cfg["evaluation"]
    rows = [
        ("Base model", "Qwen3.5-9B, BF16, SDPA attention"),
        ("Max. prompt length", f"{da['max_length']:,} tokens (longest seen: "
                               f"{full['views']['max_tokens']:,})"),
        ("LoRA", f"rank {mo['lora_rank']}, $\\alpha$={mo['lora_alpha']}, dropout {mo['lora_dropout']}, rsLoRA"),
        ("LoRA targets", "q,k,v,o,gate,up,down (language model)"),
        ("Optimiser", f"AdamW, lr ${op['learning_rate'] * 1e5:g}\\times10^{{-5}}$, "
                      f"weight decay {op['weight_decay']:g}"),
        ("LoRA+ ratio", f"{op['lora_b_multiplier']:g} (lr of $B$ / lr of $A$)"),
        ("Schedule", f"{op['scheduler']}, warm-up {op['warmup_frac']:.0%}, {op['epochs']:g} epochs"),
        ("Batch", f"{op['groups_per_batch']} pairs $\\times$ {op['grad_accum']} accumulation = 8 primary rows"),
        ("Gradient clipping", f"global norm {op['max_grad_norm']:.1f}"),
        ("Loss weights", f"$\\lambda_{{\\mathrm{{cf}}}}$={lo['cf_weight']}, $\\lambda_{{\\mathrm{{pc}}}}$={lo['pcr_weight']}, "
                         f"$\\lambda_{{\\mathrm{{nr}}}}$={lo['nr_weight']}, $\\lambda_{{\\mathrm{{emd}}}}$={lo['ordinal_emd_weight']}"),
        ("Margins", f"$m$={lo['cf_margin']:g}, $\\varepsilon$={lo['nr_margin']:g}; ramp {lo['reg_warmup_frac']:.0%} of steps"),
        ("Length bucketing", f"{op['length_bucket_groups']} groups per bucket"),
        ("Selection", f"every {op['eval_every_frac']:.0%} of training, lowest selection NLL"),
        ("Evaluation", f"$K$={ev['permutation_ensemble']} permutations, ECE with {ev['ece_bins']} bins"),
        ("Calibration", "contextual $T(x)$, L-BFGS, 120 iterations"),
        ("Seeds", "17, 18, 19 (ablations: 17)"),
    ]
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows).replace("%", "\\%")
    write("hparams", f"""\\begin{{table}}[t]
\\centering
\\caption{{Hyper-parameters of PACT (read from the resolved run configuration). The
Nimble recipe differs only in: 1 epoch, linear schedule, no rsLoRA, LoRA+ ratio 1, no extra
views or terms.}}
\\label{{tab:hparams}}
\\setlength{{\\tabcolsep}}{{4pt}}
\\footnotesize
\\rowcolors{{2}}{{white}}{{rowgray}}
\\begin{{tabular}}{{l >{{\\raggedright\\arraybackslash}}p{{5.2cm}}}}
\\toprule
\\rowcolor{{hdr}}
\\hd{{Setting}} & \\hd{{Value}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")


def numbers():
    """Print the derived numbers the prose quotes, as a cross-check."""
    out = {}
    for r in MAIN:
        for mode in ("single", "single_calibrated", "ensemble"):
            for key in ("accuracy", "pair_accuracy", "nll", "ece", "expected_score_mae"):
                mean, sd, _ = stat(r, key, mode)
                out[f"{r}.{mode}.{key}"] = (round(mean, 4), round(sd, 4))
    for r in ABL + ("pact_full", "ce_only"):
        out[f"{r}.s17"] = {k: round(m(r, 17, k), 4) for k in ("accuracy", "pair_accuracy", "ece", "aurc")}
    (TAB / "numbers.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    fig_main()
    fig_calibration()
    fig_position_bias()
    fig_training()
    fig_domains()
    fig_rl()
    fig_selection()
    fig_temperature()
    fig_errors()
    tab_main()
    tab_data()
    tab_cost()
    tab_hparams()
    tab_ensemble()
    tab_ablation()
    tab_significance()
    tab_kind()
    tab_calibration()
    numbers()
