"""Charts, drawn from finished runs.

Five figures, each answering one question the paper asks:

  reliability.png      does the stated probability match how often it is right,
                       before and after the fitted temperature?
  risk_coverage.png    if we only answer the most confident share, how does the
                       error rate fall?
  permutation.png      how much does relabelling the answer codes move the
                       distribution?
  runs_comparison.png  how do the runs and ablations compare on the holdout?
  training.png         did the loss terms and the inner validation behave?

Everything a figure plots is also written to ``figure_data.json``, so the
numbers survive even when matplotlib is missing.
"""

import json
from pathlib import Path

from pact.metrics import reliability_bins, selective_risk

# Publication palette. Runs are colour-coded by role, not by plot order, so the
# same run means the same colour in every figure: grey for the untouched base
# model, warm amber/plum for the two reproduced controls, dark teal for the
# full method, and a light-to-dark blue ramp for the leave-one-out ablations
# (closer to pact_full's teal = closer to the full recipe).
INK = "#1b1f27"
GRID = "#d8dce1"
ROLE_COLOR = {
    "base_model": "#9aa0a8",
    "nimble_recipe": "#d98a3d",
    "ce_only": "#8a5fb0",
    "pact_full": "#0f6b66",
}
ABLATION_RAMP = ["#8fb8d9", "#6f9fc7", "#5087b5", "#356f9e", "#1c5786"]


def _color_for(name):
    if name in ROLE_COLOR:
        return ROLE_COLOR[name]
    ablations = [n for n in ("no_cf", "no_pcr", "no_nr", "no_emd", "no_perm_views") if n]
    if name in ablations:
        return ABLATION_RAMP[ablations.index(name) % len(ABLATION_RAMP)]
    return "#4c78a8"


def _apply_style(plt):
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 220, "font.size": 10.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": INK, "axes.edgecolor": "#5b6270", "axes.labelcolor": INK,
        "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlepad": 8,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": "#3a3f47", "ytick.color": "#3a3f47",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _load(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rows(run_dir, name):
    payload = _load(Path(run_dir) / name)
    return payload.get("rows") if payload else None


def _pyplot(log):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _apply_style(plt)
        return plt
    except ImportError:
        log("[figures] matplotlib is not installed; writing figure_data.json only "
            "(pip install matplotlib to get the charts)")
        return None


def gather(run_dir, output_dir, bins=15):
    """Collect everything the figures need, without drawing anything."""
    run_dir = Path(run_dir)
    data = {"run_dir": str(run_dir)}
    raw = _rows(run_dir, "holdout_single.json")
    if raw:
        data["reliability_uncalibrated"] = reliability_bins(raw, bins)
        data["risk_coverage_uncalibrated"] = selective_risk(raw)
    tuned = _rows(run_dir, "holdout_single_calibrated.json")
    if tuned:
        data["reliability_calibrated"] = reliability_bins(tuned, bins)
        data["risk_coverage_calibrated"] = selective_risk(tuned)
    ensembled = _rows(run_dir, "holdout_ensemble.json")
    if ensembled:
        values = [row["permutation_tv"] for row in ensembled
                  if row.get("permutation_tv") is not None]
        if values:
            data["permutation_tv"] = values
            data["permutation_flip_rate"] = sum(
                row.get("permutation_flip", 0) for row in ensembled) / len(ensembled)
    aggregate = _load(Path(output_dir) / "results_aggregate.json")
    if aggregate:
        data["aggregate"] = aggregate
    from pact import BUNDLE_ROOT

    sweep = _load(BUNDLE_ROOT / "results" / "ensemble_sweep.json")
    if sweep:
        data["ensemble_sweep"] = {name: points for name, points in
                                  (("pact_full", sweep.get("pact_full_seed18")),
                                   ("base_model", sweep.get("base_model")))
                                  if points}
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        steps = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    steps.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        data["train_log"] = steps
    history = _load(run_dir / "dev_history.json")
    if history:
        data["dev_history"] = history
    return data


def draw(data, destination, log=print):
    """Write the PNGs. Returns the list of files actually produced."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "figure_data.json").write_text(
        json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    plt = _pyplot(log)
    if plt is None:
        return []
    written = []

    if data.get("reliability_uncalibrated"):
        figure, axis = plt.subplots(figsize=(4.6, 4.4))
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.3, color="#a9afb8",
                  label="perfect", zorder=1)
        style = {"reliability_uncalibrated": ("uncalibrated", "#c94f4f", "s", "--"),
                 "reliability_calibrated": ("calibrated", "#0f6b66", "o", "-")}
        for key, (label, color, marker, ls) in style.items():
            bins = data.get(key)
            if not bins:
                continue
            points = [(b["confidence"], b["accuracy"], b["count"]) for b in bins if b["count"]]
            if points:
                axis.plot([p[0] for p in points], [p[1] for p in points],
                          marker=marker, markersize=5, linewidth=1.8, linestyle=ls,
                          color=color, label=label, zorder=3,
                          markeredgecolor="white", markeredgewidth=0.6)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("stated probability")
        axis.set_ylabel("observed accuracy")
        axis.set_title("Reliability on the holdout")
        axis.legend(loc="upper left")
        written.append(_save(figure, destination / "reliability.png"))

    if data.get("risk_coverage_uncalibrated"):
        figure, axis = plt.subplots(figsize=(4.6, 4.4))
        style = {"risk_coverage_uncalibrated": ("uncalibrated", "#c94f4f", "--"),
                 "risk_coverage_calibrated": ("calibrated", "#0f6b66", "-")}
        for key, (label, color, ls) in style.items():
            curve = data.get(key)
            if not curve:
                continue
            axis.plot([point["coverage"] for point in curve["curve"]],
                      [point["risk"] for point in curve["curve"]],
                      linewidth=2, linestyle=ls, color=color,
                      label=f"{label} (AURC {curve['aurc']:.3f})")
        axis.set_xlim(-0.02, 1.02)
        axis.set_xlabel("coverage (most confident share answered)")
        axis.set_ylabel("error rate on what was answered")
        axis.set_title("Selective risk")
        axis.legend(loc="upper left")
        written.append(_save(figure, destination / "risk_coverage.png"))

    if data.get("permutation_tv"):
        figure, axis = plt.subplots(figsize=(4.6, 3.6))
        axis.hist(data["permutation_tv"], bins=20, color="#5087b5",
                 edgecolor="white", linewidth=0.6)
        axis.set_xlabel("total variation between answer-code orderings")
        axis.set_ylabel("examples")
        axis.set_title("Position bias left after training\n"
                       f"(answer flips on {data['permutation_flip_rate']:.1%} of examples)",
                       fontsize=10.5)
        written.append(_save(figure, destination / "permutation.png"))

    sweep = data.get("ensemble_sweep")
    if sweep:
        figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
        for run_name, points in sweep.items():
            color = _color_for(run_name)
            ks = [p["k"] for p in points]
            axes[0].plot(ks, [p["accuracy"] for p in points], marker="o",
                        color=color, label=run_name, linewidth=2, markersize=4)
            axes[1].plot(ks, [p["permutation_tv"] for p in points], marker="o",
                        color=color, label=run_name, linewidth=2, markersize=4)
        axes[0].set_xlabel("K (permutations averaged)")
        axes[0].set_ylabel("holdout accuracy")
        axes[0].set_title("Accuracy vs. ensemble size", fontsize=10.5)
        axes[1].set_xlabel("K (permutations averaged)")
        axes[1].set_ylabel("mean pairwise TV")
        axes[1].set_title("Residual position bias vs. K", fontsize=10.5)
        axes[0].set_xticks(sorted({p["k"] for pts in sweep.values() for p in pts}))
        axes[1].set_xticks(sorted({p["k"] for pts in sweep.values() for p in pts}))
        axes[0].legend(fontsize=8)
        written.append(_save(figure, destination / "ensemble_sweep.png"))

    aggregate = data.get("aggregate", {}).get("single")
    if aggregate:
        metrics = [("accuracy", "Accuracy", True), ("pair_accuracy", "Pair accuracy", True),
                   ("ece", "ECE (lower better)", False)]
        from pact.report import ordered_names

        names = [name for name in ordered_names(aggregate)
                 if any(metric in aggregate[name] for metric, _, _ in metrics)]
        if names:
            colors = [_color_for(name) for name in names]
            figure, axes = plt.subplots(1, len(metrics), figsize=(4.3 * len(metrics), 4.2))
            for axis, (metric, title, _) in zip(axes, metrics):
                values = [aggregate[name].get(metric, {}).get("mean") for name in names]
                errors = [aggregate[name].get(metric, {}).get("std", 0) for name in names]
                heights = [value if value is not None else 0 for value in values]
                positions = list(range(len(names)))
                axis.bar(positions, heights, capsize=3, color=colors,
                         edgecolor="white", linewidth=0.7,
                         yerr=[error if value is not None else 0
                               for value, error in zip(values, errors)],
                         error_kw={"ecolor": "#3a3f47", "elinewidth": 1.1, "capthick": 1.1})
                # Bars keep a zero baseline - honest, but small gaps are then
                # hard to see, so print the number on each bar.
                for position, value in zip(positions, values):
                    if value is None:
                        continue
                    axis.text(position, value, f"{value:.3f}", ha="center", va="bottom",
                              fontsize=7.5, color=INK)
                axis.set_ylim(0, max(heights) * 1.18 if max(heights) else 1)
                axis.set_xticks(positions)
                axis.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
                axis.set_title(title, fontsize=10.5)
            figure.suptitle("Holdout, single-pass decoding (mean ± sd over seeds)",
                            fontsize=12, fontweight="bold")
            written.append(_save(figure, destination / "runs_comparison.png"))

    steps = data.get("train_log")
    if steps:
        figure, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
        term_colors = {"ce": "#3a3f47", "cf": "#0f6b66", "pcr": "#d98a3d",
                       "nr": "#8a5fb0", "emd": "#c94f4f"}
        term_labels = {"ce": "cross-entropy", "cf": "pair margin", "pcr": "permutation",
                      "nr": "necessity", "emd": "ordinal"}
        for key, label in term_labels.items():
            series = [(record["step"], record[key]) for record in steps if key in record]
            if series:
                axes[0].plot([point[0] for point in series],
                             [point[1] for point in series], label=label,
                             linewidth=1.6, color=term_colors[key], alpha=0.9)
        axes[0].set_xlabel("optimiser step")
        axes[0].set_ylabel("loss term")
        axes[0].set_title("Training", fontsize=10.5)
        axes[0].legend(fontsize=8)
        history = data.get("dev_history") or []
        points = [(entry["step"], entry["summary"]["all"]) for entry in history
                  if entry.get("summary", {}).get("all")]
        if points:
            axes[1].plot([p[0] for p in points], [p[1]["nll"] for p in points],
                         marker="o", color="#0f6b66", linewidth=1.8, markersize=4,
                         label="NLL (selection criterion)")
            axes[1].grid(False)
            twin = axes[1].twinx()
            twin.plot([p[0] for p in points], [p[1]["accuracy"] for p in points],
                      marker="s", color="#d98a3d", linewidth=1.8, markersize=4,
                      label="accuracy")
            twin.set_ylabel("accuracy")
            twin.grid(False)
            axes[1].set_ylabel("NLL")
            handles = axes[1].get_lines() + twin.get_lines()
            axes[1].legend(handles, [handle.get_label() for handle in handles],
                           fontsize=8, loc="center right")
        axes[1].set_xlabel("optimiser step")
        axes[1].set_title("Inner validation (selection)", fontsize=10.5)
        written.append(_save(figure, destination / "training.png"))

    log(f"[figures] wrote {len(written)} chart(s) to {destination}")
    return written


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    import matplotlib.pyplot as plt

    plt.close(figure)
    return str(path)


def build(run_dir, output_dir, destination, bins=15, log=print):
    """Gather and draw in one call."""
    return draw(gather(run_dir, output_dir, bins), destination, log)
