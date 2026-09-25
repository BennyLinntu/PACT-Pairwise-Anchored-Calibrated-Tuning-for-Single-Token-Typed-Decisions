"""Turn run outputs into the tables the paper reports.

Reads every ``<output_dir>/<run>/seed-<n>/results.json`` written by
``train_all.py`` and emits, side by side, a Markdown and a LaTeX version of:

1. the headline comparison on the frozen 324-example holdout;
2. the leave-one-out ablation of the four PACT terms;
3. calibration before and after the contextual temperature;
4. the per-task-type breakdown.

Numbers are only ever copied from results files - nothing here invents a
result, and a missing run shows up as an empty cell.
"""

import json
import math
import statistics
from pathlib import Path

HEADLINE = [
    ("accuracy", "Accuracy", 4, True),
    ("pair_accuracy", "Pair accuracy", 4, True),
    ("nll", "NLL", 4, False),
    ("brier", "Brier", 4, False),
    ("ece", "ECE", 4, False),
    ("permutation_tv", "Perm. TV", 4, False),
    ("permutation_flip_rate", "Perm. flip", 4, False),
]
SCORE_METRICS = [("expected_score_mae", "Score MAE", 4, False)]
# Reading order for the tables: references first, then the method, then the
# leave-one-out ablations. Unlisted runs keep alphabetical order at the end.
PREFERRED_ORDER = ["base_model", "nimble_recipe", "ce_only", "pact_full", "pact_full_qlora",
                   "ce_only_qlora", "no_cf", "no_pcr", "no_nr", "no_emd", "no_perm_views"]


def ordered_names(aggregated):
    listed = [name for name in PREFERRED_ORDER if name in aggregated]
    return listed + sorted(set(aggregated) - set(listed))


def collect(output_dir):
    """All results files under an output directory, newest layout only."""
    records = []
    for path in sorted(Path(output_dir).glob("*/seed-*/results.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["path"] = str(path)
        records.append(payload)
    baseline = Path(output_dir) / "baselines.json"
    if baseline.exists():
        records.append(json.loads(baseline.read_text(encoding="utf-8")))
    return records


def _pick(record, decoding, metric, kind="all"):
    node = record.get("holdout", {}).get(decoding, {}).get("summary", {}).get(kind, {})
    value = node.get(metric)
    return value if isinstance(value, (int, float)) else None


def aggregate(records, decoding="single", kind="all"):
    """Mean and standard deviation over seeds, per run."""
    runs = {}
    for record in records:
        runs.setdefault(record.get("run", "unknown"), []).append(record)
    out = {}
    for name, group in runs.items():
        entry = {"seeds": sorted(r.get("seed") for r in group if r.get("seed") is not None),
                 "runs": len(group), "description": group[0].get("description", "")}
        for metric, _, _, _ in HEADLINE + SCORE_METRICS:
            values = [v for v in (_pick(r, decoding, metric, kind) for r in group) if v is not None]
            if not values:
                continue
            entry[metric] = {"mean": statistics.fmean(values),
                             "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                             "n": len(values)}
        out[name] = entry
    return out


def _cell(entry, metric, digits):
    value = entry.get(metric)
    if value is None:
        return "--"
    if value["n"] > 1:
        return f"{value['mean']:.{digits}f} ± {value['std']:.{digits}f}"
    return f"{value['mean']:.{digits}f}"


def markdown_table(aggregated, metrics, title, order=None):
    names = order or ordered_names(aggregated)
    names = [name for name in names if name in aggregated]
    header = "| Run | " + " | ".join(label for _, label, _, _ in metrics) + " |"
    rule = "| --- |" + " ---: |" * len(metrics)
    lines = [f"**{title}**", "", header, rule]
    for name in names:
        entry = aggregated[name]
        cells = [_cell(entry, metric, digits) for metric, _, digits, _ in metrics]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def latex_table(aggregated, metrics, caption, label, order=None):
    # Wide tables (many metric columns, ± sd) overflow a standard \textwidth
    # at normal size -- shrink-wrap them rather than let them bleed into the
    # margin. Small tables are left alone so they don't get stretched.
    wide = len(metrics) >= 5
    names = order or ordered_names(aggregated)
    names = [name for name in names if name in aggregated]
    columns = "l" + "r" * len(metrics)
    lines = ["\\begin{table}[t]", "\\centering"]
    if wide:
        lines += ["\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\resizebox{\\textwidth}{!}{%"]
    lines += [f"\\begin{{tabular}}{{{columns}}}", "\\toprule",
             "Run & " + " & ".join(label for _, label, _, _ in metrics) + " \\\\", "\\midrule"]
    for name in names:
        entry = aggregated[name]
        cells = [_cell(entry, metric, digits).replace("±", "$\\pm$")
                 for metric, _, digits, _ in metrics]
        lines.append(name.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}" + ("}" if wide else ""),
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}", ""]
    return "\n".join(lines)


def calibration_table(records):
    rows = ["**Calibration on the inner split (fitted), holdout ECE (applied)**", "",
            "| Run | Seed | Mode | ECE before | ECE after | Holdout ECE (uncal.) | "
            "Holdout ECE (cal.) |", "| --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for record in records:
        calibration = record.get("calibration")
        if not calibration:
            continue
        uncalibrated = record.get("holdout", {}).get("single", {}).get("summary", {}).get("all", {})
        calibrated = record.get("holdout", {}).get("single_calibrated", {}).get(
            "summary", {}).get("all", {})
        rows.append("| {run} | {seed} | {mode} | {before} | {after} | {raw} | {tuned} |".format(
            run=record.get("run", "?"), seed=record.get("seed", "-"),
            mode=calibration.get("mode", "-"),
            before=_format(calibration.get("ece_before")),
            after=_format(calibration.get("ece_after")),
            raw=_format(uncalibrated.get("ece")), tuned=_format(calibrated.get("ece"))))
    return "\n".join(rows) + "\n"


def _format(value, digits=4):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and math.isfinite(value) else "--"


def kind_table(records, decoding="single"):
    rows = ["**Accuracy by task type (holdout)**", "",
            "| Run | Seed | Choice | Noul | Score | Score MAE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for record in records:
        summary = record.get("holdout", {}).get(decoding, {}).get("summary", {})
        if not summary:
            continue
        rows.append("| {run} | {seed} | {choice} | {noul} | {score} | {mae} |".format(
            run=record.get("run", "?"), seed=record.get("seed", "-"),
            choice=_format(summary.get("choice", {}).get("accuracy")),
            noul=_format(summary.get("noul", {}).get("accuracy")),
            score=_format(summary.get("score", {}).get("accuracy")),
            mae=_format(summary.get("score", {}).get("expected_score_mae"))))
    return "\n".join(rows) + "\n"


def build(output_dir, paper_dir=None, log=print):
    records = collect(output_dir)
    if not records:
        log(f"[report] no results under {output_dir}")
        return None
    single = aggregate(records, "single")
    ensemble = aggregate(records, "ensemble")
    metrics = HEADLINE + SCORE_METRICS
    markdown = [
        "# PACT results",
        "",
        f"Runs found: {len(records)}. Every number below is copied from a "
        "`results.json` written by `train_all.py`.",
        "",
        markdown_table(single, metrics, "Single-pass decoding (holdout, 324 examples)"),
        "",
        markdown_table(ensemble, metrics, "Permutation-ensembled decoding (holdout)"),
        "",
        kind_table(records),
        "",
        calibration_table(records),
    ]
    latex = [
        latex_table(single, metrics,
                    "Single-pass decoding on the frozen 324-example holdout.", "tab:main"),
        latex_table(ensemble, metrics,
                    "Permutation-ensembled decoding on the same holdout.", "tab:ensemble"),
    ]
    output = Path(output_dir)
    (output / "results_tables.md").write_text("\n".join(markdown), encoding="utf-8")
    (output / "results_tables.tex").write_text("\n".join(latex), encoding="utf-8")
    (output / "results_aggregate.json").write_text(
        json.dumps({"single": single, "ensemble": ensemble}, indent=2), encoding="utf-8")
    if paper_dir:
        paper = Path(paper_dir)
        paper.mkdir(parents=True, exist_ok=True)
        (paper / "tables.md").write_text("\n".join(markdown), encoding="utf-8")
        (paper / "tables.tex").write_text("\n".join(latex), encoding="utf-8")
    log(f"[report] wrote tables to {output / 'results_tables.md'}")
    return {"single": single, "ensemble": ensemble, "records": len(records)}
