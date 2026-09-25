"""Package the finished sweep into one folder a human can open.

Called automatically at the end of ``train_all.py``. It picks the model to
ship, copies its weights, its calibrator and its contract, gathers the tables,
the figures and every run's summary, and writes a README for the folder.

**How the shipped model is chosen.** By inner-validation NLL, across the seeds
of the primary run - never by holdout score. Choosing on the frozen holdout
would turn the only clean number in the project into a training signal.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ADAPTER_FILES = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                 "schema_config.json", "calibration.json", "tokenizer.json",
                 "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                 "vocab.json", "merges.txt", "tokenizer.model"]


def load_results(output_dir):
    records = []
    for path in sorted(Path(output_dir).glob("*/seed-*/results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        payload["_path"] = str(path)
        payload["_dir"] = str(path.parent)
        records.append(payload)
    return records


def inner_score(record):
    """The inner-validation NLL used for selection, or None."""
    best = (record.get("train") or {}).get("best") or {}
    if isinstance(best.get("nll"), (int, float)):
        return best["nll"], best.get("accuracy")
    history = record.get("dev_history") or []
    for entry in reversed(history):
        summary = (entry.get("summary") or {}).get("all") or {}
        if isinstance(summary.get("nll"), (int, float)):
            return summary["nll"], summary.get("accuracy")
    return None, None


def select_model(records, cfg, log=print):
    """Pick the run x seed to ship, using inner validation only."""
    candidates = [r for r in records if r.get("run") == cfg.release.primary_run]
    if not candidates:
        candidates = [r for r in records if r.get("run") not in (None, "base_model")]
    if not candidates:
        return None
    scored = []
    for record in candidates:
        nll, accuracy = inner_score(record)
        scored.append({"record": record, "nll": nll, "accuracy": accuracy})
    usable = [item for item in scored if item["nll"] is not None]
    if cfg.release.selection == "first" or not usable:
        chosen = scored[0]
        reason = "first completed run (no inner-validation score available)"
    elif cfg.release.selection == "inner_accuracy":
        chosen = max(usable, key=lambda item: (item["accuracy"] or 0, -item["nll"]))
        reason = "highest inner-validation accuracy"
    else:
        chosen = min(usable, key=lambda item: item["nll"])
        reason = "lowest inner-validation NLL"
    record = chosen["record"]
    log(f"[release] shipping {record['run']} seed {record['seed']} ({reason}"
        + (f", inner NLL {chosen['nll']:.4f}" if chosen["nll"] is not None else "") + ")")
    return {"record": record, "reason": reason, "inner_nll": chosen["nll"],
            "inner_accuracy": chosen["accuracy"]}


def export_model(selection, cfg, destination, log=print):
    """Copy the adapter, and write a plain torch state dict beside it."""
    record = selection["record"]
    source = Path(record.get("adapter") or "")
    if not source.exists():
        for candidate in (Path(record["_dir"]) / "best", Path(record["_dir"]) / "final"):
            if (candidate / "adapter_config.json").exists():
                source = candidate
                break
    if not source.exists():
        log(f"[release] no adapter directory for {record['run']} seed {record['seed']}")
        return None
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ADAPTER_FILES:
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
            copied.append(name)
    for extra in ("calibration.json",):
        path = Path(record["_dir"]) / extra
        if path.exists() and not (destination / extra).exists():
            shutil.copy2(path, destination / extra)
            copied.append(extra)
    exported = {"source": str(source), "files": copied}
    if cfg.release.save_pth:
        exported["state_dict"] = _write_state_dict(source, destination / "adapter.pth", log)
    if cfg.release.merge_full_model:
        exported["merged"] = _merge_full_model(record, source, destination.parent / "model_merged",
                                               log)
    return exported


def _write_state_dict(source, destination, log):
    """adapter.pth: the trained LoRA tensors as a plain torch state dict."""
    import torch

    safetensors = source / "adapter_model.safetensors"
    if safetensors.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    elif (source / "adapter_model.bin").exists():
        state = torch.load(str(source / "adapter_model.bin"), map_location="cpu")
    else:
        log("[release] no adapter weights to convert")
        return None
    torch.save(state, destination)
    size = destination.stat().st_size / 1024 ** 2
    log(f"[release] wrote {destination.name} ({size:.1f} MB, {len(state)} tensors)")
    return {"path": str(destination), "tensors": len(state), "megabytes": round(size, 1)}


def _merge_full_model(record, adapter, destination, log):
    """Optional: base weights with the adapter folded in, on CPU."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoTokenizer

        from pact.model import _load_base
        from pact.config import PactConfig

        cfg = PactConfig.from_dict(record["config"])
        log("[release] merging the adapter into the base weights on CPU "
            "(this needs RAM and disk, and takes a while)")
        base, _ = _load_base(cfg, "cpu", torch.bfloat16, log)
        merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload(safe_merge=True)
        destination = Path(destination)
        merged.save_pretrained(str(destination), safe_serialization=True)
        AutoTokenizer.from_pretrained(str(adapter)).save_pretrained(str(destination))
        shutil.copy2(adapter / "schema_config.json", destination / "schema_config.json")
        log(f"[release] merged model written to {destination}")
        return str(destination)
    except Exception as error:  # noqa: BLE001 - never lose the run over packaging
        log(f"[release] merge skipped: {type(error).__name__}: {error}")
        return None


def _headline(aggregate, decoding="single"):
    table = aggregate.get(decoding, {}) if aggregate else {}
    order = ["base_model", "nimble_recipe", "ce_only", "pact_full"]
    names = [name for name in order if name in table] + \
            [name for name in sorted(table) if name not in order]
    lines = ["| Run | Accuracy | Pair accuracy | ECE | NLL |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        entry = table[name]

        def cell(metric):
            value = entry.get(metric)
            if not value:
                return "--"
            if value.get("n", 1) > 1:
                return f"{value['mean']:.4f} ± {value['std']:.4f}"
            return f"{value['mean']:.4f}"

        lines.append(f"| {name} | {cell('accuracy')} | {cell('pair_accuracy')} | "
                     f"{cell('ece')} | {cell('nll')} |")
    return "\n".join(lines)


def write_summary(path, cfg, records, selection, aggregate, figures, exported, output_dir):
    """The one file to open when the sweep is done."""
    record = selection["record"] if selection else None
    finished = [r for r in records if r.get("holdout")]
    lines = [
        "# PACT results",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`{output_dir}`.",
        "",
        f"* runs with a holdout score: **{len(finished)}** of {len(records)} recorded",
        f"* base model: `{cfg.model.model_id}` @ `{cfg.model.revision}`",
        f"* holdout: the frozen `data/eval.jsonl`, used for scoring only",
        f"* figures drawn: {len(figures)}" if figures else "* figures: none drawn",
        "",
        "## Headline (single-pass decoding, mean ± sd over seeds)",
        "",
        _headline(aggregate),
        "",
    ]
    if record:
        lines += [
            "## Shipped model",
            "",
            f"* run **{record['run']}**, seed **{record['seed']}** — chosen by "
            f"{selection['reason']}"
            + (f" (inner NLL {selection['inner_nll']:.4f})" if selection["inner_nll"] else ""),
            f"* weights: `model/` (PEFT adapter"
            + (", `model/adapter.pth` as a plain state dict"
               if exported and exported.get("state_dict") else "") + ")",
            "* temperature: `model/calibration.json` (applied automatically by `PactScorer`)",
            "* contract: `model/schema_config.json` — records the base model, the revision "
            "and the SHA-256 of the scoring prompt this adapter was trained against",
            "",
            "```python",
            "from pact.inference import PactScorer",
            "",
            "scorer = PactScorer('results/model', ensemble=1)",
            "print(scorer.score('The payment service is down for all customers.', schema))",
            "```",
            "",
        ]
        holdout = (record.get("holdout") or {}).get("single", {}).get("summary", {}).get("all")
        if holdout:
            lines += ["Its own holdout numbers:", "",
                      f"* accuracy {holdout['accuracy']:.4f} "
                      f"({holdout['correct']}/{holdout['count']})",
                      f"* pair accuracy {holdout.get('pair_accuracy', float('nan')):.4f}",
                      f"* NLL {holdout['nll']:.4f}, Brier {holdout['brier']:.4f}, "
                      f"ECE {holdout['ece']:.4f}", ""]
    lines += [
        "## What is in this folder",
        "",
        "| Path | What |",
        "| --- | --- |",
        "| `SUMMARY.md` | this file |",
        "| `model/` | the shipped adapter, its calibrator and its contract |",
        "| `tables/results_tables.md` | every run, every metric, Markdown |",
        "| `tables/results_tables.tex` | the same as LaTeX, for the paper |",
        "| `tables/results_aggregate.json` | the same as JSON |",
        "| `figures/*.png` | reliability, selective risk, position bias, run comparison, training |",
        "| `figures/figure_data.json` | the numbers behind the figures |",
        "| `runs/<run>/seed-<n>/results.json` | one run's full record, including its config |",
        "| `holdout_rows.json` | the shipped model's per-example holdout predictions |",
        "| `all_results.json` | every run's summaries in one file |",
        "",
        "## Reading the ablations",
        "",
        "The ablation runs are a test, not a search. `paper/paper.md` §5.5 states in "
        "advance what each one should do if its term is working: compare against that, "
        "not against the best cell in the table. Seed-to-seed spread is in the ± column.",
        "",
        "## Caveats",
        "",
        "* The holdout is 324 synthetic, model-labelled examples from six source families.",
        "* The shipped model was chosen on inner validation, so its holdout number is "
        "an honest estimate; the per-run table is not a leaderboard to pick from.",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mirror_to_paper(figures, log):
    """Put the same PNGs next to the LaTeX, so \\includegraphics just works."""
    if not figures:
        return
    from pact import BUNDLE_ROOT

    target = BUNDLE_ROOT / "paper" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    for path in figures:
        shutil.copy2(path, target / Path(path).name)
    log(f"[release] mirrored {len(figures)} figure(s) to {target}")


def package(cfg, output_dir, destination, log=print):
    """Collect weights, tables, figures and summaries into one folder."""
    from pact import figures as figure_module

    output_dir, destination = Path(output_dir), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = load_results(output_dir)
    if not records:
        log(f"[release] no results under {output_dir}; nothing to package")
        return None
    selection = select_model(records, cfg, log)
    aggregate = {}
    aggregate_path = output_dir / "results_aggregate.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    tables = destination / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for name in ("results_tables.md", "results_tables.tex", "results_aggregate.json",
                 "preflight.json", "config.resolved.json", "failures.json"):
        path = output_dir / name
        if path.exists():
            shutil.copy2(path, tables / name if name.startswith("results")
                         else destination / name)

    runs = destination / "runs"
    for record in records:
        target = runs / record["run"] / f"seed-{record['seed']}"
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["_path"], target / "results.json")
        for extra in ("train_report.json", "dev_history.json", "calibration.json",
                      "train_plan.json"):
            path = Path(record["_dir"]) / extra
            if path.exists():
                shutil.copy2(path, target / extra)
    baseline = output_dir / "baselines.json"
    if baseline.exists():
        shutil.copy2(baseline, destination / "baselines.json")

    summaries = [{k: v for k, v in record.items()
                  if k in ("run", "seed", "description", "holdout", "calibration",
                           "holdout_examples", "adapter")} for record in records]
    (destination / "all_results.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")

    exported, drawn = None, []
    if selection:
        exported = export_model(selection, cfg, destination / "model", log)
        if cfg.release.copy_rows:
            rows = Path(selection["record"]["_dir"]) / "holdout_single.json"
            if rows.exists():
                shutil.copy2(rows, destination / "holdout_rows.json")
        if cfg.release.figures:
            drawn = figure_module.build(selection["record"]["_dir"], output_dir,
                                        destination / "figures",
                                        cfg.evaluation.ece_bins, log)
            _mirror_to_paper(drawn, log)
    write_summary(destination / "SUMMARY.md", cfg, records, selection, aggregate, drawn,
                  exported, output_dir)
    log(f"[release] packaged {len(records)} run(s) into {destination}")
    return {"destination": str(destination), "records": len(records),
            "selected": (selection or {}).get("record", {}).get("run"),
            "seed": (selection or {}).get("record", {}).get("seed"),
            "model": exported, "figures": drawn}
