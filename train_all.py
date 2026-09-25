#!/usr/bin/env python
"""PACT - one command that runs the whole pipeline.

    cd new_methods
    CUDA_VISIBLE_DEVICES=0,1 nohup python train_all.py > train.log 2>&1 &

With no arguments it reads ``configs/server_2x48gb.json`` and runs every stage
in order:

  preflight  verify the frozen data, the pair structure, the reconstructable
             evidence ablations and the config. No GPU, no model download.
  build      encode every training view once per distinct run configuration,
             with the pinned tokenizer, and cache it. Needs the tokenizer only.
  train      for every run x seed: fit the adapter with the PACT objective,
             validating on inner families held out of training.
  calibrate  fit the contextual temperature on the inner calibration split.
  evaluate   score the frozen 324-example holdout - single-pass, calibrated
             single-pass, and permutation-ensembled - plus the untouched base
             model for reference.
  report     aggregate every run into Markdown and LaTeX tables.
  package    collect the shipped weights (PEFT adapter + adapter.pth), the
             calibrator, the tables, the figures and every run's record into
             ./results/, with a SUMMARY.md to open first.

**Several GPUs.** The plan is a sweep of independent run x seed jobs, so the
work is split *by job*: one worker process per visible GPU, each holding a
complete model and running one job at a time, with no gradient traffic between
them. Two GPUs therefore do two jobs at once and stay busy for the whole sweep.
The shared view caches are built once, in this process, before any worker
starts. Use ``--no-parallel`` to force everything through a single process, or
``--devices 0`` to pin one.

The frozen holdout is never used for training, checkpoint selection or
calibration; all three use source families held out of the training file.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

from pact import BUNDLE_ROOT, bundle_path  # noqa: E402
from pact.config import PactConfig  # noqa: E402

STAGES = ("preflight", "build", "train", "calibrate", "evaluate", "report",
          "package")
WORK_STAGES = ("train", "calibrate", "evaluate")
DEFAULT_CONFIG = BUNDLE_ROOT / "configs" / "server_2x48gb.json"


def make_logger(path=None):
    handle = open(path, "a", encoding="utf-8") if path else None

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if handle:
            handle.write(line + "\n")
            handle.flush()

    return log


def write_atomic(path, payload):
    """Write JSON so that two workers can never leave a half file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


# ------------------------------------------------------------------ stages


def stage_preflight(cfg, output_dir, log):
    """Everything that can be checked on a laptop, before any GPU time."""
    from pact import ablation
    from pact.data import build_pairs, verify_dataset
    from pact.model import memory_estimate

    cfg.validate()
    report, train_rows, eval_rows = verify_dataset(cfg)
    groups, skipped = build_pairs(train_rows, True, cfg.data.limit_pairs)
    paired = [group for group in groups if group["paired"]]
    coverage = ablation.coverage(train_rows)
    payload = {
        "dataset": report,
        "groups": len(groups),
        "contrastive_pairs": len(paired),
        "unpaired_or_skipped": skipped,
        "evidence_ablation": coverage,
        "holdout_rows": len(eval_rows),
        "memory_estimate": memory_estimate(cfg),
        "config": cfg.to_dict(),
    }
    if coverage["ablation_built"] < coverage["with_certificate"]:
        log(f"[preflight] warning: {coverage['with_certificate'] - coverage['ablation_built']} "
            "records have no reconstructable ablation; those pairs train without the "
            "necessity term")
    write_atomic(output_dir / "preflight.json", payload)
    log("[preflight] " + json.dumps({k: payload[k] for k in
                                     ("groups", "contrastive_pairs", "holdout_rows")}))
    log(f"[preflight] ablation coverage: {coverage['ablation_built']}/{coverage['records']}")
    return payload


def stage_build(cfg, runs, log):
    """Build every distinct view cache once, before any worker needs it."""
    from pact.data import build_views, cache_fingerprint, verify_dataset
    from pact.model import load_tokenizer

    tokenizer = load_tokenizer(cfg)
    report, _, _ = verify_dataset(cfg)
    seen, built = set(), []
    for spec in runs:
        run_cfg = cfg.with_overrides(spec.get("overrides", {}))
        digest, _ = cache_fingerprint(run_cfg, report)
        if digest in seen:
            continue
        seen.add(digest)
        log(f"[build] view cache {digest} for run {spec['name']}")
        manifest, path = build_views(run_cfg, tokenizer, log=log)
        built.append({"digest": digest, "runs": spec["name"], "path": str(path),
                      "views": manifest["views"]})
    log(f"[build] {len(built)} distinct view cache(s) ready")
    return built


# --------------------------------------------------------------- one job


def run_once(base_cfg, spec, seed, output_dir, arguments, log):
    """Train (or reuse) one adapter, calibrate it, and score the holdout."""
    import torch

    from pact.calibrate import Calibrator, fit_calibrator
    from pact.data import dev_examples, holdout_examples, load_views
    from pact.evaluate import cached_dev_scorer, calibration_records, evaluate_examples
    from pact.model import LogitReader, build_model, load_adapter_weights, load_tokenizer
    from pact.trainer import PactTrainer

    overrides = dict(spec.get("overrides", {}))
    overrides["optim.seed"] = seed
    cfg = base_cfg.with_overrides(overrides).validate()
    run_dir = output_dir / spec["name"] / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.json"
    if results_path.exists() and not arguments.force:
        log(f"[run] {spec['name']}/seed-{seed}: results exist, skipping (use --force to redo)")
        return json.loads(results_path.read_text(encoding="utf-8"))

    tokenizer = load_tokenizer(cfg)
    store = load_views(cfg, tokenizer, log=log)
    results = {"run": spec["name"], "seed": seed, "description": spec.get("description", ""),
               "config": cfg.to_dict(), "views": store.summary(),
               "device_env": os.environ.get("CUDA_VISIBLE_DEVICES", "")}

    adapter_dir = None
    for candidate in (run_dir / "best", run_dir / "final"):
        if (candidate / "adapter_config.json").exists():
            adapter_dir = candidate
            break

    if "train" in arguments.stages and not (adapter_dir and arguments.skip_existing_train):
        trainer = PactTrainer(cfg, store, run_dir, log=log)
        trainer.prepare()
        device = trainer.device
        scorer = cached_dev_scorer(cfg, store, "dev_select", device,
                                   cfg.evaluation.ece_bins,
                                   store.manifest.get("pad_token_id", 0))
        train_report = trainer.train(dev_scorer=scorer)
        results["train"] = {k: v for k, v in train_report.items() if k != "history"}
        results["dev_history"] = train_report.get("history", [])
        model, reader = trainer.model, trainer.reader
        adapter_dir = run_dir / ("best" if (run_dir / "best" / "adapter_config.json").exists()
                                 else "final")
        if trainer.best and (run_dir / "best" / "adapter_config.json").exists():
            load_adapter_weights(model, run_dir / "best")
            log(f"[run] restored the best inner-validation checkpoint "
                f"(step {trainer.best['step']})")
    else:
        if adapter_dir is None:
            raise FileNotFoundError(f"No adapter found under {run_dir}; run the train stage")
        model, info = build_model(cfg, log)
        device = info["device"]
        load_adapter_weights(model, adapter_dir)
        reader = LogitReader()
        log(f"[run] loaded adapter from {adapter_dir}")
    model.eval()
    results["adapter"] = str(adapter_dir)

    calibrator = None
    if "calibrate" in arguments.stages and cfg.evaluation.calibrate:
        calibration_split = dev_examples(cfg, store, "dev_calib")
        if calibration_split:
            records, raw_summary = calibration_records(model, reader, tokenizer,
                                                       calibration_split, cfg, device)
            calibrator, stats = fit_calibrator(records, cfg.evaluation.calibration_model, log)
            calibrator.save(run_dir / "calibration.json")
            if adapter_dir is not None:
                calibrator.save(Path(adapter_dir) / "calibration.json")
            results["calibration"] = stats
            results["calibration_split"] = {"rows": len(records),
                                            "accuracy": raw_summary["all"]["accuracy"]}
        else:
            log("[calibrate] no calibration split available; skipping")
    elif (run_dir / "calibration.json").exists():
        calibrator = Calibrator.load(run_dir / "calibration.json")

    if "evaluate" in arguments.stages:
        holdout = holdout_examples(cfg)
        if cfg.evaluation.holdout_limit:
            holdout = holdout[:cfg.evaluation.holdout_limit]
            log(f"[eval] WARNING: holdout truncated to {len(holdout)} examples "
                "(evaluation.holdout_limit); this is a smoke setting, not a result")
        results["holdout"] = {}
        results["holdout_examples"] = len(holdout)
        single = evaluate_examples(model, reader, tokenizer, holdout, cfg, device,
                                   ensemble=1, calibrator=None, log=log)
        results["holdout"]["single"] = {"summary": single["summary"]}
        write_atomic(run_dir / "holdout_single.json", single)
        if calibrator is not None:
            tuned = evaluate_examples(model, reader, tokenizer, holdout, cfg, device,
                                      ensemble=1, calibrator=calibrator, log=log)
            results["holdout"]["single_calibrated"] = {"summary": tuned["summary"]}
            write_atomic(run_dir / "holdout_single_calibrated.json", tuned)
        if cfg.evaluation.permutation_ensemble > 1:
            ensembled = evaluate_examples(model, reader, tokenizer, holdout, cfg, device,
                                          ensemble=cfg.evaluation.permutation_ensemble,
                                          calibrator=calibrator, log=log)
            results["holdout"]["ensemble"] = {"summary": ensembled["summary"]}
            write_atomic(run_dir / "holdout_ensemble.json", ensembled)
        baseline_path = output_dir / "baselines.json"
        if (cfg.evaluation.score_base_model and arguments.baseline
                and not baseline_path.exists()):
            _score_base_model(model, reader, tokenizer, holdout, cfg, device,
                              baseline_path, log)

    write_atomic(results_path, results)
    log(f"[run] wrote {results_path}")
    del model
    try:
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - CPU runs have nothing to free
        pass
    return results


def _score_base_model(model, reader, tokenizer, holdout, cfg, device, path, log):
    """The same evaluation with the adapter switched off."""
    from pact.evaluate import evaluate_examples

    log("[baseline] scoring the untouched base model (adapter disabled)")
    with model.disable_adapter():
        single = evaluate_examples(model, reader, tokenizer, holdout, cfg, device,
                                   ensemble=1, calibrator=None, log=log)
        payload = {"run": "base_model", "seed": None,
                   "description": f"{cfg.model.model_id} @ {cfg.model.revision}, no adapter",
                   "holdout": {"single": {"summary": single["summary"]}}}
        if cfg.evaluation.permutation_ensemble > 1:
            ensembled = evaluate_examples(model, reader, tokenizer, holdout, cfg, device,
                                          ensemble=cfg.evaluation.permutation_ensemble,
                                          calibrator=None, log=log)
            payload["holdout"]["ensemble"] = {"summary": ensembled["summary"]}
    write_atomic(path, payload)


# ----------------------------------------------------------- GPU scheduling


def visible_devices(explicit=None):
    """GPU ids to use, as this process sees them in its own environment."""
    if explicit:
        return [item.strip() for item in explicit.split(",") if item.strip()]
    environment = os.environ.get("CUDA_VISIBLE_DEVICES")
    if environment is not None and environment.strip():
        return [item.strip() for item in environment.split(",") if item.strip()]
    try:
        import torch

        return [str(index) for index in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 - no torch, no GPUs
        return []


def spawn_worker(job, device, config_path, output_dir, stages, arguments, allow_baseline):
    """One job, pinned to one GPU, in its own process."""
    spec, seed = job
    run_dir = Path(output_dir) / spec["name"] / f"seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(SCRIPT), "--config", str(config_path),
               "--output-dir", str(output_dir), "--stages", *stages,
               "--only-run", spec["name"], "--only-seed", str(seed), "--no-parallel"]
    if arguments.force:
        command.append("--force")
    if arguments.skip_existing_train:
        command.append("--skip-existing-train")
    command.append("--allow-baseline" if allow_baseline else "--no-baseline")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = device
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    environment["PACT_WORKER"] = f"{spec['name']}:{seed}:gpu{device}"
    handle = (run_dir / "worker.log").open("a", encoding="utf-8")
    process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                               env=environment, cwd=str(BUNDLE_ROOT))
    return {"process": process, "job": job, "device": device, "handle": handle,
            "started": time.monotonic(), "log": run_dir / "worker.log"}


def run_parallel(jobs, devices, config_path, output_dir, stages, arguments, log):
    """Dispatch jobs across GPUs, one job per GPU at a time."""
    queue = list(jobs)
    active, finished, failures = {}, [], []
    baseline_claimed = False
    log(f"[sweep] {len(queue)} job(s) across GPU(s) {', '.join(devices)}")
    while queue or active:
        for device in devices:
            if device in active or not queue:
                continue
            job = queue.pop(0)
            allow_baseline = not baseline_claimed
            baseline_claimed = True
            active[device] = spawn_worker(job, device, config_path, output_dir, stages,
                                          arguments, allow_baseline)
            log(f"[sweep] GPU {device} <- {job[0]['name']} seed {job[1]} "
                f"({len(queue)} queued)")
        time.sleep(3)
        for device in list(active):
            worker = active[device]
            code = worker["process"].poll()
            if code is None:
                continue
            worker["handle"].close()
            spec, seed = worker["job"]
            elapsed = time.monotonic() - worker["started"]
            if code == 0:
                finished.append((spec["name"], seed))
                log(f"[sweep] GPU {device} done: {spec['name']} seed {seed} "
                    f"in {elapsed / 60:.1f} min ({len(finished)}/{len(jobs)})")
            else:
                failures.append({"run": spec["name"], "seed": seed, "exit_code": code,
                                 "log": str(worker["log"])})
                log(f"[sweep] GPU {device} FAILED: {spec['name']} seed {seed} "
                    f"(exit {code}) - see {worker['log']}")
            del active[device]
    if failures:
        write_atomic(Path(output_dir) / "failures.json", failures)
        log(f"[sweep] {len(failures)} job(s) failed; details in failures.json")
    return finished, failures


# ------------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    parser.add_argument("--output-dir", type=Path, help="Override the config's output_dir")
    parser.add_argument("--only-run", nargs="+", help="Run only these run names")
    parser.add_argument("--only-seed", nargs="+", type=int)
    parser.add_argument("--devices", help="Comma-separated GPU ids (default: CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--no-parallel", action="store_true",
                        help="One process for everything, even with several GPUs")
    parser.add_argument("--force", action="store_true", help="Redo runs that already have results")
    parser.add_argument("--skip-existing-train", action="store_true",
                        help="Reuse an adapter already on disk instead of retraining")
    parser.add_argument("--allow-baseline", dest="baseline", action="store_true", default=True,
                        help=argparse.SUPPRESS)
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and stop")
    arguments = parser.parse_args()

    config_path = arguments.config if arguments.config.is_absolute() \
        else (BUNDLE_ROOT / arguments.config if not arguments.config.exists()
              else arguments.config)
    cfg = PactConfig.load(config_path) if config_path.exists() else PactConfig()
    cfg.validate()
    output_dir = Path(arguments.output_dir) if arguments.output_dir \
        else bundle_path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    worker = os.environ.get("PACT_WORKER")
    log = make_logger(output_dir / ("worker.log.jsonl" if worker else "train_all.log"))

    runs = [spec for spec in cfg.runs
            if not arguments.only_run or spec["name"] in arguments.only_run]
    jobs = [(spec, seed) for spec in runs for seed in (spec.get("seeds") or cfg.seeds)
            if not arguments.only_seed or seed in arguments.only_seed]
    devices = visible_devices(arguments.devices)

    if not worker:
        log(f"[start] config={config_path.name} stages={arguments.stages} output={output_dir}")
        log(f"[start] bundle={BUNDLE_ROOT} python={sys.version.split()[0]} "
            f"gpus={devices or 'none'}")
        log("[plan] " + json.dumps([{"run": spec["name"], "seed": seed} for spec, seed in jobs]))
        write_atomic(output_dir / "config.resolved.json", cfg.to_dict())
    if arguments.dry_run:
        return

    if "preflight" in arguments.stages:
        stage_preflight(cfg, output_dir, log)
    if "build" in arguments.stages:
        stage_build(cfg, runs, log)

    stages = [stage for stage in arguments.stages if stage in WORK_STAGES]
    if stages and jobs:
        parallel = len(devices) > 1 and len(jobs) > 1 and not arguments.no_parallel
        if parallel:
            started = time.monotonic()
            run_parallel(jobs, devices, config_path, output_dir, stages, arguments, log)
            log(f"[sweep] all jobs finished in {(time.monotonic() - started) / 3600:.2f} h")
        else:
            if len(devices) > 1:
                log(f"[sweep] single process on {len(devices)} visible GPU(s); "
                    "one job uses one GPU")
            for spec, seed in jobs:
                started = time.monotonic()
                log(f"[run] === {spec['name']} seed {seed} ===")
                try:
                    run_once(cfg, spec, seed, output_dir, arguments, log)
                except Exception as error:  # noqa: BLE001 - one bad job must not sink the sweep
                    log(f"[run] FAILED {spec['name']} seed {seed}: "
                        f"{type(error).__name__}: {error}")
                    write_atomic(output_dir / spec["name"] / f"seed-{seed}" / "failure.json",
                                 {"error": f"{type(error).__name__}: {error}"})
                    if worker:
                        raise
                log(f"[run] {spec['name']} seed {seed} took {time.monotonic() - started:.0f}s")

    if "report" in arguments.stages:
        from pact.report import build

        try:
            build(output_dir, BUNDLE_ROOT / "paper", log=log)
        except Exception as error:  # noqa: BLE001 - a table must not lose a sweep
            log(f"[report] FAILED: {type(error).__name__}: {error}")

    results_dir = None
    if "package" in arguments.stages:
        from pact.release import package

        try:
            summary = package(cfg, output_dir, bundle_path(cfg.release.results_dir), log)
            results_dir = summary["destination"] if summary else None
        except Exception as error:  # noqa: BLE001 - packaging must never lose a sweep
            log(f"[package] FAILED: {type(error).__name__}: {error}")

    if results_dir:
        log(f"[done] everything is in {results_dir} - open SUMMARY.md")
    else:
        log(f"[done] outputs are in {output_dir}")


if __name__ == "__main__":
    main()
