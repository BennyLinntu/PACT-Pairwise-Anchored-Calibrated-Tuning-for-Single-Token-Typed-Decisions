"""The PACT training loop.

A small, explicit loop rather than a subclass of a generic Trainer: the
objective couples several rows of the same micro-batch, the sampler must keep
those rows together, and the optimiser uses two learning rates. All three are
easier to read - and to verify - written out.

The saved adapter keeps Nimble's ``schema_config.json`` contract, so the
released scorers load a PACT adapter without any change.
"""

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch

from nimble.scoring.parallel_schema import SYSTEM_PROMPT

from pact.losses import PactLoss
from pact.model import LogitReader, build_model, candidate_logits, memory_estimate, parameter_groups
from pact.sampler import Collator, GroupSampler


def prompt_code_hash():
    """SHA-256 of the scoring-prompt module that is actually imported.

    Hashing the loaded module - rather than a fixed path - means the contract
    records the code this adapter was really trained against, whether that is
    the bundled copy or a checkout's.
    """
    from nimble.scoring import parallel_schema

    return hashlib.sha256(Path(parallel_schema.__file__).read_bytes()).hexdigest()


def schedule_factor(step, total, warmup, kind):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if kind == "constant" or total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    if kind == "linear":
        return 1.0 - progress
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class PactTrainer:
    def __init__(self, cfg, store, output_dir, log=print, model=None):
        self.cfg = cfg
        # An already-built model can be injected (tests, or a caller that wants
        # to reuse a loaded checkpoint); otherwise prepare() builds one.
        self.model = model
        self.model_info = None
        self.store = store
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.reader = LogitReader()
        self.loss_fn = PactLoss(cfg.loss)
        self.history = []
        self.best = None
        self._plan = None

    # ---------------------------------------------------------------- setup

    def contract(self, extra=None):
        payload = {
            "task": "schema_candidate_classification_v1",
            "model": self.cfg.model.model_id,
            "revision": self.cfg.model.revision,
            "system_prompt": SYSTEM_PROMPT,
            "prompt_code_sha256": prompt_code_hash(),
            "max_length": self.cfg.data.max_length,
            "lora_rank": self.cfg.model.lora_rank,
            "seed": self.cfg.optim.seed,
            "learning_rate": self.cfg.optim.learning_rate,
            "lr_scheduler": self.cfg.optim.scheduler,
            "inference_precision": "bf16_autocast",
            "method": "pact-v1",
            "pact": self.cfg.to_dict(),
        }
        if extra:
            payload.update(extra)
        return payload

    def prepare(self):
        if getattr(self, "_plan", None) is not None:
            return self._plan          # prepare() is idempotent: never load twice
        torch.manual_seed(self.cfg.optim.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.optim.seed)
        if self.model is None:
            self.model, self.model_info = build_model(self.cfg, self.log)
        else:
            from pact.model import resolve_device

            self.model_info = self.model_info or {"device": resolve_device(self.cfg.model.device),
                                                  "loader": "injected"}
        self.device = self.model_info["device"]
        self.autocast_dtype = torch.bfloat16 if self.device == "cuda" else None
        groups = self.store.groups("train")
        if not groups:
            raise ValueError("No training groups in the view cache")
        self.sampler = GroupSampler(groups, self.cfg.optim.groups_per_batch,
                                    self.cfg.optim.length_bucket_groups, self.cfg.optim.seed)
        pad_id = self.store.manifest.get("pad_token_id", 0)
        self.collator = Collator(pad_id, self.cfg.optim.perm_view_prob,
                                 self.cfg.optim.necessity_prob, self.cfg.optim.seed)
        batches = len(self.sampler)
        per_epoch = max(1, math.ceil(batches / self.cfg.optim.grad_accum))
        self.total_steps = self.cfg.optim.max_steps or max(1, int(round(per_epoch * self.cfg.optim.epochs)))
        self.warmup_steps = int(self.total_steps * self.cfg.optim.warmup_frac)
        groups_param = parameter_groups(self.model, self.cfg.optim.learning_rate,
                                        self.cfg.optim.lora_b_multiplier,
                                        self.cfg.optim.weight_decay)
        self.optimizer = torch.optim.AdamW(groups_param, lr=self.cfg.optim.learning_rate,
                                           betas=(0.9, 0.999), eps=1e-8)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        plan = {"train_groups": len(groups), "micro_batches_per_epoch": batches,
                "optimizer_steps_per_epoch": per_epoch, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps,
                "effective_pairs_per_step": self.cfg.optim.groups_per_batch * self.cfg.optim.grad_accum,
                "memory_estimate": memory_estimate(self.cfg)}
        self.log("[train] " + json.dumps(plan))
        (self.output_dir / "train_plan.json").write_text(
            json.dumps({"plan": plan, "model": self.model_info,
                        "views": self.store.summary()}, indent=2) + "\n", encoding="utf-8")
        return plan

    # ---------------------------------------------------------------- steps

    def forward_batch(self, batch, reg_scale):
        if self.autocast_dtype is not None:
            context = torch.autocast(self.device, dtype=self.autocast_dtype)
        else:
            context = torch.enable_grad()
        with context:
            slots = candidate_logits(self.reader, self.model, batch)
        return self.loss_fn(slots, batch, reg_scale)

    def train(self, dev_scorer=None):
        plan = self.prepare()
        step, micro, started = 0, 0, time.monotonic()
        accumulated = []
        self.model.train()
        log_path = self.output_dir / "train_log.jsonl"
        log_file = log_path.open("a", encoding="utf-8")
        eval_every = max(1, int(self.total_steps * self.cfg.optim.eval_every_frac))
        save_every = max(1, int(self.total_steps * self.cfg.optim.save_every_frac))
        stop = False
        for epoch in range(10_000):
            if stop:
                break
            for groups in self.sampler.epoch(epoch):
                batch = self.collator(groups, self.device, training=True)
                if batch is None:
                    continue
                loss, parts, _, _ = self.forward_batch(
                    batch, self._reg_scale(step))
                (loss / self.cfg.optim.grad_accum).backward()
                accumulated.append({key: float(value) for key, value in parts.items()})
                micro += 1
                if micro % self.cfg.optim.grad_accum:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.cfg.optim.max_grad_norm)
                factor = schedule_factor(step, self.total_steps, self.warmup_steps,
                                         self.cfg.optim.scheduler)
                for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                    group["lr"] = base * factor
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % self.cfg.optim.log_every == 0 or step == 1:
                    record = {"step": step, "epoch": round(micro / max(1, len(self.sampler)), 3),
                              "lr": self.base_lrs[0] * factor, "grad_norm": float(norm),
                              "tokens": batch["tokens"], "rows": len(batch["rows"]),
                              "seconds": round(time.monotonic() - started, 1)}
                    for key in accumulated[0]:
                        record[key] = sum(item.get(key, 0.0) for item in accumulated) / len(accumulated)
                    log_file.write(json.dumps(record) + "\n")
                    log_file.flush()
                    self.log("[train] " + json.dumps(record))
                accumulated = []
                if dev_scorer is not None and step % eval_every == 0:
                    self.validate(dev_scorer, step)
                if step % save_every == 0:
                    self.save(self.output_dir / f"checkpoint-{step}")
                if step >= self.total_steps:
                    stop = True
                    break
        if dev_scorer is not None:
            self.validate(dev_scorer, step)
        self.save(self.output_dir / "final")
        log_file.close()
        report = {"plan": plan, "steps": step, "history": self.history, "best": self.best,
                  "wall_seconds": round(time.monotonic() - started, 1),
                  "logit_mode": self.reader.mode}
        if torch.cuda.is_available():
            report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 1024 ** 3
        (self.output_dir / "train_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report

    def _reg_scale(self, step):
        ramp = self.cfg.loss.reg_warmup_frac * self.total_steps
        return 1.0 if ramp <= 0 else min(1.0, (step + 1) / ramp)

    def validate(self, dev_scorer, step):
        was_training = self.model.training
        self.model.eval()
        summary = dev_scorer(self.model, self.reader)
        self.model.train(was_training)
        entry = {"step": step, "summary": summary}
        self.history.append(entry)
        self.log("[dev] " + json.dumps({"step": step, "all": summary.get("all")}))
        score = summary.get("all", {}).get("nll")
        if score is not None and (self.best is None or score < self.best["nll"] - 1e-6):
            self.best = {"step": step, "nll": score,
                         "accuracy": summary.get("all", {}).get("accuracy")}
            self.save(self.output_dir / "best")
        (self.output_dir / "dev_history.json").write_text(
            json.dumps(self.history, indent=2) + "\n", encoding="utf-8")

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory))
        (directory / "schema_config.json").write_text(
            json.dumps(self.contract(), indent=2) + "\n", encoding="utf-8")
        self._prune_checkpoints()
        return directory

    def _prune_checkpoints(self):
        keep = self.cfg.optim.keep_checkpoints
        checkpoints = sorted(self.output_dir.glob("checkpoint-*"),
                             key=lambda path: int(path.name.split("-")[-1]))
        for path in checkpoints[:-keep] if keep else checkpoints:
            shutil.rmtree(path, ignore_errors=True)
