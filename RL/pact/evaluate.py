"""Scoring and evaluation.

Two decoding modes are supported:

* single pass - exactly what Nimble does today: one prompt, one forward, read
  the candidate logits at the final position;
* permutation-ensembled - the same context scored under K relabellings of the
  choice codes, averaged in canonical space. Training with the consistency term
  is what makes K = 1 nearly as good as K = 4; the gap between them is reported
  as the residual position bias of the model.

The frozen holdout is only ever *scored* here. Model selection and the
calibrator use the inner splits carved from training families.
"""

import json
import math
import random
from pathlib import Path

import torch

from pact.canon import permutation_plan, total_variation
from pact.data import encode_view
from pact.losses import masked_log_softmax, scatter_canonical
from pact.metrics import row_metrics, summarize
from pact.model import candidate_logits
from pact.sampler import Collator


def make_view_rows(tokenizer, example, cfg, ensemble, rng):
    """Encode one example under `ensemble` code orderings."""
    plans = permutation_plan(len(example["canonical"]), ensemble - 1, rng)
    rows = []
    for order, permutation in enumerate(plans):
        encoded = encode_view(tokenizer, example["context"], example["schema"],
                              example["field"], example["canonical"], permutation,
                              cfg.data.max_length)
        rows.append({
            "uid": f"{example['id']}#eval{order}",
            "example_id": example["id"],
            "family": example["family"],
            "source_family": example.get("source_family", example["family"]),
            "domain": example.get("domain", "unknown"),
            "kind": example["kind"],
            "variant": example.get("variant", "base"),
            "split": example.get("split", "holdout"),
            "view": "primary" if order == 0 else f"perm{order}",
            "source": example.get("source", "nimble"),
            "weight": 1.0,
            "n_choices": len(example["canonical"]),
            "gold": example["gold"],
            "contrast": example.get("partner_gold", -1),
            **encoded,
        })
    return rows


@torch.no_grad()
def canonical_logits_for(model, reader, rows, collator, device, rows_per_batch,
                         autocast_dtype=torch.bfloat16):
    """Canonical-space logits for a list of encoded views, batched by length."""
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["input_ids"]))
    out = [None] * len(rows)
    for start in range(0, len(order), rows_per_batch):
        chunk = [rows[i] for i in order[start:start + rows_per_batch]]
        batch = collator(chunk, device, training=False)
        if device == "cuda" and autocast_dtype is not None:
            context = torch.autocast("cuda", dtype=autocast_dtype)
        else:
            context = torch.inference_mode()
        with context:
            slots = candidate_logits(reader, model, batch)
        logits, valid = scatter_canonical(slots.float(), batch["canon_index"],
                                          batch["slot_mask"])
        logits, valid = logits.cpu(), valid.cpu()
        for position, index in enumerate(order[start:start + rows_per_batch]):
            count = rows[index]["n_choices"]
            out[index] = logits[position, :count].tolist()
    return out


def _probabilities(logits, temperature=1.0):
    scaled = [value / temperature for value in logits]
    highest = max(scaled)
    weights = [math.exp(value - highest) for value in scaled]
    total = sum(weights)
    return [weight / total for weight in weights]


def _average_log_probs(views):
    """Geometric mean of the per-view distributions, renormalised."""
    length = len(views[0])
    logs = [0.0] * length
    for view in views:
        for index, probability in enumerate(view):
            logs[index] += math.log(max(probability, 1e-15))
    logs = [value / len(views) for value in logs]
    highest = max(logs)
    weights = [math.exp(value - highest) for value in logs]
    total = sum(weights)
    return [weight / total for weight in weights]


def evaluate_examples(model, reader, tokenizer, examples, cfg, device="cuda",
                      ensemble=None, calibrator=None, seed=17, log=print,
                      keep_logits=False):
    """Score a list of canonical examples and summarise them."""
    ensemble = ensemble or cfg.evaluation.permutation_ensemble
    rng = random.Random(seed)
    collator = Collator(tokenizer.pad_token_id or 0)
    results, flat = [], []
    for example in examples:
        rows = make_view_rows(tokenizer, example, cfg, ensemble, rng)
        results.append(rows)
        flat.extend(rows)
    autocast = torch.bfloat16 if device == "cuda" else None
    logits = canonical_logits_for(model, reader, flat, collator, device,
                                  cfg.evaluation.rows_per_batch, autocast)
    for row, values in zip(flat, logits):
        row["logits"] = values
    scored = []
    for example, rows in zip(examples, results):
        views = []
        for row in rows:
            temperature = 1.0
            if calibrator is not None:
                temperature = calibrator.temperature(
                    kind=row["kind"], n_choices=row["n_choices"],
                    prompt_tokens=len(row["input_ids"]))
            views.append(_probabilities(row["logits"], temperature))
        probabilities = views[0] if len(views) == 1 else _average_log_probs(views)
        record = row_metrics(probabilities, example["gold"], example["kind"],
                             example["canonical"])
        record.update(id=example["id"], family=example["family"],
                      source_family=example.get("source_family", example["family"]),
                      domain=example.get("domain", "unknown"), kind=example["kind"],
                      variant=example.get("variant", "base"),
                      gold=example["gold"], n_choices=len(example["canonical"]),
                      probabilities=probabilities)
        if len(views) > 1:
            distances = [total_variation(views[i], views[j])
                         for i in range(len(views)) for j in range(i + 1, len(views))]
            answers = {max(range(len(view)), key=lambda k: view[k]) for view in views}
            record["permutation_tv"] = sum(distances) / len(distances)
            record["permutation_flip"] = int(len(answers) > 1)
        if keep_logits:
            record["view_logits"] = [row["logits"] for row in rows]
            record["prompt_tokens"] = len(rows[0]["input_ids"])
        scored.append(record)
    summary = summarize(scored, cfg.evaluation.ece_bins)
    log("[eval] " + json.dumps({"count": len(scored), "ensemble": ensemble,
                                "accuracy": summary["all"]["accuracy"],
                                "nll": summary["all"]["nll"],
                                "ece": summary["all"]["ece"]}))
    return {"summary": summary, "rows": scored, "ensemble": ensemble,
            "calibrated": calibrator is not None}


def calibration_records(model, reader, tokenizer, examples, cfg, device="cuda", seed=101):
    """Uncalibrated single-pass logits on the inner calibration split."""
    report = evaluate_examples(model, reader, tokenizer, examples, cfg, device,
                               ensemble=1, calibrator=None, seed=seed,
                               log=lambda *_: None, keep_logits=True)
    return [{"logits": row["view_logits"][0], "gold": row["gold"], "kind": row["kind"],
             "n_choices": row["n_choices"], "prompt_tokens": row["prompt_tokens"]}
            for row in report["rows"]], report["summary"]


def cached_dev_scorer(cfg, store, split, device, ece_bins=15, pad_id=0):
    """A closure the trainer calls for inner validation during training."""
    rows = store.split_rows(split, "primary")
    if not rows:
        return None
    collator = Collator(pad_id)

    def score(model, reader):
        autocast = torch.bfloat16 if device == "cuda" else None
        logits = canonical_logits_for(model, reader, rows, collator, device,
                                      cfg.evaluation.rows_per_batch, autocast)
        scored = []
        for row, values in zip(rows, logits):
            probabilities = _probabilities(values)
            choices = [str(index) for index in range(row["n_choices"])] \
                if row["kind"] == "score" else ["false", "true"] \
                if row["kind"] == "noul" else list(range(row["n_choices"]))
            record = row_metrics(probabilities, row["gold"], row["kind"], choices)
            record.update(id=row["example_id"], family=row["family"], kind=row["kind"],
                          domain=row["domain"])
            scored.append(record)
        return summarize(scored, ece_bins)

    return score


def write_report(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")
