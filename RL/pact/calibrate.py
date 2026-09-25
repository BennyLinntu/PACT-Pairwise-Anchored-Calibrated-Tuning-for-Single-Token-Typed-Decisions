"""Post-hoc calibration of the candidate distribution.

Nimble's README is explicit that its probabilities are not calibrated and that
no temperature was ever fitted. Fitting one is cheap, but a single global
temperature is the wrong shape for this task: a field with two choices and a
field with twelve are not miscalibrated by the same factor, and neither are a
300-token and a 1,900-token prompt.

PACT fits a *contextual* temperature

    T(x) = softplus(a + b * log C + c * log(L / 1000))

with C the number of allowed answers and L the prompt length in tokens, by
minimising negative log-likelihood on the inner calibration split. It costs
three parameters, no extra forward passes at serving time, and it cannot
change any argmax when b = c = 0, so it never trades accuracy for calibration
unless the data asks for it.
"""

import json
import math
from pathlib import Path

import torch

from pact.metrics import expected_calibration_error, row_metrics, summarize

MODES = ("scalar", "per_kind", "contextual")
KIND_INDEX = {"choice": 0, "noul": 1, "score": 2}


def _softplus(value):
    return math.log1p(math.exp(-abs(value))) + max(value, 0.0)


class Calibrator:
    """Maps prompt statistics to a temperature."""

    def __init__(self, mode="scalar", weights=None):
        if mode not in MODES:
            raise ValueError(f"Unknown calibration mode: {mode}")
        self.mode = mode
        self.weights = weights or {"bias": [0.541324854612918] * 3, "log_choices": 0.0,
                                   "log_tokens": 0.0}

    def temperature(self, kind="choice", n_choices=2, prompt_tokens=1000):
        index = KIND_INDEX.get(kind, 0) if self.mode == "per_kind" else 0
        value = self.weights["bias"][index if self.mode == "per_kind" else 0]
        if self.mode == "contextual":
            value = value + self.weights["log_choices"] * math.log(max(n_choices, 1))
            value = value + self.weights["log_tokens"] * math.log(max(prompt_tokens, 1) / 1000)
        return max(_softplus(value), 1e-3)

    def to_dict(self):
        return {"mode": self.mode, "weights": self.weights}

    @staticmethod
    def from_dict(payload):
        return Calibrator(payload["mode"], payload["weights"])

    @staticmethod
    def load(path):
        return Calibrator.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _tensors(records, device="cpu"):
    width = max(len(record["logits"]) for record in records)
    logits = torch.full((len(records), width), -1e30, device=device)
    for index, record in enumerate(records):
        logits[index, :len(record["logits"])] = torch.tensor(record["logits"], device=device)
    gold = torch.tensor([record["gold"] for record in records], device=device)
    kinds = torch.tensor([KIND_INDEX.get(record["kind"], 0) for record in records], device=device)
    choices = torch.tensor([float(record["n_choices"]) for record in records], device=device)
    tokens = torch.tensor([float(record["prompt_tokens"]) for record in records], device=device)
    return logits, gold, kinds, choices, tokens


def fit_calibrator(records, mode="contextual", log=print, max_iterations=120):
    """Fit the temperature model by NLL on held-out inner-split logits."""
    if not records:
        raise ValueError("No calibration records")
    if mode != "scalar" and len(records) < 50:
        log(f"[calibrate] only {len(records)} records; falling back to a scalar temperature")
        mode = "scalar"
    logits, gold, kinds, choices, tokens = _tensors(records)
    bias = torch.full((3,), 0.541324854612918, requires_grad=True)
    slope_choices = torch.zeros((), requires_grad=True)
    slope_tokens = torch.zeros((), requires_grad=True)
    parameters = [bias] + ([slope_choices, slope_tokens] if mode == "contextual" else [])
    optimizer = torch.optim.LBFGS(parameters, lr=0.2, max_iter=max_iterations,
                                  tolerance_grad=1e-9, tolerance_change=1e-11,
                                  line_search_fn="strong_wolfe")

    def temperatures():
        value = bias[kinds] if mode == "per_kind" else bias[0].expand(len(records))
        if mode == "contextual":
            value = value + slope_choices * choices.clamp(min=1).log() \
                + slope_tokens * (tokens.clamp(min=1) / 1000).log()
        return torch.nn.functional.softplus(value).clamp(min=1e-3)

    def closure():
        optimizer.zero_grad()
        scaled = logits / temperatures().unsqueeze(1)
        loss = torch.nn.functional.cross_entropy(scaled, gold)
        loss.backward()
        return loss

    before = float(torch.nn.functional.cross_entropy(logits, gold))
    optimizer.step(closure)
    with torch.no_grad():
        after = float(torch.nn.functional.cross_entropy(
            logits / temperatures().unsqueeze(1), gold))
        fitted = temperatures()
        stats = {"mode": mode, "records": len(records), "nll_before": before,
                 "nll_after": after,
                 "temperature_mean": float(fitted.mean()),
                 "temperature_min": float(fitted.min()),
                 "temperature_max": float(fitted.max())}
    if not math.isfinite(after) or after > before:
        log("[calibrate] fit did not improve NLL; keeping temperature 1.0")
        calibrator = Calibrator("scalar")
        stats["applied"] = False
    else:
        calibrator = Calibrator(mode, {"bias": [float(v) for v in bias.detach()],
                                       "log_choices": float(slope_choices.detach()),
                                       "log_tokens": float(slope_tokens.detach())})
        stats["applied"] = True
    stats.update(_calibration_quality(records, calibrator))
    if (stats["applied"] and stats["ece_after"] is not None
            and stats["ece_before"] is not None
            and stats["ece_after"] > stats["ece_before"] + 1e-6):
        log("[calibrate] note: the NLL-optimal temperature raises ECE on this split "
            f"({stats['ece_before']:.4f} -> {stats['ece_after']:.4f}); both are reported")
    log("[calibrate] " + json.dumps(stats))
    return calibrator, stats


def _calibration_quality(records, calibrator):
    """ECE and accuracy on the calibration split, before and after scaling."""
    def score(temperature_of):
        rows = []
        for record in records:
            temperature = temperature_of(record)
            scaled = [value / temperature for value in record["logits"]]
            highest = max(scaled)
            weights = [math.exp(value - highest) for value in scaled]
            total = sum(weights)
            probabilities = [weight / total for weight in weights]
            choices = [str(i) for i in range(record["n_choices"])] if record["kind"] == "score" \
                else ["false", "true"] if record["kind"] == "noul" \
                else list(range(record["n_choices"]))
            row = row_metrics(probabilities, record["gold"], record["kind"], choices)
            row.update(kind=record["kind"], family=record.get("family", record["kind"]),
                       domain=record.get("domain", "unknown"))
            rows.append(row)
        return rows

    raw = score(lambda record: 1.0)
    tuned = score(lambda record: calibrator.temperature(
        kind=record["kind"], n_choices=record["n_choices"],
        prompt_tokens=record["prompt_tokens"]))
    return {"ece_before": expected_calibration_error(raw),
            "ece_after": expected_calibration_error(tuned),
            "accuracy_before": summarize(raw)["all"]["accuracy"],
            "accuracy_after": summarize(tuned)["all"]["accuracy"]}
