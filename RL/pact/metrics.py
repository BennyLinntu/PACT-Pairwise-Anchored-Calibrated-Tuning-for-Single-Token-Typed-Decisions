"""Metrics for typed decisions.

Beyond accuracy this reports what the Nimble README flags as untested:
calibration (ECE, Brier, NLL), the cost of ordinal mistakes, the stability of
an answer under relabelled choice codes, and whether both members of a
contrastive pair are handled correctly rather than one of them by luck.

Pure Python and numpy-free, so it runs anywhere, including the preflight pass
on a laptop.
"""

import math
from collections import defaultdict

KINDS = ("choice", "noul", "score")


def _safe_log(value):
    return math.log(max(value, 1e-15))


def row_metrics(probabilities, gold, kind, choices):
    """Per-example metrics from a canonical probability vector."""
    if not probabilities:
        raise ValueError("Empty probability vector")
    total = sum(probabilities)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-5:
        raise ValueError(f"Probabilities must sum to 1, got {total}")
    predicted = max(range(len(probabilities)), key=lambda i: probabilities[i])
    result = {
        "prediction": predicted,
        "correct": int(predicted == gold),
        "confidence": probabilities[predicted],
        "reference_probability": probabilities[gold],
        "nll": -_safe_log(probabilities[gold]),
        "brier": sum((p - (1.0 if i == gold else 0.0)) ** 2
                     for i, p in enumerate(probabilities)),
    }
    if kind == "noul":
        keys = [str(value).lower() for value in choices]
        if "true" in keys:
            result["probability_true"] = probabilities[keys.index("true")]
    if kind == "score":
        levels = [float(value) for value in choices]
        expected = sum(level * p for level, p in zip(levels, probabilities))
        result["expected_score"] = expected
        result["absolute_error"] = abs(expected - levels[gold])
        result["level_error"] = abs(levels[predicted] - levels[gold])
    return result


def expected_calibration_error(rows, bins=15):
    """Equal-width ECE over top-choice confidence."""
    if not rows:
        return None
    buckets = [[] for _ in range(bins)]
    for row in rows:
        index = min(int(row["confidence"] * bins), bins - 1)
        buckets[index].append(row)
    error, total = 0.0, len(rows)
    for bucket in buckets:
        if not bucket:
            continue
        confidence = sum(r["confidence"] for r in bucket) / len(bucket)
        accuracy = sum(r["correct"] for r in bucket) / len(bucket)
        error += len(bucket) / total * abs(confidence - accuracy)
    return error


def reliability_bins(rows, bins=15):
    """Bin contents for a reliability diagram."""
    out = []
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [r for r in rows
                  if (low <= r["confidence"] < high) or (index == bins - 1 and r["confidence"] == 1.0)]
        if not bucket:
            out.append({"bin": index, "low": low, "high": high, "count": 0})
            continue
        out.append({"bin": index, "low": low, "high": high, "count": len(bucket),
                    "confidence": sum(r["confidence"] for r in bucket) / len(bucket),
                    "accuracy": sum(r["correct"] for r in bucket) / len(bucket)})
    return out


def selective_risk(rows):
    """Risk-coverage curve and its area, ranking by confidence."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda r: -r["confidence"])
    errors, curve = 0, []
    for index, row in enumerate(ordered, start=1):
        errors += 1 - row["correct"]
        curve.append({"coverage": index / len(ordered), "risk": errors / index})
    area = sum(point["risk"] for point in curve) / len(curve)
    return {"aurc": area, "curve": curve[:: max(1, len(curve) // 50)]}


def pair_consistency(rows):
    """Share of contrastive pairs where both members are answered correctly.

    A model that always answers with the family's majority label scores well on
    accuracy and badly here, which is exactly the shortcut contrastive data is
    meant to expose.
    """
    families = defaultdict(list)
    for row in rows:
        families[row["family"]].append(row)
    pairs = [members for members in families.values() if len(members) == 2]
    if not pairs:
        return None
    both = sum(1 for members in pairs if all(m["correct"] for m in members))
    either = sum(1 for members in pairs if any(m["correct"] for m in members))
    return {"pairs": len(pairs), "both_correct": both, "pair_accuracy": both / len(pairs),
            "at_least_one": either / len(pairs)}


def permutation_sensitivity(rows):
    """Mean total-variation distance between code orderings of one example."""
    values = [row["permutation_tv"] for row in rows if row.get("permutation_tv") is not None]
    flips = [row["permutation_flip"] for row in rows if row.get("permutation_flip") is not None]
    if not values:
        return None
    return {"mean_total_variation": sum(values) / len(values),
            "answer_flip_rate": (sum(flips) / len(flips)) if flips else None}


def summarize(rows, bins=15):
    """Aggregate per-example rows into the table the paper reports."""
    summary = {}
    for kind in ("all",) + KINDS:
        selected = [r for r in rows if kind == "all" or r["kind"] == kind]
        if not selected:
            continue
        group = {
            "count": len(selected),
            "correct": sum(r["correct"] for r in selected),
            "accuracy": sum(r["correct"] for r in selected) / len(selected),
            "nll": sum(r["nll"] for r in selected) / len(selected),
            "brier": sum(r["brier"] for r in selected) / len(selected),
            "ece": expected_calibration_error(selected, bins),
            "mean_confidence": sum(r["confidence"] for r in selected) / len(selected),
        }
        errors = [r["absolute_error"] for r in selected if "absolute_error" in r]
        if errors:
            group["expected_score_mae"] = sum(errors) / len(errors)
            group["level_mae"] = sum(r["level_error"] for r in selected
                                     if "level_error" in r) / len(errors)
        consistency = pair_consistency(selected)
        if consistency:
            group["pair_accuracy"] = consistency["pair_accuracy"]
            group["pairs"] = consistency["pairs"]
        sensitivity = permutation_sensitivity(selected)
        if sensitivity:
            group["permutation_tv"] = sensitivity["mean_total_variation"]
            group["permutation_flip_rate"] = sensitivity["answer_flip_rate"]
        risk = selective_risk(selected)
        if risk:
            group["aurc"] = risk["aurc"]
        summary[kind] = group
    by_domain = defaultdict(list)
    for row in rows:
        by_domain[row.get("domain", "unknown")].append(row)
    summary["by_domain"] = {
        domain: {"count": len(items),
                 "accuracy": sum(r["correct"] for r in items) / len(items)}
        for domain, items in sorted(by_domain.items())}
    return summary


def compare(before, after):
    """Point differences between two summaries, for the ablation table."""
    out = {}
    for kind, group in after.items():
        if kind == "by_domain" or kind not in before:
            continue
        out[kind] = {}
        for key, value in group.items():
            other = before[kind].get(key)
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                out[kind][key] = value - other
    return out
