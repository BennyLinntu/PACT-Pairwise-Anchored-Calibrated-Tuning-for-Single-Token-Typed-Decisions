"""Post-hoc significance and error analysis on saved holdout predictions.

Everything here reads the per-example rows already written by a finished
sweep (``holdout_single_calibrated.json`` per run/seed, ``baselines.json``
for the untouched base model) and needs no GPU and no retraining. It answers
three questions the headline table cannot: is the accuracy gap over
``nimble_recipe`` bigger than seed noise (McNemar, paired bootstrap), where
do the two models disagree (per-domain, per-kind breakdown), and does the
model that is "confidently wrong" concentrate anywhere in particular.
"""

import json
import math
import random
from pathlib import Path

from scipy import stats


def load_rows(run_dir):
    payload = json.loads((Path(run_dir) / "holdout_single_calibrated.json").read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["rows"]}


def load_baseline_rows(cache_root):
    """Per-example base-model rows, if a local re-eval produced them (baselines.json
    only kept the aggregate summary, not the rows, when it was packaged)."""
    path = Path(cache_root) / "base_model_rows.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: row for row in payload["rows"]}


def mcnemar(rows_a, rows_b, ids):
    """Exact McNemar's test on paired correctness, plus the chi-square approximation."""
    b = c = 0  # b: A right/B wrong, c: A wrong/B right
    for uid in ids:
        ca, cb = rows_a[uid]["correct"], rows_b[uid]["correct"]
        if ca and not cb:
            b += 1
        elif cb and not ca:
            c += 1
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "n_discordant": 0, "exact_p": 1.0, "chi2_p": 1.0}
    exact_p = stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    chi2 = (abs(b - c) - 1) ** 2 / n
    chi2_p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"b": b, "c": c, "n_discordant": n, "exact_p": exact_p, "chi2_p": chi2_p}


def paired_bootstrap(rows_a, rows_b, ids, n_resamples=10000, seed=0):
    """Bootstrap CI for the accuracy difference (A - B) over the shared holdout."""
    rng = random.Random(seed)
    ids = list(ids)
    n = len(ids)
    diffs = [int(rows_a[i]["correct"]) - int(rows_b[i]["correct"]) for i in ids]
    point = sum(diffs) / n
    resampled = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        resampled.append(total / n)
    resampled.sort()
    lo = resampled[int(0.025 * n_resamples)]
    hi = resampled[int(0.975 * n_resamples) - 1]
    return {"point": point, "ci95": [lo, hi],
            "excludes_zero": lo > 0 or hi < 0}


def per_domain(rows_a, rows_b, ids):
    domains = {}
    for uid in ids:
        d = rows_a[uid]["domain"]
        entry = domains.setdefault(d, {"n": 0, "a_correct": 0, "b_correct": 0})
        entry["n"] += 1
        entry["a_correct"] += int(rows_a[uid]["correct"])
        entry["b_correct"] += int(rows_b[uid]["correct"])
    out = {}
    for d, e in sorted(domains.items()):
        out[d] = {"n": e["n"], "a_acc": e["a_correct"] / e["n"], "b_acc": e["b_correct"] / e["n"],
                   "delta": (e["a_correct"] - e["b_correct"]) / e["n"]}
    return out


def per_kind(rows_a, rows_b, ids):
    kinds = {}
    for uid in ids:
        k = rows_a[uid]["kind"]
        entry = kinds.setdefault(k, {"n": 0, "a_correct": 0, "b_correct": 0})
        entry["n"] += 1
        entry["a_correct"] += int(rows_a[uid]["correct"])
        entry["b_correct"] += int(rows_b[uid]["correct"])
    out = {}
    for k, e in sorted(kinds.items()):
        out[k] = {"n": e["n"], "a_acc": e["a_correct"] / e["n"], "b_acc": e["b_correct"] / e["n"],
                   "delta": (e["a_correct"] - e["b_correct"]) / e["n"]}
    return out


def confidently_wrong(rows, threshold=0.9):
    """Examples answered with stated confidence above threshold but wrong."""
    out = [row for row in rows.values() if not row["correct"] and row["confidence"] >= threshold]
    return sorted(out, key=lambda r: -r["confidence"])


def compare(cache_root, run_a, run_b, seed):
    root = Path(cache_root)
    if run_a == "base_model":
        rows_a = load_baseline_rows(root)
    else:
        rows_a = load_rows(root / run_a / f"seed-{seed}")
    if run_b == "base_model":
        rows_b = load_baseline_rows(root)
    else:
        rows_b = load_rows(root / run_b / f"seed-{seed}")
    ids = sorted(set(rows_a) & set(rows_b))
    assert len(ids) == 324, f"expected 324 shared holdout ids, got {len(ids)}"
    return {
        "run_a": run_a, "run_b": run_b, "seed": seed, "n": len(ids),
        "accuracy_a": sum(rows_a[i]["correct"] for i in ids) / len(ids),
        "accuracy_b": sum(rows_b[i]["correct"] for i in ids) / len(ids),
        "mcnemar": mcnemar(rows_a, rows_b, ids),
        "bootstrap": paired_bootstrap(rows_a, rows_b, ids),
        "by_domain": per_domain(rows_a, rows_b, ids),
        "by_kind": per_kind(rows_a, rows_b, ids),
    }, rows_a, rows_b


def main():
    cache_root = Path(r"C:\Users\Administrator\Desktop\PACT\.cache\runs\pact-9b")
    out = {"comparisons": [], "confidently_wrong": {}}
    pairs = [("pact_full", "nimble_recipe"), ("pact_full", "ce_only")]
    if (cache_root / "base_model_rows.json").exists():
        pairs.append(("pact_full", "base_model"))
    for run_a, run_b in pairs:
        for seed in (17, 18, 19):
            record, rows_a, rows_b = compare(cache_root, run_a, run_b, seed)
            out["comparisons"].append(record)
            print(f"[{run_a} vs {run_b} | seed {seed}] "
                  f"acc {record['accuracy_a']:.4f} vs {record['accuracy_b']:.4f}  "
                  f"McNemar b={record['mcnemar']['b']} c={record['mcnemar']['c']} "
                  f"p={record['mcnemar']['exact_p']:.4g}  "
                  f"bootstrap 95% CI={record['bootstrap']['ci95']}")

    # Confidently-wrong audit for the shipped model (pact_full, seed 18) vs base model.
    shipped = load_rows(cache_root / "pact_full" / "seed-18")
    out["confidently_wrong"]["pact_full_seed18"] = [
        {"id": r["id"], "domain": r["domain"], "kind": r["kind"],
         "confidence": r["confidence"], "gold": r["gold"], "prediction": r["prediction"]}
        for r in confidently_wrong(shipped)]
    if (cache_root / "base_model_rows.json").exists():
        base = load_baseline_rows(cache_root)
        out["confidently_wrong"]["base_model"] = [
            {"id": r["id"], "domain": r["domain"], "kind": r["kind"],
             "confidence": r["confidence"], "gold": r["gold"], "prediction": r.get("prediction")}
            for r in confidently_wrong(base)]

    out_path = Path(r"C:\Users\Administrator\Desktop\PACT\results\significance.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"pact_full seed18 confidently-wrong (conf>=0.9, still incorrect): "
          f"{len(out['confidently_wrong']['pact_full_seed18'])} / 324")
    if "base_model" in out["confidently_wrong"]:
        print(f"base_model confidently-wrong (conf>=0.9, still incorrect): "
              f"{len(out['confidently_wrong']['base_model'])} / 324")


if __name__ == "__main__":
    main()
