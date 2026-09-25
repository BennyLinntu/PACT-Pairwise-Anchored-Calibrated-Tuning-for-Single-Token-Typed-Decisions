"""Evaluate the existing Qwen scorer on saved typed-decision datasets without tuning."""

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from nimble.paths import PROJECT_ROOT

from nimble.scoring.parallel_schema import validate_schema


def adapt_input(input_data):
    """Accept only model inputs, so references/teacher outputs cannot leak in."""
    state = input_data["state"]
    context = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    schema = {}
    for name, question in input_data["questions"].items():
        kind, criteria = question["type"], question["criteria"]
        field = {"description": question["instructions"]}
        if kind == "choice":
            field.update(type="enum", choices=list(criteria), choice_descriptions=criteria)
        elif kind == "noul":
            field.update(type="boolean", choices=[False, True], choice_descriptions=criteria)
        elif kind == "score":
            field.update(type="enum", choices=[str(i) for i in range(len(criteria))],
                         choice_descriptions={str(i): value for i, value in enumerate(criteria)})
        else:
            raise ValueError(f"Unsupported question type: {kind}")
        schema[name] = field
    validate_schema(schema)
    return context, schema


def target_key(target, kind):
    if kind == "noul":
        if type(target) is not bool:
            raise ValueError("Noul reference must be boolean")
        return str(target).lower()
    if kind == "score" and type(target) is not int:
        raise ValueError("Score reference must be an integer level")
    return str(target)


def assess(probabilities, target, kind):
    if (not probabilities or any(not math.isfinite(p) or not 0 <= p <= 1
                                 for p in probabilities.values())
            or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6)):
        raise ValueError("Invalid probability distribution")
    gold = target_key(target, kind)
    if gold not in probabilities:
        raise ValueError("Reference target absent from candidates")
    best = max(probabilities, key=probabilities.get)
    prediction = (best == "true") if kind == "noul" else int(best) if kind == "score" else best
    result = {
        "prediction": prediction, "correct": best == gold,
        "probabilities": probabilities, "reference_probability": probabilities[gold],
        "top_probability": probabilities[best],
        "negative_log_likelihood": -math.log(max(probabilities[gold], 1e-15)),
        "multiclass_brier": sum((p - int(k == gold)) ** 2 for k, p in probabilities.items()),
    }
    if kind == "noul":
        result["binary_brier"] = (probabilities["true"] - int(target)) ** 2
    if kind == "score":
        expected = sum(int(k) * p for k, p in probabilities.items())
        result.update(expected_score=expected, absolute_score_error=abs(expected - target))
    return result


def summarize(rows, model):
    result = {}
    for kind in ("all", "choice", "noul", "score"):
        selected = [r[model] for r in rows if kind == "all" or r["type"] == kind]
        if not selected:
            continue
        group = {"count": len(selected), "correct": sum(r["correct"] for r in selected)}
        group["accuracy"] = group["correct"] / group["count"]
        for metric in ("negative_log_likelihood", "multiclass_brier", "binary_brier", "absolute_score_error"):
            if all(metric in r for r in selected):
                group["mean_" + metric] = statistics.mean(r[metric] for r in selected)
        result[kind] = group
    return result


def markdown_report(result):
    sources = sorted({r["reference"].get("source", "unspecified") for r in result["rows"]})
    reviewed = sum(r["reference"].get("human_reviewed", False) for r in result["rows"])
    lines = [f"# Qwen evaluation: {Path(result['dataset']).parent.name}", "",
             "All dataset rows evaluated together; original split ignored. No training or prompt/temperature tuning.",
             f"Evaluated {len(result['rows'])} examples. Reference sources: {', '.join(sources)}. Human-reviewed: {reviewed}/{len(result['rows'])}. "
             "For unreviewed references, the metrics measure agreement with provisional labels, not established ground-truth accuracy.", "",
             "| Type | Qwen reference matches | Saved teacher reference matches |", "|---|---:|---:|"]
    for kind, group in result["summary"]["qwen"].items():
        teacher = result["summary"]["teacher"][kind]
        lines.append(f"| {kind} | {group['correct']}/{group['count']} | {teacher['correct']}/{teacher['count']} |")
    if "comparison" in result:
        lines += ["", "## Direct comparison", "", "```json", json.dumps(result["comparison"], indent=2), "```", "",
                  "## Domain breakdown", "", "| Domain | Qwen reference matches | Teacher reference matches |", "|---|---:|---:|"]
        for domain, summaries in result["by_domain"].items():
            q, t = summaries["qwen"]["all"], summaries["teacher"]["all"]
            lines.append(f"| {domain} | {q['correct']}/{q['count']} | {t['correct']}/{t['count']} |")
    lines += ["", "## Per-example results", "", "Probabilities below are assigned to the reference label, not TypeSafe's separate confidence statistic.", "",
              "| ID | Task | Reference | Qwen prediction | Qwen P(reference) | Teacher P(reference) |",
              "|---|---|---|---|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['id']} | {row['family']} | {json.dumps(row['reference']['target'])} | "
                     f"{json.dumps(row['qwen']['prediction'])} | {row['qwen']['reference_probability']:.6f} | "
                     f"{row['teacher']['reference_probability']:.6f} |")
    lines += ["", "## Probability metrics", "", "Lower is better. NLL uses natural logarithms and clips zero probabilities to 1e-15. "
              "Multiclass Brier sums squared errors over candidates; binary Brier uses P(true) only. "
              "Score MAE compares the probability-weighted level index with the reference integer.", "",
              "| Metric | Qwen | Saved teacher |", "|---|---:|---:|"]
    for label, kind, metric in [
        ("NLL, all rows", "all", "mean_negative_log_likelihood"),
        ("Multiclass Brier, all rows", "all", "mean_multiclass_brier"),
        ("Binary Brier, Noul rows", "noul", "mean_binary_brier"),
        ("Expected score MAE, Score rows", "score", "mean_absolute_score_error"),
    ]:
        lines.append(f"| {label} | {result['summary']['qwen'][kind][metric]:.6f} | {result['summary']['teacher'][kind][metric]:.6f} |")
    lines += ["", "## Ordered score outputs", "", "| ID | Reference level | Qwen expected score | Teacher expected score |", "|---|---:|---:|---:|"]
    for row in result["rows"]:
        if row["type"] == "score":
            lines.append(f"| {row['id']} | {row['reference']['target']} | {row['qwen']['expected_score']:.6f} | {row['teacher']['expected_score']:.6f} |")
    lines += ["", "## Method", "", "Qwen3.5-4B, existing MLX parallel scorer, BF16 weights and FP32 candidate projection, temperature 1.0, thinking disabled. "
              "Choice order and all instructions/criteria preserved. Structured state serialized as JSON. "
              "Noul mapped to false/true; Score mapped to categorical level indices, then averaged by probability. "
              "Only input.state and input.questions are passed to Qwen. Teacher responses are reused from the dataset; no teacher API calls.", "",
              "Each record has one field, so this run exercises the parallel scorer's single-field path, not a multi-field speedup. "
              "Teacher probability metrics use normalized teacher_targets when present; original rounded API answers are retained. "
              "Choice accuracy uses the saved teacher choice; Noul uses P(true) > 0.5; Score uses the most probable level. "
              "Probability ties use candidate order. Zero teacher probabilities make NLL sensitive to the stated clipping floor.", "",
              f"Reproduce: `.venv-mlx/bin/python -m nimble.evaluation.evaluate_pilot --data {result['dataset']} "
              f"--output-dir evaluations/{Path(result['dataset']).parent.name}`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=(PROJECT_ROOT / "evaluations/typesafe_diverse_300_gpt56"))
    args = parser.parse_args()
    raw = args.data.read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if not records or len({r["id"] for r in records}) != len(records):
        raise ValueError("Dataset must be nonempty with unique IDs")
    prepared = []
    for row in records:
        if list(row["input"]["questions"]) != ["decision"]:
            raise ValueError("Evaluator expects one decision question per record")
        context, schema = adapt_input(row["input"])
        kind = row["input"]["questions"]["decision"]["type"]
        answer = row["teacher"]["answers"]["decision"]
        probs = (row["teacher_targets"]["probabilities"] if "teacher_targets" in row else
                 {"false": 1 - answer["noul"], "true": answer["noul"]}
                 if kind == "noul" else answer["probabilities"])
        teacher = assess(probs, row["reference"]["target"], kind)
        if kind == "choice":
            if answer["choice"] not in probs:
                raise ValueError("Teacher choice absent from candidates")
            teacher["prediction"] = answer["choice"]
            teacher["correct"] = answer["choice"] == row["reference"]["target"]
        if set(probs) != set(str(v).lower() if isinstance(v, bool) else v for v in schema["decision"]["choices"]):
            raise ValueError("Teacher and model candidates differ")
        prepared.append((row, context, schema, kind, teacher))
    from nimble.scoring.parallel_scorer import ParallelScorer
    started = time.perf_counter()
    scorer = ParallelScorer(temperature=1.0)
    load_seconds = time.perf_counter() - started
    results = []
    for row, context, schema, kind, teacher in prepared:
        output = scorer.score(context, schema)
        qwen = assess(output["fields"]["decision"]["scores"], row["reference"]["target"], kind)
        results.append({"id": row["id"], "family": row["family"], "type": kind,
                        "domain": row.get("domain", "unspecified"), "split": row.get("split", "unspecified"),
                        "subtopic": row.get("subtopic"), "diversity": row.get("diversity"),
                        "reference": row["reference"], "qwen": qwen, "teacher": teacher,
                        "raw_teacher": row["teacher"],
                        "adapted_input": {"context": context, "schema": schema}, "raw_qwen": output})
        print(json.dumps({"progress": f"{len(results)}/{len(records)}", "id": row["id"],
                          "qwen": qwen["prediction"], "teacher": teacher["prediction"],
                          "reference": row["reference"]["target"], "reference_probability": qwen["reference_probability"]}), flush=True)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset": str(args.data.resolve()), "dataset_sha256": hashlib.sha256(raw).hexdigest(),
              "split_policy": "all rows; split ignored", "temperature_fitted": False,
              "model": output["model"], "revision": output["revision"],
              "versions": {"python": platform.python_version(), **{k: version(k) for k in ("mlx", "mlx-lm", "transformers")}},
              "load_seconds": load_seconds, "wall_seconds": time.perf_counter() - started,
              "summary": {name: summarize(results, name) for name in ("qwen", "teacher")}, "rows": results}
    for axis in ("domain", "split"):
        result["by_" + axis] = {
            value: {name: summarize([r for r in results if r[axis] == value], name)
                    for name in ("qwen", "teacher")}
            for value in sorted({r[axis] for r in results})}
    agrees = sum(r["qwen"]["prediction"] == r["teacher"]["prediction"] for r in results)
    result["comparison"] = {
        "qwen_teacher_agree": agrees, "count": len(results), "agreement_rate": agrees / len(results),
        "both_match_reference": sum(r["qwen"]["correct"] and r["teacher"]["correct"] for r in results),
        "qwen_only_matches_reference": sum(r["qwen"]["correct"] and not r["teacher"]["correct"] for r in results),
        "teacher_only_matches_reference": sum(not r["qwen"]["correct"] and r["teacher"]["correct"] for r in results),
        "neither_matches_reference": sum(not r["qwen"]["correct"] and not r["teacher"]["correct"] for r in results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (args.output_dir / "report.md").write_text(markdown_report(result))
    review = [r for r in results if not r["qwen"]["correct"] or not r["teacher"]["correct"]]
    (args.output_dir / "disagreements.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in review))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
