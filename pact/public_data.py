"""Optional auxiliary training data from public, human-labelled corpora.

Nimble's own README notes that the model is trained on ten synthetic domains
and should not be expected to generalise far beyond them. The cheapest way to
widen it without weakening the contrastive signal is to mix in a small,
down-weighted share of human-labelled decisions in exactly the same typed
format, so they share the prompt, the one-token codes and the loss.

These rows carry no counterfactual partner, so they contribute to the
cross-entropy term only: the paired and necessity terms skip them. Keep the
weight low (0.2-0.4); the point is regularisation, not a new training set.

Converted files land in ``.cache/public/<name>.jsonl`` inside this bundle and are
referenced from a config as::

    "public_mix": [{"path": ".cache/public/boolq.jsonl",
                    "weight": 0.3, "max_rows": 3000}]
"""

import argparse
import json
from pathlib import Path

from pact import bundle_path

MAX_CONTEXT_CHARS = 6000


def _boolean_schema(question, yes, no):
    return {"decision": {"type": "boolean", "description": question,
                         "choice_descriptions": {"true": yes, "false": no}}}


def _enum_schema(question, criteria):
    return {"decision": {"type": "enum", "description": question,
                         "choices": list(criteria), "choice_descriptions": dict(criteria)}}


def _row(identifier, source, kind, context, schema, target):
    return {"id": f"{source}-{identifier}", "source": source, "domain": f"public/{source}",
            "kind": kind, "field": "decision", "context": context[:MAX_CONTEXT_CHARS],
            "schema": schema, "target": target}


def convert_boolq(split="train", limit=3000):
    from datasets import load_dataset

    data = load_dataset("google/boolq", split=split)
    schema_question = ("Decide whether the passage supports a yes answer to the question. "
                       "Use only the passage.")
    rows = []
    for index, item in enumerate(data):
        if len(rows) >= limit:
            break
        context = f"Passage: {item['passage']}\n\nQuestion: {item['question']}"
        schema = _boolean_schema(schema_question,
                                 "The passage supports a yes answer.",
                                 "The passage does not support a yes answer.")
        rows.append(_row(index, "boolq", "noul", context, schema, bool(item["answer"])))
    return rows


def convert_paws(split="train", limit=3000, config="labeled_final"):
    from datasets import load_dataset

    data = load_dataset("google-research-datasets/paws", config, split=split)
    rows = []
    for index, item in enumerate(data):
        if len(rows) >= limit:
            break
        context = f"Sentence A: {item['sentence1']}\n\nSentence B: {item['sentence2']}"
        schema = _boolean_schema(
            "Decide whether the two sentences state the same thing.",
            "The two sentences state the same thing.",
            "The two sentences differ in what they state.")
        rows.append(_row(index, "paws", "noul", context, schema, bool(item["label"])))
    return rows


def convert_vitaminc(split="train", limit=4000):
    from datasets import load_dataset

    data = load_dataset("tals/vitaminc", split=split)
    criteria = {
        "SUPPORTS": "The evidence supports the claim.",
        "REFUTES": "The evidence contradicts the claim.",
        "NOT_ENOUGH_INFO": "The evidence neither supports nor contradicts the claim.",
    }
    mapping = {"SUPPORTS": "SUPPORTS", "REFUTES": "REFUTES",
               "NOT ENOUGH INFO": "NOT_ENOUGH_INFO"}
    rows = []
    for index, item in enumerate(data):
        if len(rows) >= limit:
            break
        label = mapping.get(str(item["label"]).upper())
        if label is None:
            continue
        context = f"Evidence: {item['evidence']}\n\nClaim: {item['claim']}"
        schema = _enum_schema("Judge the claim against the evidence only.", criteria)
        rows.append(_row(index, "vitaminc", "choice", context, schema, label))
    return rows


def convert_multinli(split="train", limit=4000):
    from datasets import load_dataset

    data = load_dataset("nyu-mll/multi_nli", split=split)
    criteria = {
        "entailment": "The hypothesis must be true given the premise.",
        "neutral": "The hypothesis may or may not be true given the premise.",
        "contradiction": "The hypothesis cannot be true given the premise.",
    }
    names = ["entailment", "neutral", "contradiction"]
    rows = []
    for index, item in enumerate(data):
        if len(rows) >= limit:
            break
        if not 0 <= int(item["label"]) < 3:
            continue
        context = f"Premise: {item['premise']}\n\nHypothesis: {item['hypothesis']}"
        schema = _enum_schema("Decide how the hypothesis relates to the premise.", criteria)
        rows.append(_row(index, "multinli", "choice", context, schema, names[int(item["label"])]))
    return rows


CONVERTERS = {"boolq": convert_boolq, "paws": convert_paws,
              "vitaminc": convert_vitaminc, "multinli": convert_multinli}


def write_rows(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["boolq"], choices=sorted(CONVERTERS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--out-dir", default=str(bundle_path(".cache/public")))
    arguments = parser.parse_args()
    for name in arguments.datasets:
        rows = CONVERTERS[name](split=arguments.split, limit=arguments.limit)
        path = write_rows(rows, Path(arguments.out_dir) / f"{name}.jsonl")
        print(json.dumps({"dataset": name, "rows": len(rows), "path": str(path)}))


if __name__ == "__main__":
    main()
