"""Dataset construction for PACT.

The prompt that reaches the model is produced by Nimble's own
``prepare_prompts``, unchanged, so a PACT adapter stays a drop-in replacement
for a Nimble adapter. What changes is *what we encode*:

1. every training example is encoded once in its canonical choice order and
   once or more under a random permutation of the choice codes;
2. every contrastive pair additionally gets one evidence-ablated encoding;
3. training examples are grouped by ``family`` so that both members of a pair
   land in the same micro-batch, which is what makes the paired losses possible.

Nothing here touches ``data/eval.jsonl`` for training or selection: the inner
validation and calibration splits are carved out of the *training* source
families only.
"""

import hashlib
import json
import random
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path

from nimble.evaluation.evaluate_pilot import adapt_input
from nimble.scoring.parallel_schema import choice_key, choices_for, prepare_prompts
from nimble.training.schema_data import as_scoring, validate_separation

from pact import ablation
from pact.canon import canon_index_for, canonical_keys, permutation_plan, presented_choices
from pact.config import dataset_path, resolve

VIEW_FORMAT = "pact-views-v1"
KINDS = ("choice", "noul", "score")


# ---------------------------------------------------------------- loading


def load_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No records in {path}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate record ids in {path}")
    return rows


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_dataset(cfg):
    """Check the frozen files and the train/holdout separation before training."""
    train_path = dataset_path(cfg, "train_file")
    eval_path = dataset_path(cfg, "eval_file")
    manifest_path = dataset_path(cfg, "manifest_file")
    report = {"train_file": str(train_path), "eval_file": str(eval_path),
              "train_sha256": file_digest(train_path), "eval_sha256": file_digest(eval_path)}
    train_rows, eval_rows = load_jsonl(train_path), load_jsonl(eval_path)
    report.update(train_rows=len(train_rows), eval_rows=len(eval_rows))
    if cfg.data.verify_manifest and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = manifest.get("file_sha256", {})
        for key, digest in (("train.jsonl", report["train_sha256"]),
                            ("eval.jsonl", report["eval_sha256"])):
            if key in expected and expected[key] != digest:
                raise ValueError(f"Frozen dataset bytes changed: {key}")
        report["manifest_checked"] = bool(expected)
    validate_separation(train_rows, eval_rows)
    families = defaultdict(list)
    for row in train_rows:
        families[row["family"]].append(row)
    sizes = Counter(len(members) for members in families.values())
    report.update(train_families=len(families), family_sizes=dict(sizes),
                  train_source_families=len({r["source_family"] for r in train_rows}),
                  eval_source_families=len({r.get("source_family", r["family"]) for r in eval_rows}),
                  separation_checked=True)
    return report, train_rows, eval_rows


# ------------------------------------------------------------- examples


def to_example(row, training):
    """One record -> the fields PACT needs, in canonical choice space."""
    scoring = as_scoring(row, training)
    field = scoring["schema"][scoring["field"]]
    canonical = list(choices_for(field))
    keys = canonical_keys(canonical)
    target = choice_key(scoring["target"])
    if target not in keys:
        raise ValueError(f"{row['id']}: reference target is not an allowed choice")
    return {
        "id": row["id"],
        "family": row["family"],
        "source_family": scoring["source_family"],
        "domain": row.get("domain", "unknown"),
        "kind": row["input"]["questions"][scoring["field"]]["type"],
        "variant": row.get("variant", "base"),
        "context": scoring["context"],
        "schema": scoring["schema"],
        "field": scoring["field"],
        "canonical": canonical,
        "gold": keys.index(target),
        "source": "nimble",
        "weight": 1.0,
    }


def build_pairs(rows, training=True, limit_pairs=0):
    """Group training records into contrastive pairs, keeping singletons usable."""
    examples = [to_example(row, training) for row in rows]
    by_id = {row["id"]: row for row in rows}
    families = defaultdict(list)
    for example in examples:
        families[example["family"]].append(example)
    groups, skipped = [], Counter()
    for family in sorted(families):
        members = sorted(families[family], key=lambda e: (e["variant"] != "base", e["id"]))
        if len(members) == 2:
            base, counter = members
            if base["canonical"] != counter["canonical"]:
                skipped["choice_set_mismatch"] += 1
                continue
            if base["gold"] == counter["gold"]:
                skipped["labels_not_flipped"] += 1
                continue
            base["partner_gold"], counter["partner_gold"] = counter["gold"], base["gold"]
            base["raw"], counter["raw"] = by_id[base["id"]], by_id[counter["id"]]
            groups.append({"family": family, "members": members, "paired": True})
        else:
            for member in members:
                member["partner_gold"] = -1
                member["raw"] = by_id[member["id"]]
            skipped["unpaired_family"] += 1
            groups.append({"family": family, "members": members, "paired": False})
    if limit_pairs:
        groups = groups[:limit_pairs]
    return groups, dict(skipped)


def assign_splits(groups, cfg):
    """Hold out whole source families for inner validation and calibration."""
    families = sorted({member["source_family"] for group in groups for member in group["members"]})
    rng = random.Random(cfg.data.split_seed)
    shuffled = families[:]
    rng.shuffle(shuffled)
    held = shuffled[: max(0, min(cfg.data.inner_holdout_families, len(shuffled) - 1))]
    split_of = {}
    for index, family in enumerate(held):
        split_of[family] = "dev_select" if index % 2 == 0 else "dev_calib"
    for family in families:
        split_of.setdefault(family, "train")
    for group in groups:
        splits = {split_of[member["source_family"]] for member in group["members"]}
        if len(splits) != 1:
            raise ValueError(f"Family {group['family']} spans several splits")
        group["split"] = splits.pop()
        for member in group["members"]:
            member["split"] = group["split"]
    counts = Counter(group["split"] for group in groups)
    if counts.get("train", 0) == 0:
        raise ValueError("Split assignment left no training families")
    return {"split_of_source_family": split_of, "groups_per_split": dict(counts),
            "held_out_source_families": held}


# --------------------------------------------------------------- encoding


def encode_view(tokenizer, context, schema, field, canonical, permutation, max_length):
    """Render one prompt with the choice codes in a given permutation."""
    rendered = json.loads(json.dumps(schema))
    presented = presented_choices(canonical, permutation)
    if rendered[field]["type"] == "enum":
        rendered[field]["choices"] = list(presented)
    else:
        rendered[field]["choices"] = list(presented)
    prompt = prepare_prompts(tokenizer, context, rendered, max_length)
    index = prompt.names.index(field)
    shown = prompt.choices[index]
    return {
        "input_ids": prompt.full_ids[index],
        "candidate_ids": prompt.candidate_ids[index],
        "canon_index": canon_index_for(canonical, shown),
    }


def _view_record(example, encoded, view, gold, necessity_pair=None):
    record = {
        "uid": f"{example['id']}#{view}",
        "example_id": example["id"],
        "family": example["family"],
        "source_family": example["source_family"],
        "domain": example["domain"],
        "kind": example["kind"],
        "variant": example["variant"],
        "split": example["split"],
        "view": view,
        "source": example["source"],
        "weight": example["weight"],
        "n_choices": len(example["canonical"]),
        "gold": gold,
        "contrast": example.get("partner_gold", -1),
        "input_ids": encoded["input_ids"],
        "candidate_ids": encoded["candidate_ids"],
        "canon_index": encoded["canon_index"],
    }
    if necessity_pair is not None:
        record["necessity_pair"] = list(necessity_pair)
    return record


def cache_fingerprint(cfg, dataset_report):
    payload = {
        "format": VIEW_FORMAT,
        "model": cfg.model.model_id,
        "revision": cfg.model.revision,
        "max_length": cfg.data.max_length,
        "permutation_views": cfg.data.permutation_views,
        "necessity_views": cfg.data.necessity_views,
        "necessity_drop": cfg.data.necessity_drop,
        "split_seed": cfg.data.split_seed,
        "inner_holdout_families": cfg.data.inner_holdout_families,
        "limit_pairs": cfg.data.limit_pairs,
        "public_mix": cfg.data.public_mix,
        "train_sha256": dataset_report["train_sha256"],
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return digest, payload


def build_views(cfg, tokenizer, log=print):
    """Encode every training view once and cache it as JSONL."""
    report, train_rows, _ = verify_dataset(cfg)
    groups, skipped = build_pairs(train_rows, True, cfg.data.limit_pairs)
    split_info = assign_splits(groups, cfg)
    digest, payload = cache_fingerprint(cfg, report)
    cache_dir = resolve(cfg.data.view_cache) / digest
    views_path, manifest_path = cache_dir / "views.jsonl", cache_dir / "manifest.json"
    if manifest_path.exists() and views_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == payload and manifest.get("complete"):
            log(f"[data] reusing cached views: {views_path}")
            return manifest, views_path
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.data.split_seed + 991)
    counts, too_long, started = Counter(), [], time.monotonic()
    with views_path.open("w", encoding="utf-8") as handle:
        for position, group in enumerate(groups):
            wanted = cfg.data.permutation_views if group["split"] == "train" else 0
            for member in group["members"]:
                plans = permutation_plan(len(member["canonical"]), wanted, rng)
                for order, permutation in enumerate(plans):
                    try:
                        encoded = encode_view(tokenizer, member["context"], member["schema"],
                                              member["field"], member["canonical"], permutation,
                                              cfg.data.max_length)
                    except ValueError as error:
                        too_long.append({"id": member["id"], "view": order, "error": str(error)})
                        continue
                    view = "primary" if order == 0 else f"perm{order}"
                    handle.write(json.dumps(_view_record(member, encoded, view, member["gold"]),
                                            ensure_ascii=False) + "\n")
                    counts[view] += 1
            if (cfg.data.necessity_views and group["paired"] and group["split"] == "train"):
                record = _necessity_record(cfg, tokenizer, group, rng, too_long)
                if record is not None:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts["necessity"] += 1
            if position % 100 == 0:
                log(f"[data] encoded {position}/{len(groups)} groups "
                    f"({time.monotonic() - started:.0f}s)")
        counts.update(_encode_public(cfg, tokenizer, handle, too_long, log))
    manifest = {
        "format": VIEW_FORMAT,
        "fingerprint": payload,
        "pad_token_id": int(tokenizer.pad_token_id or 0),
        "dataset": report,
        "splits": split_info,
        "skipped_families": skipped,
        "views": dict(counts),
        "rejected_prompts": too_long[:50],
        "rejected_count": len(too_long),
        "complete": True,
        "built_seconds": round(time.monotonic() - started, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    log(f"[data] wrote {sum(counts.values())} views to {views_path}")
    return manifest, views_path


def _necessity_record(cfg, tokenizer, group, rng, too_long):
    base = next((m for m in group["members"] if m["variant"] == "base"), group["members"][0])
    counter = next(m for m in group["members"] if m is not base)
    changed = ablation.changed_slot(base["raw"])
    if cfg.data.necessity_drop == "random" or changed is None:
        slot = rng.randrange(2) if changed is None else rng.choice([changed, 1 - changed])
    elif cfg.data.necessity_drop == "changed":
        slot = changed
    else:
        slot = 1 - changed
    reduced = ablation.ablate_input(base["raw"], slot)
    if reduced is None:
        return None
    context, schema = adapt_input(reduced)
    identity = tuple(range(len(base["canonical"])))
    try:
        encoded = encode_view(tokenizer, context, schema, base["field"], base["canonical"],
                              identity, cfg.data.max_length)
    except ValueError as error:
        too_long.append({"id": base["id"], "view": "necessity", "error": str(error)})
        return None
    ablated = dict(base)
    ablated["id"] = base["id"] + "|necessity"
    return _view_record(ablated, encoded, "necessity", -1,
                        necessity_pair=(base["gold"], counter["gold"]))


def _encode_public(cfg, tokenizer, handle, too_long, log):
    """Optional auxiliary public data, encoded as ordinary single examples."""
    counts = Counter()
    for spec in cfg.data.public_mix:
        path = resolve(spec["path"])
        if not Path(path).exists():
            raise FileNotFoundError(f"public_mix file missing: {path} "
                                    "(run download_all.py --public ...)")
        weight = float(spec.get("weight", 0.3))
        limit = int(spec.get("max_rows", 0))
        rows = load_jsonl(path)
        if limit:
            rows = rows[:limit]
        for row in rows:
            canonical = list(choices_for(row["schema"][row["field"]]))
            keys = canonical_keys(canonical)
            target = choice_key(row["target"])
            if target not in keys:
                continue
            example = {"id": row["id"], "family": f"public::{row['id']}",
                       "source_family": f"public::{row.get('source', Path(path).stem)}",
                       "domain": row.get("domain", "public"), "kind": row["kind"],
                       "variant": "public", "split": "train", "canonical": canonical,
                       "gold": keys.index(target), "source": row.get("source", Path(path).stem),
                       "weight": weight, "partner_gold": -1}
            try:
                encoded = encode_view(tokenizer, row["context"], row["schema"], row["field"],
                                      canonical, tuple(range(len(canonical))), cfg.data.max_length)
            except ValueError as error:
                too_long.append({"id": row["id"], "view": "public", "error": str(error)})
                continue
            handle.write(json.dumps(_view_record(example, encoded, "primary", example["gold"]),
                                    ensure_ascii=False) + "\n")
            counts["public"] += 1
        log(f"[data] mixed in {counts['public']} public rows from {path}")
    return counts


# ------------------------------------------------------------- view store


class ViewStore:
    """Cached views, grouped by family, with token ids held as compact arrays."""

    def __init__(self, views_path, manifest):
        self.manifest = manifest
        self.rows = []
        with Path(views_path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                record["input_ids"] = array("i", record["input_ids"])
                record["candidate_ids"] = array("i", record["candidate_ids"])
                record["canon_index"] = array("i", record["canon_index"])
                self.rows.append(record)
        self.by_family = defaultdict(list)
        for record in self.rows:
            self.by_family[(record["split"], record["family"])].append(record)

    def groups(self, split):
        """Training units: all views of one family, longest prompt first."""
        out = []
        for (row_split, family), records in self.by_family.items():
            if row_split != split:
                continue
            out.append({"family": family, "rows": records,
                        "length": max(len(r["input_ids"]) for r in records)})
        out.sort(key=lambda group: group["family"])
        return out

    def split_rows(self, split, view="primary"):
        return [r for r in self.rows if r["split"] == split and r["view"] == view]

    def summary(self):
        counts = Counter((r["split"], r["view"]) for r in self.rows)
        kinds = Counter((r["split"], r["kind"]) for r in self.rows if r["view"] == "primary")
        return {"views": {f"{s}/{v}": n for (s, v), n in sorted(counts.items())},
                "primary_kinds": {f"{s}/{k}": n for (s, k), n in sorted(kinds.items())},
                "max_tokens": max(len(r["input_ids"]) for r in self.rows)}


def load_views(cfg, tokenizer, log=print):
    manifest, views_path = build_views(cfg, tokenizer, log=log)
    return ViewStore(views_path, manifest)


def holdout_examples(cfg):
    """The frozen 324-example evaluation set, as canonical-space examples."""
    _, _, eval_rows = verify_dataset(cfg)
    examples = []
    for row in eval_rows:
        example = to_example(row, False)
        example["split"] = "holdout"
        example["partner_gold"] = -1
        examples.append(example)
    by_family = defaultdict(list)
    for example in examples:
        by_family[example["family"]].append(example)
    for members in by_family.values():
        if len(members) == 2 and members[0]["canonical"] == members[1]["canonical"]:
            members[0]["partner_gold"], members[1]["partner_gold"] = members[1]["gold"], members[0]["gold"]
    return examples


def dev_examples(cfg, store, split):
    """Rebuild canonical examples for an inner split from the cached views."""
    rows = store.split_rows(split, "primary")
    ids = {row["example_id"] for row in rows}
    _, train_rows, _ = verify_dataset(cfg)
    examples = []
    for row in train_rows:
        if row["id"] not in ids:
            continue
        example = to_example(row, True)
        example["split"] = split
        example["partner_gold"] = -1
        examples.append(example)
    by_family = defaultdict(list)
    for example in examples:
        by_family[example["family"]].append(example)
    for members in by_family.values():
        if len(members) == 2 and members[0]["canonical"] == members[1]["canonical"]:
            members[0]["partner_gold"], members[1]["partner_gold"] = members[1]["gold"], members[0]["gold"]
    return examples
