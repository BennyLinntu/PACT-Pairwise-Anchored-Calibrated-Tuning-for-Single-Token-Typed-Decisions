"""Group-aware batching.

Two constraints shape a PACT micro-batch:

* both members of a contrastive pair, and every view of an example, must be in
  the same micro-batch, otherwise the paired losses have nothing to pair;
* prompts here run to 2,048 tokens and padding is pure waste, so batches are
  built from length-sorted buckets.

The unit of batching is therefore a *group* (one family: its two members and
all of their views), and groups are shuffled, bucketed by length, and then cut
into micro-batches.
"""

import random

import torch

from pact.losses import MAX_CHOICES


class GroupSampler:
    """Deterministic, length-bucketed, pair-preserving group batches."""

    def __init__(self, groups, groups_per_batch, bucket_groups, seed):
        self.groups = list(groups)
        self.groups_per_batch = groups_per_batch
        self.bucket_groups = max(bucket_groups, groups_per_batch)
        self.seed = seed

    def __len__(self):
        return (len(self.groups) + self.groups_per_batch - 1) // self.groups_per_batch

    def epoch(self, index):
        rng = random.Random(self.seed * 1000 + index)
        order = self.groups[:]
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.bucket_groups):
            bucket = sorted(order[start:start + self.bucket_groups],
                            key=lambda group: group["length"], reverse=True)
            for cut in range(0, len(bucket), self.groups_per_batch):
                batches.append(bucket[cut:cut + self.groups_per_batch])
        rng.shuffle(batches)
        return batches


class Collator:
    """Turn a list of groups into padded tensors plus the loss index tables."""

    def __init__(self, pad_id, perm_view_prob=1.0, necessity_prob=1.0, seed=17):
        self.pad_id = pad_id
        self.perm_view_prob = perm_view_prob
        self.necessity_prob = necessity_prob
        self.rng = random.Random(seed)

    def select_rows(self, groups, training=True):
        rows = []
        for group in groups:
            for row in group["rows"]:
                if row["view"] == "necessity":
                    if not training or self.rng.random() > self.necessity_prob:
                        continue
                elif row["view"] != "primary":
                    if not training or self.rng.random() > self.perm_view_prob:
                        continue
                rows.append(row)
        return rows

    def __call__(self, groups, device="cpu", training=True):
        rows = self.select_rows(groups, training) if isinstance(groups[0], dict) \
            and "rows" in groups[0] else list(groups)
        if not rows:
            return None
        length = max(len(row["input_ids"]) for row in rows)
        input_ids, attention, candidates, slot_mask, canon_index = [], [], [], [], []
        gold, weight, is_score = [], [], []
        for row in rows:
            ids = list(row["input_ids"])
            pad = length - len(ids)
            input_ids.append([self.pad_id] * pad + ids)
            attention.append([0] * pad + [1] * len(ids))
            codes = list(row["candidate_ids"])
            canon = list(row["canon_index"])
            fill = MAX_CHOICES - len(codes)
            if fill < 0:
                raise ValueError("More candidates than the 26 allowed codes")
            candidates.append(codes + [0] * fill)
            canon_index.append(canon + [0] * fill)
            slot_mask.append([True] * len(codes) + [False] * fill)
            gold.append(row["gold"])
            weight.append(float(row.get("weight", 1.0)))
            is_score.append(row["kind"] == "score")

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        batch = {
            "input_ids": tensor(input_ids, torch.long),
            "attention_mask": tensor(attention, torch.long),
            "candidate_ids": tensor(candidates, torch.long),
            "canon_index": tensor(canon_index, torch.long),
            "slot_mask": tensor(slot_mask, torch.bool),
            "gold": tensor(gold, torch.long),
            "weight": tensor(weight, torch.float32),
            "is_score": tensor(is_score, torch.bool),
            "rows": rows,
            "tokens": length * len(rows),
        }
        batch["labelled_rows"] = tensor([i for i, value in enumerate(gold) if value >= 0],
                                        torch.long)
        pair_a, pair_b, pair_gold_a, pair_gold_b = _pair_tables(rows)
        view_a, view_b = _view_tables(rows)
        nec_rows, nec_a, nec_b, nec_ref = _necessity_tables(rows)
        for name, values in (("pair_a", pair_a), ("pair_b", pair_b),
                             ("pair_gold_a", pair_gold_a), ("pair_gold_b", pair_gold_b),
                             ("view_a", view_a), ("view_b", view_b),
                             ("nec_rows", nec_rows), ("nec_choice_a", nec_a),
                             ("nec_choice_b", nec_b), ("nec_reference", nec_ref)):
            batch[name] = tensor(values, torch.long)
        return batch


def _pair_tables(rows):
    """Contrastive pairs: same family, same view, flipped reference labels."""
    buckets = {}
    for index, row in enumerate(rows):
        if row["gold"] < 0 or row["view"] == "necessity":
            continue
        buckets.setdefault((row["family"], row["view"]), []).append(index)
    first, second, gold_first, gold_second = [], [], [], []
    for members in buckets.values():
        if len(members) != 2:
            continue
        left, right = members
        if rows[left]["gold"] == rows[right]["gold"]:
            continue
        first.append(left)
        second.append(right)
        gold_first.append(rows[left]["gold"])
        gold_second.append(rows[right]["gold"])
    return first, second, gold_first, gold_second


def _view_tables(rows):
    """Every unordered pair of distinct views of the same example."""
    buckets = {}
    for index, row in enumerate(rows):
        if row["view"] == "necessity":
            continue
        buckets.setdefault(row["example_id"], []).append(index)
    first, second = [], []
    for members in buckets.values():
        for position, left in enumerate(members):
            for right in members[position + 1:]:
                first.append(left)
                second.append(right)
    return first, second


def _necessity_tables(rows):
    """Ablated rows with the two choices the removed evidence decided."""
    reference = {}
    for index, row in enumerate(rows):
        if row["view"] == "primary" and row["variant"] == "base":
            reference[row["family"]] = index
    indices, first, second, anchors = [], [], [], []
    for index, row in enumerate(rows):
        if row["view"] != "necessity":
            continue
        pair = row.get("necessity_pair")
        if not pair or len(pair) != 2 or pair[0] == pair[1]:
            continue
        indices.append(index)
        first.append(pair[0])
        second.append(pair[1])
        anchors.append(reference.get(row["family"], index))
    return indices, first, second, anchors
