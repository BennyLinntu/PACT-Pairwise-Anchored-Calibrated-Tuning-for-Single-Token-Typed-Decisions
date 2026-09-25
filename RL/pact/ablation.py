"""Reconstruct the evidence-ablated ("necessity") view of a contrastive pair.

Nimble's curation pipeline verified, for every retained pair, that deleting
either of the two focus-evidence sentences makes the focus fact *unknown* even
with all the remaining text present. The released recipe throws those ablated
contexts away, because "missing evidence" has no label.

PACT keeps them, unlabelled. Section 3.4 of the paper uses them as a constraint
instead of a target: with the focus evidence gone, the two outcomes that the
focus fact discriminates must become indistinguishable to the model.

Everything needed to rebuild the ablated context is already inside each record:

* ``evidence_certificate.spec.focus_evidence`` - the two sentences and a path
  into ``input.state``;
* ``evidence_certificate.verified_pair`` - the base and counterfactual wording
  of each sentence, which identifies the one the counterfactual edits.
"""

import copy
import json
import re


def _string_leaves(node, prefix=()):
    """Yield ``(path, text)`` for every string leaf in a nested JSON value."""
    if isinstance(node, str):
        yield prefix, node
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _string_leaves(value, prefix + (index,))
    elif isinstance(node, dict):
        for key, value in node.items():
            yield from _string_leaves(value, prefix + (key,))


def _resolve(node, path):
    for step in path:
        if isinstance(node, list):
            index = int(step)
            if not 0 <= index < len(node):
                return None
            node = node[index]
        elif isinstance(node, dict):
            if step not in node:
                return None
            node = node[step]
        else:
            return None
    return node


def _normalised_path(state, certificate_path):
    """Turn a certificate path (strings for list indices) into a usable path."""
    path, node = [], state
    for step in certificate_path:
        if isinstance(node, list):
            try:
                index = int(step)
            except (TypeError, ValueError):
                return None
            if not 0 <= index < len(node):
                return None
            path.append(index)
            node = node[index]
        elif isinstance(node, dict):
            if step not in node:
                return None
            path.append(step)
            node = node[step]
        else:
            return None
    return tuple(path)


def locate(state, sentence, certificate_path=None):
    """Find the string leaf that carries ``sentence``.

    Returns ``(path, exact)`` or ``None``. ``exact`` is True when the leaf is
    exactly the sentence, which lets us drop the whole conversational turn
    instead of editing text inside it.
    """
    if not sentence:
        return None
    target = sentence.strip()
    if certificate_path:
        path = _normalised_path(state, certificate_path)
        if path is not None:
            value = _resolve(state, path)
            if isinstance(value, str):
                if value.strip() == target:
                    return path, True
                if target in value:
                    return path, False
    leaves = list(_string_leaves(state))
    for path, value in leaves:
        if value.strip() == target:
            return path, True
    for path, value in leaves:
        if target in value:
            return path, False
    return None


def _drop_at(state, path, sentence, exact):
    """Delete the sentence, dropping its turn when the turn is only that text."""
    state = copy.deepcopy(state)
    if not path:  # the whole context is a single string
        if not isinstance(state, str):
            return None
        return _tidy(state.replace(sentence.strip(), " "))

    parent = _resolve(state, path[:-1])
    key = path[-1]
    if not isinstance(parent, (list, dict)):
        return None
    remaining = "" if exact else _tidy(parent[key].replace(sentence.strip(), " "))
    if remaining:
        parent[key] = remaining
        return state
    # The turn no longer carries text: remove the whole element when it sits in
    # a list, otherwise blank the field.
    list_depth = max((index for index, step in enumerate(path) if isinstance(step, int)),
                     default=None)
    if list_depth is None:
        parent[key] = ""
        return state
    container = _resolve(state, path[:list_depth])
    if not isinstance(container, list):
        return None
    del container[path[list_depth]]
    return state


def _tidy(text):
    return re.sub(r"\s{2,}", " ", text).strip()


def focus_sentences(row):
    """The two focus-evidence sentences of a record, base wording first."""
    spec = row["evidence_certificate"]["spec"]
    evidence = spec.get("focus_evidence") or []
    if len(evidence) != 2:
        return None
    pair = row["evidence_certificate"].get("verified_pair", {})
    slots = []
    for position, item in enumerate(evidence):
        side = "left" if position == 0 else "right"
        if pair.get("left", "").strip() == item["text"].strip():
            side = "left"
        elif pair.get("right", "").strip() == item["text"].strip():
            side = "right"
        slots.append({
            "side": side,
            "path": tuple(item.get("path", ())),
            "text": item["text"],
            "counterfactual_text": pair.get("negative_" + side, item["text"]),
        })
    return slots


def changed_slot(row):
    """Index (0 or 1) of the focus sentence that the counterfactual rewrites."""
    slots = focus_sentences(row)
    if slots is None:
        return None
    for index, slot in enumerate(slots):
        if slot["counterfactual_text"].strip() != slot["text"].strip():
            return index
    return None


def ablate_input(row, slot_index):
    """Return a copy of ``row['input']`` with one focus sentence deleted.

    Works for both members of a pair: the counterfactual member carries the
    edited wording of the changed sentence, so both wordings are tried.
    """
    slots = focus_sentences(row)
    if slots is None or not 0 <= slot_index < len(slots):
        return None
    slot = slots[slot_index]
    state = row["input"]["state"]
    for sentence in (slot["text"], slot["counterfactual_text"]):
        found = locate(state, sentence, slot["path"])
        if found is None:
            continue
        path, exact = found
        reduced = _drop_at(state, path, sentence, exact)
        if reduced is None:
            continue
        serialised = reduced if isinstance(reduced, str) else json.dumps(reduced, ensure_ascii=False)
        if sentence.strip() in serialised:
            continue  # the sentence survives somewhere else; refuse the view
        if not serialised.strip():
            continue
        payload = copy.deepcopy(row["input"])
        payload["state"] = reduced
        return payload
    return None


def coverage(rows):
    """Preflight statistics: how many records admit a necessity view."""
    stats = {"records": 0, "with_certificate": 0, "changed_slot_known": 0,
             "ablation_built": 0, "failures": []}
    for row in rows:
        stats["records"] += 1
        if focus_sentences(row) is None:
            continue
        stats["with_certificate"] += 1
        index = changed_slot(row)
        if index is None:
            index = 1
        else:
            stats["changed_slot_known"] += 1
        if ablate_input(row, index) is not None:
            stats["ablation_built"] += 1
        elif len(stats["failures"]) < 20:
            stats["failures"].append(row["id"])
    return stats
