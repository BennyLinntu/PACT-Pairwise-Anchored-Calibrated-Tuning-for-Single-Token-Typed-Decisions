"""Canonical choice space.

Nimble presents each allowed answer as a one-letter code (A, B, C, ...) and
reads the logit of that single token. The mapping from a letter to a meaning is
therefore an arbitrary, per-example artefact of how the schema was rendered.

Every PACT loss and metric works in *canonical* space instead: choice index
``c`` always means the same answer for a given field, whatever letter it was
shown as. A view's ``canon_index[k]`` says which canonical choice was presented
at code position ``k``.
"""

import math

from nimble.scoring.parallel_schema import choice_key

MAX_CHOICES = 26


def canonical_keys(choices):
    """Stable string keys for a canonical choice list (booleans -> true/false)."""
    return [choice_key(value) for value in choices]


def permutation_plan(n_choices, extra_views, rng):
    """Identity permutation first, then ``extra_views`` distinct permutations.

    A permutation ``p`` means: code position ``k`` shows canonical choice
    ``p[k]``. The identity view is always kept so that the primary training
    view matches what a caller would send at inference time.
    """
    identity = tuple(range(n_choices))
    plans = [identity]
    if extra_views <= 0:
        return plans
    distinct = math.factorial(n_choices) if n_choices <= 8 else float("inf")
    seen = {identity}
    attempts = 0
    while len(plans) < extra_views + 1 and attempts < 64 * (extra_views + 1):
        attempts += 1
        candidate = list(range(n_choices))
        rng.shuffle(candidate)
        candidate = tuple(candidate)
        if candidate in seen and len(seen) < distinct:
            continue
        seen.add(candidate)
        plans.append(candidate)
    while len(plans) < extra_views + 1:  # n_choices == 1: nothing to permute
        plans.append(identity)
    return plans


def presented_choices(canonical, permutation):
    """Order the canonical choices the way a permuted prompt shows them."""
    return [canonical[index] for index in permutation]


def canon_index_for(canonical, presented):
    """Map each presented code position back to its canonical choice index."""
    keys = canonical_keys(canonical)
    lookup = {key: index for index, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError("Canonical choices are not unique after key normalisation")
    mapping = []
    for value in presented:
        key = choice_key(value)
        if key not in lookup:
            raise ValueError(f"Presented choice {key!r} is not in the canonical list")
        mapping.append(lookup[key])
    if sorted(mapping) != list(range(len(keys))):
        raise ValueError("Presented choices are not a permutation of the canonical ones")
    return mapping


def total_variation(first, second):
    """Total-variation distance between two equal-length probability vectors."""
    return 0.5 * sum(abs(a - b) for a, b in zip(first, second))
