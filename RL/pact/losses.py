"""The PACT training objective.

All terms operate on *canonical* candidate logits: the letter a choice was
shown as has already been undone (see ``pact.canon``), so a choice index means
the same thing across permutations and across the two members of a pair.

    L = w_ce * L_CE
      + w_cf * L_CF     counterfactual difference-in-differences margin
      + w_pc * L_PC     permutation consistency
      + w_nr * L_NR     evidence necessity
      + w_emd * L_EMD   ordinal transport cost on Score fields

Every term is a mean over the units it applies to, so changing the mix of
views in a batch does not silently rescale the objective.
"""

import torch
import torch.nn.functional as F

NEG = -1e30
MAX_CHOICES = 26


def scatter_canonical(slot_logits, canon_index, slot_mask, n_canon=MAX_CHOICES):
    """Move per-code-position logits into canonical choice positions.

    ``slot_logits[b, k]`` is the logit of the code shown at position ``k``;
    ``canon_index[b, k]`` says which canonical choice that was. Masked slots are
    parked in a scratch column that is dropped afterwards, so they can never
    overwrite a real choice.
    """
    batch = slot_logits.shape[0]
    scratch = torch.full_like(canon_index, n_canon)
    index = torch.where(slot_mask, canon_index, scratch)
    logits = torch.where(slot_mask, slot_logits, torch.full_like(slot_logits, NEG))
    spread = slot_logits.new_full((batch, n_canon + 1), NEG)
    spread.scatter_(1, index, logits)
    valid = torch.zeros((batch, n_canon + 1), dtype=torch.bool, device=slot_logits.device)
    valid.scatter_(1, index, slot_mask)
    return spread[:, :n_canon], valid[:, :n_canon]


def masked_log_softmax(logits, valid):
    filled = torch.where(valid, logits, torch.full_like(logits, NEG))
    return filled.log_softmax(dim=-1)


def _gather(values, index):
    return values.gather(1, index.unsqueeze(1)).squeeze(1)


def _entropy(log_probs, valid):
    probs = log_probs.exp()
    terms = torch.where(valid, -probs * log_probs, torch.zeros_like(probs))
    return terms.sum(dim=-1)


def cross_entropy_term(log_probs, valid, gold, weight, label_smoothing):
    """Weighted candidate cross-entropy with optional uniform smoothing."""
    nll = -_gather(log_probs, gold)
    if label_smoothing > 0:
        counts = valid.sum(dim=-1).clamp(min=1)
        uniform = -torch.where(valid, log_probs, torch.zeros_like(log_probs)).sum(-1) / counts
        nll = (1 - label_smoothing) * nll + label_smoothing * uniform
    return (nll * weight).sum() / weight.sum().clamp(min=1e-6)


def ordinal_soft_targets(valid, gold, tau):
    """q_j proportional to exp(-|j - y| / tau) over the valid ordered levels."""
    levels = torch.arange(valid.shape[1], device=valid.device).unsqueeze(0)
    distance = (levels - gold.unsqueeze(1)).abs().float()
    scores = torch.where(valid, -distance / tau, torch.full_like(distance, NEG))
    return scores.softmax(dim=-1)


def soft_cross_entropy(log_probs, targets, weight):
    terms = torch.where(targets > 0, -targets * log_probs, torch.zeros_like(targets))
    loss = terms.sum(dim=-1)
    return (loss * weight).sum() / weight.sum().clamp(min=1e-6)


def ordinal_emd(log_probs, gold, weight):
    """Squared earth-mover distance between predicted and one-hot level CDFs.

    Score fields are ordered rubrics, so being one level off should cost less
    than being three levels off. Cross-entropy alone cannot see that.
    """
    probs = log_probs.exp()
    predicted = probs.cumsum(dim=-1)
    levels = torch.arange(probs.shape[1], device=probs.device).unsqueeze(0)
    target = (levels >= gold.unsqueeze(1)).float()
    cost = (predicted - target).pow(2).sum(dim=-1)
    return (cost * weight).sum() / weight.sum().clamp(min=1e-6)


def counterfactual_margin(logits, pair_a, pair_b, gold_a, gold_b, margin):
    """Difference-in-differences margin over a contrastive pair.

        d = [z_a(y_a) - z_a(y_b)] + [z_b(y_b) - z_b(y_a)]

    Any logit offset the model assigns to the shared scenario cancels in ``d``,
    so the term can only be reduced by responding to the one edited fact. A
    hinge on ``d`` therefore supervises the *effect of the evidence* rather than
    the label of either example on its own.
    """
    first, second = logits.index_select(0, pair_a), logits.index_select(0, pair_b)
    delta = (_gather(first, gold_a) - _gather(first, gold_b)) \
        + (_gather(second, gold_b) - _gather(second, gold_a))
    return F.softplus(margin - delta).mean(), delta.detach()


def permutation_consistency(log_probs, valid, view_a, view_b):
    """Jensen-Shannon divergence between two code orderings of one example.

    The answer cannot depend on whether a choice was printed as A or as D, so
    any divergence here is position bias. Penalising it directly removes a bias
    that otherwise shows up as miscalibration.
    """
    first = log_probs.index_select(0, view_a)
    second = log_probs.index_select(0, view_b)
    keep = valid.index_select(0, view_a) & valid.index_select(0, view_b)
    p, q = first.exp(), second.exp()
    mixture = (0.5 * (p + q)).clamp(min=1e-12)
    log_mixture = mixture.log()
    left = torch.where(keep & (p > 0), p * (first - log_mixture), torch.zeros_like(p))
    right = torch.where(keep & (q > 0), q * (second - log_mixture), torch.zeros_like(q))
    return (0.5 * left.sum(-1) + 0.5 * right.sum(-1)).mean()


def necessity_penalty(logits, log_probs, valid, rows, choice_a, choice_b, margin,
                      reference_entropy=None, entropy_margin=0.0):
    """Force indifference between the two outcomes the removed fact decided.

    The curation certificate proves that, with this sentence deleted, the focus
    fact is unknown even given all remaining text. The model must therefore not
    separate ``y_a`` from ``y_b`` on this context - whatever it does with the
    other choices, which the remaining evidence may still rule out.
    """
    selected = logits.index_select(0, rows)
    gap = (_gather(selected, choice_a) - _gather(selected, choice_b)).abs()
    loss = F.relu(gap - margin).pow(2).mean()
    detail = {"necessity_gap": gap.mean().detach()}
    if reference_entropy is not None and entropy_margin > 0:
        entropy = _entropy(log_probs.index_select(0, rows), valid.index_select(0, rows))
        shortfall = F.relu(reference_entropy + entropy_margin - entropy)
        loss = loss + shortfall.mean()
        detail["necessity_entropy"] = entropy.mean().detach()
    return loss, detail


class PactLoss:
    """Assemble the weighted objective for one micro-batch."""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, slot_logits, batch, reg_scale=1.0):
        loss_cfg = self.cfg
        logits, valid = scatter_canonical(slot_logits.float(), batch["canon_index"],
                                          batch["slot_mask"])
        log_probs = masked_log_softmax(logits, valid)
        parts, total = {}, slot_logits.new_zeros((), dtype=torch.float32)

        labelled = batch["labelled_rows"]
        if labelled.numel():
            rows_log = log_probs.index_select(0, labelled)
            rows_valid = valid.index_select(0, labelled)
            gold = batch["gold"].index_select(0, labelled)
            weight = batch["weight"].index_select(0, labelled)
            is_score = batch["is_score"].index_select(0, labelled)
            if loss_cfg.ordinal_smoothing_tau > 0 and bool(is_score.any()):
                targets = F.one_hot(gold, rows_log.shape[1]).float()
                soft = ordinal_soft_targets(rows_valid, gold, loss_cfg.ordinal_smoothing_tau)
                targets = torch.where(is_score.unsqueeze(1), soft, targets)
                ce = soft_cross_entropy(rows_log, targets, weight)
            else:
                ce = cross_entropy_term(rows_log, rows_valid, gold, weight,
                                        loss_cfg.label_smoothing)
            parts["ce"] = ce.detach()
            total = total + loss_cfg.ce_weight * ce
            if loss_cfg.ordinal_emd_weight > 0 and bool(is_score.any()):
                emd = ordinal_emd(rows_log[is_score], gold[is_score], weight[is_score])
                parts["emd"] = emd.detach()
                total = total + reg_scale * loss_cfg.ordinal_emd_weight * emd

        if loss_cfg.cf_weight > 0 and batch["pair_a"].numel():
            margin_loss, delta = counterfactual_margin(
                logits, batch["pair_a"], batch["pair_b"], batch["pair_gold_a"],
                batch["pair_gold_b"], loss_cfg.cf_margin)
            parts["cf"] = margin_loss.detach()
            parts["cf_delta"] = delta.mean()
            total = total + reg_scale * loss_cfg.cf_weight * margin_loss

        if loss_cfg.pcr_weight > 0 and batch["view_a"].numel():
            consistency = permutation_consistency(log_probs, valid, batch["view_a"],
                                                  batch["view_b"])
            parts["pcr"] = consistency.detach()
            total = total + reg_scale * loss_cfg.pcr_weight * consistency

        if loss_cfg.nr_weight > 0 and batch["nec_rows"].numel():
            reference = None
            if loss_cfg.nr_entropy_weight > 0 and batch["nec_reference"].numel():
                reference = _entropy(log_probs.index_select(0, batch["nec_reference"]),
                                     valid.index_select(0, batch["nec_reference"])).detach()
            necessity, detail = necessity_penalty(
                logits, log_probs, valid, batch["nec_rows"], batch["nec_choice_a"],
                batch["nec_choice_b"], loss_cfg.nr_margin, reference,
                loss_cfg.nr_entropy_margin)
            parts["nr"] = necessity.detach()
            parts.update(detail)
            total = total + reg_scale * loss_cfg.nr_weight * necessity

        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite PACT loss: {parts}")
        parts["total"] = total.detach()
        return total, parts, logits, valid
