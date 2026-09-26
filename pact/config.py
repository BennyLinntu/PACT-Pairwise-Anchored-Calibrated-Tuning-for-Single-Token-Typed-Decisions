"""Typed, JSON-backed configuration for PACT runs.

A run plan is a single JSON file. ``train_all.py`` reads it, expands the
``runs`` list into concrete configurations (each run may override any field
with dotted keys), and executes the stages in order.
"""

import json
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from pathlib import Path


@dataclass
class DataConfig:
    # Empty = the frozen dataset bundled next to this package. Set it only to
    # point at a checkout's data/ directory instead.
    data_root: str = ""
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/eval.jsonl"
    manifest_file: str = "data/manifest.json"
    view_cache: str = ".cache/views"
    max_length: int = 2048
    # Extra permuted encodings of every training example (0 disables the
    # permutation-consistency regulariser).
    permutation_views: int = 1
    # Build the evidence-ablated ("necessity") view for every contrastive pair.
    necessity_views: bool = True
    # Which focus sentence to delete: "changed" (the one the counterfactual
    # edits), "other", or "random".
    necessity_drop: str = "changed"
    # Source families held out of training for model selection and for fitting
    # the calibrator. The frozen data/eval.jsonl is never used for either.
    inner_holdout_families: int = 6
    split_seed: int = 17
    # Optional auxiliary public data:
    # [{"path": "...jsonl", "weight": 0.3, "max_rows": 4000}]
    public_mix: list = field(default_factory=list)
    verify_manifest: bool = True
    # Cap on contrastive pairs, for smoke runs (0 = no cap).
    limit_pairs: int = 0


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3.5-9B"
    revision: str = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    dtype: str = "bfloat16"
    device: str = "auto"
    # "single": the whole model on one GPU (the default; the sweep then runs
    # one job per GPU, which keeps every card busy). "auto"/"balanced": shard
    # one model across all visible GPUs - only useful for a single large job.
    device_map: str = "single"
    attn_implementation: str = "sdpa"
    load_in_4bit: bool = False
    bnb_compute_dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False
    # Only linear layers whose module path contains this substring are adapted.
    # Nimble uses ".language_model." to skip the vision tower. Empty string
    # means "every linear layer outside the output head".
    target_scope: str = ".language_model."
    target_suffixes: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    gradient_checkpointing: bool = True
    trust_remote_code: bool = False


@dataclass
class LossConfig:
    ce_weight: float = 1.0
    # Counterfactual difference-in-differences margin (paper Sec. IV-B).
    cf_weight: float = 0.5
    cf_margin: float = 2.0
    # Permutation consistency (Section 3.3).
    pcr_weight: float = 0.5
    # Evidence necessity (Section 3.4).
    nr_weight: float = 0.2
    nr_margin: float = 0.0
    nr_entropy_weight: float = 0.0
    nr_entropy_margin: float = 0.0
    # Ordinal treatment of Score fields (Section 3.5).
    ordinal_emd_weight: float = 0.3
    ordinal_smoothing_tau: float = 0.0
    label_smoothing: float = 0.0
    # Regularisers ramp in linearly over this fraction of total steps.
    reg_warmup_frac: float = 0.1


@dataclass
class OptimConfig:
    epochs: float = 2.0
    max_steps: int = 0          # 0 = derive from epochs
    learning_rate: float = 5e-5
    # LoRA+ : the B matrices get learning_rate * lora_b_multiplier.
    lora_b_multiplier: float = 8.0
    weight_decay: float = 0.0
    warmup_frac: float = 0.1
    scheduler: str = "cosine"   # cosine | linear | constant
    groups_per_batch: int = 2   # one group = one contrastive pair
    grad_accum: int = 2
    # Share of steps that carry the extra views. Lower them on a small GPU:
    # the method still works, it just sees the regularisers less often.
    perm_view_prob: float = 1.0
    necessity_prob: float = 1.0
    max_grad_norm: float = 1.0
    seed: int = 17
    length_bucket_groups: int = 64
    eval_every_frac: float = 0.34
    save_every_frac: float = 0.5
    keep_checkpoints: int = 2
    log_every: int = 10


@dataclass
class EvalConfig:
    # Permutation-ensembled decoding at test time (1 = single pass).
    permutation_ensemble: int = 4
    rows_per_batch: int = 4
    ece_bins: int = 15
    # Also score the untouched base model once, for the comparison table.
    score_base_model: bool = True
    # Score only the first N holdout examples (0 = all 324). Smoke runs only:
    # a truncated holdout is not a result.
    holdout_limit: int = 0
    calibrate: bool = True
    # Temperature model: "scalar", "per_kind", "contextual".
    calibration_model: str = "contextual"


@dataclass
class ReleaseConfig:
    """What the pipeline leaves behind when every run has finished."""

    # Everything a human wants to look at is copied here, inside the bundle.
    results_dir: str = "results"
    # The run whose best seed becomes the shipped model.
    primary_run: str = "pact_full"
    # How that seed is chosen. "inner_nll" uses the inner validation split;
    # the frozen holdout is never used to choose anything.
    selection: str = "inner_nll"
    # Save the LoRA weights as a plain torch state dict (adapter.pth) next to
    # the PEFT directory.
    save_pth: bool = True
    # Also write base+LoRA merged weights (~18 GB, loads the base on CPU).
    merge_full_model: bool = False
    # Copy the per-example holdout rows of the shipped model.
    copy_rows: bool = True
    figures: bool = True


@dataclass
class RunSpec:
    name: str = "pact_full"
    description: str = ""
    overrides: dict = field(default_factory=dict)
    seeds: list = field(default_factory=list)


@dataclass
class PactConfig:
    output_dir: str = ".cache/runs/pact"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    seeds: list = field(default_factory=lambda: [17])
    runs: list = field(default_factory=lambda: [asdict(RunSpec())])

    # ---- construction -------------------------------------------------

    @staticmethod
    def from_dict(payload):
        return _build(PactConfig, payload)

    @staticmethod
    def load(path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return PactConfig.from_dict(payload)

    def to_dict(self):
        return asdict(self)

    def with_overrides(self, overrides):
        """Return a copy with dotted-key overrides applied (``loss.cf_weight``)."""
        payload = self.to_dict()
        for key, value in overrides.items():
            node = payload
            parts = key.split(".")
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    raise KeyError(f"Unknown config section in override: {key}")
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(f"Unknown config field in override: {key}")
            node[parts[-1]] = value
        return PactConfig.from_dict(payload)

    def validate(self):
        if self.data.max_length <= 0:
            raise ValueError("data.max_length must be positive")
        if self.data.necessity_drop not in ("changed", "other", "random"):
            raise ValueError("data.necessity_drop must be changed|other|random")
        if self.data.permutation_views < 0:
            raise ValueError("data.permutation_views must be >= 0")
        if self.loss.pcr_weight > 0 and self.data.permutation_views < 1:
            raise ValueError("permutation consistency needs data.permutation_views >= 1")
        if self.loss.nr_weight > 0 and not self.data.necessity_views:
            raise ValueError("necessity regularisation needs data.necessity_views")
        if self.optim.groups_per_batch <= 0 or self.optim.grad_accum <= 0:
            raise ValueError("batch sizes must be positive")
        if not 0 <= self.optim.warmup_frac < 1:
            raise ValueError("optim.warmup_frac must be in [0, 1)")
        if self.optim.scheduler not in ("cosine", "linear", "constant"):
            raise ValueError("optim.scheduler must be cosine|linear|constant")
        if self.evaluation.calibration_model not in ("scalar", "per_kind", "contextual"):
            raise ValueError("calibration_model must be scalar|per_kind|contextual")
        if self.evaluation.permutation_ensemble < 1:
            raise ValueError("permutation_ensemble must be >= 1")
        if self.model.lora_rank <= 0:
            raise ValueError("model.lora_rank must be positive")
        if self.model.device_map not in ("single", "auto", "balanced"):
            raise ValueError("model.device_map must be single|auto|balanced")
        if self.release.selection not in ("inner_nll", "inner_accuracy", "first"):
            raise ValueError("release.selection must be inner_nll|inner_accuracy|first")
        return self


def _build(cls, payload):
    if not isinstance(payload, dict):
        raise TypeError(f"{cls.__name__} expects an object, got {type(payload).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(payload) - set(known)
    if unknown:
        raise KeyError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, spec in known.items():
        if name not in payload:
            continue
        value = payload[name]
        if isinstance(spec.type, type) and is_dataclass(spec.type):
            kwargs[name] = _build(spec.type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def resolve(path, root=""):
    """Resolve a relative path against ``root``, or against the bundle."""
    from pact import BUNDLE_ROOT

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root or BUNDLE_ROOT) / candidate


def dataset_path(cfg, attribute):
    """Resolve one of the frozen dataset files for a config."""
    from pact import DATA_ROOT

    candidate = Path(getattr(cfg.data, attribute))
    if candidate.is_absolute():
        return candidate
    return Path(cfg.data.data_root or DATA_ROOT) / candidate
