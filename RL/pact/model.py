"""Model, adapter and logit-reading utilities.

The adapter is an ordinary PEFT LoRA adapter over the same linear layers Nimble
adapts, so anything that can load a Nimble adapter can load a PACT one. The
extras here are optional and all reversible: rsLoRA scaling, DoRA, LoRA+
(a larger step size on the B factors), and 4-bit base weights for small GPUs.
"""

import json
from pathlib import Path

import torch

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def resolve_dtype(name, device):
    if name != "auto":
        if name not in DTYPES:
            raise ValueError(f"Unsupported dtype: {name}")
        return DTYPES[name]
    return torch.bfloat16 if device == "cuda" else torch.float32


def resolve_device(requested):
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_tokenizer(cfg):
    from transformers import AutoTokenizer

    kwargs = {"trust_remote_code": cfg.model.trust_remote_code}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_id, **kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_base(cfg, device, dtype, log):
    import transformers

    kwargs = {"dtype": dtype, "attn_implementation": cfg.model.attn_implementation,
              "trust_remote_code": cfg.model.trust_remote_code}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if cfg.model.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=DTYPES[cfg.model.bnb_compute_dtype])
        kwargs["device_map"] = {"": 0}
    elif cfg.model.device_map != "single" and device == "cuda":
        # Shard one model across every visible GPU. Slower than one job per
        # GPU (the cards take turns), so use it only for a single big job.
        kwargs["device_map"] = cfg.model.device_map
    candidates = ["AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration",
                  "AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModel"]
    errors = {}
    for name in candidates:
        loader = getattr(transformers, name, None)
        if loader is None:
            continue
        try:
            model = loader.from_pretrained(cfg.model.model_id, **kwargs)
        except Exception as error:  # noqa: BLE001 - report every failed loader
            errors[name] = f"{type(error).__name__}: {error}"
            continue
        log(f"[model] loaded {cfg.model.model_id} with {name}")
        return model, name
    raise RuntimeError("No transformers loader accepted this checkpoint: "
                       + json.dumps(errors, indent=2))


def find_target_modules(model, scope, suffixes, log=print):
    """Linear layers to adapt: language-model projections, never the head."""
    def matches(name, module):
        if not isinstance(module, torch.nn.Linear):
            return False
        if "lm_head" in name or name.endswith("output_embeddings"):
            return False
        return not suffixes or any(name.endswith("." + suffix) or name == suffix
                                   for suffix in suffixes)

    named = list(model.named_modules())
    targets = [name for name, module in named if scope in name and matches(name, module)]
    if not targets and scope:
        log(f"[model] no linear layer matched scope {scope!r}; adapting every language layer")
        targets = [name for name, module in named if matches(name, module)]
    if not targets:
        targets = [name for name, module in named
                   if isinstance(module, torch.nn.Linear) and "lm_head" not in name]
    if not targets:
        raise RuntimeError("No adaptable linear layers found")
    return targets


def build_model(cfg, log=print):
    """Load the base checkpoint and attach a trainable LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    device = resolve_device(cfg.model.device)
    dtype = resolve_dtype(cfg.model.dtype, device)
    if device == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16; set model.dtype explicitly")
    base, loader = _load_base(cfg, device, dtype, log)
    if cfg.model.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=cfg.model.gradient_checkpointing)
    elif device == "cuda" and cfg.model.device_map == "single":
        base = base.to("cuda")
    base.config.use_cache = False
    targets = find_target_modules(base, cfg.model.target_scope, cfg.model.target_suffixes, log)
    lora_kwargs = {}
    if cfg.model.use_rslora:
        lora_kwargs["use_rslora"] = True
    if cfg.model.use_dora:
        lora_kwargs["use_dora"] = True
    model = get_peft_model(base, LoraConfig(
        r=cfg.model.lora_rank, lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout, target_modules=targets, bias="none",
        task_type="CAUSAL_LM", **lora_kwargs))
    if cfg.model.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    info = {"loader": loader, "device": device, "dtype": str(dtype),
            "target_modules": len(targets), "target_example": targets[:4],
            "trainable_parameters": trainable, "total_parameters": total,
            "trainable_fraction": trainable / max(total, 1),
            "load_in_4bit": cfg.model.load_in_4bit, "use_rslora": cfg.model.use_rslora,
            "use_dora": cfg.model.use_dora}
    log("[model] " + json.dumps(info))
    return model, info


def load_adapter_weights(model, directory):
    """Load saved LoRA weights into an already-built PEFT model, in place."""
    from peft import set_peft_model_state_dict

    directory = Path(directory)
    safetensors = directory / "adapter_model.safetensors"
    if safetensors.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors))
    else:
        binary = directory / "adapter_model.bin"
        if not binary.exists():
            raise FileNotFoundError(f"No adapter weights in {directory}")
        state = torch.load(str(binary), map_location="cpu")
    outcome = set_peft_model_state_dict(model, state)
    missing = getattr(outcome, "unexpected_keys", None)
    if missing:
        raise RuntimeError(f"Adapter weights did not match the model: {missing[:5]}")
    return model


def parameter_groups(model, learning_rate, b_multiplier, weight_decay):
    """LoRA+: the B factors take a larger step than the A factors."""
    slow, fast = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (fast if "lora_B" in name else slow).append(parameter)
    groups = [{"params": slow, "lr": learning_rate, "weight_decay": weight_decay}]
    if fast:
        groups.append({"params": fast, "lr": learning_rate * b_multiplier,
                       "weight_decay": weight_decay})
    return groups


class LogitReader:
    """Read the next-token logits of the final position, cheaply.

    Only one position matters, so we ask the model to materialise one row of
    the vocabulary projection. The keyword for that moved between transformers
    releases, so the working call is discovered once and then reused.
    """

    MODES = ("logits_to_keep", "num_logits_to_keep", "full")

    def __init__(self):
        self.mode = None

    def __call__(self, model, input_ids, attention_mask):
        modes = (self.mode,) if self.mode else self.MODES
        last_error = None
        for mode in modes:
            kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
                      "use_cache": False}
            if mode != "full":
                kwargs[mode] = 1
            try:
                logits = model(**kwargs).logits
            except TypeError as error:
                last_error = error
                continue
            except ValueError as error:
                last_error = error
                continue
            self.mode = mode
            return logits[:, -1, :]
        raise RuntimeError(f"Could not read last-token logits: {last_error}")


def candidate_logits(reader, model, batch):
    """Logits of the allowed one-token answer codes, in code-position order."""
    logits = reader(model, batch["input_ids"], batch["attention_mask"]).float()
    selected = logits.gather(1, batch["candidate_ids"].clamp(min=0))
    return selected


def memory_estimate(cfg, parameters=9e9, layers=36, hidden=3584, vocab=152_000):
    """A rough per-GPU VRAM budget, printed before a run so a small card fails early.

    Assumes gradient checkpointing: the dominant term is one saved input per
    layer, plus the recomputation of a single layer. LoRA keeps gradients and
    optimiser state tiny. The real figure is measured during the run and
    written to ``train_report.json`` as ``peak_gpu_gib``.
    """
    gib = 1024 ** 3
    bytes_per_weight = 0.55 if cfg.model.load_in_4bit else 2.0
    weights = parameters * bytes_per_weight
    trainable = 4 * cfg.model.lora_rank * layers * hidden * 7      # LoRA A+B, 7 projections
    optimizer = trainable * (2 + 2 + 2)                            # weights, grads, Adam states
    rows = cfg.optim.groups_per_batch * (2 + cfg.data.permutation_views * 2
                                         + int(cfg.data.necessity_views))
    tokens = rows * cfg.data.max_length
    checkpoints = layers * tokens * hidden * 2
    recompute = 12 * tokens * hidden * 2
    logits = rows * vocab * 4
    activations = checkpoints + recompute + logits
    total = weights + optimizer + activations
    return {"rows_per_micro_batch": rows,
            "tokens_per_micro_batch": tokens,
            "weights_gib": round(weights / gib, 2),
            "optimizer_gib": round(optimizer / gib, 2),
            "activations_gib": round(activations / gib, 2),
            "estimated_peak_gib": round(total / gib, 2),
            "assumes": {"parameters": parameters, "layers": layers, "hidden": hidden,
                        "gradient_checkpointing": cfg.model.gradient_checkpointing},
            "note": "rough estimate; the measured peak is in train_report.json"}
