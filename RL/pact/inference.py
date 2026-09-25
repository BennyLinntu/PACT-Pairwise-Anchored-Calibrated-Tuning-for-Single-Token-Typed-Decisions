"""Serving a PACT adapter.

``PactScorer`` returns the same shape as Nimble's own scorers - a typed output
plus per-candidate logits and probabilities - and adds two things the training
recipe earned:

* optional permutation-ensembled decoding (``ensemble > 1``);
* the fitted contextual temperature, applied before the softmax.

With ``ensemble=1`` and no calibration file it is, deliberately, the same
computation Nimble does today, so any difference in the numbers comes from the
adapter and not from the serving path.
"""

import hashlib
import json
import random
from pathlib import Path

import torch

from nimble.scoring.parallel_schema import choice_key, choices_for, prepare_prompts, validate_schema

from pact.calibrate import Calibrator
from pact.canon import canon_index_for, permutation_plan, presented_choices
from pact.evaluate import _average_log_probs, _probabilities


class PactScorer:
    def __init__(self, adapter, model_id=None, revision=None, device="auto",
                 ensemble=1, calibration=None, check_prompt_contract=True, dtype=None):
        adapter = Path(adapter)
        self.contract = json.loads((adapter / "schema_config.json").read_text(encoding="utf-8"))
        if check_prompt_contract:
            from nimble.scoring import parallel_schema

            expected = hashlib.sha256(
                Path(parallel_schema.__file__).read_bytes()).hexdigest()
            if self.contract.get("prompt_code_sha256") != expected:
                raise ValueError("Adapter was trained against a different scoring prompt")
        from peft import PeftModel
        from transformers import AutoTokenizer

        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.max_length = self.contract.get("max_length", 2048)
        self.ensemble = max(1, ensemble)
        self.tokenizer = AutoTokenizer.from_pretrained(str(adapter))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = self._load_base(model_id or self.contract["model"],
                               revision or self.contract.get("revision"), dtype)
        self.model = PeftModel.from_pretrained(base, str(adapter)).eval()
        path = Path(calibration) if calibration else adapter / "calibration.json"
        self.calibrator = Calibrator.load(path) if path.exists() else None
        from pact.model import LogitReader

        self.reader = LogitReader()

    def _load_base(self, model_id, revision, dtype):
        import transformers

        resolved = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        kwargs = {"dtype": resolved}
        if revision:
            kwargs["revision"] = revision
        errors = {}
        for name in ("AutoModelForCausalLM", "Qwen3_5ForConditionalGeneration",
                     "AutoModelForImageTextToText", "AutoModel"):
            loader = getattr(transformers, name, None)
            if loader is None:
                continue
            try:
                model = loader.from_pretrained(model_id, **kwargs)
            except Exception as error:  # noqa: BLE001
                errors[name] = str(error)
                continue
            model.config.use_cache = False
            return model.to(self.device)
        raise RuntimeError("Could not load the base model: " + json.dumps(errors, indent=2))

    @torch.no_grad()
    def score(self, context, schema, ensemble=None):
        validate_schema(schema)
        ensemble = max(1, ensemble or self.ensemble)
        rng = random.Random(0)
        fields = {}
        for name, definition in schema.items():
            canonical = list(choices_for(definition))
            views = []
            for permutation in permutation_plan(len(canonical), ensemble - 1, rng):
                rendered = json.loads(json.dumps(schema))
                rendered[name]["choices"] = presented_choices(canonical, permutation)
                prompt = prepare_prompts(self.tokenizer, context, rendered, self.max_length)
                index = prompt.names.index(name)
                mapping = canon_index_for(canonical, prompt.choices[index])
                logits = self._last_logits(prompt.full_ids[index], prompt.candidate_ids[index])
                ordered = [0.0] * len(canonical)
                for slot, canonical_index in enumerate(mapping):
                    ordered[canonical_index] = logits[slot]
                temperature = 1.0
                if self.calibrator is not None:
                    temperature = self.calibrator.temperature(
                        kind="noul" if definition["type"] == "boolean" else "choice",
                        n_choices=len(canonical),
                        prompt_tokens=len(prompt.full_ids[index]))
                views.append({"logits": ordered,
                              "probabilities": _probabilities(ordered, temperature)})
            probabilities = views[0]["probabilities"] if len(views) == 1 else \
                _average_log_probs([view["probabilities"] for view in views])
            best = max(range(len(canonical)), key=lambda i: probabilities[i])
            fields[name] = {
                "prediction": canonical[best],
                "scores": {choice_key(value): probability
                           for value, probability in zip(canonical, probabilities)},
                "probabilities": {choice_key(value): probability
                                  for value, probability in zip(canonical, probabilities)},
                "logits": {choice_key(value): logit
                           for value, logit in zip(canonical, views[0]["logits"])},
                "ensemble": len(views),
            }
        return {"output": {name: result["prediction"] for name, result in fields.items()},
                "fields": fields}

    def _last_logits(self, input_ids, candidate_ids):
        ids = torch.tensor([list(input_ids)], device=self.device)
        mask = torch.ones_like(ids)
        if self.device == "cuda":
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context = torch.inference_mode()
        with context:
            logits = self.reader(self.model, ids, mask).float()[0]
        return [float(logits[code]) for code in candidate_ids]
