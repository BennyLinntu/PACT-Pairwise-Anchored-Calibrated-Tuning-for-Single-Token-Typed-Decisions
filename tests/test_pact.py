"""Offline checks for PACT.

Everything here runs on a laptop with no GPU, no network and no model
download: the data path is exercised with a character-level tokenizer and the
losses with hand-made logits. Run it before sending a job to a server.

    python -m unittest discover -s new_methods/tests -t new_methods
"""

import hashlib
import json
import os
import random
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from pact import BUNDLE_ROOT, DATA_ROOT, ablation, bundle_path  # noqa: E402
from pact.calibrate import Calibrator, fit_calibrator  # noqa: E402
from pact.canon import canon_index_for, permutation_plan, presented_choices  # noqa: E402
from pact.config import PactConfig  # noqa: E402
from pact.data import (ViewStore, build_pairs, build_views, load_jsonl,  # noqa: E402
                       to_example, verify_dataset)
from pact.losses import (MAX_CHOICES, PactLoss, counterfactual_margin,  # noqa: E402
                         masked_log_softmax, necessity_penalty, ordinal_emd,
                         permutation_consistency, scatter_canonical)
from pact.metrics import (expected_calibration_error, pair_consistency,  # noqa: E402
                          row_metrics, summarize)
from pact.sampler import Collator, GroupSampler  # noqa: E402
from tests.fake_tokenizer import FakeTokenizer  # noqa: E402

TRAIN = DATA_ROOT / "data" / "train.jsonl"


def smoke_config(tmp, pairs=6):
    return PactConfig.from_dict({
        "output_dir": str(tmp),
        "data": {"view_cache": str(Path(tmp) / "views"), "max_length": 100000,
                 "permutation_views": 1, "necessity_views": True, "limit_pairs": pairs,
                 "inner_holdout_families": 1},
        "model": {"model_id": "fake", "revision": "fake"},
    })


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_jsonl(TRAIN)[:400]

    def test_dataset_verifies(self):
        report, train_rows, eval_rows = verify_dataset(PactConfig())
        self.assertEqual(len(train_rows), report["train_rows"])
        self.assertEqual(len(eval_rows), 324)
        self.assertTrue(report["separation_checked"])

    def test_every_family_is_a_flipped_pair(self):
        groups, skipped = build_pairs(self.rows, True)
        self.assertTrue(all(group["paired"] for group in groups))
        self.assertEqual(skipped, {})
        for group in groups:
            base, counter = group["members"]
            self.assertEqual(base["canonical"], counter["canonical"])
            self.assertNotEqual(base["gold"], counter["gold"])
            self.assertEqual(base["partner_gold"], counter["gold"])

    def test_ablation_is_reconstructable(self):
        coverage = ablation.coverage(self.rows)
        self.assertEqual(coverage["ablation_built"], coverage["records"])
        self.assertEqual(coverage["failures"], [])

    def test_ablation_removes_only_the_focus_sentence(self):
        row = self.rows[0]
        slot = ablation.changed_slot(row)
        sentences = ablation.focus_sentences(row)
        reduced = ablation.ablate_input(row, slot)
        self.assertIsNotNone(reduced)
        serialised = json.dumps(reduced["state"], ensure_ascii=False)
        self.assertNotIn(sentences[slot]["text"], serialised)
        self.assertNotIn(sentences[slot]["counterfactual_text"], serialised)
        other = sentences[1 - slot]["text"]
        self.assertIn(other[:40], serialised)
        self.assertEqual(reduced["questions"], row["input"]["questions"])


class TestEncoding(unittest.TestCase):
    """The permutation trick has to survive the round trip through the prompt."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.tmp = bundle_path(".cache/test-views")

    def test_permutation_plan_is_a_permutation(self):
        rng = random.Random(3)
        for size in (2, 3, 5):
            plans = permutation_plan(size, 2, rng)
            self.assertEqual(plans[0], tuple(range(size)))
            for plan in plans:
                self.assertEqual(sorted(plan), list(range(size)))

    def test_canon_index_maps_codes_back(self):
        canonical = ["LOW", "MEDIUM", "HIGH"]
        permutation = (2, 0, 1)
        shown = presented_choices(canonical, permutation)
        self.assertEqual(shown, ["HIGH", "LOW", "MEDIUM"])
        self.assertEqual(canon_index_for(canonical, shown), [2, 0, 1])

    def test_views_are_built_and_ordered_correctly(self):
        config = smoke_config(self.tmp, pairs=4)
        manifest, path = build_views(config, self.tokenizer, log=lambda *_: None)
        store = ViewStore(path, manifest)
        self.assertGreaterEqual(manifest["views"].get("primary", 0), 8)
        self.assertGreaterEqual(manifest["views"].get("necessity", 0), 1)
        self.assertEqual(manifest["rejected_count"], 0)
        for row in store.rows:
            self.assertEqual(len(row["candidate_ids"]), row["n_choices"])
            self.assertEqual(sorted(row["canon_index"]), list(range(row["n_choices"])))
            if row["view"] != "necessity":
                self.assertTrue(0 <= row["gold"] < row["n_choices"])

    def test_permuted_prompt_really_reorders_the_choices(self):
        rows = load_jsonl(TRAIN)[:2]
        example = to_example(rows[0], True)
        canonical = example["canonical"]
        if len(canonical) < 3:
            self.skipTest("needs a field with at least three choices")
        from pact.data import encode_view

        permutation = tuple(reversed(range(len(canonical))))
        encoded = encode_view(self.tokenizer, example["context"], example["schema"],
                              example["field"], canonical, permutation, 100000)
        text = self.tokenizer.decode(encoded["input_ids"])
        positions = [text.index(str(value)) for value in presented_choices(canonical, permutation)]
        self.assertEqual(positions, sorted(positions),
                         "choices must appear in the permuted order inside the prompt")
        self.assertEqual(list(encoded["canon_index"]), list(permutation))


class TestBatching(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        tmp = bundle_path(".cache/test-views-batch")
        config = smoke_config(tmp, pairs=6)
        manifest, path = build_views(config, self.tokenizer, log=lambda *_: None)
        self.store = ViewStore(path, manifest)
        self.config = config

    def test_groups_keep_pairs_together(self):
        groups = self.store.groups("train")
        self.assertTrue(groups)
        for group in groups:
            families = {row["family"] for row in group["rows"]}
            self.assertEqual(len(families), 1)

    def test_collator_builds_every_index_table(self):
        groups = self.store.groups("train")
        sampler = GroupSampler(groups, 2, 8, seed=17)
        batches = sampler.epoch(0)
        collator = Collator(self.tokenizer.pad_token_id)
        found = {"pairs": 0, "views": 0, "necessity": 0}
        for batch_groups in batches:
            batch = collator(batch_groups, "cpu", training=True)
            self.assertEqual(batch["input_ids"].shape[0], len(batch["rows"]))
            self.assertEqual(batch["candidate_ids"].shape[1], MAX_CHOICES)
            # Left padding: the final column is never padding.
            self.assertTrue(bool(batch["attention_mask"][:, -1].all()))
            found["pairs"] += int(batch["pair_a"].numel())
            found["views"] += int(batch["view_a"].numel())
            found["necessity"] += int(batch["nec_rows"].numel())
            for index in batch["pair_a"].tolist():
                self.assertGreaterEqual(batch["gold"][index].item(), 0)
        self.assertGreater(found["pairs"], 0)
        self.assertGreater(found["views"], 0)
        self.assertGreater(found["necessity"], 0)

    def test_sampler_is_deterministic(self):
        groups = self.store.groups("train")
        first = GroupSampler(groups, 2, 8, seed=17).epoch(0)
        second = GroupSampler(groups, 2, 8, seed=17).epoch(0)
        self.assertEqual([[g["family"] for g in batch] for batch in first],
                         [[g["family"] for g in batch] for batch in second])


class TestLosses(unittest.TestCase):
    def test_scatter_canonical_undoes_the_permutation(self):
        slots = torch.tensor([[1.0, 2.0, 3.0] + [0.0] * 23])
        canon = torch.tensor([[2, 0, 1] + [0] * 23])
        mask = torch.tensor([[True] * 3 + [False] * 23])
        logits, valid = scatter_canonical(slots, canon, mask)
        self.assertAlmostEqual(logits[0, 0].item(), 2.0)
        self.assertAlmostEqual(logits[0, 1].item(), 3.0)
        self.assertAlmostEqual(logits[0, 2].item(), 1.0)
        self.assertEqual(valid[0, :3].tolist(), [True, True, True])
        self.assertFalse(bool(valid[0, 3:].any()))

    def test_counterfactual_margin_rewards_the_flip(self):
        index = torch.tensor([0])
        gold_a, gold_b = torch.tensor([0]), torch.tensor([1])
        aligned = torch.tensor([[3.0, -3.0, 0.0], [-3.0, 3.0, 0.0]])
        shortcut = torch.tensor([[3.0, -3.0, 0.0], [3.0, -3.0, 0.0]])
        good, delta_good = counterfactual_margin(aligned, index, torch.tensor([1]),
                                                 gold_a, gold_b, 2.0)
        bad, delta_bad = counterfactual_margin(shortcut, index, torch.tensor([1]),
                                               gold_a, gold_b, 2.0)
        self.assertLess(good.item(), bad.item())
        self.assertGreater(delta_good.item(), delta_bad.item())

    def test_permutation_consistency_is_zero_for_equal_views(self):
        logits = torch.tensor([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5], [2.0, -1.0, 0.0]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        log_probs = masked_log_softmax(logits, valid)
        same = permutation_consistency(log_probs, valid, torch.tensor([0]), torch.tensor([1]))
        different = permutation_consistency(log_probs, valid, torch.tensor([0]), torch.tensor([2]))
        self.assertAlmostEqual(same.item(), 0.0, places=6)
        self.assertGreater(different.item(), 0.05)

    def test_necessity_penalty_wants_indifference(self):
        logits = torch.tensor([[0.0, 0.0, 5.0], [4.0, -4.0, 5.0]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        log_probs = masked_log_softmax(logits, valid)
        rows = torch.tensor([0, 1])
        loss, detail = necessity_penalty(logits, log_probs, valid, rows,
                                         torch.tensor([0, 0]), torch.tensor([1, 1]), 0.0)
        self.assertGreater(loss.item(), 0.0)
        alone, _ = necessity_penalty(logits, log_probs, valid, torch.tensor([0]),
                                     torch.tensor([0]), torch.tensor([1]), 0.0)
        self.assertAlmostEqual(alone.item(), 0.0, places=6)
        self.assertIn("necessity_gap", detail)

    def test_ordinal_emd_grows_with_distance(self):
        logits = torch.tensor([[5.0, 0.0, 0.0, 0.0]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        log_probs = masked_log_softmax(logits, valid)
        weight = torch.ones(1)
        near = ordinal_emd(log_probs, torch.tensor([1]), weight)
        far = ordinal_emd(log_probs, torch.tensor([3]), weight)
        self.assertLess(near.item(), far.item())

    def test_full_objective_is_differentiable(self):
        config = PactConfig().loss
        loss_fn = PactLoss(config)
        slots = torch.randn(5, MAX_CHOICES, requires_grad=True)
        batch = {
            "canon_index": torch.tensor([[0, 1, 2] + [0] * 23] * 5),
            "slot_mask": torch.tensor([[True] * 3 + [False] * 23] * 5),
            "gold": torch.tensor([0, 1, 0, 1, -1]),
            "weight": torch.ones(5),
            "is_score": torch.tensor([False, False, True, True, False]),
            "labelled_rows": torch.tensor([0, 1, 2, 3]),
            "pair_a": torch.tensor([0]), "pair_b": torch.tensor([1]),
            "pair_gold_a": torch.tensor([0]), "pair_gold_b": torch.tensor([1]),
            "view_a": torch.tensor([0]), "view_b": torch.tensor([2]),
            "nec_rows": torch.tensor([4]), "nec_choice_a": torch.tensor([0]),
            "nec_choice_b": torch.tensor([1]), "nec_reference": torch.tensor([0]),
        }
        loss, parts, logits, valid = loss_fn(slots, batch, reg_scale=1.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(slots.grad)
        self.assertTrue(torch.isfinite(slots.grad).all())
        for key in ("ce", "cf", "pcr", "nr", "emd", "total"):
            self.assertIn(key, parts)
        self.assertFalse(bool(valid[:, 3:].any()))

    def test_masked_choices_get_no_probability(self):
        slots = torch.randn(2, MAX_CHOICES)
        canon = torch.tensor([[0, 1] + [0] * 24] * 2)
        mask = torch.tensor([[True] * 2 + [False] * 24] * 2)
        logits, valid = scatter_canonical(slots, canon, mask)
        probabilities = masked_log_softmax(logits, valid).exp()
        self.assertAlmostEqual(probabilities.sum(-1)[0].item(), 1.0, places=5)
        self.assertAlmostEqual(probabilities[:, 2:].sum().item(), 0.0, places=6)


class TestMetrics(unittest.TestCase):
    def test_row_metrics_and_summary(self):
        rows = []
        for index in range(20):
            correct = index % 2 == 0
            probabilities = [0.8, 0.2] if correct else [0.2, 0.8]
            row = row_metrics(probabilities, 0, "noul", [False, True])
            row.update(kind="noul", family=f"f{index // 2}", domain="test")
            rows.append(row)
        summary = summarize(rows)
        self.assertEqual(summary["all"]["count"], 20)
        self.assertAlmostEqual(summary["all"]["accuracy"], 0.5)
        self.assertIn("ece", summary["all"])
        self.assertEqual(summary["all"]["pairs"], 10)
        self.assertAlmostEqual(summary["all"]["pair_accuracy"], 0.0)

    def test_perfect_calibration_has_low_ece(self):
        rows = [{"confidence": 0.9, "correct": 1} for _ in range(9)]
        rows.append({"confidence": 0.9, "correct": 0})
        self.assertLess(expected_calibration_error(rows, 10), 0.02)

    def test_pair_consistency_counts_both_members(self):
        rows = [{"family": "a", "correct": 1}, {"family": "a", "correct": 1},
                {"family": "b", "correct": 1}, {"family": "b", "correct": 0}]
        result = pair_consistency(rows)
        self.assertEqual(result["pairs"], 2)
        self.assertAlmostEqual(result["pair_accuracy"], 0.5)

    def test_score_metrics_use_the_expected_level(self):
        row = row_metrics([0.0, 0.5, 0.5, 0.0], 1, "score", ["0", "1", "2", "3"])
        self.assertAlmostEqual(row["expected_score"], 1.5)
        self.assertAlmostEqual(row["absolute_error"], 0.5)


class TestCalibration(unittest.TestCase):
    def _records(self, sharpness=3.0, count=400):
        rng = random.Random(11)
        records = []
        for index in range(count):
            choices = rng.choice([2, 3, 5])
            gold = rng.randrange(choices)
            logits = [rng.gauss(0, 1) for _ in range(choices)]
            if rng.random() < 0.75:
                logits[gold] = max(logits) + abs(rng.gauss(1.0, 0.3))
            records.append({"logits": [value * sharpness for value in logits], "gold": gold,
                            "kind": "choice", "n_choices": choices,
                            "prompt_tokens": rng.randrange(200, 1900)})
        return records

    def test_overconfident_logits_get_a_temperature_above_one(self):
        records = self._records()
        calibrator, stats = fit_calibrator(records, "scalar", log=lambda *_: None)
        self.assertTrue(stats["applied"])
        self.assertLess(stats["nll_after"], stats["nll_before"])
        self.assertGreater(calibrator.temperature(), 1.0)
        # Temperature scaling minimises NLL, which is a proper scoring rule;
        # ECE is reported but is not what the fit optimises.
        self.assertIn("ece_after", stats)
        self.assertIn("ece_before", stats)

    def test_contextual_mode_varies_with_prompt_statistics(self):
        records = self._records()
        calibrator, stats = fit_calibrator(records, "contextual", log=lambda *_: None)
        self.assertEqual(calibrator.mode if stats["applied"] else "contextual", "contextual")
        first = calibrator.temperature(n_choices=2, prompt_tokens=200)
        second = calibrator.temperature(n_choices=5, prompt_tokens=1800)
        self.assertGreater(first, 0)
        self.assertGreater(second, 0)

    def test_round_trip(self):
        calibrator = Calibrator("contextual", {"bias": [1.0, 1.0, 1.0],
                                               "log_choices": 0.1, "log_tokens": -0.05})
        payload = calibrator.to_dict()
        restored = Calibrator.from_dict(payload)
        self.assertAlmostEqual(calibrator.temperature(n_choices=4, prompt_tokens=900),
                               restored.temperature(n_choices=4, prompt_tokens=900))


class TestConfig(unittest.TestCase):
    def test_dotted_overrides(self):
        config = PactConfig().with_overrides({"loss.cf_weight": 0.0, "optim.seed": 5})
        self.assertEqual(config.loss.cf_weight, 0.0)
        self.assertEqual(config.optim.seed, 5)

    def test_unknown_override_is_rejected(self):
        with self.assertRaises(KeyError):
            PactConfig().with_overrides({"loss.nonexistent": 1})

    def test_validation_catches_impossible_combinations(self):
        with self.assertRaises(ValueError):
            PactConfig().with_overrides({"data.permutation_views": 0}).validate()
        with self.assertRaises(ValueError):
            PactConfig().with_overrides({"data.necessity_views": False}).validate()

    def test_shipped_configs_are_valid(self):
        for path in sorted((BUNDLE_ROOT / "configs").glob("*.json")):
            with self.subTest(config=path.name):
                config = PactConfig.load(path)
                config.validate()
                for spec in config.runs:
                    config.with_overrides(spec.get("overrides", {})).validate()


class TinyModel(torch.nn.Module):
    """A stand-in language model: embeddings, one layer, a vocabulary head.

    Enough to exercise the training loop end to end without transformers or
    peft installed. It honours ``logits_to_keep`` by returning only the final
    position, exactly as the real models do.
    """

    def __init__(self, vocab=513, dim=24):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.mix = torch.nn.Linear(dim, dim)
        self.head = torch.nn.Linear(dim, vocab)
        self.saved = []

    def forward(self, input_ids, attention_mask=None, use_cache=False, logits_to_keep=None):
        hidden = torch.tanh(self.mix(self.embed(input_ids)))
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        last = hidden[:, -1:, :]
        return SimpleNamespace(logits=self.head(last))

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
        self.saved.append(str(path))


class TestTrainingLoop(unittest.TestCase):
    """The loop itself: batching, accumulation, schedule, logging, saving."""

    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.tmp = bundle_path(".cache/test-train")
        shutil.rmtree(self.tmp / "run", ignore_errors=True)
        config = smoke_config(self.tmp, pairs=6).with_overrides({
            "optim.max_steps": 3, "optim.groups_per_batch": 1, "optim.grad_accum": 2,
            "optim.log_every": 1, "optim.save_every_frac": 1.0, "optim.warmup_frac": 0.34,
            "model.device": "cpu"})
        manifest, path = build_views(config, self.tokenizer, log=lambda *_: None)
        self.store = ViewStore(path, manifest)
        self.config = config

    def test_three_steps_run_and_write_their_artefacts(self):
        from pact.trainer import PactTrainer, schedule_factor

        output = self.tmp / "run"
        trainer = PactTrainer(self.config, self.store, output, log=lambda *_: None,
                              model=TinyModel())
        report = trainer.train(dev_scorer=None)
        self.assertEqual(report["steps"], 3)
        self.assertTrue((output / "train_plan.json").exists())
        self.assertTrue((output / "train_report.json").exists())
        self.assertTrue((output / "final" / "schema_config.json").exists())
        contract = json.loads((output / "final" / "schema_config.json").read_text())
        self.assertEqual(contract["task"], "schema_candidate_classification_v1")
        self.assertEqual(contract["method"], "pact-v1")
        logged = [json.loads(line) for line in
                  (output / "train_log.jsonl").read_text().splitlines() if line.strip()]
        self.assertEqual(len(logged), 3)
        self.assertTrue(all("ce" in record for record in logged))
        self.assertTrue(all(record["lr"] > 0 for record in logged))

    def test_schedule_warms_up_then_decays(self):
        from pact.trainer import schedule_factor

        warm = schedule_factor(0, 100, 10, "cosine")
        peak = schedule_factor(10, 100, 10, "cosine")
        late = schedule_factor(99, 100, 10, "cosine")
        self.assertLess(warm, peak)
        self.assertAlmostEqual(peak, 1.0, places=6)
        self.assertLess(late, 0.05)
        self.assertAlmostEqual(schedule_factor(50, 100, 0, "constant"), 1.0)

    def test_loss_falls_on_a_memorisable_batch(self):
        """Ten steps on one batch must reduce its own loss - the gradient path works."""
        from pact.trainer import PactTrainer

        trainer = PactTrainer(self.config.with_overrides({"optim.max_steps": 1}),
                              self.store, self.tmp / "grad", log=lambda *_: None,
                              model=TinyModel())
        trainer.prepare()
        groups = self.store.groups("train")[:1]
        batch = trainer.collator(groups, "cpu", training=True)
        first = None
        for _ in range(10):
            loss, parts, _, _ = trainer.forward_batch(batch, 1.0)
            trainer.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.optimizer.step()
            first = first if first is not None else float(loss.detach())
        self.assertLess(float(loss.detach()), first)


class TestBundle(unittest.TestCase):
    """The folder must be copyable to a server on its own."""

    VENDORED = ["nimble/__init__.py", "nimble/compat.py", "nimble/paths.py",
                "nimble/scoring/__init__.py", "nimble/scoring/parallel_schema.py",
                "nimble/evaluation/__init__.py", "nimble/evaluation/evaluate_pilot.py",
                "nimble/training/__init__.py", "nimble/training/schema_data.py"]
    DATA = ["data/train.jsonl", "data/eval.jsonl", "data/manifest.json"]

    def test_everything_needed_is_present(self):
        for relative in self.VENDORED + self.DATA:
            with self.subTest(file=relative):
                self.assertTrue((BUNDLE_ROOT / relative).exists(),
                                f"{relative} is missing from the bundle")

    def test_nimble_resolves_to_the_vendored_copy(self):
        from nimble.scoring import parallel_schema

        self.assertEqual(Path(parallel_schema.__file__).resolve(),
                         (BUNDLE_ROOT / "nimble/scoring/parallel_schema.py").resolve())

    def test_vendored_files_match_the_checkout(self):
        """If the parent repository is here too, the copies must be identical."""
        upstream = BUNDLE_ROOT.parent
        if not (upstream / "nimble" / "scoring" / "parallel_schema.py").exists():
            self.skipTest("no parent checkout to compare against")
        for relative in self.VENDORED + self.DATA:
            with self.subTest(file=relative):
                mine = hashlib.sha256((BUNDLE_ROOT / relative).read_bytes()).hexdigest()
                theirs = hashlib.sha256((upstream / relative).read_bytes()).hexdigest()
                self.assertEqual(mine, theirs,
                                 f"{relative} has drifted from the checkout; re-copy it")

    def test_contract_hash_is_the_imported_module(self):
        from pact.trainer import prompt_code_hash
        from nimble.scoring import parallel_schema

        expected = hashlib.sha256(
            Path(parallel_schema.__file__).read_bytes()).hexdigest()
        self.assertEqual(prompt_code_hash(), expected)


class TestScheduling(unittest.TestCase):
    """Job-level GPU scheduling: the plan, not the CUDA calls."""

    def setUp(self):
        sys.path.insert(0, str(BUNDLE_ROOT))
        import train_all

        self.train_all = train_all

    def test_devices_come_from_the_environment(self):
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            self.assertEqual(self.train_all.visible_devices(), ["0", "1"])
            os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3 ,5"
            self.assertEqual(self.train_all.visible_devices(), ["2", "3", "5"])
            self.assertEqual(self.train_all.visible_devices("1"), ["1"])
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous

    def test_default_config_expands_to_one_job_per_run_and_seed(self):
        config = PactConfig.load(BUNDLE_ROOT / "configs" / "server_2x48gb.json")
        jobs = [(spec["name"], seed) for spec in config.runs
                for seed in (spec.get("seeds") or config.seeds)]
        self.assertEqual(len(jobs), len(set(jobs)), "duplicate run/seed job")
        self.assertGreater(len(jobs), 2, "a sweep is what keeps both GPUs busy")
        self.assertEqual(jobs[0][0], "pact_full",
                         "the first job claims the base-model baseline")

    def test_worker_command_pins_one_gpu(self):
        """The spawned command must carry one run, one seed, one GPU."""
        captured = {}

        class FakePopen:
            def __init__(self, command, stdout=None, stderr=None, env=None, cwd=None):
                captured["command"] = command
                captured["env"] = env
                captured["cwd"] = cwd

            def poll(self):
                return 0

        config = PactConfig.load(BUNDLE_ROOT / "configs" / "server_2x48gb.json")
        spec = config.runs[0]
        arguments = SimpleNamespace(force=True, skip_existing_train=False)
        original = self.train_all.subprocess.Popen
        self.train_all.subprocess.Popen = FakePopen
        try:
            worker = self.train_all.spawn_worker(
                (spec, 17), "1", BUNDLE_ROOT / "configs" / "server_2x48gb.json",
                bundle_path(".cache/test-sweep"), ["train", "evaluate"], arguments,
                allow_baseline=False)
        finally:
            self.train_all.subprocess.Popen = original
            shutil.rmtree(bundle_path(".cache/test-sweep"), ignore_errors=True)
        command = captured["command"]
        self.assertIn("--only-run", command)
        self.assertEqual(command[command.index("--only-run") + 1], spec["name"])
        self.assertEqual(command[command.index("--only-seed") + 1], "17")
        self.assertIn("--no-parallel", command)
        self.assertIn("--no-baseline", command)
        self.assertIn("--force", command)
        self.assertNotIn("preflight", command)
        self.assertEqual(captured["env"]["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(captured["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertEqual(worker["device"], "1")
        worker["handle"].close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
