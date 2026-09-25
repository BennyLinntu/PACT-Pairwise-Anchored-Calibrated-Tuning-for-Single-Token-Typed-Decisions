"""Offline checks for the last stage: figures, weights, tables, SUMMARY.md.

A finished sweep is faked on disk - results, logs, holdout rows and a small
adapter - and then packaged, so the whole tail of the pipeline is exercised
without a GPU or a model.

    python -m unittest discover -s tests -t .
"""

import json
import shutil
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from pact import bundle_path  # noqa: E402
from pact.config import PactConfig  # noqa: E402


def fake_run(root, run, seed, inner_nll, accuracy, with_adapter=True):
    """A plausible finished run on disk."""
    directory = root / run / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)

    def summary(value):
        return {"all": {"count": 20, "correct": int(value * 20), "accuracy": value,
                        "nll": 0.4, "brier": 0.2, "ece": 0.07, "mean_confidence": 0.8,
                        "pair_accuracy": value - 0.05, "pairs": 10,
                        "permutation_tv": 0.03, "permutation_flip_rate": 0.05,
                        "expected_score_mae": 0.3, "aurc": 0.1},
                "choice": {"count": 8, "accuracy": value},
                "noul": {"count": 6, "accuracy": value},
                "score": {"count": 6, "accuracy": value, "expected_score_mae": 0.3},
                "by_domain": {"commerce": {"count": 20, "accuracy": value}}}

    rows = []
    for index in range(20):
        correct = index % 5 != 0
        rows.append({"id": f"r{index}", "family": f"f{index // 2}", "kind": "choice",
                     "domain": "commerce", "gold": 0, "n_choices": 3,
                     "prediction": 0 if correct else 1, "correct": int(correct),
                     "confidence": 0.85, "reference_probability": 0.85 if correct else 0.1,
                     "nll": 0.2, "brier": 0.2, "probabilities": [0.85, 0.1, 0.05],
                     "permutation_tv": 0.02 + index * 0.001,
                     "permutation_flip": int(index % 7 == 0)})
    for name in ("holdout_single.json", "holdout_single_calibrated.json",
                 "holdout_ensemble.json"):
        (directory / name).write_text(json.dumps({"summary": summary(accuracy), "rows": rows}),
                                      encoding="utf-8")
    (directory / "train_log.jsonl").write_text("\n".join(
        json.dumps({"step": step, "lr": 5e-5, "ce": 1.0 / (step + 1), "cf": 0.5,
                    "pcr": 0.1, "nr": 0.05, "emd": 0.2, "total": 1.5})
        for step in range(1, 6)), encoding="utf-8")
    (directory / "dev_history.json").write_text(json.dumps(
        [{"step": 2, "summary": {"all": {"nll": inner_nll + 0.1, "accuracy": 0.70}}},
         {"step": 4, "summary": {"all": {"nll": inner_nll, "accuracy": 0.75}}}]),
        encoding="utf-8")
    adapter = directory / "best"
    if with_adapter:
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
        from safetensors.torch import save_file

        save_file({"base_model.model.layers.0.lora_A.weight": torch.zeros(4, 8),
                   "base_model.model.layers.0.lora_B.weight": torch.zeros(8, 4)},
                  str(adapter / "adapter_model.safetensors"))
        (adapter / "schema_config.json").write_text(json.dumps(
            {"task": "schema_candidate_classification_v1", "method": "pact-v1"}),
            encoding="utf-8")
        (directory / "calibration.json").write_text(json.dumps(
            {"mode": "scalar", "weights": {"bias": [0.9, 0.9, 0.9],
                                           "log_choices": 0.0, "log_tokens": 0.0}}),
            encoding="utf-8")
    (directory / "results.json").write_text(json.dumps({
        "run": run, "seed": seed, "description": f"{run} seed {seed}",
        "adapter": str(adapter), "holdout_examples": 20,
        "calibration": {"mode": "scalar", "ece_before": 0.1, "ece_after": 0.06,
                        "nll_before": 0.5, "nll_after": 0.45, "applied": True},
        "train": {"steps": 5, "best": {"step": 4, "nll": inner_nll, "accuracy": 0.75}},
        "holdout": {"single": {"summary": summary(accuracy)},
                    "single_calibrated": {"summary": summary(accuracy)},
                    "ensemble": {"summary": summary(accuracy + 0.01)}}}),
        encoding="utf-8")
    return directory


class TestPackaging(unittest.TestCase):
    """The last stage: weights, tables, figures and a summary in one folder."""

    def setUp(self):
        self.root = bundle_path(".cache/test-release")
        shutil.rmtree(self.root, ignore_errors=True)
        self.sweep = self.root / "sweep"
        self.sweep.mkdir(parents=True)
        fake_run(self.sweep, "pact_full", 17, inner_nll=0.50, accuracy=0.90)
        fake_run(self.sweep, "pact_full", 18, inner_nll=0.42, accuracy=0.88)
        fake_run(self.sweep, "ce_only", 17, inner_nll=0.61, accuracy=0.86)
        self.cfg = PactConfig().with_overrides(
            {"release.results_dir": str(self.root / "results")})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_selection_uses_inner_validation_not_the_holdout(self):
        from pact.release import load_results, select_model

        chosen = select_model(load_results(self.sweep), self.cfg, log=lambda *_: None)
        # Seed 18 has the better inner NLL and the worse holdout accuracy.
        # Choosing it anyway is the whole point: the holdout stays untouched.
        self.assertEqual(chosen["record"]["run"], "pact_full")
        self.assertEqual(chosen["record"]["seed"], 18)
        self.assertAlmostEqual(chosen["inner_nll"], 0.42)

    def test_package_writes_weights_tables_figures_and_summary(self):
        from pact.release import package
        from pact.report import build

        build(self.sweep, None, log=lambda *_: None)
        destination = self.root / "results"
        result = package(self.cfg, self.sweep, destination, log=lambda *_: None)
        self.assertIsNotNone(result)
        self.assertEqual(result["seed"], 18)

        summary = (destination / "SUMMARY.md").read_text(encoding="utf-8")
        self.assertIn("pact_full", summary)
        self.assertIn("Shipped model", summary)
        self.assertIn("Headline", summary)

        for relative in ("model/adapter_config.json", "model/adapter_model.safetensors",
                         "model/schema_config.json", "model/calibration.json",
                         "model/adapter.pth", "tables/results_tables.md",
                         "tables/results_tables.tex", "tables/results_aggregate.json",
                         "all_results.json", "holdout_rows.json",
                         "runs/pact_full/seed-17/results.json",
                         "runs/ce_only/seed-17/results.json"):
            with self.subTest(file=relative):
                self.assertTrue((destination / relative).exists(), f"missing {relative}")

        state = torch.load(destination / "model" / "adapter.pth", map_location="cpu")
        self.assertEqual(len(state), 2)
        self.assertTrue(all(hasattr(tensor, "shape") for tensor in state.values()))
        recorded = json.loads((destination / "all_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(recorded), 3)

    def test_figures_are_drawn_or_their_data_is_kept(self):
        from pact.figures import build as build_figures
        from pact.report import build

        build(self.sweep, None, log=lambda *_: None)
        destination = self.root / "figures"
        written = build_figures(self.sweep / "pact_full" / "seed-18", self.sweep,
                                destination, log=lambda *_: None)
        data = json.loads((destination / "figure_data.json").read_text(encoding="utf-8"))
        for key in ("reliability_uncalibrated", "risk_coverage_uncalibrated",
                    "permutation_tv", "train_log", "dev_history", "aggregate"):
            self.assertIn(key, data)
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.assertEqual(written, [])
            self.skipTest("matplotlib not installed; the JSON fallback was checked")
        names = {Path(path).name for path in written}
        for expected in ("reliability.png", "risk_coverage.png", "permutation.png",
                         "runs_comparison.png", "training.png"):
            self.assertIn(expected, names)
        for path in written:
            self.assertGreater(Path(path).stat().st_size, 1000)

    def test_package_survives_an_empty_sweep(self):
        from pact.release import package

        empty = self.root / "empty"
        empty.mkdir(parents=True)
        self.assertIsNone(package(self.cfg, empty, self.root / "results-empty",
                                  log=lambda *_: None))

    def test_package_runs_without_an_adapter_on_disk(self):
        """A crashed save must not stop the tables and figures from appearing."""
        from pact.release import package

        shutil.rmtree(self.sweep / "pact_full" / "seed-18" / "best")
        shutil.rmtree(self.sweep / "pact_full" / "seed-17" / "best")
        destination = self.root / "results-no-adapter"
        result = package(self.cfg, self.sweep, destination, log=lambda *_: None)
        self.assertIsNotNone(result)
        self.assertTrue((destination / "SUMMARY.md").exists())
        self.assertTrue((destination / "all_results.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
