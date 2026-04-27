"""
Юнит-тесты для experiment_runner.py.
Используют mock-диспатчер — никаких LLM-вызовов.

Запуск: python -m agent_dialogue_sim.test_experiment_runner
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_dialogue_sim.experiment_runner import (
    ArmConfig,
    DialogueConfig,
    AnalysisConfig,
    ExperimentConfig,
    ExperimentRunner,
    mock_dispatcher_factory,
)


class TestExperimentConfigYAML(unittest.TestCase):
    def test_load_h6(self):
        cfg = ExperimentConfig.from_yaml("agent_dialogue_sim/experiments/configs/h6_cot.yaml")
        self.assertEqual(cfg.hypothesis, "H6")
        self.assertEqual(len(cfg.arms), 2)
        self.assertEqual(cfg.arms[0].n_dialogues, 55)
        self.assertEqual(cfg.analysis.test, "welch_t_test")

    def test_load_h1(self):
        cfg = ExperimentConfig.from_yaml("agent_dialogue_sim/experiments/configs/h1_extraversion.yaml")
        self.assertEqual(len(cfg.arms), 3)
        self.assertEqual(cfg.analysis.test, "anova_oneway")
        self.assertEqual(cfg.arms[2].agent_overrides["big_five"]["extraversion"], 0.8)

    def test_load_h5(self):
        cfg = ExperimentConfig.from_yaml("agent_dialogue_sim/experiments/configs/h5_validation.yaml")
        self.assertEqual(cfg.analysis.test, "z_test_two_proportions")
        self.assertFalse(cfg.arms[1].dialogue_overrides["enable_validation"])

    def test_roundtrip_dict(self):
        cfg = ExperimentConfig.from_yaml("agent_dialogue_sim/experiments/configs/h6_cot.yaml")
        d = cfg.to_dict()
        cfg2 = ExperimentConfig.from_dict(d)
        self.assertEqual(cfg.id, cfg2.id)
        self.assertEqual(cfg.arms[0].n_dialogues, cfg2.arms[0].n_dialogues)


class TestRunnerWithMock(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_h6_config(self, n_per_arm: int = 60) -> ExperimentConfig:
        return ExperimentConfig(
            id="test_h6_mock",
            hypothesis="H6",
            description="mock test",
            arms=[
                ArmConfig(name="cot_on", n_dialogues=n_per_arm, seed_offset=0),
                ArmConfig(name="baseline", n_dialogues=n_per_arm, seed_offset=100),
            ],
            dialogue=DialogueConfig(n_turns=20),
            analysis=AnalysisConfig(
                test="welch_t_test",
                metric="actionability_rate",
                params={"alpha": 0.05, "alternative": "greater", "d_threshold": 0.5},
            ),
            seed_base=6000,
            parallelism=4,
        )

    def test_h6_mock_strong_effect_rejects_H0(self):
        # Сильный эффект: cot_on имеет mean=0.7, baseline=0.4 (Δ=0.3 при sd≈0.1 → d≈3)
        cfg = self._make_h6_config(n_per_arm=55)
        dispatcher = mock_dispatcher_factory(
            metric_means={"cot_on": 0.7, "baseline": 0.4},
            metric_sd=0.1,
            metric_name="actionability_rate",
        )
        runner = ExperimentRunner(cfg, dispatcher=dispatcher)
        outcome = runner.run(self.tmpdir)
        self.assertEqual(outcome.test_result["decision"], "reject_H0")
        self.assertLess(outcome.test_result["p_value"], 0.001)
        self.assertGreater(outcome.test_result["effect_size"], 0.5)
        # Проверяем, что summary.json и arm_*.jsonl созданы
        out_dir = Path(outcome.output_dir)
        self.assertTrue((out_dir / "summary.json").exists())
        self.assertTrue((out_dir / "arm_cot_on.jsonl").exists())
        self.assertTrue((out_dir / "arm_baseline.jsonl").exists())
        self.assertTrue((out_dir / "config.yaml").exists())

    def test_h6_mock_no_effect_fails_to_reject(self):
        cfg = self._make_h6_config(n_per_arm=55)
        dispatcher = mock_dispatcher_factory(
            metric_means={"cot_on": 0.5, "baseline": 0.5},
            metric_sd=0.1,
            metric_name="actionability_rate",
        )
        runner = ExperimentRunner(cfg, dispatcher=dispatcher)
        outcome = runner.run(self.tmpdir)
        self.assertEqual(outcome.test_result["decision"], "fail_to_reject_H0")

    def test_h1_mock_three_groups_anova(self):
        cfg = ExperimentConfig(
            id="test_h1_mock",
            hypothesis="H1",
            description="mock test for ANOVA",
            arms=[
                ArmConfig(name="low", n_dialogues=30, seed_offset=0),
                ArmConfig(name="mid", n_dialogues=30, seed_offset=100),
                ArmConfig(name="high", n_dialogues=30, seed_offset=200),
            ],
            dialogue=DialogueConfig(n_turns=20),
            analysis=AnalysisConfig(
                test="anova_oneway",
                metric="avg_reply_length",
                params={"alpha": 0.05, "eta_sq_threshold": 0.06},
            ),
            seed_base=1000,
            parallelism=4,
        )
        # Большие различия: 10, 20, 30 → должно быть reject
        dispatcher = mock_dispatcher_factory(
            metric_means={"low": 5, "mid": 10, "high": 15},
            metric_sd=2.0,
            metric_name="avg_reply_length",
        )
        runner = ExperimentRunner(cfg, dispatcher=dispatcher)
        outcome = runner.run(self.tmpdir)
        self.assertEqual(outcome.test_result["decision"], "reject_H0")
        self.assertEqual(len(outcome.arms), 3)
        self.assertIn("tukey_hsd", outcome.test_result.get("extra_runner", {}))

    def test_summary_json_structure(self):
        cfg = self._make_h6_config(n_per_arm=20)
        dispatcher = mock_dispatcher_factory(
            metric_means={"cot_on": 0.6, "baseline": 0.4},
            metric_sd=0.1,
            metric_name="actionability_rate",
        )
        runner = ExperimentRunner(cfg, dispatcher=dispatcher)
        outcome = runner.run(self.tmpdir)
        with open(Path(outcome.output_dir) / "summary.json") as f:
            summary = json.load(f)
        for key in ("config_id", "hypothesis", "timestamp", "arm_metrics", "test_result"):
            self.assertIn(key, summary)
        self.assertIn("cot_on", summary["arm_metrics"])
        self.assertIn("baseline", summary["arm_metrics"])
        self.assertEqual(len(summary["arm_metrics"]["cot_on"]), 20)

    def test_seeds_are_deterministic(self):
        """Две прогонки с одинаковым seed_base должны дать одинаковые результаты."""
        cfg = self._make_h6_config(n_per_arm=20)
        dispatcher = mock_dispatcher_factory(
            metric_means={"cot_on": 0.6, "baseline": 0.4},
            metric_sd=0.1,
            metric_name="actionability_rate",
        )
        r1 = ExperimentRunner(cfg, dispatcher=dispatcher).run(self.tmpdir / "a")
        r2 = ExperimentRunner(cfg, dispatcher=dispatcher).run(self.tmpdir / "b")
        # одинаковые метрики (отсортированные по seed)
        m1 = sorted(r1.arms["cot_on"], key=lambda x: x["seed"])
        m2 = sorted(r2.arms["cot_on"], key=lambda x: x["seed"])
        for a, b in zip(m1, m2):
            self.assertAlmostEqual(a["actionability_rate"], b["actionability_rate"], places=10)


class TestArmExtraction(unittest.TestCase):
    def test_missing_metric_raises(self):
        cfg = ExperimentConfig(
            id="bad_metric",
            hypothesis="H?",
            description="",
            arms=[ArmConfig(name="a", n_dialogues=5)],
            dialogue=DialogueConfig(n_turns=5),
            analysis=AnalysisConfig(test="welch_t_test", metric="nonexistent_metric"),
        )
        # Mock дает metric_value под другим именем — должен поднять KeyError
        dispatcher = mock_dispatcher_factory(
            metric_means={"a": 1.0}, metric_name="something_else",
        )
        runner = ExperimentRunner(cfg, dispatcher=dispatcher)
        tmp = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(KeyError):
                runner.run(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
