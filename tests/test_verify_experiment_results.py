"""
Юнит-тесты для verify_experiment_results.py.

Создают временные результаты эксперимента (clean / contaminated / partial)
и проверяют что верификатор корректно их классифицирует.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from agent_dialogue_sim.science.verify_experiment_results import verify, render_text


def _write_results_dir(tmp: Path, config_id: str, arms: dict[str, list[dict]]) -> Path:
    """Создаёт минимальный output dir с config.yaml и arm_*.jsonl."""
    out = tmp / config_id
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "id": config_id,
        "hypothesis": "test",
        "description": "",
        "arms": [{"name": n, "n_dialogues": len(rs)} for n, rs in arms.items()],
        "dialogue": {"n_turns": 5},
        "analysis": {"test": "welch_t_test", "metric": "actionability_rate"},
        "seed_base": 1000,
        "parallelism": 1,
    }
    with (out / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    for name, records in arms.items():
        with (out / f"arm_{name}.jsonl").open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
    return out


def _clean_dialogue(seed: int, n_turns: int = 5) -> dict:
    return {
        "seed": seed,
        "arm": "?",
        "n_turns": n_turns,
        "actionability_rate": 0.5,
        "history": [
            {"speaker": "Alex", "target": "Sam",
             "reply": f"That's a real reply for turn {i}, with substance."}
            for i in range(n_turns)
        ],
    }


def _contaminated_dialogue(seed: int, n_turns: int = 5) -> dict:
    return {
        "seed": seed,
        "arm": "?",
        "n_turns": n_turns,
        "actionability_rate": 0.0,
        "history": [
            {"speaker": "Alex", "target": "Sam",
             "reply": "I'm here, but having trouble responding right now..."}
            for _ in range(n_turns)
        ],
    }


def _error_dialogue(seed: int, msg: str = "API down") -> dict:
    return {"seed": seed, "arm": "?", "error": msg}


class TestVerifyClean(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_results_pass(self):
        out = _write_results_dir(self.tmp, "clean_test", {
            "treatment": [_clean_dialogue(i) for i in range(5)],
            "control":   [_clean_dialogue(100 + i) for i in range(5)],
        })
        report = verify(out, strict=True)
        self.assertTrue(report.is_valid, f"clean results should pass; issues: {report.issues}")
        self.assertEqual(report.arm_stats["treatment"]["n_real_replies"], 25)
        self.assertEqual(report.arm_stats["treatment"]["n_fallback_replies"], 0)


class TestVerifyContaminated(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_2026_04_26_h6_failure_pattern(self):
        """Воспроизводим точную картину провального H6: cot_on имеет смесь
        реальных и fallback, baseline — 100% fallback."""
        out = _write_results_dir(self.tmp, "h6_failure_pattern", {
            "cot_on": [
                _clean_dialogue(i) if i < 2 else _contaminated_dialogue(i)
                for i in range(5)
            ],
            "baseline": [_contaminated_dialogue(100 + i) for i in range(5)],
        })
        report = verify(out, strict=True)
        self.assertFalse(report.is_valid, "contaminated results must fail strict verification")
        # Должно быть найдено в обоих arm
        issues_text = " ".join(report.issues)
        self.assertIn("cot_on", issues_text)
        self.assertIn("baseline", issues_text)
        # Конкретные числа: baseline 5*5=25 fallback, cot_on 3*5=15 fallback
        self.assertEqual(report.arm_stats["baseline"]["n_fallback_replies"], 25)
        self.assertEqual(report.arm_stats["cot_on"]["n_fallback_replies"], 15)

    def test_lax_mode_demotes_to_warnings(self):
        out = _write_results_dir(self.tmp, "demote_test", {
            "a": [_contaminated_dialogue(i) for i in range(3)],
        })
        # min_per_arm=0 — не блокировать на этом
        report = verify(out, strict=False, min_per_arm=0)
        self.assertTrue(report.is_valid, "lax mode should not have issues")
        self.assertGreater(len(report.warnings), 0)


class TestVerifyErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_error_dialogues_blocked(self):
        out = _write_results_dir(self.tmp, "error_test", {
            "a": [_error_dialogue(i) for i in range(3)] + [_clean_dialogue(10)],
        })
        report = verify(out, strict=True, min_per_arm=2)
        # 1 успешный из 4 — но min_per_arm=2 → invalid
        self.assertFalse(report.is_valid)
        self.assertTrue(any("only 1 successful" in i for i in report.issues),
                        f"expected min_per_arm violation; got: {report.issues}")

    def test_strict_blocks_any_error(self):
        out = _write_results_dir(self.tmp, "any_error_test", {
            "a": [_clean_dialogue(i) for i in range(5)] + [_error_dialogue(99)],
        })
        report = verify(out, strict=True, min_per_arm=1)
        # 5 успешных + 1 error — min_per_arm passes, но errors блокируют в strict
        self.assertFalse(report.is_valid)


class TestVerifyMissingFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_config(self):
        out = self.tmp / "no_config"
        out.mkdir()
        report = verify(out)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("config.yaml" in i for i in report.issues))

    def test_missing_arm_jsonl(self):
        out = self.tmp / "no_jsonl"
        out.mkdir()
        config = {
            "id": "no_jsonl", "hypothesis": "x", "description": "",
            "arms": [{"name": "missing", "n_dialogues": 3}],
            "dialogue": {"n_turns": 5}, "analysis": {"test": "welch_t_test", "metric": "x"},
            "seed_base": 1, "parallelism": 1,
        }
        (out / "config.yaml").write_text(yaml.safe_dump(config))
        report = verify(out)
        self.assertFalse(report.is_valid)
        self.assertTrue(any("arm_missing.jsonl" in i for i in report.issues))


class TestRenderText(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_renders_valid_status(self):
        out = _write_results_dir(self.tmp, "render_test", {
            "a": [_clean_dialogue(i) for i in range(3)],
        })
        report = verify(out)
        text = render_text(report)
        self.assertIn("VALID", text)
        self.assertIn("a:", text)

    def test_renders_invalid_with_issues(self):
        out = _write_results_dir(self.tmp, "render_invalid", {
            "a": [_contaminated_dialogue(i) for i in range(3)],
        })
        report = verify(out, strict=True)
        text = render_text(report)
        self.assertIn("INVALID", text)
        self.assertIn("Issues", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
