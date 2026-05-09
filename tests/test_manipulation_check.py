"""
Юнит-тесты для manipulation_check.py — без реальных LLM-вызовов.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agent_dialogue_sim.science.manipulation_check import (
    BigFiveJudge,
    TraitRating,
    check_manipulation,
    sample_replies_from_arm,
    _parse_judge_response,
)


class TestParseResponse(unittest.TestCase):
    def test_valid_json(self):
        content = (
            '{"openness": 5, "conscientiousness": 4, "extraversion": 7, '
            '"agreeableness": 6, "neuroticism": 2, "reasoning": "very chatty"}'
        )
        rating = _parse_judge_response(content)
        self.assertEqual(rating.extraversion, 7)
        self.assertEqual(rating.openness, 5)

    def test_clamps_out_of_range(self):
        content = (
            '{"openness": 99, "conscientiousness": -5, "extraversion": 4, '
            '"agreeableness": 4, "neuroticism": 4, "reasoning": "x"}'
        )
        rating = _parse_judge_response(content)
        self.assertEqual(rating.openness, 7)
        self.assertEqual(rating.conscientiousness, 1)

    def test_handles_extra_text(self):
        content = (
            "Here is my rating:\n"
            '{"openness": 4, "conscientiousness": 4, "extraversion": 5, '
            '"agreeableness": 4, "neuroticism": 4, "reasoning": "ok"}'
        )
        rating = _parse_judge_response(content)
        self.assertEqual(rating.extraversion, 5)


class TestMockJudge(unittest.TestCase):
    def test_pattern_matching(self):
        mapping = {
            "AWESOME": TraitRating(4, 4, 7, 4, 4),
            "boring": TraitRating(4, 4, 2, 4, 4),
        }
        judge = BigFiveJudge(client=None, mock_mapping=mapping)
        self.assertEqual(judge.rate("That's AWESOME!").extraversion, 7)
        self.assertEqual(judge.rate("This is boring stuff").extraversion, 2)
        # Default neutral
        self.assertEqual(judge.rate("normal text").extraversion, 4)

    def test_call_count(self):
        judge = BigFiveJudge(client=None)
        judge.rate("a")
        judge.rate("b")
        self.assertEqual(judge.n_calls, 2)


class TestRealClientCall(unittest.TestCase):
    def test_calls_llm_correctly(self):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = (
            '{"openness": 5, "conscientiousness": 6, "extraversion": 6, '
            '"agreeableness": 5, "neuroticism": 3, "reasoning": "expressive"}'
        )
        client.chat.completions.create.return_value.choices = [choice]

        judge = BigFiveJudge(client=client, model="x/y")
        rating = judge.rate("I love new ideas, let's do this!")
        self.assertEqual(rating.extraversion, 6)
        # Проверка системного промпта
        msgs = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertIn("Big Five", msgs[0]["content"])
        self.assertIn("OCEAN", msgs[0]["content"])


class TestCheckManipulationSuccess(unittest.TestCase):
    """Манипуляция должна детектироваться когда наблюдаемая оценка коррелирует с заданной."""

    def test_strong_extraversion_manipulation_detected(self):
        # Создаём mock-сценарий: low (0.2) → judge даёт 2; high (0.8) → 7
        mapping = {
            "[LOW]": TraitRating(4, 4, 2, 4, 4),
            "[MID]": TraitRating(4, 4, 4, 4, 4),
            "[HIGH]": TraitRating(4, 4, 7, 4, 4),
        }
        judge = BigFiveJudge(client=None, mock_mapping=mapping)
        replies_by_level = {
            0.2: [f"[LOW] reply {i}" for i in range(10)],
            0.5: [f"[MID] reply {i}" for i in range(10)],
            0.8: [f"[HIGH] reply {i}" for i in range(10)],
        }
        result = check_manipulation(replies_by_level, "extraversion", judge)
        self.assertTrue(result.manipulation_successful)
        self.assertGreater(result.spearman_rho, 0.4)
        self.assertLess(result.p_value, 0.05)
        self.assertEqual(result.n_replies, 30)

    def test_failed_manipulation_when_no_correlation(self):
        # Все уровни → одинаковая оценка
        mapping = {
            "reply": TraitRating(4, 4, 4, 4, 4),
        }
        judge = BigFiveJudge(client=None, mock_mapping=mapping)
        replies_by_level = {
            0.2: [f"reply {i}" for i in range(10)],
            0.5: [f"reply {i}" for i in range(10)],
            0.8: [f"reply {i}" for i in range(10)],
        }
        result = check_manipulation(replies_by_level, "extraversion", judge)
        self.assertFalse(result.manipulation_successful)
        self.assertIn("FAILED", result.interpretation.upper())

    def test_reverse_correlation_also_flags(self):
        # Манипуляция работает в обратную сторону — ИЛИ rho<0 ИЛИ rho>0 при success
        # check_manipulation использует |rho| ≥ threshold
        mapping = {
            "[LOW]": TraitRating(4, 4, 7, 4, 4),
            "[HIGH]": TraitRating(4, 4, 2, 4, 4),
        }
        judge = BigFiveJudge(client=None, mock_mapping=mapping)
        replies_by_level = {
            0.2: [f"[LOW] r{i}" for i in range(10)],
            0.8: [f"[HIGH] r{i}" for i in range(10)],
        }
        result = check_manipulation(replies_by_level, "extraversion", judge)
        # |rho| высокий, но в обратную сторону → строго говоря успешно по критерию
        self.assertGreater(abs(result.spearman_rho), 0.4)
        self.assertLess(result.p_value, 0.05)
        self.assertTrue(result.manipulation_successful)
        # Но это знак неправильной манипуляции, что должно быть видно по знаку
        self.assertLess(result.spearman_rho, 0)

    def test_invalid_trait_raises(self):
        judge = BigFiveJudge(client=None)
        with self.assertRaises(ValueError):
            check_manipulation({0.5: ["a"]}, "made_up_trait", judge)

    def test_insufficient_data(self):
        judge = BigFiveJudge(client=None)
        result = check_manipulation(
            {0.5: ["a"], 0.8: ["b"]},
            "extraversion", judge,
        )
        # n=2 < 6 — не считаем
        self.assertFalse(result.manipulation_successful)
        self.assertIn("insufficient data", result.interpretation)


class TestSampleRepliesFromArm(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_arm_jsonl(self, n_dialogues: int, n_turns: int) -> Path:
        path = self.tmp / "arm_test.jsonl"
        with path.open("w") as f:
            for d in range(n_dialogues):
                rec = {
                    "seed": 1000 + d,
                    "arm": "test",
                    "history": [
                        {"speaker": "Alex", "target": "Sam",
                         "reply": f"dialogue{d}_turn{t}"}
                        for t in range(n_turns)
                    ],
                }
                f.write(json.dumps(rec) + "\n")
        return path

    def test_extracts_one_reply_per_dialogue(self):
        path = self._make_arm_jsonl(n_dialogues=10, n_turns=20)
        replies = sample_replies_from_arm(str(path), n_per_dialogue=1)
        self.assertEqual(len(replies), 10)

    def test_skips_error_dialogues(self):
        path = self.tmp / "arm.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"seed": 1, "arm": "x", "error": "API down"}) + "\n")
            f.write(json.dumps({"seed": 2, "arm": "x",
                                "history": [{"speaker": "A", "reply": "ok"}]}) + "\n")
        replies = sample_replies_from_arm(str(path))
        self.assertEqual(len(replies), 1)

    def test_max_dialogues_limit(self):
        path = self._make_arm_jsonl(n_dialogues=20, n_turns=5)
        replies = sample_replies_from_arm(str(path), n_per_dialogue=1, max_dialogues=5)
        self.assertEqual(len(replies), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
