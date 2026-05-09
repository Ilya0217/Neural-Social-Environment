"""
Юнит-тесты для toxicity_judge.py — без реальных LLM-вызовов.
Проверяют: парсинг ответа судьи, mock-режим, классификацию диалога,
корректность Cohen's κ.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from agent_dialogue_sim.science.toxicity_judge import (
    ToxicityJudge,
    ToxicityLabel,
    JudgeError,
    _parse_judge_response,
    compute_kappa,
    classify_dialogue_replies,
)


class TestParseResponse(unittest.TestCase):
    def test_valid_json(self):
        content = '{"is_violation": true, "principle": "non_maleficence", "reasoning": "insulting"}'
        label = _parse_judge_response(content)
        self.assertTrue(label.is_violation)
        self.assertEqual(label.principle, "non_maleficence")

    def test_extra_text_around_json(self):
        # LLM иногда добавляет преамбулу
        content = (
            "Here is my evaluation:\n"
            '{"is_violation": false, "principle": "none", "reasoning": "neutral"}'
            "\nLet me know if you need more."
        )
        label = _parse_judge_response(content)
        self.assertFalse(label.is_violation)
        self.assertEqual(label.principle, "none")

    def test_invalid_principle_normalized(self):
        content = '{"is_violation": true, "principle": "MADE_UP", "reasoning": ""}'
        label = _parse_judge_response(content)
        self.assertEqual(label.principle, "none")  # нормализуется

    def test_no_json_raises(self):
        with self.assertRaises(JudgeError):
            _parse_judge_response("Sorry, I cannot evaluate this.")

    def test_malformed_json_raises(self):
        with self.assertRaises(JudgeError):
            _parse_judge_response('{"is_violation": tru, "principle": ...}')


class TestMockJudge(unittest.TestCase):
    def test_exact_match(self):
        judge = ToxicityJudge(client=None,
                               mock_responses={"You are stupid": True,
                                                "Hello there": False})
        self.assertTrue(judge.classify("You are stupid").is_violation)
        self.assertFalse(judge.classify("Hello there").is_violation)

    def test_substring_match(self):
        judge = ToxicityJudge(client=None,
                               mock_responses={"stupid": True})
        # частичное совпадение по подстроке
        result = judge.classify("That idea is stupid, frankly.")
        self.assertTrue(result.is_violation)

    def test_default_no_match(self):
        judge = ToxicityJudge(client=None, mock_responses={"hate": True})
        result = judge.classify("Just a normal sentence.")
        self.assertFalse(result.is_violation)

    def test_usage_counts(self):
        judge = ToxicityJudge(client=None, mock_responses={"bad": True})
        judge.classify("bad thing")
        judge.classify("ok message")
        judge.classify("very bad")
        self.assertEqual(judge.usage.n_calls, 3)
        self.assertEqual(judge.usage.n_violations, 2)


class TestRealClient(unittest.TestCase):
    def test_calls_client_with_correct_messages(self):
        client = MagicMock()
        # Имитация ответа OpenAI SDK
        choice = MagicMock()
        choice.message.content = (
            '{"is_violation": true, "principle": "non_maleficence", "reasoning": "insult"}'
        )
        client.chat.completions.create.return_value.choices = [choice]

        judge = ToxicityJudge(client=client, model="test/model")
        label = judge.classify("You're an idiot",
                                context={"speaker": "A", "target": "B"})
        self.assertTrue(label.is_violation)
        self.assertEqual(label.principle, "non_maleficence")

        # Проверяем что system prompt и user message были переданы
        call = client.chat.completions.create.call_args
        msgs = call.kwargs["messages"]
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("Beauchamp", msgs[0]["content"])
        self.assertEqual(msgs[1]["role"], "user")
        self.assertIn("A → B", msgs[1]["content"])
        self.assertIn("You're an idiot", msgs[1]["content"])

    def test_client_failure_raises_judge_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        judge = ToxicityJudge(client=client)
        with self.assertRaises(JudgeError):
            judge.classify("test")
        self.assertEqual(judge.usage.n_errors, 1)


class TestBatchClassify(unittest.TestCase):
    def test_batch_with_skip_on_error(self):
        # Mock client, который чередует success/failure
        responses = []
        client = MagicMock()
        def side_effect(*args, **kwargs):
            i = len(responses)
            responses.append(i)
            if i == 1:
                raise RuntimeError("fail on second call")
            choice = MagicMock()
            choice.message.content = (
                '{"is_violation": false, "principle": "none", "reasoning": "ok"}'
            )
            r = MagicMock()
            r.choices = [choice]
            return r
        client.chat.completions.create.side_effect = side_effect

        judge = ToxicityJudge(client=client)
        results = judge.classify_batch(
            ["a", "b", "c"], on_error="skip",
        )
        self.assertEqual(len(results), 3)
        self.assertIsNotNone(results[0])
        self.assertIsNone(results[1])
        self.assertIsNotNone(results[2])


class TestKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        a = [True, False, True, True, False] * 4
        b = list(a)
        report = compute_kappa(a, b)
        self.assertAlmostEqual(report.kappa, 1.0, places=5)
        self.assertTrue(report.is_acceptable)
        self.assertEqual(report.interpretation, "almost perfect")

    def test_substantial_agreement(self):
        # 17 совпадений из 20, baseline ~50/50
        a = [True] * 10 + [False] * 10
        b = [True] * 8 + [False] * 2 + [False] * 8 + [True] * 2
        report = compute_kappa(a, b, threshold=0.5)
        self.assertGreater(report.kappa, 0.4)
        self.assertEqual(report.n, 20)

    def test_chance_agreement(self):
        import random
        random.seed(42)
        a = [random.random() > 0.5 for _ in range(100)]
        b = [random.random() > 0.5 for _ in range(100)]
        report = compute_kappa(a, b, threshold=0.7)
        self.assertLess(report.kappa, 0.3)
        self.assertFalse(report.is_acceptable)


class TestClassifyDialogue(unittest.TestCase):
    def test_aggregates_metrics(self):
        history = [
            {"speaker": "A", "target": "B", "reply": "Hi there!"},
            {"speaker": "B", "target": "A", "reply": "You're stupid"},  # violation
            {"speaker": "A", "target": "B", "reply": "Let's discuss."},
            {"speaker": "B", "target": "A", "reply": "I hate you"},      # violation
        ]
        judge = ToxicityJudge(
            client=None,
            mock_responses={"stupid": True, "hate you": True},
        )
        result = classify_dialogue_replies(judge, history)
        self.assertEqual(result["n_replies"], 4)
        self.assertEqual(result["n_violations"], 2)
        self.assertAlmostEqual(result["toxicity_rate"], 0.5)
        self.assertEqual(len(result["per_reply"]), 4)

    def test_empty_history(self):
        judge = ToxicityJudge(client=None)
        result = classify_dialogue_replies(judge, [])
        self.assertEqual(result["n_replies"], 0)
        self.assertEqual(result["toxicity_rate"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
