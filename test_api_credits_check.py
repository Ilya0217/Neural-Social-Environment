"""
Юнит-тесты для api_credits_check.py.
Не делает реальных HTTP-вызовов — мокает urlopen.

Запуск: python -m agent_dialogue_sim.test_api_credits_check
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch, MagicMock

from agent_dialogue_sim.api_credits_check import (
    CreditsInfo,
    CreditsCheckError,
    CreditsExhaustedError,
    check_credits,
    ensure_sufficient,
    estimate_experiment_cost,
    _is_openrouter,
)


def _mock_urlopen_response(payload: dict):
    """Контекст-менеджер, имитирующий http.client.HTTPResponse."""
    resp = MagicMock()
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


class TestIsOpenRouter(unittest.TestCase):
    def test_openrouter_url(self):
        self.assertTrue(_is_openrouter("https://openrouter.ai/api/v1"))
        self.assertTrue(_is_openrouter("https://OPENROUTER.AI/api/v1"))

    def test_openai_url_is_not_openrouter(self):
        self.assertFalse(_is_openrouter("https://api.openai.com/v1"))

    def test_empty_url(self):
        self.assertFalse(_is_openrouter(""))
        self.assertFalse(_is_openrouter(None))


class TestCheckCredits(unittest.TestCase):
    def test_returns_remaining_balance(self):
        payload = {"data": {"total_credits": 5.00, "total_usage": 1.23}}
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=_mock_urlopen_response(payload)):
            info = check_credits(api_key="sk-or-v1-test",
                                  base_url="https://openrouter.ai/api/v1")
        self.assertTrue(info.is_openrouter)
        self.assertAlmostEqual(info.total_credits, 5.00)
        self.assertAlmostEqual(info.total_usage, 1.23)
        self.assertAlmostEqual(info.remaining_usd, 3.77)

    def test_skip_when_not_openrouter(self):
        # base_url не OpenRouter → возвращаем infinity, не блокируем
        info = check_credits(api_key="sk-test", base_url="https://api.openai.com/v1")
        self.assertFalse(info.is_openrouter)
        self.assertEqual(info.remaining_usd, float("inf"))

    def test_missing_api_key_raises(self):
        with self.assertRaises(CreditsCheckError) as ctx:
            check_credits(api_key="", base_url="https://openrouter.ai/api/v1")
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    def test_invalid_json_raises(self):
        bad_resp = MagicMock()
        bad_resp.read.return_value = b"not json at all"
        bad_resp.__enter__.return_value = bad_resp
        bad_resp.__exit__.return_value = False
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=bad_resp):
            with self.assertRaises(CreditsCheckError):
                check_credits(api_key="sk-or-v1-test",
                              base_url="https://openrouter.ai/api/v1")


class TestEnsureSufficient(unittest.TestCase):
    def test_passes_when_balance_sufficient(self):
        payload = {"data": {"total_credits": 5.00, "total_usage": 1.0}}
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=_mock_urlopen_response(payload)):
            info = ensure_sufficient(min_usd=0.50,
                                      api_key="sk-or-v1-test",
                                      base_url="https://openrouter.ai/api/v1")
        self.assertEqual(info.remaining_usd, 4.0)

    def test_raises_when_balance_insufficient(self):
        payload = {"data": {"total_credits": 1.0, "total_usage": 0.95}}
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=_mock_urlopen_response(payload)):
            with self.assertRaises(CreditsExhaustedError) as ctx:
                ensure_sufficient(min_usd=0.50,
                                   api_key="sk-or-v1-test",
                                   base_url="https://openrouter.ai/api/v1")
        # remaining = 0.05, min = 0.50 → должен сработать
        self.assertIn("0.50", str(ctx.exception))
        self.assertIn("openrouter.ai/settings/credits", str(ctx.exception))

    def test_raises_on_negative_balance(self):
        # Случай 2026-04-26: usage > total_credits
        payload = {"data": {"total_credits": 0.0, "total_usage": 0.23}}
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=_mock_urlopen_response(payload)):
            with self.assertRaises(CreditsExhaustedError):
                ensure_sufficient(min_usd=0.10,
                                   api_key="sk-or-v1-test",
                                   base_url="https://openrouter.ai/api/v1")


class TestEstimateCost(unittest.TestCase):
    def test_h6_full_run_estimate(self):
        # 110 диалогов × 20 ходов = 2200 calls
        cost = estimate_experiment_cost(n_dialogues=110, n_turns=20)
        # Ожидаем что-то около $0.50–0.60
        self.assertGreater(cost, 0.40)
        self.assertLess(cost, 1.00)

    def test_h1_full_run_estimate(self):
        # 480 диалогов × 20 ходов = 9600 calls
        cost = estimate_experiment_cost(n_dialogues=480, n_turns=20)
        self.assertGreater(cost, 1.5)
        self.assertLess(cost, 5.0)

    def test_smoke_test_estimate(self):
        # 4 диалога × 4 хода = 16 calls
        cost = estimate_experiment_cost(n_dialogues=4, n_turns=4)
        # Должно быть < $0.01
        self.assertLess(cost, 0.01)


class TestRunnerCreditsIntegration(unittest.TestCase):
    """ExperimentRunner должен вызывать pre-flight check и блокировать запуск
    при недостатке баланса."""

    def test_runner_blocks_on_insufficient_balance(self):
        from agent_dialogue_sim.experiment_runner import (
            ArmConfig, DialogueConfig, AnalysisConfig, ExperimentConfig,
            ExperimentRunner, real_dispatcher,
        )
        cfg = ExperimentConfig(
            id="test_credits_block",
            hypothesis="HX",
            description="",
            arms=[ArmConfig(name="a", n_dialogues=110)],
            dialogue=DialogueConfig(n_turns=20),
            analysis=AnalysisConfig(test="welch_t_test", metric="actionability_rate"),
        )
        runner = ExperimentRunner(cfg, dispatcher=real_dispatcher,
                                   check_credits=True)
        # Mock баланс = $0
        payload = {"data": {"total_credits": 0.0, "total_usage": 0.0}}
        with patch("agent_dialogue_sim.api_credits_check.urlopen",
                   return_value=_mock_urlopen_response(payload)):
            # Также мокаем base_url через config
            with patch("agent_dialogue_sim.api_credits_check._is_openrouter",
                       return_value=True):
                from agent_dialogue_sim.api_credits_check import CreditsExhaustedError
                with self.assertRaises(CreditsExhaustedError):
                    runner._preflight_credits()

    def test_runner_skips_check_when_disabled(self):
        from agent_dialogue_sim.experiment_runner import (
            ArmConfig, DialogueConfig, AnalysisConfig, ExperimentConfig,
            ExperimentRunner, real_dispatcher,
        )
        cfg = ExperimentConfig(
            id="test_skip", hypothesis="HX", description="",
            arms=[ArmConfig(name="a", n_dialogues=10)],
            dialogue=DialogueConfig(n_turns=5),
            analysis=AnalysisConfig(test="welch_t_test", metric="actionability_rate"),
        )
        runner = ExperimentRunner(cfg, dispatcher=real_dispatcher,
                                   check_credits=False)
        # urlopen НЕ должен вызываться
        with patch("agent_dialogue_sim.api_credits_check.urlopen") as mock_urlopen:
            runner._preflight_credits()
            mock_urlopen.assert_not_called()

    def test_runner_skips_for_mock_dispatcher(self):
        from agent_dialogue_sim.experiment_runner import (
            ArmConfig, DialogueConfig, AnalysisConfig, ExperimentConfig,
            ExperimentRunner, mock_dispatcher_factory,
        )
        cfg = ExperimentConfig(
            id="test_mock_skip", hypothesis="HX", description="",
            arms=[ArmConfig(name="a", n_dialogues=10)],
            dialogue=DialogueConfig(n_turns=5),
            analysis=AnalysisConfig(test="welch_t_test", metric="actionability_rate"),
        )
        dispatcher = mock_dispatcher_factory({"a": 0.5}, metric_name="actionability_rate")
        runner = ExperimentRunner(cfg, dispatcher=dispatcher, check_credits=True)
        # urlopen НЕ должен вызываться даже при check_credits=True для mock
        with patch("agent_dialogue_sim.api_credits_check.urlopen") as mock_urlopen:
            runner._preflight_credits()
            mock_urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
