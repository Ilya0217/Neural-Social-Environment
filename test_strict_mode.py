"""
Юнит-тесты для strict_mode в DialogueManager.

Цель: убедиться, что при API-сбоях strict_mode=True поднимает RuntimeError,
а strict_mode=False (default) генерирует fallback-ответ.

НЕ требует реальных LLM-вызовов — model_turn мокается.

Запуск: python -m agent_dialogue_sim.test_strict_mode
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from agent_dialogue_sim.agents import Agent
from agent_dialogue_sim.dialogue_manager import DialogueManager


def _make_agents() -> list[Agent]:
    return [
        Agent(name="Alex", nature="explorer", color="#000"),
        Agent(name="Sam", nature="facilitator", color="#111"),
        Agent(name="Jordan", nature="critic", color="#222"),
    ]


def _make_dm(strict_mode: bool, enable_validation: bool = False) -> DialogueManager:
    client = MagicMock()  # клиент не вызывается напрямую — model_turn мокается
    return DialogueManager(
        client=client,
        agents=_make_agents(),
        env_context="test environment",
        human_io=None,
        enable_validation=enable_validation,
        strict_mode=strict_mode,
    )


class TestStrictModeAPIFailure(unittest.TestCase):
    """API-сбой в model_turn → RuntimeError при strict_mode=True, fallback при False."""

    def test_strict_mode_raises_on_api_failure(self):
        dm = _make_dm(strict_mode=True)
        # model_turn кидает RuntimeError (как делает реальный код при исчерпании
        # всех попыток API-вызовов)
        with patch.object(dm, "model_turn", side_effect=RuntimeError("API error: 402")):
            with self.assertRaises(RuntimeError) as ctx:
                dm.step()
            self.assertIn("strict_mode=True", str(ctx.exception))
            self.assertIn("no fallback", str(ctx.exception))

    def test_default_mode_falls_back_on_api_failure(self):
        dm = _make_dm(strict_mode=False)
        with patch.object(dm, "model_turn", side_effect=RuntimeError("API error: 402")):
            # должно вернуть fallback вместо raise
            record, _ = dm.step()
            self.assertIn("having trouble responding", record["reply"])
            self.assertEqual(record["tone"], "neutral")

    def test_history_not_polluted_in_strict_mode(self):
        """Главное свойство для научной валидности: в strict_mode сбой ломает диалог,
        но не оставляет в self.history fallback-реплику."""
        dm = _make_dm(strict_mode=True)
        initial_history_len = len(dm.history)
        with patch.object(dm, "model_turn", side_effect=RuntimeError("API down")):
            try:
                dm.step()
            except RuntimeError:
                pass
        # history не должна вырасти на одну fallback-запись
        self.assertEqual(len(dm.history), initial_history_len,
                         "strict_mode must not append fallback turns to history")

    def test_default_mode_appends_fallback_to_history(self):
        """Контрольный тест: при strict_mode=False fallback ДОЛЖЕН попасть в history
        (это и есть текущий проблемный сценарий, который мы фиксим только в strict_mode)."""
        dm = _make_dm(strict_mode=False)
        with patch.object(dm, "model_turn", side_effect=RuntimeError("API down")):
            dm.step()
        self.assertEqual(len(dm.history), 1)
        self.assertIn("having trouble", dm.history[0]["reply"])


class TestStrictModeFlag(unittest.TestCase):
    """Поле strict_mode корректно прокидывается в dataclass."""

    def test_default_value_is_false(self):
        dm = _make_dm(strict_mode=False)
        self.assertFalse(dm.strict_mode)

    def test_can_set_true(self):
        dm = _make_dm(strict_mode=True)
        self.assertTrue(dm.strict_mode)


class TestRunnerIntegration(unittest.TestCase):
    """experiment_runner._run_arm должен корректно ловить RuntimeError из strict_mode."""

    def test_run_arm_records_error_on_strict_failure(self):
        from agent_dialogue_sim.experiment_runner import (
            ArmConfig, DialogueConfig, AnalysisConfig, ExperimentConfig,
            ExperimentRunner,
        )

        # Кастомный диспатчер, который имитирует API-сбой (как было бы в strict_mode)
        def failing_dispatcher(seed, arm, dialogue_config):
            raise RuntimeError(f"simulated API failure for seed={seed}")

        cfg = ExperimentConfig(
            id="strict_failure_test",
            hypothesis="HX",
            description="",
            arms=[ArmConfig(name="a", n_dialogues=3)],
            dialogue=DialogueConfig(n_turns=5),
            analysis=AnalysisConfig(test="welch_t_test", metric="actionability_rate"),
        )
        runner = ExperimentRunner(cfg, dispatcher=failing_dispatcher)

        import tempfile, shutil
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        try:
            results = runner._run_arm(cfg.arms[0])
            # все 3 диалога должны быть помечены как error и не упасть всей серии
            self.assertEqual(len(results), 3)
            for r in results:
                self.assertIn("error", r)
                self.assertIn("simulated API failure", r["error"])
            # _extract_metric должна корректно отфильтровать error-диалоги
            metric_values = runner._extract_metric(results)
            self.assertEqual(len(metric_values), 0,
                             "error dialogues must be excluded from metric extraction")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
