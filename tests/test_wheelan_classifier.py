"""
Юнит-тесты для wheelan_classifier.py.

Синтетические диалоги с принудительными паттернами:
  - storming pattern: много "but/disagree/no"
  - norming pattern: много "agree/let's/together"
  - performing pattern: высокий task + взаимность + actionability
  - forming pattern: много вопросов, низкая task-доля
"""

from __future__ import annotations

import unittest

from agent_dialogue_sim.science.wheelan_classifier import (
    StageFeatures,
    classify_stage,
    extract_features,
    classify_dialogue_windows,
    stage_sequence,
    transition_matrix,
)


def _msg(speaker: str, target: str, reply: str, tone: str = "neutral") -> dict:
    return {"speaker": speaker, "target": target, "reply": reply,
            "tone": tone, "emotion": "neutral"}


def _storming_dialogue(n: int = 10) -> list[dict]:
    msgs = []
    for i in range(n):
        if i % 3 == 0:
            msgs.append(_msg("Alex", "Sam",
                              "I disagree with that approach. The problem is it's wrong.",
                              tone="negative"))
        elif i % 3 == 1:
            msgs.append(_msg("Sam", "Alex",
                              "But that's not right either. There's an issue with your idea.",
                              tone="negative"))
        else:
            msgs.append(_msg("Jordan", "Alex",
                              "I'm against this. It won't work, however we look at it.",
                              tone="negative"))
    return msgs


def _norming_dialogue(n: int = 10) -> list[dict]:
    msgs = []
    for i in range(n):
        if i % 3 == 0:
            msgs.append(_msg("Alex", "Sam",
                              "I agree with you. Let's work together on this. Sounds good!",
                              tone="positive"))
        elif i % 3 == 1:
            msgs.append(_msg("Sam", "Alex",
                              "Good idea! I'm on board. Let's cooperate on the next step.",
                              tone="positive"))
        else:
            msgs.append(_msg("Jordan", "Sam",
                              "Agree. Let's together plan our work. Perfect approach.",
                              tone="positive"))
    return msgs


def _performing_dialogue(n: int = 10) -> list[dict]:
    """Высокий task ratio (модальные глаголы), reciprocity (A→B and B→A), actionability."""
    pairs = [("Alex", "Sam"), ("Sam", "Alex"), ("Alex", "Jordan"), ("Jordan", "Alex"),
             ("Sam", "Jordan"), ("Jordan", "Sam")]
    msgs = []
    for i in range(n):
        s, t = pairs[i % len(pairs)]
        msgs.append(_msg(s, t,
                          f"We should plan the next step. I will own the deadline. "
                          f"How will we measure that?",
                          tone="neutral"))
    return msgs


def _forming_dialogue(n: int = 10) -> list[dict]:
    msgs = []
    for i in range(n):
        msgs.append(_msg("Alex", "Sam",
                          f"Hi everyone, who are you? What is this about?",
                          tone="neutral"))
    return msgs


class TestStormingDetection(unittest.TestCase):
    def test_classifies_storming(self):
        feats = extract_features(_storming_dialogue(10))
        result = classify_stage(feats)
        self.assertEqual(result.stage, "storming")
        self.assertGreater(result.confidence, 0.3)
        # Ожидаем явный negative_social ИЛИ conflict_word
        self.assertTrue(feats.negative_social_ratio > 0.25
                         or feats.conflict_word_rate > 0.20)


class TestNormingDetection(unittest.TestCase):
    def test_classifies_norming(self):
        feats = extract_features(_norming_dialogue(10))
        result = classify_stage(feats)
        self.assertEqual(result.stage, "norming")
        self.assertGreater(feats.positive_social_ratio, 0.3)
        self.assertGreater(feats.cooperation_word_rate, 0.15)


class TestPerformingDetection(unittest.TestCase):
    def test_classifies_performing(self):
        feats = extract_features(_performing_dialogue(10))
        result = classify_stage(feats)
        self.assertEqual(result.stage, "performing")
        self.assertGreater(feats.reciprocity, 0.5)
        self.assertGreater(feats.actionability_rate, 0.4)


class TestFormingDetection(unittest.TestCase):
    def test_default_to_forming(self):
        feats = extract_features(_forming_dialogue(10))
        result = classify_stage(feats)
        self.assertEqual(result.stage, "forming")


class TestEmptyAndEdgeCases(unittest.TestCase):
    def test_empty_window(self):
        feats = extract_features([])
        result = classify_stage(feats)
        self.assertEqual(result.stage, "forming")
        self.assertEqual(feats.n_turns, 0)

    def test_single_message(self):
        feats = extract_features([_msg("Alex", "Sam", "Hi.")])
        result = classify_stage(feats)
        # Один ход с приветствием → forming
        self.assertEqual(result.stage, "forming")


class TestWindowedClassification(unittest.TestCase):
    def test_dialogue_with_phase_transitions(self):
        # forming → storming → norming → performing
        history = (_forming_dialogue(10) + _storming_dialogue(10)
                   + _norming_dialogue(10) + _performing_dialogue(10))
        seq = stage_sequence(history, window_size=10, step=5)
        # Должны увидеть как минимум 2-3 разные стадии
        unique = set(seq)
        self.assertGreaterEqual(len(unique), 3,
                                 f"expected ≥3 distinct stages, got: {seq}")
        # Первое окно — forming
        self.assertEqual(seq[0], "forming")

    def test_short_history_returns_single_classification(self):
        history = _forming_dialogue(3)
        results = classify_dialogue_windows(history, window_size=10, step=5)
        self.assertEqual(len(results), 1)


class TestTransitionMatrix(unittest.TestCase):
    def test_count_transitions(self):
        stages = ["forming", "forming", "storming", "storming", "norming",
                   "performing", "performing"]
        m = transition_matrix(stages)
        self.assertEqual(m[("forming", "forming")], 1)
        self.assertEqual(m[("forming", "storming")], 1)
        self.assertEqual(m[("storming", "storming")], 1)
        self.assertEqual(m[("storming", "norming")], 1)
        self.assertEqual(m[("norming", "performing")], 1)
        self.assertEqual(m[("performing", "performing")], 1)


class TestIntegrationWithStatTests(unittest.TestCase):
    """Совместное использование с statistical_tests для H3."""

    def test_chi_square_on_transition_matrix(self):
        """Воспроизводит проверку H3: матрица переходов vs равномерная."""
        from agent_dialogue_sim.science.statistical_tests import chi_square_goodness_of_fit
        # Симулируем 30 диалогов, в каждом 10 переходов между стадиями
        # Один сильно forming-доминирующий, остальное случайное
        stages_dominant_forming = ["forming"] * 8 + ["storming"] * 1 + ["norming"] * 1
        m = transition_matrix(stages_dominant_forming)
        # Готовим observed/expected для χ²: 4 стадии × 4 → 16 ячеек
        all_stages = ["forming", "storming", "norming", "performing"]
        observed = []
        expected = []
        for a in all_stages:
            for b in all_stages:
                observed.append(m.get((a, b), 0))
                expected.append(1.0)  # равномерное предположение
        result = chi_square_goodness_of_fit(observed, expected, alpha=0.05)
        self.assertEqual(result.test_name, "chi_square_gof")
        # Не проверяем конкретный decision (мало данных), но что тест запустился
        self.assertIsNotNone(result.p_value)


if __name__ == "__main__":
    unittest.main(verbosity=2)
