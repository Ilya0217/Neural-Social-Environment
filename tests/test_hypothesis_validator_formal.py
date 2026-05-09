"""
Тесты формального статистического слоя hypothesis_validator.py.

Цель: убедиться что _compute_formal_stats() возвращает корректный p-value/effect
для разных категорий гипотез на синтетическом metrics_history.

Не подменяет существующие эвристические тесты — проверяет только новый формальный путь.
"""

from __future__ import annotations

import unittest

from agent_dialogue_sim.analytics.hypothesis_validator import HypothesisValidator


def _snapshot(turn: int, **scientific_kwargs) -> dict:
    """Build a minimal metrics dict matching expected structure."""
    return {
        "summary": {
            "window": {"avg_tone": scientific_kwargs.pop("avg_tone", 0.0),
                       "actionability_rate": scientific_kwargs.pop("actionability_rate", 0.5)},
            "all": {},
        },
        "scientific": scientific_kwargs,
    }


class TestFormalNetworkStructure(unittest.TestCase):
    def test_high_centrality_rejects_H0(self):
        v = HypothesisValidator()
        # 5 снимков с высокой центральностью (Sam ≈ 0.85, реалистичный шум)
        for t, c in enumerate([0.83, 0.85, 0.88, 0.81, 0.86]):
            v.add_metrics_snapshot(t, {
                "summary": {"window": {}, "all": {}},
                "scientific": {"centrality_metrics": {"in_degree": {"Sam": c, "Alex": 0.4, "Jordan": 0.3}}},
            })
        result = v._compute_formal_stats("Network Structure", "Sam dominates")
        self.assertIsNotNone(result)
        self.assertEqual(result["test_used"], "one_sample_t_test")
        self.assertLess(result["p_value"], 0.05)
        self.assertGreater(result["effect_size"], 0.5)
        self.assertEqual(result["decision_formal"], "reject_H0")

    def test_balanced_centrality_fails_to_reject(self):
        v = HypothesisValidator()
        for t, c in enumerate([0.42, 0.40, 0.43, 0.39, 0.41]):
            v.add_metrics_snapshot(t, {
                "summary": {"window": {}, "all": {}},
                "scientific": {"centrality_metrics": {"in_degree": {"Sam": c, "Alex": 0.4, "Jordan": 0.4}}},
            })
        result = v._compute_formal_stats("Network Structure", "balanced")
        self.assertIsNotNone(result)
        self.assertEqual(result["decision_formal"], "fail_to_reject_H0")

    def test_insufficient_history_returns_none(self):
        v = HypothesisValidator()
        v.add_metrics_snapshot(0, {"summary": {"window": {}, "all": {}}, "scientific": {}})
        result = v._compute_formal_stats("Network Structure", "x")
        self.assertIsNone(result)


class TestFormalEmotionalClimate(unittest.TestCase):
    def test_consistently_positive_tone_rejects_H0(self):
        v = HypothesisValidator()
        for t, tone in enumerate([0.4, 0.5, 0.45, 0.6, 0.5]):
            v.add_metrics_snapshot(t, _snapshot(t, avg_tone=tone))
        result = v._compute_formal_stats("Emotional Climate", "")
        self.assertIsNotNone(result)
        self.assertLess(result["p_value"], 0.05)
        self.assertEqual(result["decision_formal"], "reject_H0")

    def test_neutral_tone_fails_to_reject(self):
        v = HypothesisValidator()
        for t, tone in enumerate([0.0, 0.05, -0.02, 0.01, -0.03]):
            v.add_metrics_snapshot(t, _snapshot(t, avg_tone=tone))
        result = v._compute_formal_stats("Emotional Climate", "")
        self.assertEqual(result["decision_formal"], "fail_to_reject_H0")


class TestFormalSocialCapital(unittest.TestCase):
    def test_high_cohesion_rejects(self):
        v = HypothesisValidator()
        for t, c in enumerate([0.82, 0.79, 0.85, 0.78, 0.81]):
            v.add_metrics_snapshot(t, _snapshot(t, group_cohesion=c))
        result = v._compute_formal_stats("Social Capital", "")
        self.assertEqual(result["decision_formal"], "reject_H0")


class TestFormalTurnTaking(unittest.TestCase):
    def test_high_gini_rejects(self):
        v = HypothesisValidator()
        for t, g in enumerate([0.72, 0.68, 0.75, 0.69, 0.73]):
            v.add_metrics_snapshot(t, _snapshot(t, turn_taking={"gini_coefficient": g}))
        result = v._compute_formal_stats("Turn-Taking", "")
        self.assertEqual(result["decision_formal"], "reject_H0")

    def test_balanced_gini_fails_to_reject(self):
        v = HypothesisValidator()
        for t, g in enumerate([0.32, 0.28, 0.31, 0.29, 0.30]):
            v.add_metrics_snapshot(t, _snapshot(t, turn_taking={"gini_coefficient": g}))
        result = v._compute_formal_stats("Turn-Taking", "")
        self.assertEqual(result["decision_formal"], "fail_to_reject_H0")


class TestFormalGroupDevelopment(unittest.TestCase):
    def test_dominant_stage_rejects(self):
        v = HypothesisValidator()
        for t in range(8):
            v.add_metrics_snapshot(t, _snapshot(t, group_stage={"stage": "norming", "confidence": 0.7}))
        result = v._compute_formal_stats("Group Development", "")
        self.assertEqual(result["decision_formal"], "reject_H0")
        self.assertEqual(result["test_used"], "binomial_test")

    def test_random_stages_fails(self):
        v = HypothesisValidator()
        # 4 разные стадии равномерно
        for t, stage in enumerate(["forming", "storming", "norming", "performing"] * 2):
            v.add_metrics_snapshot(t, _snapshot(t, group_stage={"stage": stage}))
        result = v._compute_formal_stats("Group Development", "")
        self.assertEqual(result["decision_formal"], "fail_to_reject_H0")


class TestFormalDialogueStructure(unittest.TestCase):
    def test_increasing_actionability_detects_trend(self):
        v = HypothesisValidator()
        for t, ar in enumerate([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            v.add_metrics_snapshot(t, _snapshot(t, actionability_rate=ar))
        result = v._compute_formal_stats("Dialogue Structure", "")
        self.assertIsNotNone(result)
        self.assertEqual(result["test_used"], "spearman_correlation")
        self.assertGreater(result["effect_size"], 0.9)
        self.assertEqual(result["decision_formal"], "reject_H0")

    def test_no_trend(self):
        v = HypothesisValidator()
        for t, ar in enumerate([0.5, 0.5, 0.5, 0.5, 0.5]):
            v.add_metrics_snapshot(t, _snapshot(t, actionability_rate=ar))
        result = v._compute_formal_stats("Dialogue Structure", "")
        # При плоской серии — может быть None из-за NaN корреляции, но не reject
        if result is not None:
            self.assertEqual(result["decision_formal"], "fail_to_reject_H0")


class TestEndToEndAttachment(unittest.TestCase):
    """Полный путь: validate_hypothesis() прикрепляет formal stats."""

    def test_validate_attaches_p_value(self):
        v = HypothesisValidator()
        for t in range(5):
            v.add_metrics_snapshot(t, _snapshot(t, avg_tone=0.5,
                                                 centrality_metrics={"in_degree": {"Sam": 0.4}},
                                                 group_stage={"stage": "norming", "confidence": 0.7},
                                                 group_cohesion=0.7))
        hyp = {
            "framework": "Cowen-Keltner",
            "category": "Emotional Climate",
            "finding": "Positive atmosphere",
        }
        result = v.validate_hypothesis(hyp, current_metrics=_snapshot(5, avg_tone=0.5))
        self.assertIsNotNone(result.p_value, "formal p_value must be attached")
        self.assertEqual(result.test_used, "one_sample_t_test")
        self.assertIn("Формальный тест", "\n".join(result.evidence))


if __name__ == "__main__":
    unittest.main(verbosity=2)
