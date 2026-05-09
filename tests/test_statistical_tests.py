"""
Юнит-тесты для statistical_tests.py.

Идея: для каждой функции — два кейса.
  1. SCENARIO_TRUE_H1: данные генерируются под H1, ожидаем decision='reject_H0'.
  2. SCENARIO_NULL_H0: данные генерируются под H0, ожидаем decision='fail_to_reject_H0'.

Запуск: python -m agent_dialogue_sim.test_statistical_tests
"""

from __future__ import annotations

import unittest

import numpy as np

from agent_dialogue_sim.science.statistical_tests import (
    anova_oneway,
    tukey_hsd_pairs,
    chi_square_goodness_of_fit,
    cox_stuart_trend,
    z_test_two_proportions,
    welch_t_test,
    mann_whitney_u,
    tost_equivalence,
    spearman_correlation,
    cohens_kappa,
    holm_bonferroni,
    evaluate_hypothesis_family,
    cohens_d,
    eta_squared_oneway,
)

RNG = np.random.default_rng(42)


class TestAnova(unittest.TestCase):
    """H1: ANOVA для трёх уровней экстраверсии."""

    def test_groups_differ_strongly(self):
        # Сильный эффект: средние 5, 10, 15, sd=2 → ожидаем reject H0
        rng = np.random.default_rng(1)
        groups = [rng.normal(5, 2, 20), rng.normal(10, 2, 20), rng.normal(15, 2, 20)]
        result = anova_oneway(groups)
        self.assertLess(result.p_value, 0.001)
        self.assertGreater(result.effect_size, 0.5)
        self.assertEqual(result.decision, "reject_H0")
        self.assertEqual(result.effect_size_name, "eta_squared")

    def test_groups_equal(self):
        # Под H0: все три выборки из одного распределения
        rng = np.random.default_rng(2)
        groups = [rng.normal(10, 2, 20), rng.normal(10, 2, 20), rng.normal(10, 2, 20)]
        result = anova_oneway(groups)
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")

    def test_significant_but_small_effect(self):
        # p<alpha но η² < 0.06 — НЕ должны отвергать H0 по правилу
        rng = np.random.default_rng(3)
        # Большая выборка с очень маленькой разницей средних
        groups = [rng.normal(10.0, 2, 500), rng.normal(10.2, 2, 500), rng.normal(10.4, 2, 500)]
        result = anova_oneway(groups)
        # Возможно p<alpha из-за большой N, но эффект медленный
        if result.p_value < 0.05 and result.effect_size < 0.06:
            self.assertEqual(result.decision, "fail_to_reject_H0",
                             "должны не отвергать H0 при малом эффекте")

    def test_tukey_hsd_pairs(self):
        rng = np.random.default_rng(4)
        groups = [rng.normal(5, 2, 20), rng.normal(10, 2, 20), rng.normal(15, 2, 20)]
        pairs = tukey_hsd_pairs(groups, labels=["low", "mid", "high"])
        self.assertEqual(len(pairs), 3)  # 3C2 = 3 пары
        self.assertTrue(all(p["significant"] for p in pairs))


class TestChiSquare(unittest.TestCase):
    """H3: χ² для матрицы переходов марковской цепи."""

    def test_non_uniform_matrix(self):
        # Распределение явно не равномерное
        observed = [50, 10, 10, 10]  # сильное отклонение от равномерного
        expected = [20, 20, 20, 20]
        result = chi_square_goodness_of_fit(observed, expected, alpha=0.01)
        self.assertLess(result.p_value, 0.001)
        self.assertGreater(result.effect_size, 0.3)  # cohens_w
        self.assertEqual(result.decision, "reject_H0")

    def test_uniform_matrix(self):
        # Под H0: распределение равномерное (с шумом)
        observed = [20, 22, 19, 21]
        expected = [20, 20, 20, 20]
        result = chi_square_goodness_of_fit(observed, expected, alpha=0.01)
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")


class TestCoxStuart(unittest.TestCase):
    """H3: тест тренда (forming → performing должен быть монотонный)."""

    def test_increasing_trend(self):
        # 1, 2, 3, ... 20 — явный возрастающий тренд
        values = list(range(1, 21))
        result = cox_stuart_trend(values, alternative="greater")
        self.assertLess(result.p_value, 0.01)
        self.assertEqual(result.decision, "reject_H0")

    def test_no_trend(self):
        rng = np.random.default_rng(5)
        values = rng.normal(10, 2, 30).tolist()
        result = cox_stuart_trend(values, alternative="greater")
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")


class TestZTestTwoProportions(unittest.TestCase):
    """H5: валидация снижает долю токсичности."""

    def test_validation_reduces_toxicity(self):
        # control: 60/300 = 20% toxic; treatment: 9/300 = 3% toxic
        result = z_test_two_proportions(
            successes_a=9, n_a=300,
            successes_b=60, n_b=300,
            alternative="less", alpha=0.01, min_abs_diff=0.05,
        )
        self.assertLess(result.p_value, 0.001)
        self.assertLess(result.effect_size, -0.05)
        self.assertEqual(result.decision, "reject_H0")

    def test_no_validation_effect(self):
        # Обе группы 10% — никакого эффекта валидации
        result = z_test_two_proportions(
            successes_a=30, n_a=300,
            successes_b=30, n_b=300,
            alternative="less", alpha=0.01, min_abs_diff=0.05,
        )
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")

    def test_significant_but_practically_small(self):
        # Огромная выборка, реальный эффект 1% — статистически значим, но мал
        result = z_test_two_proportions(
            successes_a=900, n_a=10000,
            successes_b=1000, n_b=10000,
            alternative="less", alpha=0.01, min_abs_diff=0.05,
        )
        if result.p_value < 0.01 and abs(result.effect_size) < 0.05:
            self.assertEqual(result.decision, "fail_to_reject_H0",
                             "размер эффекта < 5 п.п. → не отвергаем H0")


class TestWelchT(unittest.TestCase):
    """H6: CoT vs baseline по actionability."""

    def test_cot_better_than_baseline(self):
        rng = np.random.default_rng(6)
        cot = rng.normal(0.6, 0.15, 30)
        baseline = rng.normal(0.4, 0.15, 30)
        result = welch_t_test(cot, baseline, alternative="greater")
        self.assertLess(result.p_value, 0.001)
        self.assertGreater(result.effect_size, 0.5)
        self.assertEqual(result.decision, "reject_H0")

    def test_cot_no_difference(self):
        rng = np.random.default_rng(7)
        cot = rng.normal(0.5, 0.15, 30)
        baseline = rng.normal(0.5, 0.15, 30)
        result = welch_t_test(cot, baseline, alternative="greater")
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")


class TestTOST(unittest.TestCase):
    """H7: эквивалентность LLM-Gini и human-Gini."""

    def test_distributions_equivalent(self):
        # Обе выборки имеют близкую медиану ~ 0.3
        rng = np.random.default_rng(8)
        llm = rng.normal(0.30, 0.05, 50)
        human = rng.normal(0.32, 0.05, 50)
        result = tost_equivalence(llm, human, low=-0.1, high=0.1)
        self.assertEqual(result.decision, "reject_H0",
                         "эквивалентность должна быть подтверждена")
        self.assertTrue(result.extra["equivalent"])

    def test_distributions_different(self):
        # LLM = 0.7, human = 0.3 — далеко за пределами Δ=0.1
        rng = np.random.default_rng(9)
        llm = rng.normal(0.70, 0.05, 50)
        human = rng.normal(0.30, 0.05, 50)
        result = tost_equivalence(llm, human, low=-0.1, high=0.1)
        self.assertEqual(result.decision, "fail_to_reject_H0",
                         "эквивалентность должна провалиться при разных распределениях")


class TestMannWhitney(unittest.TestCase):
    def test_different_distributions(self):
        rng = np.random.default_rng(10)
        a = rng.normal(5, 1, 30)
        b = rng.normal(7, 1, 30)
        result = mann_whitney_u(a, b)
        self.assertLess(result.p_value, 0.001)
        self.assertEqual(result.decision, "reject_H0")


class TestSpearman(unittest.TestCase):
    def test_strong_positive_correlation(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, 50)
        y = 0.8 * x + rng.normal(0, 0.5, 50)
        result = spearman_correlation(x, y, alternative="greater")
        self.assertLess(result.p_value, 0.001)
        self.assertGreater(result.effect_size, 0.3)
        self.assertEqual(result.decision, "reject_H0")

    def test_no_correlation(self):
        rng = np.random.default_rng(12)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        result = spearman_correlation(x, y, alternative="two-sided")
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")


class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        a = ["pos", "neg", "pos", "neg", "pos"] * 10
        b = list(a)
        result = cohens_kappa(a, b)
        self.assertAlmostEqual(result.statistic, 1.0, places=5)
        self.assertEqual(result.decision, "reject_H0")

    def test_chance_agreement(self):
        rng = np.random.default_rng(13)
        a = rng.choice(["pos", "neg"], 100).tolist()
        b = rng.choice(["pos", "neg"], 100).tolist()
        result = cohens_kappa(a, b, kappa_threshold=0.7)
        # При случайной разметке κ близок к 0, должны fail_to_reject
        self.assertLess(result.statistic, 0.5)
        self.assertEqual(result.decision, "fail_to_reject_H0")


class TestHolmBonferroni(unittest.TestCase):
    def test_holm_basic(self):
        # 5 p-values: только самые маленькие проходят
        ps = [0.001, 0.008, 0.04, 0.05, 0.5]
        result = holm_bonferroni(ps, alpha=0.05)
        # p=0.001 * 5 = 0.005 < 0.05 → reject
        self.assertTrue(result[0]["reject"])
        # p=0.5 → точно не reject
        self.assertFalse(result[4]["reject"])
        # adjusted монотонно неубывает в порядке отсортированных p
        sorted_adj = sorted([(r["p_raw"], r["p_adj"]) for r in result])
        for i in range(1, len(sorted_adj)):
            self.assertGreaterEqual(sorted_adj[i][1], sorted_adj[i - 1][1])

    def test_holm_all_significant(self):
        # Все p очень маленькие → все reject
        ps = [0.001, 0.001, 0.001, 0.001, 0.001]
        result = holm_bonferroni(ps, alpha=0.05)
        self.assertTrue(all(r["reject"] for r in result))


class TestEffectSizes(unittest.TestCase):
    def test_cohens_d_known(self):
        # μ_a=10, μ_b=8, sd=2 → d = 1.0
        rng = np.random.default_rng(14)
        a = rng.normal(10, 2, 1000)
        b = rng.normal(8, 2, 1000)
        d = cohens_d(a, b)
        self.assertAlmostEqual(d, 1.0, delta=0.1)

    def test_eta_squared_zero_when_equal(self):
        groups = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        eta = eta_squared_oneway(groups)
        self.assertAlmostEqual(eta, 0.0, places=5)


class TestEvaluateFamily(unittest.TestCase):
    """Высокоуровневая интеграция: семейство гипотез H1-H7."""

    def test_family_with_holm(self):
        rng = np.random.default_rng(15)
        # H1 — strong effect
        h1 = anova_oneway([rng.normal(5, 2, 20), rng.normal(10, 2, 20), rng.normal(15, 2, 20)])
        # H6 — strong effect
        h6 = welch_t_test(rng.normal(0.6, 0.15, 30), rng.normal(0.4, 0.15, 30),
                          alternative="greater")
        # Фейковая гипотеза с большим p
        h_null = welch_t_test(rng.normal(0.5, 0.15, 30), rng.normal(0.5, 0.15, 30),
                              alternative="greater")
        family = evaluate_hypothesis_family({"H1": h1, "H6": h6, "Hnull": h_null})
        self.assertEqual(family["H1"]["final_decision"], "reject_H0")
        self.assertEqual(family["H6"]["final_decision"], "reject_H0")
        self.assertEqual(family["Hnull"]["final_decision"], "fail_to_reject_H0")
        # Holm adjusted p-values присутствуют
        for name in ("H1", "H6", "Hnull"):
            self.assertIn("p_adj_holm", family[name])
            self.assertIn("reject_after_holm", family[name])


class TestAssumptionChecking(unittest.TestCase):
    """Тесты для проверки предпосылок параметрических тестов."""

    def test_normal_data_passes(self):
        from agent_dialogue_sim.science.statistical_tests import check_test_assumptions
        rng = np.random.default_rng(20)
        a = rng.normal(10, 2, 50)
        b = rng.normal(10, 2, 50)
        report = check_test_assumptions(a, b)
        self.assertTrue(report.assumptions_met)
        self.assertEqual(len(report.normality_per_group), 2)
        self.assertTrue(all(ng["normal_at_05"] for ng in report.normality_per_group))

    def test_skewed_data_flags_normality(self):
        from agent_dialogue_sim.science.statistical_tests import check_test_assumptions
        rng = np.random.default_rng(21)
        # Сильно скошенное распределение (экспоненциальное)
        a = rng.exponential(2, 100)
        b = rng.exponential(2, 100)
        report = check_test_assumptions(a, b)
        self.assertFalse(report.assumptions_met)
        self.assertIn("non-parametric", report.recommendation.lower())

    def test_unequal_variances_flags_homogeneity(self):
        from agent_dialogue_sim.science.statistical_tests import check_test_assumptions
        rng = np.random.default_rng(22)
        a = rng.normal(10, 1, 50)
        b = rng.normal(10, 5, 50)  # дисперсия в 25 раз больше
        report = check_test_assumptions(a, b)
        self.assertFalse(report.homogeneity["equal_variances_at_05"])
        self.assertIn("Welch", report.recommendation)

    def test_too_small_sample(self):
        from agent_dialogue_sim.science.statistical_tests import check_test_assumptions
        report = check_test_assumptions([1, 2], [3, 4])  # n=2 для каждой
        # Не должно падать
        self.assertEqual(len(report.normality_per_group), 2)
        # n<3 → пропуск shapiro
        self.assertIsNone(report.normality_per_group[0]["normal_at_05"])

    def test_auto_choose_test_picks_welch_for_normal(self):
        from agent_dialogue_sim.science.statistical_tests import auto_choose_test_two_samples
        rng = np.random.default_rng(23)
        a = rng.normal(10, 2, 30)
        b = rng.normal(8, 2, 30)
        result = auto_choose_test_two_samples(a, b, alternative="greater")
        self.assertEqual(result.extra["auto_chosen"], "welch_t_test")
        self.assertIn("assumption_report", result.extra)

    def test_auto_choose_test_picks_mwu_for_skewed(self):
        from agent_dialogue_sim.science.statistical_tests import auto_choose_test_two_samples
        rng = np.random.default_rng(24)
        a = rng.exponential(2, 100)
        b = rng.exponential(3, 100)
        result = auto_choose_test_two_samples(a, b)
        self.assertEqual(result.extra["auto_chosen"], "mann_whitney_u")
        self.assertTrue(result.extra.get("normality_violated"))


class TestMixedEffectsModel(unittest.TestCase):
    """Тесты иерархической модели — реплики внутри диалога."""

    def test_detects_treatment_effect_with_clustered_data(self):
        from agent_dialogue_sim.science.statistical_tests import mixed_effects_two_groups
        rng = np.random.default_rng(30)
        # 20 диалогов, по 10 реплик в каждом, эффект treatment = +0.5 + диалоговый шум
        values, groups, clusters = [], [], []
        for d_idx in range(40):
            dialogue_offset = rng.normal(0, 0.3)  # random intercept
            grp = "treatment" if d_idx < 20 else "control"
            grp_effect = 0.5 if grp == "treatment" else 0.0
            for _ in range(10):
                values.append(0.5 + grp_effect + dialogue_offset + rng.normal(0, 0.2))
                groups.append(grp)
                clusters.append(d_idx)
        result = mixed_effects_two_groups(values, groups, clusters, alpha=0.05)
        self.assertEqual(result.test_name, "mixed_effects_lmm")
        self.assertLess(result.p_value, 0.05)
        self.assertEqual(result.extra["n_clusters"], 40)

    def test_no_effect_fails_to_reject(self):
        from agent_dialogue_sim.science.statistical_tests import mixed_effects_two_groups
        rng = np.random.default_rng(31)
        values, groups, clusters = [], [], []
        for d_idx in range(30):
            dialogue_offset = rng.normal(0, 0.3)
            grp = "treatment" if d_idx < 15 else "control"
            for _ in range(8):
                values.append(0.5 + dialogue_offset + rng.normal(0, 0.2))
                groups.append(grp)
                clusters.append(d_idx)
        result = mixed_effects_two_groups(values, groups, clusters, alpha=0.05)
        self.assertGreater(result.p_value, 0.05)
        self.assertEqual(result.decision, "fail_to_reject_H0")

    def test_random_intercept_variance_estimated(self):
        from agent_dialogue_sim.science.statistical_tests import mixed_effects_two_groups
        rng = np.random.default_rng(32)
        values, groups, clusters = [], [], []
        for d_idx in range(20):
            offset = rng.normal(0, 1.0)  # большая дисперсия между диалогами
            grp = "a" if d_idx < 10 else "b"
            for _ in range(8):
                values.append(offset + rng.normal(0, 0.1))
                groups.append(grp)
                clusters.append(d_idx)
        result = mixed_effects_two_groups(values, groups, clusters)
        var = result.extra.get("random_intercept_var")
        self.assertIsNotNone(var)
        self.assertGreater(var, 0.1)

    def test_validation_2_groups_required(self):
        from agent_dialogue_sim.science.statistical_tests import mixed_effects_two_groups
        with self.assertRaises(ValueError):
            mixed_effects_two_groups(
                [1, 2, 3, 4, 5, 6],
                ["a", "a", "b", "b", "c", "c"],  # 3 группы
                [1, 1, 2, 2, 3, 3],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
