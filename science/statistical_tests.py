"""
Статистические тесты для проверки гипотез H1, H3, H5, H6, H7 из preregistration.md.

Все функции возвращают TestResult — единый dataclass с p-value, размером эффекта,
доверительным интервалом и решением по H0 при заданном alpha.

Решающее правило везде: H0 отвергается, если p < alpha И размер эффекта >= порога.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Sequence, Literal, Iterable

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.weightstats import ttest_ind as sm_ttest_ind


Alternative = Literal["two-sided", "greater", "less"]


def _to_statsmodels_alt(alt: Alternative) -> str:
    """scipy convention ('greater'/'less') -> statsmodels ('larger'/'smaller')."""
    return {"two-sided": "two-sided", "greater": "larger", "less": "smaller"}[alt]


@dataclass
class TestResult:
    """Унифицированный результат любого статистического теста."""
    test_name: str
    statistic: float
    p_value: float
    n: int | tuple[int, ...]
    effect_size: float | None = None
    effect_size_name: str | None = None  # "cohens_d", "eta_squared", "cramers_w", "phi", ...
    ci_95: tuple[float, float] | None = None
    alpha: float = 0.05
    decision: Literal["reject_H0", "fail_to_reject_H0"] = "fail_to_reject_H0"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


# ---------------------------------------------------------------------------
# Effect-size helpers
# ---------------------------------------------------------------------------

def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d для двух независимых выборок (pooled SD)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def eta_squared_oneway(groups: Sequence[Sequence[float]]) -> float:
    """η² для однофакторного ANOVA: SS_between / SS_total."""
    flat = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand_mean = flat.mean()
    ss_total = float(((flat - grand_mean) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = float(sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups))
    return ss_between / ss_total


def cramers_w(observed: np.ndarray, expected: np.ndarray) -> float:
    """Cohen's w (≡ Cramér's φ для goodness-of-fit) — размер эффекта для χ²."""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    n = observed.sum()
    if n == 0:
        return float("nan")
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    return math.sqrt(chi2 / n)


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation для Mann–Whitney U."""
    if n1 == 0 or n2 == 0:
        return float("nan")
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def _decision(p: float, alpha: float, effect: float | None,
              effect_threshold: float | None) -> Literal["reject_H0", "fail_to_reject_H0"]:
    if math.isnan(p):
        return "fail_to_reject_H0"
    if p >= alpha:
        return "fail_to_reject_H0"
    if effect_threshold is None or effect is None or math.isnan(effect):
        return "reject_H0"
    return "reject_H0" if abs(effect) >= effect_threshold else "fail_to_reject_H0"


# ---------------------------------------------------------------------------
# H1 — ANOVA + Tukey HSD (влияние OCEAN-уровня на длину реплики и т.п.)
# ---------------------------------------------------------------------------

def anova_oneway(groups: Sequence[Sequence[float]],
                 alpha: float = 0.05,
                 eta_sq_threshold: float = 0.06) -> TestResult:
    """
    H0: μ_1 = μ_2 = ... = μ_k.
    Решение: reject H0 если p<alpha И η² ≥ eta_sq_threshold (medium по Cohen).
    """
    arrs = [np.asarray(g, dtype=float) for g in groups]
    if any(len(a) < 2 for a in arrs):
        raise ValueError("each group needs >= 2 observations")
    f_stat, p = stats.f_oneway(*arrs)
    eta2 = eta_squared_oneway(arrs)
    return TestResult(
        test_name="anova_oneway",
        statistic=float(f_stat),
        p_value=float(p),
        n=tuple(len(a) for a in arrs),
        effect_size=eta2,
        effect_size_name="eta_squared",
        alpha=alpha,
        decision=_decision(p, alpha, eta2, eta_sq_threshold),
        extra={"eta_sq_threshold": eta_sq_threshold, "k_groups": len(arrs)},
    )


def tukey_hsd_pairs(groups: Sequence[Sequence[float]],
                    labels: Sequence[str] | None = None,
                    alpha: float = 0.05) -> list[dict]:
    """
    Пост-хок Tukey HSD после ANOVA. Возвращает список парных сравнений.
    Использует scipy.stats.tukey_hsd (>=1.8).
    """
    res = stats.tukey_hsd(*groups)
    if labels is None:
        labels = [f"g{i}" for i in range(len(groups))]
    out = []
    k = len(groups)
    for i in range(k):
        for j in range(i + 1, k):
            out.append({
                "pair": (labels[i], labels[j]),
                "mean_diff": float(res.statistic[i, j]),
                "p_value": float(res.pvalue[i, j]),
                "ci_low": float(res.confidence_interval(confidence_level=1 - alpha).low[i, j]),
                "ci_high": float(res.confidence_interval(confidence_level=1 - alpha).high[i, j]),
                "significant": float(res.pvalue[i, j]) < alpha,
            })
    return out


# ---------------------------------------------------------------------------
# H3 — χ² goodness-of-fit для матрицы переходов марковской цепи + Cox–Stuart trend
# ---------------------------------------------------------------------------

def chi_square_goodness_of_fit(observed: Sequence[float],
                               expected: Sequence[float],
                               alpha: float = 0.01,
                               w_threshold: float = 0.3) -> TestResult:
    """
    H0: наблюдаемое распределение совпадает с expected.
    Размер эффекта: Cohen's w.
    """
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have same shape")
    # scipy.chisquare требует одинаковых сумм; масштабируем expected к total observed
    obs_sum = observed.sum()
    exp_sum = expected.sum()
    if exp_sum == 0:
        raise ValueError("expected frequencies sum to 0")
    expected_scaled = expected * (obs_sum / exp_sum)
    chi2, p = stats.chisquare(f_obs=observed, f_exp=expected_scaled)
    w = cramers_w(observed, expected_scaled)
    return TestResult(
        test_name="chi_square_gof",
        statistic=float(chi2),
        p_value=float(p),
        n=int(observed.sum()),
        effect_size=w,
        effect_size_name="cohens_w",
        alpha=alpha,
        decision=_decision(p, alpha, w, w_threshold),
        extra={"w_threshold": w_threshold, "df": int(len(observed) - 1)},
    )


def cox_stuart_trend(values: Sequence[float],
                     alpha: float = 0.05,
                     alternative: Alternative = "greater") -> TestResult:
    """
    Тест Кокса–Стюарта на тренд: знаковый тест для пар (x_i, x_{i+m}),
    где m = n // 2. H0 — нет тренда. Эффект — доля положительных пар.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("need >= 4 points")
    m = n // 2
    pairs = list(zip(x[:m], x[-m:]))
    diffs = [b - a for a, b in pairs if b != a]
    if not diffs:
        return TestResult(
            test_name="cox_stuart",
            statistic=0.0,
            p_value=1.0,
            n=0,
            effect_size=0.0,
            effect_size_name="prop_positive",
            alpha=alpha,
            decision="fail_to_reject_H0",
            extra={"reason": "all_pairs_equal"},
        )
    pos = sum(1 for d in diffs if d > 0)
    n_eff = len(diffs)
    p_alt = {"two-sided": "two-sided", "greater": "greater", "less": "less"}[alternative]
    res = stats.binomtest(pos, n_eff, p=0.5, alternative=p_alt)
    prop = pos / n_eff
    return TestResult(
        test_name="cox_stuart",
        statistic=float(pos),
        p_value=float(res.pvalue),
        n=n_eff,
        effect_size=float(prop),
        effect_size_name="prop_positive",
        ci_95=(float(res.proportion_ci(confidence_level=0.95).low),
               float(res.proportion_ci(confidence_level=0.95).high)),
        alpha=alpha,
        decision=_decision(res.pvalue, alpha, abs(prop - 0.5), 0.0),
        extra={"alternative": alternative, "n_total": n, "n_pairs": n_eff,
               "n_positive": pos},
    )


# ---------------------------------------------------------------------------
# H5 — z-тест двух пропорций (доля токсичности с/без валидации)
# ---------------------------------------------------------------------------

def z_test_two_proportions(successes_a: int, n_a: int,
                            successes_b: int, n_b: int,
                            alternative: Alternative = "less",
                            alpha: float = 0.01,
                            min_abs_diff: float = 0.05) -> TestResult:
    """
    H0: p_A = p_B.
    alternative='less' — H1: p_A < p_B (одностороннее).
    Решение: reject H0 если p<alpha И |p_A - p_B| >= min_abs_diff.
    """
    if min(n_a, n_b) == 0:
        raise ValueError("n_a and n_b must be > 0")
    counts = np.array([successes_a, successes_b])
    nobs = np.array([n_a, n_b])
    z_stat, p = proportions_ztest(counts, nobs, alternative=_to_statsmodels_alt(alternative))
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_a - p_b
    ci_a = proportion_confint(successes_a, n_a, alpha=0.05, method="wilson")
    ci_b = proportion_confint(successes_b, n_b, alpha=0.05, method="wilson")
    return TestResult(
        test_name="z_test_two_proportions",
        statistic=float(z_stat),
        p_value=float(p),
        n=(n_a, n_b),
        effect_size=float(diff),
        effect_size_name="prop_diff",
        alpha=alpha,
        decision=_decision(p, alpha, abs(diff), min_abs_diff),
        extra={
            "p_a": p_a, "p_b": p_b,
            "ci_a_95": tuple(map(float, ci_a)),
            "ci_b_95": tuple(map(float, ci_b)),
            "alternative": alternative,
            "min_abs_diff_threshold": min_abs_diff,
        },
    )


# ---------------------------------------------------------------------------
# H6 — Welch t-test (CoT vs baseline по actionability)
# ---------------------------------------------------------------------------

def welch_t_test(a: Sequence[float], b: Sequence[float],
                 alternative: Alternative = "greater",
                 alpha: float = 0.05,
                 d_threshold: float = 0.5) -> TestResult:
    """
    Welch t-test (equal_var=False) для двух независимых выборок.
    H0: μ_A = μ_B.
    Решение: reject H0 если p<alpha И |Cohen's d| >= d_threshold.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample needs >= 2 observations")
    t_stat, p, df = sm_ttest_ind(a, b, alternative=_to_statsmodels_alt(alternative), usevar="unequal")
    d = cohens_d(a, b)
    return TestResult(
        test_name="welch_t_test",
        statistic=float(t_stat),
        p_value=float(p),
        n=(len(a), len(b)),
        effect_size=d,
        effect_size_name="cohens_d",
        alpha=alpha,
        decision=_decision(p, alpha, d, d_threshold),
        extra={"df": float(df), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
               "alternative": alternative, "d_threshold": d_threshold},
    )


# ---------------------------------------------------------------------------
# H7 — TOST (two one-sided tests) для эквивалентности LLM ≈ human
# ---------------------------------------------------------------------------

def mann_whitney_u(a: Sequence[float], b: Sequence[float],
                   alternative: Alternative = "two-sided",
                   alpha: float = 0.05) -> TestResult:
    """Двухвыборочный тест Манна–Уитни U; size effect = rank-biserial."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    u_stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    rb = rank_biserial(u_stat, len(a), len(b))
    return TestResult(
        test_name="mann_whitney_u",
        statistic=float(u_stat),
        p_value=float(p),
        n=(len(a), len(b)),
        effect_size=float(rb),
        effect_size_name="rank_biserial",
        alpha=alpha,
        decision=_decision(p, alpha, abs(rb), 0.0),
        extra={"median_a": float(np.median(a)), "median_b": float(np.median(b)),
               "alternative": alternative},
    )


def tost_equivalence(a: Sequence[float], b: Sequence[float],
                     low: float, high: float,
                     alpha: float = 0.05) -> TestResult:
    """
    Two One-Sided Tests на эквивалентность медиан.
    H0: median(a) - median(b) <= low ИЛИ >= high (различие практически значимо).
    H1: low < median(a) - median(b) < high (эквивалентны).

    Реализация: два теста Манна–Уитни на смещённых данных.
    Эквивалентность подтверждается, если МАКСИМУМ p_low, p_high < alpha.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if low >= high:
        raise ValueError("low must be < high")
    # Тест 1: H1_low — median(a) - median(b) > low  ⇔  median(a - low) > median(b)
    _, p_low = stats.mannwhitneyu(a - low, b, alternative="greater")
    # Тест 2: H1_high — median(a) - median(b) < high  ⇔  median(a - high) < median(b)
    _, p_high = stats.mannwhitneyu(a - high, b, alternative="less")
    p_max = max(float(p_low), float(p_high))
    median_diff = float(np.median(a) - np.median(b))
    equivalent = p_max < alpha
    return TestResult(
        test_name="tost_equivalence",
        statistic=median_diff,
        p_value=p_max,
        n=(len(a), len(b)),
        effect_size=median_diff,
        effect_size_name="median_diff",
        alpha=alpha,
        decision="reject_H0" if equivalent else "fail_to_reject_H0",
        extra={"p_low": float(p_low), "p_high": float(p_high),
               "bounds": (low, high), "equivalent": equivalent},
    )


# ---------------------------------------------------------------------------
# Дополнительные общеупотребительные тесты
# ---------------------------------------------------------------------------

def spearman_correlation(x: Sequence[float], y: Sequence[float],
                         alternative: Alternative = "two-sided",
                         alpha: float = 0.05,
                         rho_threshold: float = 0.3) -> TestResult:
    """Корреляция Спирмена. Размер эффекта = |ρ|; medium = 0.3."""
    rho, p = stats.spearmanr(x, y, alternative=alternative)
    return TestResult(
        test_name="spearman_correlation",
        statistic=float(rho),
        p_value=float(p),
        n=len(x),
        effect_size=float(rho),
        effect_size_name="spearman_rho",
        alpha=alpha,
        decision=_decision(p, alpha, abs(rho), rho_threshold),
        extra={"alternative": alternative, "rho_threshold": rho_threshold},
    )


def cohens_kappa(rater_a: Sequence, rater_b: Sequence,
                 alpha: float = 0.05,
                 kappa_threshold: float = 0.7) -> TestResult:
    """
    Cohen's κ — согласие двух экспертов на категориальной разметке.
    Тест значимости: z-тест относительно κ=0.
    Решение: 'reject_H0' (κ значимо отличается от 0 И κ >= kappa_threshold)
    интерпретируется как 'achieved acceptable agreement'.
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if len(a) != len(b):
        raise ValueError("raters must have same length")
    cats = sorted(set(a.tolist()) | set(b.tolist()))
    cat_idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(a)
    cm = np.zeros((k, k))
    for ai, bi in zip(a, b):
        cm[cat_idx[ai], cat_idx[bi]] += 1
    po = cm.diagonal().sum() / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if pe == 1:
        kappa, se, z, p = 1.0, 0.0, float("inf"), 0.0
    else:
        kappa = (po - pe) / (1 - pe)
        # SE по Fleiss (1969) для approximate z-test
        se = math.sqrt(po * (1 - po) / (n * (1 - pe) ** 2))
        z = kappa / se if se > 0 else float("inf")
        p = 2 * (1 - stats.norm.cdf(abs(z)))
    return TestResult(
        test_name="cohens_kappa",
        statistic=float(kappa),
        p_value=float(p),
        n=n,
        effect_size=float(kappa),
        effect_size_name="kappa",
        alpha=alpha,
        decision="reject_H0" if (p < alpha and kappa >= kappa_threshold) else "fail_to_reject_H0",
        extra={"po": float(po), "pe": float(pe), "z": float(z),
               "kappa_threshold": kappa_threshold, "n_categories": k},
    )


# ---------------------------------------------------------------------------
# Поправка на множественные сравнения
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mixed-effects модель (для иерархических данных: реплики внутри диалога)
# ---------------------------------------------------------------------------

def mixed_effects_two_groups(values: Sequence[float],
                              group: Sequence[str],
                              cluster: Sequence,
                              alpha: float = 0.05,
                              d_threshold: float = 0.5) -> TestResult:
    """
    Линейная смешанная модель: value ~ group + (1 | cluster).

    Когда применять: данные иерархические (например, реплики внутри одного диалога —
    они зависимы; стандартный t-test даёт заниженные стандартные ошибки).

    Параметры:
      values  — зависимая переменная
      group   — индикатор treatment vs control (2 уровня, строки)
      cluster — id кластера (например, dialogue_id) — рандомный эффект

    Тест значимости: фиксированный эффект group (Wald z-test).
    Эффект-сайз: разность средних, стандартизованная по pooled SD.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    if not (len(values) == len(group) == len(cluster)):
        raise ValueError("values, group, cluster must have same length")
    if len(values) < 4:
        raise ValueError("need >= 4 observations for mixed-effects model")

    df = pd.DataFrame({"value": values, "group": list(group), "cluster": list(cluster)})
    levels = sorted(df["group"].unique())
    if len(levels) != 2:
        raise ValueError(f"mixed_effects_two_groups requires 2 groups, got {levels}")

    # Reference категория = первая по алфавиту; коэффициент при group[T.<other>]
    md = smf.mixedlm("value ~ C(group)", df, groups=df["cluster"])
    try:
        result = md.fit(method="lbfgs", disp=False)
    except Exception:
        result = md.fit(reml=False, method="powell", disp=False)

    # Вытаскиваем коэффициент при второй группе
    treat_label = [name for name in result.params.index if name.startswith("C(group)")]
    if not treat_label:
        raise RuntimeError("mixed_effects: failed to identify group coefficient")
    coef_name = treat_label[0]
    beta = float(result.params[coef_name])
    se = float(result.bse[coef_name])
    z = beta / se if se > 0 else float("nan")
    p = float(result.pvalues[coef_name])

    # Cohen's d через pooled SD сырых значений (приближение)
    a_vals = df.loc[df["group"] == levels[0], "value"].to_numpy()
    b_vals = df.loc[df["group"] == levels[1], "value"].to_numpy()
    d = cohens_d(b_vals, a_vals)  # b vs a в порядке "T - reference"

    decision = _decision(p, alpha, d, d_threshold)
    return TestResult(
        test_name="mixed_effects_lmm",
        statistic=float(z) if not math.isnan(z) else 0.0,
        p_value=p,
        n=(len(a_vals), len(b_vals)),
        effect_size=d,
        effect_size_name="cohens_d",
        alpha=alpha,
        decision=decision,
        extra={
            "fixed_effect_coef": beta,
            "fixed_effect_se": se,
            "reference_group": levels[0],
            "treatment_group": levels[1],
            "n_clusters": int(df["cluster"].nunique()),
            "random_intercept_var": float(result.cov_re.iloc[0, 0])
                if hasattr(result, "cov_re") and result.cov_re is not None else None,
            "converged": bool(result.converged) if hasattr(result, "converged") else None,
            "d_threshold": d_threshold,
        },
    )


# ---------------------------------------------------------------------------
# Assumption checking (предпосылки параметрических тестов)
# ---------------------------------------------------------------------------

@dataclass
class AssumptionReport:
    """Результат проверки предпосылок параметрического теста."""
    normality_per_group: list[dict]   # [{group_idx, statistic, p_value, normal_at_05}]
    homogeneity: dict | None          # {test, statistic, p_value, equal_variances_at_05}
    n_per_group: list[int]
    assumptions_met: bool             # True если ВСЕ предпосылки выполнены при α=0.05
    recommendation: str               # текстовая рекомендация по выбору теста

    def to_dict(self) -> dict:
        return asdict(self)


def check_test_assumptions(*groups: Sequence[float],
                            normality_alpha: float = 0.05,
                            homogeneity_alpha: float = 0.05) -> AssumptionReport:
    """
    Проверяет предпосылки параметрических тестов:
      - Шапиро–Уилк на нормальность каждой группы
      - Левена на гомогенность дисперсий

    Если выборка одна — проверяется только нормальность.
    Возвращает AssumptionReport с рекомендацией.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]
    if not arrays:
        raise ValueError("at least one group required")

    normality_per_group = []
    all_normal = True
    for i, arr in enumerate(arrays):
        if len(arr) < 3:
            normality_per_group.append({
                "group_idx": i, "statistic": None, "p_value": None,
                "normal_at_05": None, "reason": "n<3 (skipped)",
            })
            continue
        try:
            stat, p = stats.shapiro(arr)
            normal = bool(p >= normality_alpha)
        except Exception as e:
            normality_per_group.append({
                "group_idx": i, "statistic": None, "p_value": None,
                "normal_at_05": None, "reason": f"shapiro_failed: {e}",
            })
            all_normal = False
            continue
        normality_per_group.append({
            "group_idx": i,
            "statistic": float(stat),
            "p_value": float(p),
            "normal_at_05": normal,
        })
        if not normal:
            all_normal = False

    homogeneity = None
    equal_var = True
    if len(arrays) >= 2 and all(len(a) >= 2 for a in arrays):
        try:
            lev_stat, lev_p = stats.levene(*arrays, center="median")
            equal_var = bool(lev_p >= homogeneity_alpha)
            homogeneity = {
                "test": "levene_brown_forsythe",
                "statistic": float(lev_stat),
                "p_value": float(lev_p),
                "equal_variances_at_05": equal_var,
            }
        except Exception as e:
            homogeneity = {"test": "levene_brown_forsythe", "error": str(e),
                           "equal_variances_at_05": None}
            equal_var = True  # не блокируем при ошибке

    assumptions_met = all_normal and equal_var

    # Рекомендация
    if assumptions_met:
        rec = "All assumptions met — parametric tests (t-test, ANOVA) are appropriate."
    elif all_normal and not equal_var:
        rec = ("Normality OK but variances differ — use Welch t-test (already non-pooled) "
               "or set ANOVA welch correction.")
    elif not all_normal and len(arrays) == 2:
        rec = ("Normality violated — switch to Mann-Whitney U (non-parametric) "
               "or rely on robustness of t-test for n>30.")
    elif not all_normal and len(arrays) > 2:
        rec = ("Normality violated for ≥1 group — consider Kruskal-Wallis "
               "instead of ANOVA, or rely on robustness for n>30 per group.")
    else:
        rec = "Mixed violations — use non-parametric test."

    return AssumptionReport(
        normality_per_group=normality_per_group,
        homogeneity=homogeneity,
        n_per_group=[len(a) for a in arrays],
        assumptions_met=assumptions_met,
        recommendation=rec,
    )


def auto_choose_test_two_samples(a: Sequence[float], b: Sequence[float],
                                   alternative: Alternative = "two-sided",
                                   alpha: float = 0.05,
                                   d_threshold: float = 0.5) -> TestResult:
    """
    Выбирает между Welch t-test и Mann-Whitney на основе предпосылок.
    Welch — если нормальность OK (даже при разных дисперсиях, она handle Welch).
    Mann-Whitney — если нормальность нарушена.
    """
    report = check_test_assumptions(a, b)
    all_normal = all(
        ng.get("normal_at_05") is True for ng in report.normality_per_group
    )
    if all_normal:
        result = welch_t_test(a, b, alternative=alternative, alpha=alpha,
                               d_threshold=d_threshold)
        result.extra["auto_chosen"] = "welch_t_test"
    else:
        result = mann_whitney_u(a, b, alternative=alternative, alpha=alpha)
        result.extra["auto_chosen"] = "mann_whitney_u"
        result.extra["normality_violated"] = True
    result.extra["assumption_report"] = report.to_dict()
    return result


def holm_bonferroni(p_values: Iterable[float],
                    alpha: float = 0.05) -> list[dict]:
    """
    Holm–Bonferroni для семейства тестов.
    Возвращает список с p_adj, reject_at_alpha — в исходном порядке.
    """
    p_arr = np.asarray(list(p_values), dtype=float)
    K = len(p_arr)
    order = np.argsort(p_arr)
    p_adj = np.empty(K, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        # Holm step-down: альфа на этом шаге = alpha / (K - rank)
        adjusted = p_arr[idx] * (K - rank)
        running_max = max(running_max, adjusted)
        p_adj[idx] = min(running_max, 1.0)
    return [
        {
            "rank": int(np.where(order == i)[0][0]) + 1,
            "p_raw": float(p_arr[i]),
            "p_adj": float(p_adj[i]),
            "reject": bool(p_adj[i] < alpha),
        }
        for i in range(K)
    ]


# ---------------------------------------------------------------------------
# Высокоуровневая обёртка для всего семейства гипотез из preregistration
# ---------------------------------------------------------------------------

def evaluate_hypothesis_family(named_results: dict[str, TestResult],
                               alpha: float = 0.05) -> dict[str, dict]:
    """
    Применяет Holm–Bonferroni к списку результатов семейства гипотез.
    Возвращает тот же dict, дополненный adjusted p-values и итоговым решением.
    """
    names = list(named_results.keys())
    p_vals = [named_results[n].p_value for n in names]
    adjusted = holm_bonferroni(p_vals, alpha=alpha)
    out = {}
    for name, adj in zip(names, adjusted):
        result = named_results[name].to_dict()
        result["p_adj_holm"] = adj["p_adj"]
        result["reject_after_holm"] = adj["reject"]
        # финальное решение: и сырой p < alpha с эффектом, и Holm-adjusted < alpha
        raw_decision = result["decision"]
        result["final_decision"] = (
            "reject_H0" if raw_decision == "reject_H0" and adj["reject"]
            else "fail_to_reject_H0"
        )
        out[name] = result
    return out
