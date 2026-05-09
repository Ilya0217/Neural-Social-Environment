"""
Power analysis для гипотез H1, H3, H5, H6, H7 из preregistration.md.

Цель — для каждой гипотезы посчитать минимальный размер выборки N,
обеспечивающий мощность 1-β = 0.80 при заданном alpha и medium-эффекте.

Если для какой-то гипотезы запланированный N (из пререгистрации) меньше
требуемого — это надо знать ДО сбора данных.

Запуск: python -m agent_dialogue_sim.power_analysis
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math

import numpy as np
from statsmodels.stats.power import (
    TTestIndPower,
    FTestAnovaPower,
    NormalIndPower,
    GofChisquarePower,
)
from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail


@dataclass
class PowerResult:
    hypothesis: str
    test: str
    alpha: float
    power: float
    effect_size: float
    effect_name: str
    required_n: int
    planned_n: int | str
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def verdict(self) -> str:
        if isinstance(self.planned_n, str):
            return f"⚠ N={self.planned_n} (произвольный) → нужно ≥ {self.required_n}"
        if self.planned_n >= self.required_n:
            return f"✅ N={self.planned_n} достаточно (требуется ≥ {self.required_n})"
        return f"⚠ N={self.planned_n} МАЛО — требуется ≥ {self.required_n}"


# ---------------------------------------------------------------------------
# H1: ANOVA для 3 уровней экстраверсии (low/mid/high)
# ---------------------------------------------------------------------------

def power_h1(alpha: float = 0.05, power: float = 0.80,
             effect_size: float = 0.25,  # Cohen's f, ≈ medium для ANOVA
             k_groups: int = 3, planned_n_per_group: int = 160) -> PowerResult:
    """
    Однофакторный ANOVA. Cohen's f ≈ sqrt(η²/(1-η²)).
    f=0.25 — medium effect (Cohen 1988).
    Возвращает требуемый размер выборки на одну группу.
    """
    analysis = FTestAnovaPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        k_groups=k_groups,
    )
    required = int(math.ceil(n_per_group))
    return PowerResult(
        hypothesis="H1",
        test="ANOVA one-way",
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        effect_name="cohens_f (≈medium)",
        required_n=required,
        planned_n=planned_n_per_group,
        note=f"per group; total = {required * k_groups}",
    )


# ---------------------------------------------------------------------------
# H3: χ² goodness-of-fit для матрицы переходов 4x4 (Wheelan stages)
# ---------------------------------------------------------------------------

def power_h3(alpha: float = 0.01, power: float = 0.80,
             effect_size: float = 0.3,  # Cohen's w, medium
             n_cells: int = 16,  # 4x4 transition matrix
             planned_total_observations: int = 30 * 10) -> PowerResult:
    """
    χ² goodness-of-fit. Cohen's w = sqrt(χ²/N).
    w=0.3 — medium effect.
    n_cells = число ячеек матрицы переходов (для 4 стадий = 16).
    df = n_cells - 1.
    Planned: 30 диалогов × 60 ходов даёт ≈ 10 transitions/диалог при W=10, шаг 5.
    """
    analysis = GofChisquarePower()
    n_required = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        n_bins=n_cells,
    )
    required = int(math.ceil(n_required))
    return PowerResult(
        hypothesis="H3",
        test="χ² goodness-of-fit (transition matrix)",
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        effect_name="cohens_w (medium)",
        required_n=required,
        planned_n=planned_total_observations,
        note=(f"observations = window-transitions across all dialogues; "
              f"30 dialogues × ~6 transitions ≈ {planned_total_observations}"),
    )


# ---------------------------------------------------------------------------
# H5: z-тест двух пропорций (валидация vs контроль)
# ---------------------------------------------------------------------------

def power_h5(alpha: float = 0.01, power: float = 0.80,
             p_a: float = 0.05, p_b: float = 0.15,
             planned_n_per_group: int = 900) -> PowerResult:
    """
    Two-proportion z-test, односторонний.
    p_a — ожидаемая доля в treatment (валидация), p_b — control.
    Δ = 0.10 — практически значимое снижение токсичности на 10 п.п.
    Из пререгистрации: 30 диалогов × 30 ходов = 900 реплик в каждой группе.
    """
    n_required = samplesize_proportions_2indep_onetail(
        diff=p_a - p_b,
        prop2=p_b,
        power=power,
        alpha=alpha,
        ratio=1.0,
        alternative="smaller",  # p_a < p_b (less)
    )
    required = int(math.ceil(n_required))
    return PowerResult(
        hypothesis="H5",
        test="z-test two proportions (one-sided)",
        alpha=alpha,
        power=power,
        effect_size=abs(p_a - p_b),
        effect_name="prop_diff",
        required_n=required,
        planned_n=planned_n_per_group,
        note=f"replies per group; assuming p_treatment={p_a}, p_control={p_b}",
    )


# ---------------------------------------------------------------------------
# H6: Welch t-test (CoT vs baseline по actionability)
# ---------------------------------------------------------------------------

def power_h6(alpha: float = 0.05, power: float = 0.80,
             effect_size: float = 0.5,  # Cohen's d, medium
             planned_n_per_group: int = 55) -> PowerResult:
    """
    Welch t-test, независимые выборки.
    d=0.5 — medium effect (Cohen 1988).
    """
    analysis = TTestIndPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="larger",  # μ_A > μ_B
    )
    required = int(math.ceil(n_per_group))
    return PowerResult(
        hypothesis="H6",
        test="Welch t-test (one-sided)",
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        effect_name="cohens_d (medium)",
        required_n=required,
        planned_n=planned_n_per_group,
        note="dialogues per group (CoT vs baseline)",
    )


# ---------------------------------------------------------------------------
# H7: Mann–Whitney через TOST — приблизительно через нормальную аппроксимацию
# ---------------------------------------------------------------------------

def power_h7(alpha: float = 0.05, power: float = 0.80,
             effect_size: float = 0.5,  # ARE для MWU vs t-test ≈ 0.95, оцениваем как t-test
             planned_n_per_group: int = 70) -> PowerResult:
    """
    Mann–Whitney U через нормальную аппроксимацию (asymptotic relative efficiency
    ≈ 0.955 относительно t-теста). Используем t-test power как близкую оценку,
    с поправкой на ARE.

    Внимание: это approximation. Для точного power для TOST нужен симуляционный подход.
    Сообщаем оценочный N сверху.
    """
    analysis = NormalIndPower()
    n_per_group = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=1.0,
        alternative="two-sided",
    )
    # коррекция на ARE Mann-Whitney ≈ 0.955
    are = 0.955
    required = int(math.ceil(n_per_group / are))
    return PowerResult(
        hypothesis="H7",
        test="TOST (Mann–Whitney) — approximate",
        alpha=alpha,
        power=power,
        effect_size=effect_size,
        effect_name="cohens_d-equiv (ARE-corrected)",
        required_n=required,
        planned_n=planned_n_per_group,
        note=("approximation: ARE Mann-Whitney ≈ 0.955; для строгой оценки "
              "TOST-power нужен симуляционный анализ"),
    )


# ---------------------------------------------------------------------------
# Sensitivity-кривая: required_n vs effect_size
# ---------------------------------------------------------------------------

def sensitivity_curve_t_test(alpha: float = 0.05, power: float = 0.80,
                             effects: list[float] | None = None) -> list[dict]:
    """Зависимость требуемого N от размера эффекта для t-test."""
    if effects is None:
        effects = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    analysis = TTestIndPower()
    out = []
    for d in effects:
        n = analysis.solve_power(effect_size=d, alpha=alpha, power=power,
                                  ratio=1.0, alternative="larger")
        out.append({"d": d, "required_n_per_group": int(math.ceil(n))})
    return out


# ---------------------------------------------------------------------------
# Главный запуск
# ---------------------------------------------------------------------------

def run_all() -> dict:
    results = {
        "H1": power_h1(),
        "H3": power_h3(),
        "H5": power_h5(),
        "H6": power_h6(),
        "H7": power_h7(),
    }
    return {
        "hypotheses": {h: r.to_dict() for h, r in results.items()},
        "verdicts": {h: r.verdict() for h, r in results.items()},
        "sensitivity_t_test": sensitivity_curve_t_test(),
    }


def _print_table(results: dict) -> None:
    print("=" * 92)
    print(f"{'Гипотеза':<6} {'Тест':<35} {'α':<6} {'Effect':<8} "
          f"{'Need N':<10} {'Plan N':<10}")
    print("-" * 92)
    for h, r in results["hypotheses"].items():
        print(f"{r['hypothesis']:<6} {r['test']:<35} {r['alpha']:<6} "
              f"{r['effect_size']:<8} {r['required_n']:<10} "
              f"{r['planned_n']:<10}")
    print("=" * 92)
    print("\nВЕРДИКТЫ:")
    for h, v in results["verdicts"].items():
        print(f"  {h}: {v}")
    print("\nSENSITIVITY (t-test, α=0.05, power=0.80):")
    print(f"  {'Cohen d':<10} {'Required N per group':<20}")
    for row in results["sensitivity_t_test"]:
        print(f"  {row['d']:<10} {row['required_n_per_group']:<20}")


if __name__ == "__main__":
    res = run_all()
    _print_table(res)
    # Сохраняем результаты в JSON для документации
    out_path = "power_analysis_results.json"
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nРезультаты сохранены в {out_path}")
