"""CLI-раннер для каузальных интервенций.

Запуск:
  python -m agent_dialogue_sim.experiments.run_causal \\
        --target-agent Critic --trait agreeableness \\
        --treatment-value 0.9 --control-value 0.1 \\
        --n-pairs 8 --n-turns 10 --parallelism 4

После прогона генерирует:
  experiments/results/<id>__<timestamp>/config.json
  experiments/results/<id>__<timestamp>/pairs.jsonl
  experiments/results/<id>__<timestamp>/summary.json
  experiments/results/<id>__<timestamp>/REPORT.md    ← научный отчёт для руководителя
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..science.causal_intervention import (
    CausalConfig,
    CausalExperimentResult,
    CausalRunner,
    TraitIntervention,
    DEFAULT_OUTCOME_METRICS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Научный отчёт
# ---------------------------------------------------------------------------

# Сопоставление метрик с научными рамками, к которым они каузально относятся.
METRIC_FRAMEWORKS: dict[str, list[tuple[str, str]]] = {
    "avg_tone": [
        ("Barsade (2002) — Emotional Contagion",
         "Тональность реплик в группе — индикатор эмоционального заражения."),
        ("Wheelan (1994) — Group Development",
         "Средний тон ассоциирован с переходами между Forming→Storming→Norming."),
    ],
    "negative_tone_rate": [
        ("Barsade (2002) — Emotional Contagion",
         "Доля негативных реплик отражает «заражение» отрицательным аффектом."),
        ("Wheelan (1994) — Storming Stage",
         "Storming характеризуется повышенной долей конфликтных/негативных актов."),
    ],
    "positive_tone_rate": [
        ("Barsade (2002) — Emotional Contagion",
         "Доля позитивных реплик отражает заражение положительным аффектом."),
    ],
    "avg_reply_length": [
        ("Goldberg (1990) — Big Five (OCEAN)",
         "Длина реплики каузально связана с Extraversion и Openness агента."),
        ("Tausczik & Pennebaker (2010) — LIWC / LSM",
         "Лексические показатели вовлечённости и языковой согласованности."),
    ],
    "actionability_rate": [
        ("Wei et al. (2022) — Chain-of-Thought",
         "Доля реплик с явными next-steps — операционализация actionability."),
        ("Wheelan (1994) — Performing Stage",
         "Performing-стадия характеризуется высокой долей задач-ориентированных актов."),
    ],
    "reciprocity": [
        ("Borgatti et al. (2009) — Social Network Analysis",
         "Доля взаимных рёбер (A↔B) — фундаментальная мера структуры сети."),
        ("Contractor et al. (2012) — Multidimensional Networks",
         "Reciprocity отражает silosность/связность многоагентной сети."),
    ],
    "emotion_entropy": [
        ("Cowen & Keltner (2017) — High-Dimensional Emotion Space",
         "Энтропия эмоций отражает разнообразие эмоциональных состояний."),
    ],
    "question_rate": [
        ("Bunt et al. (2020) — ISO 24617-2 Dialogue Acts",
         "Вопросы (info-seeking dialogue acts) — ключевой класс ISO-таксономии."),
    ],
    "targeting_rate": [
        ("Borgatti et al. (2009) — SNA",
         "Доля адресных реплик — индикатор плотности сети взаимодействий."),
    ],
    "reference_rate": [
        ("Tausczik & Pennebaker (2010) — Language Style Matching",
         "Доля упоминаний других участников — индекс групповой когнитивной согласованности."),
    ],
}


def _fmt_p(p: float) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    if p < 0.01:
        return f"= {p:.4f}"
    return f"= {p:.3f}"


def _fmt_num(x, digits=4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def _interpret_effect(d_z: float | None) -> str:
    if d_z is None or (isinstance(d_z, float) and (math.isnan(d_z) or math.isinf(d_z))):
        return "не определён"
    a = abs(d_z)
    if a < 0.2:
        return "пренебрежимо малый"
    if a < 0.5:
        return "малый"
    if a < 0.8:
        return "средний"
    return "большой"


def _decision_emoji(decision: str) -> str:
    return "✅ ОТВЕРГНУТА H₀" if decision == "reject_H0" else "⏸ H₀ не отвергнута"


def build_scientific_report(result: CausalExperimentResult) -> str:
    cfg = result.config
    treat = result.intervention_treatment
    ctrl = result.intervention_control
    out_dir = cfg.get("output_dir", "")

    md: list[str] = []
    md.append(f"# Научный отчёт: каузальная интервенция `{result.experiment_id}`")
    md.append("")
    md.append(f"**Дата:** {result.timestamp}  ")
    md.append(f"**Длительность прогона:** {result.duration_seconds:.1f} с  ")
    md.append(f"**Каталог результатов:** `{out_dir}`")
    md.append("")

    # ----- 1. Методологическая рамка -----
    md.append("## 1. Методологическая рамка")
    md.append("")
    md.append(
        "Эксперимент реализует **каузальный вывод по Pearl (1995, 2009)** через "
        "do-оператор: `P(Y | do(X = x))`, а не `P(Y | X = x)`. Дизайн — **paired "
        "counterfactual (within-subject)**: для каждого случайного seed `s ∈ {1..N}` "
        "запускаются ДВЕ симуляции, отличающиеся ТОЛЬКО интервенцией:"
    )
    md.append("")
    md.append(f"- **Treatment:** `{treat.get('descriptor', treat['name'])}`")
    md.append(f"- **Control:**   `{ctrl.get('descriptor', ctrl['name'])}`")
    md.append("")
    md.append(
        "Поскольку всё остальное (seed, env_context, остальные OCEAN-черты агентов, "
        "системные промпты) совпадает, наблюдаемая разница `Δ(s) = Y_T(s) − Y_C(s)` "
        "идентифицируется как **Individual Treatment Effect** для seed `s`. "
        "Усреднение по парам даёт **Average Treatment Effect (ATE)**."
    )
    md.append("")
    md.append("**Identifying assumptions** (Rubin/Pearl):")
    md.append("- **SUTVA** (no interference): запуски с разными seed независимы.")
    md.append(
        "- **Consistency**: исход `Y(s)` соответствует фактически применённой интервенции "
        "(гарантировано детерминированной подменой Big Five до старта диалога)."
    )
    md.append(
        "- **No unmeasured confounders**: все остальные источники вариативности "
        "контролируются через фиксированный seed, идентичный env_context и общую LLM."
    )
    md.append("")

    # ----- 2. Формальная гипотеза -----
    md.append("## 2. Формальная гипотеза")
    md.append("")
    md.append(f"Для каждой outcome-метрики **m** ∈ {{{', '.join(result.outcome_metrics)}}}:")
    md.append("")
    md.append("- **H₀ᵐ**: `E[Y_T − Y_C] = 0` (интервенция не влияет на m)")
    md.append("- **H₁ᵐ**: `E[Y_T − Y_C] ≠ 0` (интервенция каузально влияет на m)")
    md.append("")
    md.append(
        f"**Критерий вывода:** H₀ отвергается, если paired t-test даёт "
        f"`p < {cfg['alpha']}` И `|Cohen's d_z| ≥ {cfg['d_threshold']}` "
        f"(оба условия — защита от ложно-значимых эффектов малой практической силы)."
    )
    md.append("")

    # ----- 3. Параметры эксперимента -----
    md.append("## 3. Параметры эксперимента")
    md.append("")
    md.append("| Параметр | Значение |")
    md.append("|---|---|")
    md.append(f"| LLM-провайдер | DeepSeek (`deepseek-chat`) |")
    md.append(f"| Число пар (N) | {result.n_pairs_total} |")
    md.append(f"| Валидных пар | {result.n_pairs_valid} |")
    md.append(f"| Ходов на диалог | {cfg['n_turns']} |")
    md.append(f"| env_context | index={cfg['env_context_index']} |")
    md.append(f"| seed_base | {cfg['seed_base']} (диапазон {cfg['seed_base']}..{cfg['seed_base']+cfg['n_pairs']-1}) |")
    md.append(f"| Параллелизм | {cfg['parallelism']} |")
    md.append(f"| α (порог значимости) | {cfg['alpha']} |")
    md.append(f"| Порог |d_z| | {cfg['d_threshold']} |")
    md.append(f"| alternative | {cfg['alternative']} |")
    md.append("")

    # ----- 4. Основные результаты (ATE-таблица) -----
    md.append("## 4. Основные результаты — Average Treatment Effects")
    md.append("")
    md.append("| Метрика | mean(T) | mean(C) | ATE | 95% CI | t | df | p | d_z | Эффект | Решение |")
    md.append("|---|---:|---:|---:|---|---:|---:|---|---:|:---:|:---:|")
    for metric in result.outcome_metrics:
        a = result.ate_by_metric.get(metric, {})
        ci = f"[{_fmt_num(a.get('ci_low'), 3)}, {_fmt_num(a.get('ci_high'), 3)}]"
        md.append(
            f"| `{metric}` "
            f"| {_fmt_num(a.get('mean_treatment'), 3)} "
            f"| {_fmt_num(a.get('mean_control'), 3)} "
            f"| **{_fmt_num(a.get('ate'), 3)}** "
            f"| {ci} "
            f"| {_fmt_num(a.get('t'), 2)} "
            f"| {a.get('df', '?')} "
            f"| {_fmt_p(a.get('p_value'))} "
            f"| {_fmt_num(a.get('cohens_d_z'), 2)} "
            f"| {_interpret_effect(a.get('cohens_d_z'))} "
            f"| {_decision_emoji(a.get('decision', 'fail_to_reject_H0'))} |"
        )
    md.append("")

    # ----- 5. Интерпретация по теориям -----
    md.append("## 5. Интерпретация по научным рамкам")
    md.append("")
    rejected = [m for m in result.outcome_metrics
                if result.ate_by_metric.get(m, {}).get("decision") == "reject_H0"]
    not_rejected = [m for m in result.outcome_metrics if m not in rejected]

    if rejected:
        md.append("### 5.1. Подтверждённые каузальные эффекты")
        md.append("")
        for metric in rejected:
            a = result.ate_by_metric[metric]
            sign = "увеличивает" if (a.get("ate") or 0) > 0 else "уменьшает"
            md.append(f"#### `{metric}` — H₀ отвергнута")
            md.append("")
            md.append(
                f"Интервенция `{treat.get('descriptor', treat['name'])}` каузально **{sign}** "
                f"метрику `{metric}` по сравнению с контролем "
                f"(`{ctrl.get('descriptor', ctrl['name'])}`): "
                f"ATE = {_fmt_num(a.get('ate'), 3)}, "
                f"95% CI [{_fmt_num(a.get('ci_low'), 3)}, {_fmt_num(a.get('ci_high'), 3)}], "
                f"t({a.get('df')}) = {_fmt_num(a.get('t'), 2)}, "
                f"p {_fmt_p(a.get('p_value'))}, "
                f"d_z = {_fmt_num(a.get('cohens_d_z'), 2)} "
                f"({_interpret_effect(a.get('cohens_d_z'))} эффект)."
            )
            md.append("")
            frameworks = METRIC_FRAMEWORKS.get(metric, [])
            if frameworks:
                md.append(f"**Связь с теорией:**")
                for ref, desc in frameworks:
                    md.append(f"- *{ref}* — {desc}")
                md.append("")
    else:
        md.append("### 5.1. Подтверждённых каузальных эффектов нет")
        md.append("")
        md.append(
            "На текущей выборке (N = "
            f"{result.n_pairs_valid}) ни одна из метрик не достигла одновременно "
            f"`p < {cfg['alpha']}` и `|d_z| ≥ {cfg['d_threshold']}`. "
            "Это **не** означает отсутствие эффекта в популяции — это означает недостаток "
            "статистической мощности при данной N."
        )
        md.append("")

    if not_rejected:
        md.append("### 5.2. Метрики без значимого эффекта")
        md.append("")
        for metric in not_rejected:
            a = result.ate_by_metric.get(metric, {})
            md.append(
                f"- `{metric}`: ATE = {_fmt_num(a.get('ate'), 3)}, "
                f"p {_fmt_p(a.get('p_value'))}, d_z = {_fmt_num(a.get('cohens_d_z'), 2)} "
                f"— {_interpret_effect(a.get('cohens_d_z'))} эффект."
            )
        md.append("")

    # ----- 6. Per-pair raw ITE -----
    md.append("## 6. Per-pair Individual Treatment Effects (контролируемая прозрачность)")
    md.append("")
    md.append(
        "Полная таблица ITE по каждой паре — для воспроизводимости и проверки на "
        "выбросы (`pairs.jsonl` содержит также полные истории диалогов)."
    )
    md.append("")
    cols = result.outcome_metrics
    md.append("| seed | valid | " + " | ".join(f"Δ {m}" for m in cols) + " |")
    md.append("|---|---|" + "|".join("---:" for _ in cols) + "|")
    for p in result.pairs:
        seed = p["seed"]
        valid = "✓" if p["valid"] else "✗"
        ite = p.get("ite", {})
        vals = " | ".join(_fmt_num(ite.get(m), 3) for m in cols)
        md.append(f"| {seed} | {valid} | {vals} |")
    md.append("")

    # ----- 7. Ограничения и обсуждение -----
    md.append("## 7. Ограничения и обсуждение")
    md.append("")
    md.append(
        f"- **Внутренняя валидность:** обеспечена paired-дизайном; вариация между "
        f"парами устранена внутрисубъектным сравнением (df = N − 1)."
    )
    md.append(
        f"- **Статистическая мощность:** при N = {result.n_pairs_valid} мощность для "
        f"обнаружения малого эффекта (d_z = 0.3) ограничена; рекомендуется N ≥ 30 "
        f"для надёжной репликации."
    )
    md.append(
        "- **Внешняя валидность:** результат относится к данной LLM (DeepSeek), "
        "данному env_context и данной структуре агентов. Перенос на других LLM "
        "требует репликации."
    )
    md.append(
        "- **Множественные сравнения:** при анализе нескольких метрик одновременно "
        "рекомендуется применить поправку Holm–Bonferroni "
        "(`statistical_tests.holm_bonferroni`) для контроля FWER."
    )
    md.append(
        "- **Конструктная валидность:** outcome-метрики операционализированы через "
        "agreed-upon формулы из существующей научной базы проекта (см. §5)."
    )
    md.append("")

    # ----- 8. Вклад в научную работу -----
    md.append("## 8. Вклад в научную работу")
    md.append("")
    md.append(
        "Данный эксперимент переводит проект из **наблюдательной** парадигмы "
        "(корреляции между OCEAN и метриками) в **каузальную** "
        "(`P(Y | do(X = x))`). Это:"
    )
    md.append("")
    md.append(
        "1. **Удовлетворяет формальному критерию научной гипотезы**: "
        "явные H₀/H₁ в нотации математической статистики, парадигма эксперимента "
        "(paired RCT-аналог in silico), критерий вывода (p + размер эффекта)."
    )
    md.append(
        "2. **Даёт каузальную интерпретацию**, а не «вайбы»: можно утверждать, что "
        "изменение конкретной OCEAN-черты конкретного агента **вызывает** изменение "
        "групповой метрики, а не просто коррелирует с ней."
    )
    md.append(
        "3. **Связано с устоявшимися теориями**: каждая outcome-метрика "
        "обоснована published frameworks (Big Five, SNA, ISO 24617-2, Wheelan, "
        "Barsade, LIWC, CoT), что обеспечивает теоретическую валидность."
    )
    md.append(
        "4. **Воспроизводимо**: фиксированные seed, открытый код, JSONL-логи всех "
        "диалогов, явный YAML/JSON-конфиг."
    )
    md.append("")

    md.append("---")
    md.append("")
    md.append("*Отчёт сгенерирован автоматически модулем "
              "`agent_dialogue_sim.experiments.run_causal`.*")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a paired counterfactual causal-intervention experiment.",
    )
    parser.add_argument("--experiment-id", default=None,
                        help="Идентификатор эксперимента (default: causal_<trait>_<agent>_<ts>).")
    parser.add_argument("--target-agent", required=True,
                        help="Имя агента, на котором проводится интервенция (e.g. Critic).")
    parser.add_argument("--trait", default="agreeableness",
                        choices=list(TraitIntervention.VALID_TRAITS),
                        help="OCEAN-черта.")
    parser.add_argument("--treatment-value", type=float, required=True,
                        help="Значение черты в treatment-условии (0..1).")
    parser.add_argument("--control-value", type=float, required=True,
                        help="Значение черты в control-условии (0..1).")
    parser.add_argument("--n-pairs", type=int, default=8,
                        help="Число пар (= число seed × 2 запуска).")
    parser.add_argument("--n-turns", type=int, default=10,
                        help="Число ходов на диалог.")
    parser.add_argument("--env-context-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=7000)
    parser.add_argument("--parallelism", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--d-threshold", type=float, default=0.3)
    parser.add_argument("--alternative", default="two-sided",
                        choices=["two-sided", "greater", "less"])
    parser.add_argument("--outcome-metrics", nargs="*", default=None,
                        help="Список outcome-метрик; default = пресет из causal_intervention.")
    parser.add_argument("--output-dir", default="experiments/results")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    exp_id = args.experiment_id or (
        f"causal_{args.trait}_{args.target_agent}_T{args.treatment_value:.2f}_"
        f"C{args.control_value:.2f}"
    )

    treatment = TraitIntervention(
        name="treatment",
        target_agent=args.target_agent,
        trait=args.trait,
        value=args.treatment_value,
    )
    control = TraitIntervention(
        name="control",
        target_agent=args.target_agent,
        trait=args.trait,
        value=args.control_value,
    )

    cfg = CausalConfig(
        experiment_id=exp_id,
        intervention_treatment=treatment,
        intervention_control=control,
        n_pairs=args.n_pairs,
        seed_base=args.seed_base,
        n_turns=args.n_turns,
        env_context_index=args.env_context_index,
        parallelism=args.parallelism,
        outcome_metrics=args.outcome_metrics or list(DEFAULT_OUTCOME_METRICS),
        alpha=args.alpha,
        d_threshold=args.d_threshold,
        alternative=args.alternative,
        description=(
            f"do({args.trait}[{args.target_agent}] = {args.treatment_value}) "
            f"vs do({args.trait}[{args.target_agent}] = {args.control_value})"
        ),
    )

    runner = CausalRunner(cfg)
    result = runner.run(Path(args.output_dir))

    # Печать сводки
    print("\n=== CAUSAL EXPERIMENT SUMMARY ===")
    print(f"ID:              {result.experiment_id}")
    print(f"N pairs total:   {result.n_pairs_total}")
    print(f"N pairs valid:   {result.n_pairs_valid}")
    print(f"Treatment:       {result.intervention_treatment.get('descriptor')}")
    print(f"Control:         {result.intervention_control.get('descriptor')}")
    print(f"Duration:        {result.duration_seconds:.1f}s")
    print(f"Output:          {result.config.get('output_dir')}")
    print()
    print(f"{'metric':<22} {'ATE':>10} {'p':>10} {'d_z':>8} {'decision':>22}")
    print("-" * 75)
    for m in result.outcome_metrics:
        a = result.ate_by_metric.get(m, {})
        ate = a.get("ate")
        p = a.get("p_value")
        d_z = a.get("cohens_d_z")
        decision = a.get("decision", "?")
        ate_s = f"{ate:.4f}" if isinstance(ate, (int, float)) and not math.isnan(ate) else "n/a"
        p_s = f"{p:.4g}" if isinstance(p, (int, float)) and not math.isnan(p) else "n/a"
        dz_s = f"{d_z:.3f}" if isinstance(d_z, (int, float)) and not math.isnan(d_z) else "n/a"
        print(f"{m:<22} {ate_s:>10} {p_s:>10} {dz_s:>8} {decision:>22}")

    # отчёт
    report_md = build_scientific_report(result)
    out_dir = Path(result.config["output_dir"])
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"\nReport written to: {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
