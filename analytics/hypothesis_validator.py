"""
Модуль проверки гипотез, выдвинутых на основе теорий о социальных отношениях.

Проверяет гипотезы путём:
1. Сравнения метрик между временными окнами
2. Статистического анализа изменений
3. Валидации предсказаний гипотез
4. Оценки значимости эффектов
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import math
from collections import defaultdict


@dataclass
class HypothesisValidation:
    """Результат проверки гипотезы.

    Содержит ДВА уровня оценки:
      1. Эвристический (`confidence`) — legacy, для обратной совместимости с web_app.
         Считается по правилу "0.3 + бонусы за примеры/метрики". Это НЕ статистика.
      2. Формально-статистический (`p_value`, `effect_size`, `decision_formal`) —
         результат честного теста (scipy.stats / statsmodels) на временной серии
         из `metrics_history`. Это то, что должно использоваться в научной защите.

    Если formal-полей нет (None) — значит history слишком короткая (<3 снимков)
    или для этой категории формальный тест ещё не реализован.
    """
    hypothesis_id: str
    hypothesis_text: str
    framework: str
    status: str  # "confirmed", "rejected", "partial", "insufficient_data"
    confidence: float  # 0.0 - 1.0 (HEURISTIC — не статистика)
    evidence: List[str]
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None
    change_magnitude: Optional[float] = None
    statistical_significance: Optional[float] = None  # legacy, not used
    # === Формальная статистика (заполняется при достаточной history) ===
    p_value: Optional[float] = None              # raw p-value of statistical test
    effect_size: Optional[float] = None          # Cohen's d / r / etc.
    effect_size_name: Optional[str] = None       # 'cohens_d' / 'spearman_rho' / ...
    test_used: Optional[str] = None              # 'one_sample_t_test' / 'spearman' / ...
    null_hypothesis: Optional[str] = None        # текст H0 для отчёта
    alpha: float = 0.05
    n_observations: Optional[int] = None
    decision_formal: Optional[str] = None        # 'reject_H0' / 'fail_to_reject_H0' / None


class HypothesisValidator:
    """
    Валидатор гипотез на основе метрик диалога.
    
    Проверяет гипотезы, выдвинутые на основе научных теорий,
    сравнивая метрики в разные моменты времени и оценивая
    статистическую значимость изменений.
    """
    
    def __init__(self):
        self.hypothesis_history: List[Dict[str, Any]] = []
        self.metrics_history: List[Tuple[int, Dict[str, Any]]] = []  # (turn, metrics)
    
    def add_metrics_snapshot(self, turn: int, metrics: Dict[str, Any]):
        """Добавить снимок метрик для последующей проверки гипотез."""
        self.metrics_history.append((turn, metrics))
        # Храним только последние 20 снимков для экономии памяти
        if len(self.metrics_history) > 20:
            self.metrics_history = self.metrics_history[-20:]

    # =====================================================================
    # ФОРМАЛЬНЫЕ СТАТИСТИЧЕСКИЕ ТЕСТЫ
    # =====================================================================
    # В отличие от эвристического `confidence`, эти методы возвращают
    # реальный p-value через scipy.stats. Если history слишком коротка
    # (< MIN_OBS_FOR_TEST), возвращают None и валидатор работает только
    # на эвристике.
    # =====================================================================

    MIN_OBS_FOR_TEST = 3  # минимум снимков для теста

    def _extract_series(self, getter) -> List[float]:
        """Извлечь скалярную метрику из metrics_history. getter — callable(metrics_dict) -> float|None."""
        out = []
        for _turn, m in self.metrics_history:
            try:
                v = getter(m)
                if v is not None:
                    out.append(float(v))
            except (KeyError, TypeError, AttributeError):
                continue
        return out

    def _formal_one_sample_test(
        self,
        getter,
        threshold: float,
        alternative: str = "greater",
        alpha: float = 0.05,
        d_threshold: float = 0.5,
        h0_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Одновыборочный t-test: μ_metric vs threshold.
        H0: μ = threshold; H1 зависит от alternative ('greater'/'less'/'two-sided').
        Решение: reject_H0 если p < alpha И |Cohen's d| ≥ d_threshold.
        """
        from scipy import stats as scipy_stats
        values = self._extract_series(getter)
        if len(values) < self.MIN_OBS_FOR_TEST:
            return None
        try:
            t_stat, p_val = scipy_stats.ttest_1samp(
                values, popmean=threshold, alternative=alternative,
            )
        except Exception:
            return None
        mean_v = sum(values) / len(values)
        if len(values) > 1:
            var = sum((v - mean_v) ** 2 for v in values) / (len(values) - 1)
            sd = var ** 0.5
        else:
            sd = 0.0
        d = (mean_v - threshold) / sd if sd > 0 else 0.0
        decision = (
            "reject_H0"
            if (p_val < alpha and abs(d) >= d_threshold)
            else "fail_to_reject_H0"
        )
        return {
            "test_used": "one_sample_t_test",
            "p_value": float(p_val),
            "effect_size": float(d),
            "effect_size_name": "cohens_d",
            "n_observations": len(values),
            "alpha": alpha,
            "null_hypothesis": h0_text or f"μ = {threshold}",
            "decision_formal": decision,
            "_aux": {"mean": mean_v, "sd": sd, "threshold": threshold,
                     "alternative": alternative},
        }

    def _formal_trend_test(
        self,
        getter,
        alpha: float = 0.05,
        rho_threshold: float = 0.3,
        h0_text: str = "no monotonic trend over time",
    ) -> Optional[Dict[str, Any]]:
        """
        Тест монотонного тренда метрики во времени через корреляцию Спирмена.
        H0: ρ = 0 (нет тренда); H1: ρ ≠ 0.
        """
        from scipy import stats as scipy_stats
        values = self._extract_series(getter)
        if len(values) < self.MIN_OBS_FOR_TEST:
            return None
        turns = list(range(len(values)))
        try:
            rho, p_val = scipy_stats.spearmanr(turns, values)
        except Exception:
            return None
        if math.isnan(rho) or math.isnan(p_val):
            return None
        decision = (
            "reject_H0"
            if (p_val < alpha and abs(rho) >= rho_threshold)
            else "fail_to_reject_H0"
        )
        return {
            "test_used": "spearman_correlation",
            "p_value": float(p_val),
            "effect_size": float(rho),
            "effect_size_name": "spearman_rho",
            "n_observations": len(values),
            "alpha": alpha,
            "null_hypothesis": h0_text,
            "decision_formal": decision,
            "_aux": {"trend": "increasing" if rho > 0 else "decreasing"},
        }

    def _formal_proportion_test(
        self,
        successes: int,
        n: int,
        p_null: float = 0.5,
        alternative: str = "greater",
        alpha: float = 0.05,
        diff_threshold: float = 0.1,
        h0_text: str = "",
    ) -> Optional[Dict[str, Any]]:
        """
        Биномиальный тест на пропорцию.
        H0: p = p_null; H1 зависит от alternative.
        Решение: reject_H0 если p_value < alpha И |p_obs - p_null| ≥ diff_threshold.
        """
        from scipy import stats as scipy_stats
        if n <= 0:
            return None
        try:
            res = scipy_stats.binomtest(successes, n, p=p_null, alternative=alternative)
        except Exception:
            return None
        p_obs = successes / n
        diff = p_obs - p_null
        decision = (
            "reject_H0"
            if (res.pvalue < alpha and abs(diff) >= diff_threshold)
            else "fail_to_reject_H0"
        )
        return {
            "test_used": "binomial_test",
            "p_value": float(res.pvalue),
            "effect_size": float(diff),
            "effect_size_name": "prop_diff",
            "n_observations": n,
            "alpha": alpha,
            "null_hypothesis": h0_text or f"p = {p_null}",
            "decision_formal": decision,
            "_aux": {"p_observed": p_obs, "p_null": p_null,
                     "alternative": alternative},
        }

    def _attach_formal_stats(self, validation: "HypothesisValidation",
                              formal: Optional[Dict[str, Any]]) -> "HypothesisValidation":
        """Записать формальные поля в HypothesisValidation."""
        if formal is None:
            return validation
        validation.test_used = formal.get("test_used")
        validation.p_value = formal.get("p_value")
        validation.effect_size = formal.get("effect_size")
        validation.effect_size_name = formal.get("effect_size_name")
        validation.n_observations = formal.get("n_observations")
        validation.alpha = formal.get("alpha", 0.05)
        validation.null_hypothesis = formal.get("null_hypothesis")
        validation.decision_formal = formal.get("decision_formal")
        # Добавить в evidence строку с p-value
        p = formal.get("p_value")
        d = formal.get("effect_size")
        ds = formal.get("effect_size_name", "?")
        decision = formal.get("decision_formal", "?")
        n = formal.get("n_observations", "?")
        test = formal.get("test_used", "?")
        marker = "✅" if decision == "reject_H0" else "❌"
        validation.evidence.append(
            f"{marker} Формальный тест: {test}, n={n}, p={p:.4g}, "
            f"{ds}={d:.3f}, α={formal.get('alpha', 0.05)} → {decision.upper()}"
        )
        return validation
    
    def validate_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        current_metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """
        Проверить гипотезу на основе текущих и предыдущих метрик.
        
        Args:
            hypothesis: Словарь с гипотезой (из generate_scientific_hypotheses)
            current_metrics: Текущие метрики
            previous_metrics: Предыдущие метрики для сравнения (опционально)
        
        Returns:
            HypothesisValidation с результатами проверки
        """
        framework = hypothesis.get("framework", "Unknown")
        finding = hypothesis.get("finding", "")
        category = hypothesis.get("category", "General")
        
        # Если нет предыдущих метрик, используем последние из истории
        if previous_metrics is None and len(self.metrics_history) >= 2:
            previous_metrics = self.metrics_history[-2][1]
        
        # Генерируем ID гипотезы
        hyp_id = f"{category}_{framework}_{hash(finding) % 10000}"
        
        # Проверяем гипотезу в зависимости от категории — эвристический путь
        if category == "Group Development":
            validation = self._validate_group_development(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        elif category == "Network Structure":
            validation = self._validate_network_structure(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        elif category == "Social Capital":
            validation = self._validate_social_capital(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        elif category == "Emotional Climate":
            validation = self._validate_emotional_climate(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        elif category == "Dialogue Structure":
            validation = self._validate_dialogue_structure(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        elif category == "Turn-Taking":
            validation = self._validate_turn_taking(hyp_id, hypothesis, current_metrics, previous_metrics, history)
        else:
            validation = self._validate_general(hyp_id, hypothesis, current_metrics, previous_metrics, history)

        # === ФОРМАЛЬНАЯ СТАТИСТИКА — поверх эвристики ===
        # Если history достаточна, прикрепляем результат честного теста.
        # Это НЕ заменяет confidence, но даёт научно валидную оценку рядом.
        formal = self._compute_formal_stats(category, finding)
        return self._attach_formal_stats(validation, formal)

    def _compute_formal_stats(self, category: str, finding: str) -> Optional[Dict[str, Any]]:
        """
        Подбирает подходящий формальный тест под категорию гипотезы.
        Все тесты используют self.metrics_history (≥3 снимков для запуска).
        Возвращает None если данных недостаточно.

        Извлечение метрик из metrics: компонент `summary.window` или `summary.all`,
        либо `scientific.<key>` в зависимости от категории.
        """
        def s_get(m, *path, default=None):
            cur = m
            for p in path:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return default
            return cur

        if category == "Network Structure":
            # H0: max degree centrality ≤ 0.5; H1: > 0.5 (high centralization)
            return self._formal_one_sample_test(
                getter=lambda m: max(
                    (s_get(m, "scientific", "centrality_metrics", "in_degree", default={}) or {}).values() or [0]
                ) if s_get(m, "scientific", "centrality_metrics", "in_degree") else None,
                threshold=0.5,
                alternative="greater",
                d_threshold=0.5,
                h0_text="μ(max in-degree centrality) ≤ 0.5",
            )

        if category == "Emotional Climate":
            # H0: μ(avg_tone) = 0 (neutral); H1: ≠ 0
            return self._formal_one_sample_test(
                getter=lambda m: s_get(m, "summary", "window", "avg_tone"),
                threshold=0.0,
                alternative="two-sided",
                d_threshold=0.5,
                h0_text="μ(avg_tone) = 0 (нейтральная атмосфера)",
            )

        if category == "Social Capital":
            # H0: group cohesion ≤ 0.5; H1: > 0.5
            return self._formal_one_sample_test(
                getter=lambda m: s_get(m, "scientific", "group_cohesion"),
                threshold=0.5,
                alternative="greater",
                d_threshold=0.5,
                h0_text="μ(group_cohesion) ≤ 0.5",
            )

        if category == "Turn-Taking":
            # H0: Gini ≤ 0.4 (равные); H1: > 0.4 (неравные)
            return self._formal_one_sample_test(
                getter=lambda m: s_get(m, "scientific", "turn_taking", "gini_coefficient"),
                threshold=0.4,
                alternative="greater",
                d_threshold=0.5,
                h0_text="μ(Gini) ≤ 0.4 (распределение реплик равно)",
            )

        if category == "Dialogue Structure":
            # Тест на тренд: actionability_rate растёт во времени?
            return self._formal_trend_test(
                getter=lambda m: s_get(m, "summary", "window", "actionability_rate"),
                rho_threshold=0.3,
                h0_text="actionability_rate не имеет монотонного тренда во времени",
            )

        if category == "Group Development":
            # Доля снимков, классифицированных как доминирующая стадия
            stages = []
            for _t, m in self.metrics_history:
                stage = s_get(m, "scientific", "group_stage", "stage")
                if stage:
                    stages.append(stage)
            if len(stages) < self.MIN_OBS_FOR_TEST:
                return None
            from collections import Counter
            counter = Counter(stages)
            dominant_stage, hits = counter.most_common(1)[0]
            # H0: dominant_stage — случайное (p=0.25 для 4 стадий); H1: > 0.25
            return self._formal_proportion_test(
                successes=hits,
                n=len(stages),
                p_null=0.25,
                alternative="greater",
                diff_threshold=0.25,
                h0_text=f"P(stage='{dominant_stage}') = 0.25 (случайно среди 4 стадий)",
            )

        return None
    
    def _validate_group_development(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез о развитии группы."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        # Извлекаем научные метрики
        scientific = current.get("scientific")
        if not scientific:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=hypothesis.get("finding", ""),
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных для проверки"]
            )
        
        # Проверяем стадию развития группы
        stage = scientific.group_stage.get("stage", "forming")
        stage_confidence = scientific.group_stage.get("confidence", 0)
        
        evidence.append(f"📊 Метрика: Текущая стадия: {stage} (уверенность: {stage_confidence:.0%})")
        confidence = stage_confidence
        
        # Всегда добавляем примеры из диалога для текущей стадии, если есть history
        if history:
            recent_messages = history[-10:]
            if recent_messages:
                evidence.append(f"💬 Примеры текущей стадии '{stage}' (последние сообщения):")
                for r in recent_messages[:3]:
                    evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                confidence = min(1.0, confidence + 0.1)
        
        # Если есть предыдущие метрики, проверяем прогресс
        if previous:
            prev_scientific = previous.get("scientific")
            if prev_scientific:
                prev_stage = prev_scientific.group_stage.get("stage", "forming")
                if prev_stage != stage:
                    evidence.append(f"📊 Метрика: Обнаружен переход: {prev_stage} → {stage}")
                    confidence = min(1.0, confidence + 0.2)
        
        # Проверяем метрики, соответствующие стадии
        if stage == "forming":
            # На стадии формирования должна быть низкая плотность сети
            density = scientific.network_density
            if density < 0.4:
                evidence.append(f"📊 Метрика: Низкая плотность сети ({density:.0%}) соответствует стадии формирования")
                confidence = min(1.0, confidence + 0.1)
                
                # Примеры из диалога: начальные сообщения, знакомство
                if history:
                    initial_messages = [r for r in history[:10] if r.get("turn", 0) <= 5]
                    if initial_messages:
                        evidence.append(f"💬 Примеры стадии формирования (первые сообщения):")
                        for r in initial_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.1)
        elif stage == "storming":
            # На стадии конфликта должны быть негативные тона
            if history:
                negative_messages = [r for r in history[-15:] if r.get("tone") == "negative"]
                if negative_messages:
                    evidence.append(f"💬 Примеры стадии конфликта (негативные сообщения):")
                    for r in negative_messages[:3]:
                        evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                    confidence = min(1.0, confidence + 0.15)
                else:
                    # Если нет негативных, показываем последние сообщения как примеры стадии
                    recent_messages = history[-5:]
                    if recent_messages:
                        evidence.append(f"💬 Примеры стадии конфликта (последние сообщения):")
                        for r in recent_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.05)
        elif stage == "norming":
            # На стадии нормирования должны быть согласования и компромиссы
            if history:
                norming_keywords = ["agree", "compromise", "consensus", "together", "we should", "let's", "okay", "yes", "sounds good", "that works"]
                norming_messages = [
                    r for r in history[-15:]
                    if any(kw in r.get("reply", "").lower() for kw in norming_keywords)
                ]
                if norming_messages:
                    evidence.append(f"💬 Примеры стадии нормирования (согласования и компромиссы):")
                    for r in norming_messages[:3]:
                        evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                    confidence = min(1.0, confidence + 0.15)
                else:
                    # Если нет ключевых слов, показываем последние сообщения как примеры нормирования
                    recent_messages = history[-5:]
                    if recent_messages:
                        evidence.append(f"💬 Примеры стадии нормирования (последние сообщения):")
                        for r in recent_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.1)
        elif stage == "performing":
            # На стадии продуктивной работы должна быть высокая взаимность
            summary = current.get("summary", {}).get("window", {})
            reciprocity = summary.get("reciprocity", 0)
            if reciprocity > 0.6:
                evidence.append(f"📊 Метрика: Высокая взаимность ({reciprocity:.0%}) подтверждает продуктивную стадию")
                confidence = min(1.0, confidence + 0.15)
                
                # Примеры из диалога: взаимные обмены, продуктивные сообщения
                if history:
                    # Находим пары взаимных сообщений
                    recent_history = history[-15:]
                    mutual_exchanges = []
                    for i, r1 in enumerate(recent_history):
                        for r2 in recent_history[i+1:]:
                            if (r1.get("speaker") == r2.get("target") and 
                                r2.get("speaker") == r1.get("target")):
                                mutual_exchanges.append((r1, r2))
                                if len(mutual_exchanges) >= 2:
                                    break
                        if len(mutual_exchanges) >= 2:
                            break
                    
                    if mutual_exchanges:
                        evidence.append(f"💬 Примеры продуктивной стадии (взаимные обмены):")
                        for r1, r2 in mutual_exchanges[:2]:
                            evidence.append(f"  • Ход {r1.get('turn')}: {r1.get('speaker')} → {r1.get('target')}: \"{r1.get('reply', '')[:60]}...\"")
                            evidence.append(f"    Ход {r2.get('turn')}: {r2.get('speaker')} → {r2.get('target')}: \"{r2.get('reply', '')[:60]}...\"")
                        confidence = min(1.0, confidence + 0.1)
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "insufficient_data")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence,
            metrics_before=previous.get("summary", {}).get("window", {}) if previous else None,
            metrics_after=current.get("summary", {}).get("window", {})
        )
    
    def _validate_network_structure(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез о структуре сети."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        scientific = current.get("scientific")
        if not scientific:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=finding,
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных"]
            )
        
        # Проверяем централизацию
        if "центральную позицию" in finding.lower() or "central" in finding.lower():
            centrality = scientific.centrality
            if centrality:
                max_central = max(
                    centrality.items(),
                    key=lambda x: x[1].get("total_centrality", 0)
                )
                central_agent = max_central[0]
                max_val = max_central[1].get("total_centrality", 0)
                evidence.append(f"📊 Метрика: Максимальная центральность агента '{central_agent}': {max_val:.0%}")
                if max_val > 0.7:
                    evidence.append("✅ Высокая централизация подтверждает гипотезу")
                    confidence = 0.8
                else:
                    confidence = 0.4
                
                # Всегда добавляем примеры из диалога для центрального агента, если есть history
                if history:
                    central_messages = [r for r in history[-15:] if r.get("speaker") == central_agent]
                    if central_messages:
                        evidence.append(f"💬 Примеры активности центрального агента '{central_agent}' (сообщений: {len(central_messages)} из последних 15):")
                        for r in central_messages[:3]:
                            target = r.get("target", "all")
                            evidence.append(f"  • Ход {r.get('turn')}: {central_agent} → {target}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.1)
                    else:
                        # Если нет сообщений в последних 15, ищем во всей истории
                        all_central = [r for r in history if r.get("speaker") == central_agent]
                        if all_central:
                            evidence.append(f"💬 Примеры активности центрального агента '{central_agent}' (всего сообщений: {len(all_central)}):")
                            for r in all_central[-3:]:
                                target = r.get("target", "all")
                                evidence.append(f"  • Ход {r.get('turn')}: {central_agent} → {target}: \"{r.get('reply', '')[:70]}...\"")
                            confidence = min(1.0, confidence + 0.1)
        
        # Проверяем плотность сети
        if "плотность" in finding.lower() or "density" in finding.lower():
            density = scientific.network_density
            evidence.append(f"📊 Метрика: Плотность сети: {density:.0%}")
            if density < 0.5:
                evidence.append("✅ Низкая плотность подтверждает наличие структурных дыр")
                confidence = 0.75
                
                # Примеры из диалога: неполные связи между агентами
                if history:
                    agents = set()
                    connections = set()
                    for r in history[-15:]:
                        agents.add(r.get("speaker"))
                        if r.get("target"):
                            agents.add(r.get("target"))
                            connections.add((r.get("speaker"), r.get("target")))
                    
                    # Находим агентов, которые не общаются напрямую
                    isolated_pairs = []
                    agent_list = list(agents)
                    for i, a1 in enumerate(agent_list):
                        for a2 in agent_list[i+1:]:
                            if (a1, a2) not in connections and (a2, a1) not in connections:
                                isolated_pairs.append((a1, a2))
                                if len(isolated_pairs) >= 2:
                                    break
                        if len(isolated_pairs) >= 2:
                            break
                    
                    if isolated_pairs:
                        evidence.append(f"💬 Структурные дыры: отсутствуют прямые связи между парами агентов:")
                        for a1, a2 in isolated_pairs[:2]:
                            evidence.append(f"  • '{a1}' ↔ '{a2}' (нет прямого обмена сообщениями)")
                        confidence = min(1.0, confidence + 0.1)
            else:
                confidence = 0.3
        
        # Сравнение с предыдущими метриками
        if previous:
            prev_scientific = previous.get("scientific")
            if prev_scientific:
                prev_density = prev_scientific.network_density
                current_density = scientific.network_density
                if abs(current_density - prev_density) > 0.1:
                    change = current_density - prev_density
                    evidence.append(f"Изменение плотности: {change:+.0%}")
                    confidence = min(1.0, confidence + 0.15)
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "rejected")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_social_capital(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез о социальном капитале."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        scientific = current.get("scientific")
        if not scientific or not scientific.social_capital:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=finding,
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных о социальном капитале"]
            )
        
        social_cap = scientific.social_capital
        cohesion = social_cap.get("group_cohesion", 0)
        evidence.append(f"📊 Метрика: Групповая сплочённость: {cohesion:.0%}")
        
        if "низкая" in finding.lower() or "low" in finding.lower():
            if cohesion < 0.3:
                evidence.append("✅ Низкая сплочённость подтверждает гипотезу")
                confidence = 0.75
            else:
                confidence = 0.3
        
        # Проверяем изолированных участников
        if "agents" in social_cap:
            isolated = [
                a for a, data in social_cap["agents"].items()
                if data.get("sociometric_status", 0) < 0.3
            ]
            if isolated:
                evidence.append(f"📊 Метрика: Участники с низким сетевым статусом: {', '.join(isolated)}")
                confidence = min(1.0, confidence + 0.2)
                
                # Примеры из диалога: мало сообщений к изолированным агентам
                if history:
                    for agent in isolated[:2]:
                        messages_to = [r for r in history[-15:] if r.get("target") == agent]
                        messages_from = [r for r in history[-15:] if r.get("speaker") == agent]
                        evidence.append(f"💬 Пример активности агента '{agent}' (низкий сетевой статус):")
                        if messages_from:
                            # Показываем все сообщения от агента, даже если их мало
                            for r in messages_from[:2]:
                                evidence.append(f"  • Ход {r.get('turn')}: {agent} → {r.get('target', 'all')}: \"{r.get('reply', '')[:60]}...\"")
                            if len(messages_from) <= 2:
                                evidence.append(f"  • Всего сообщений от '{agent}': {len(messages_from)} (низкая активность)")
                        else:
                            evidence.append(f"  • Агент '{agent}' не отправил ни одного сообщения в последних 15 ходах")
                        if len(messages_to) <= 2:
                            evidence.append(f"  • Сообщений к '{agent}': {len(messages_to)} (низкая вовлечённость)")
                        confidence = min(1.0, confidence + 0.1)
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "rejected")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_emotional_climate(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез об эмоциональном климате."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        scientific = current.get("scientific")
        if not scientific or not scientific.emotion_distribution:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=finding,
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных об эмоциях"]
            )
        
        emotions = scientific.emotion_distribution
        dominant = scientific.dominant_emotion
        
        evidence.append(f"📊 Метрика: Доминирующая эмоция: {dominant}")
        
        # Проверяем негативные эмоции
        if "негатив" in finding.lower() or "negative" in finding.lower():
            negative = emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("sadness", 0)
            evidence.append(f"📊 Метрика: Доля негативных эмоций: {negative:.0%}")
            if negative > 0.4:
                evidence.append("✅ Преобладание негативных эмоций подтверждает гипотезу")
                confidence = 0.8
                
                # Примеры из диалога: сообщения с негативными эмоциями и тоном
                if history:
                    negative_messages = [
                        r for r in history[-15:]
                        if r.get("tone") == "negative" or 
                           r.get("emotion", "").lower() in ["angry", "fear", "sad", "concerned", "frustrated", "worried"]
                    ]
                    if negative_messages:
                        evidence.append(f"💬 Примеры негативных сообщений из диалога ({len(negative_messages)} из последних 15):")
                        for r in negative_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')} (тон: {r.get('tone')}, эмоция: {r.get('emotion')}): \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.15)
            else:
                confidence = 0.3
        
        # Проверяем позитивные эмоции
        if "позитив" in finding.lower() or "positive" in finding.lower():
            positive = emotions.get("joy", 0) + emotions.get("anticipation", 0)
            evidence.append(f"📊 Метрика: Доля позитивных эмоций: {positive:.0%}")
            if positive > 0.5:
                evidence.append("✅ Позитивный эмоциональный фон подтверждает гипотезу")
                confidence = 0.8
                
                # Примеры из диалога: сообщения с позитивными эмоциями
                if history:
                    positive_messages = [
                        r for r in history[-15:]
                        if r.get("tone") == "positive" or 
                           r.get("emotion", "").lower() in ["joy", "joyful", "excited", "happy", "enthusiastic", "optimistic"]
                    ]
                    if positive_messages:
                        evidence.append(f"💬 Примеры позитивных сообщений из диалога ({len(positive_messages)} из последних 15):")
                        for r in positive_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')} (тон: {r.get('tone')}, эмоция: {r.get('emotion')}): \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.15)
            else:
                confidence = 0.3
        
        # Сравнение с предыдущими метриками
        if previous:
            prev_scientific = previous.get("scientific")
            if prev_scientific and prev_scientific.emotion_distribution:
                prev_emotions = prev_scientific.emotion_distribution
                prev_negative = prev_emotions.get("anger", 0) + prev_emotions.get("fear", 0) + prev_emotions.get("sadness", 0)
                current_negative = emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("sadness", 0)
                change = current_negative - prev_negative
                if abs(change) > 0.1:
                    evidence.append(f"Изменение негативных эмоций: {change:+.0%}")
                    confidence = min(1.0, confidence + 0.15)
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "rejected")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_dialogue_structure(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез о структуре диалога."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        scientific = current.get("scientific")
        if not scientific or not scientific.dialogue_act_profile:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=finding,
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных о диалоговых актах"]
            )
        
        dialogue_acts = scientific.dialogue_act_profile
        task_ratio = dialogue_acts.get("task_ratio", 0)
        pos_neg_ratio = dialogue_acts.get("positive_negative_ratio", 1)
        
        evidence.append(f"📊 Метрика: Задачная ориентация: {task_ratio:.0%}")
        evidence.append(f"📊 Метрика: Соотношение позитивных/негативных актов: {pos_neg_ratio:.2f}")
        
        # Проверяем задачную ориентацию
        if "высокая задачная" in finding.lower() or "high task" in finding.lower():
            if task_ratio > 0.7:
                evidence.append("✅ Высокая задачная ориентация подтверждает гипотезу")
                confidence = 0.75
                
                # Примеры из диалога: сообщения с задачной направленностью
                if history:
                    task_keywords = ["plan", "next", "should", "need", "must", "how", "what", "when", "proposal", "suggest"]
                    task_messages = [
                        r for r in history[-15:]
                        if any(kw in r.get("reply", "").lower() for kw in task_keywords)
                    ]
                    if task_messages:
                        evidence.append(f"💬 Примеры задачно-ориентированных сообщений ({len(task_messages)} из последних 15):")
                        for r in task_messages[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.1)
            else:
                confidence = 0.3
        
        # Проверяем соотношение позитивных/негативных
        if "негативные" in finding.lower() or "negative" in finding.lower():
            if pos_neg_ratio < 1:
                evidence.append("✅ Преобладание негативных актов подтверждает гипотезу")
                confidence = 0.75
                
                # Примеры из диалога: негативные акты (disagreement, criticism)
                if history:
                    negative_acts = [
                        r for r in history[-15:]
                        if r.get("tone") == "negative" or 
                           any(word in r.get("reply", "").lower() for word in ["no", "but", "however", "disagree", "wrong", "problem", "issue", "concern"])
                    ]
                    if negative_acts:
                        evidence.append(f"💬 Примеры негативных диалоговых актов ({len(negative_acts)} из последних 15):")
                        for r in negative_acts[:3]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} → {r.get('target', 'all')}: \"{r.get('reply', '')[:70]}...\"")
                        confidence = min(1.0, confidence + 0.1)
            else:
                confidence = 0.3
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "rejected")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_turn_taking(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Проверка гипотез о чередовании реплик."""
        finding = hypothesis.get("finding", "")
        evidence = []
        confidence = 0.5
        
        scientific = current.get("scientific")
        if not scientific or not scientific.turn_taking:
            return HypothesisValidation(
                hypothesis_id=hyp_id,
                hypothesis_text=finding,
                framework=hypothesis.get("framework", ""),
                status="insufficient_data",
                confidence=0.0,
                evidence=["Недостаточно данных о чередовании реплик"]
            )
        
        turn = scientific.turn_taking
        gini = turn.get("turn_inequality_gini", 0)
        adj_rate = turn.get("adjacency_completion_rate", 0)
        
        evidence.append(f"📊 Метрика: Неравенство в репликах (коэффициент Джини): {gini:.2f}")
        evidence.append(f"📊 Метрика: Завершение смежных пар (вопрос-ответ): {adj_rate:.0%}")
        
        # Проверяем неравенство
        if "неравномерное" in finding.lower() or "inequality" in finding.lower():
            if gini > 0.3:
                evidence.append("✅ Высокое неравенство подтверждает гипотезу")
                confidence = 0.75
                
                # Примеры из диалога: подсчёт сообщений по агентам
                if history:
                    from collections import Counter
                    speaker_counts = Counter(r.get("speaker") for r in history[-15:])
                    if speaker_counts:
                        top_speaker = speaker_counts.most_common(1)[0]
                        total = sum(speaker_counts.values())
                        top_share = top_speaker[1] / total if total > 0 else 0
                        evidence.append(f"💬 Примеры неравномерного распределения (последние 15 ходов):")
                        evidence.append(f"  • '{top_speaker[0]}' говорит в {top_share:.0%} случаев ({top_speaker[1]} из {total} сообщений)")
                        for speaker, count in speaker_counts.most_common(3):
                            evidence.append(f"  • '{speaker}': {count} сообщений")
                        confidence = min(1.0, confidence + 0.1)
            else:
                confidence = 0.3
        
        # Проверяем завершение смежных пар
        if "низкий" in finding.lower() and "завершение" in finding.lower():
            if adj_rate < 0.5:
                evidence.append("✅ Низкий показатель завершения подтверждает гипотезу")
                confidence = 0.75
                
                # Примеры из диалога: неотвеченные вопросы
                if history:
                    unanswered = []
                    for i, r1 in enumerate(history[-15:]):
                        if "?" in r1.get("reply", ""):
                            # Проверяем, ответил ли адресат в следующих 3 ходах
                            target = r1.get("target")
                            if target:
                                answered = False
                                for r2 in history[i+1:i+4]:
                                    if r2.get("speaker") == target:
                                        answered = True
                                        break
                                if not answered:
                                    unanswered.append((r1, target))
                                    if len(unanswered) >= 2:
                                        break
                    
                    if unanswered:
                        evidence.append(f"💬 Примеры неотвеченных вопросов из диалога:")
                        for r, target in unanswered[:2]:
                            evidence.append(f"  • Ход {r.get('turn')}: {r.get('speaker')} задал вопрос {target}: \"{r.get('reply', '')[:60]}...\"")
                            evidence.append(f"    → Вопрос остался без ответа в следующих ходах")
                        confidence = min(1.0, confidence + 0.15)
            else:
                confidence = 0.3
        
        status = "confirmed" if confidence > 0.6 else ("partial" if confidence > 0.3 else "rejected")
        
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=finding,
            framework=hypothesis.get("framework", ""),
            status=status,
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_general(
        self,
        hyp_id: str,
        hypothesis: Dict[str, Any],
        current: Dict[str, Any],
        previous: Optional[Dict[str, Any]],
        history: Optional[List[Dict[str, Any]]] = None
    ) -> HypothesisValidation:
        """Общая проверка гипотез."""
        return HypothesisValidation(
            hypothesis_id=hyp_id,
            hypothesis_text=hypothesis.get("finding", ""),
            framework=hypothesis.get("framework", ""),
            status="insufficient_data",
            confidence=0.3,
            evidence=["Требуется специфическая проверка для данной категории"]
        )
    
    def validate_all_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        current_metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[HypothesisValidation]:
        """
        Проверить все гипотезы.
        
        Args:
            hypotheses: Список гипотез
            current_metrics: Текущие метрики
            previous_metrics: Предыдущие метрики (опционально)
            history: История диалога для извлечения примеров (опционально)
        
        Returns:
            Список результатов проверки
        """
        validations = []
        for hyp in hypotheses:
            validation = self.validate_hypothesis(hyp, current_metrics, previous_metrics, history)
            validations.append(validation)
            self.hypothesis_history.append({
                "turn": len(self.metrics_history),
                "hypothesis": hyp,
                "validation": validation
            })
        
        return validations
    
    def get_validation_summary(self, validations: List[HypothesisValidation]) -> Dict[str, Any]:
        """Получить сводку по проверке гипотез."""
        total = len(validations)
        if total == 0:
            return {
                "total": 0,
                "confirmed": 0,
                "partial": 0,
                "rejected": 0,
                "insufficient_data": 0,
                "avg_confidence": 0.0
            }
        
        confirmed = sum(1 for v in validations if v.status == "confirmed")
        partial = sum(1 for v in validations if v.status == "partial")
        rejected = sum(1 for v in validations if v.status == "rejected")
        insufficient = sum(1 for v in validations if v.status == "insufficient_data")
        avg_confidence = sum(v.confidence for v in validations) / total
        
        return {
            "total": total,
            "confirmed": confirmed,
            "partial": partial,
            "rejected": rejected,
            "insufficient_data": insufficient,
            "avg_confidence": avg_confidence,
            "confirmation_rate": confirmed / total if total > 0 else 0.0
        }
    
    def render_validation_report(
        self,
        validations: List[HypothesisValidation],
        turn: int
    ) -> str:
        """Сформировать отчёт о проверке гипотез в формате Markdown."""
        lines = []
        lines.append(f"# ✅ Проверка гипотез — Turn {turn}")
        lines.append("")
        
        lines.append("## 📋 Методология проверки")
        lines.append("")
        lines.append("Проверка гипотез проводится на основе:")
        lines.append("")
        lines.append("1. **Количественные метрики** — значения из научного анализа:")
        lines.append("   - Извлечение релевантных метрик (центральность, плотность сети, эмоции, и т.д.)")
        lines.append("   - Сравнение с научными порогами и стандартами")
        lines.append("   - Вычисление изменений относительно предыдущих метрик")
        lines.append("")
        lines.append("2. **Примеры из диалога** — конкретные сообщения, подтверждающие паттерны:")
        lines.append("   - Анализ последних 15-20 сообщений диалога")
        lines.append("   - Фильтрация по релевантным признакам (тон, эмоция, содержание)")
        lines.append("   - Извлечение цитат с указанием хода, говорящего, адресата")
        lines.append("")
        lines.append("3. **Статистическая оценка** — уровень уверенности:")
        lines.append("   - Базовая уверенность от метрик (0.0-1.0)")
        lines.append("   - Дополнительные бонусы за наличие примеров (+0.1-0.15)")
        lines.append("   - Штрафы за противоречащие данные")
        lines.append("   - Финальный статус: confirmed (>0.6), partial (0.3-0.6), rejected (<0.3)")
        lines.append("")
        
        summary = self.get_validation_summary(validations)
        lines.append("## 📊 Сводка")
        lines.append("")
        lines.append(f"- **Всего гипотез:** {summary['total']}")
        lines.append(f"- **✅ Подтверждено:** {summary['confirmed']} ({summary['confirmed']/summary['total']*100:.0f}%)" if summary['total'] > 0 else "- **✅ Подтверждено:** 0")
        lines.append(f"- **⚠️ Частично подтверждено:** {summary['partial']}")
        lines.append(f"- **❌ Отклонено:** {summary['rejected']}")
        lines.append(f"- **❓ Недостаточно данных:** {summary['insufficient_data']}")
        lines.append(f"- **📈 Средняя уверенность:** {summary['avg_confidence']:.0%}")
        lines.append("")
        
        lines.append("## 🔍 Детали проверки")
        lines.append("")
        
        for i, val in enumerate(validations, 1):
            status_icon = {
                "confirmed": "✅",
                "partial": "⚠️",
                "rejected": "❌",
                "insufficient_data": "❓"
            }.get(val.status, "❓")
            
            status_text = {
                "confirmed": "ПОДТВЕРЖДЕНО",
                "partial": "ЧАСТИЧНО ПОДТВЕРЖДЕНО",
                "rejected": "ОТКЛОНЕНО",
                "insufficient_data": "НЕДОСТАТОЧНО ДАННЫХ"
            }.get(val.status, "НЕИЗВЕСТНО")
            
            lines.append(f"### {i}. {status_icon} {val.framework}")
            lines.append("")
            lines.append(f"**Гипотеза:** {val.hypothesis_text}")
            lines.append("")
            lines.append(f"**Статус:** {status_text} (уверенность: {val.confidence:.0%})")
            lines.append("")
            
            # Добавляем теоретическое обоснование для некоторых фреймворков
            framework_lower = val.framework.lower()
            if "centrality" in framework_lower or "sna" in framework_lower or "contractor" in framework_lower or "multidimensional" in framework_lower:
                lines.append("**Теоретическое обоснование:**")
                lines.append("")
                lines.append("Согласно теории мультиплексных сетей в командах (Contractor et al., 2012), центральность агента определяется количеством прямых связей с другими участниками сети. Агент с высокой центральностью (>70%) играет ключевую роль в коммуникации, выступая как посредник и координатор взаимодействий. Высокая централизация указывает на доминирующую позицию в структуре коммуникации и способность влиять на информационные потоки в группе.")
                lines.append("")
            elif "network" in framework_lower and ("density" in val.hypothesis_text.lower() or "structural" in val.hypothesis_text.lower() or "relational" in framework_lower):
                lines.append("**Теоретическое обоснование:**")
                lines.append("")
                lines.append("Плотность сети (Network Density) отражает долю реальных связей от всех возможных связей между участниками. Низкая плотность (<50%) указывает на наличие разрывов в реляционных событиях (Leenders et al., 2016) — отсутствие прямых связей между некоторыми участниками, что может ограничивать информационный обмен и групповую сплочённость.")
                lines.append("")
            elif "team" in framework_lower or "shuffler" in framework_lower or "temporal" in framework_lower:
                lines.append("**Теоретическое обоснование:**")
                lines.append("")
                lines.append("Модель темпоральной динамики команд (Shuffler et al., 2018; Marks et al., 2001) описывает три фазы командных процессов: переходную (transition), действия (action) и межличностную (interpersonal). Каждая фаза характеризуется специфическими паттернами взаимодействия. Мета-анализ Mathieu et al. (2017) подтверждает, что динамика переходов между фазами предсказывает эффективность команды.")
                lines.append("")
            elif "network dynamics" in framework_lower or "reagans" in framework_lower or "team chemistry" in framework_lower:
                lines.append("**Теоретическое обоснование:**")
                lines.append("")
                lines.append("Динамика сетей и командная химия (Reagans et al., 2016) определяется структурой и композицией социальных связей. Участники с низким сетевым статусом (sociometric status < 0.3) имеют ограниченный доступ к информационным ресурсам группы и могут быть изолированы от основных коммуникационных потоков.")
                lines.append("")
            
            lines.append("**Доказательства:**")
            lines.append("")
            
            # Группируем доказательства по типам
            metrics_evidence = []
            examples_evidence = []
            other_evidence = []
            
            for ev in val.evidence:
                if ev.startswith("📊"):
                    metrics_evidence.append(ev.replace("📊 Метрика: ", ""))
                elif ev.startswith("💬"):
                    examples_evidence.append(ev.replace("💬 ", ""))
                elif ev.startswith("✅") or ev.startswith("⚠️"):
                    other_evidence.append(ev)
                else:
                    other_evidence.append(ev)
            
            if metrics_evidence:
                lines.append("**Количественные метрики:**")
                for ev in metrics_evidence:
                    lines.append(f"- {ev}")
                lines.append("")
            
            if examples_evidence:
                lines.append("**Примеры из диалога:**")
                lines.append("")
                for ev in examples_evidence:
                    # Если это заголовок примера (начинается с "Примеры" или "Пример")
                    if ev.startswith("Пример") and ":" in ev:
                        lines.append(f"**{ev}**")
                    elif ev.startswith("  •"):
                        lines.append(ev)
                    elif ev.startswith("  "):
                        lines.append(ev)
                    else:
                        lines.append(f"  {ev}")
                lines.append("")
            
            if other_evidence:
                for ev in other_evidence:
                    lines.append(f"- {ev}")
                lines.append("")
            
            # Добавляем сравнение метрик, если доступно
            if val.metrics_before and val.metrics_after:
                lines.append("**Сравнение метрик (до → после):**")
                key_metrics = ["avg_tone", "reciprocity", "targeting_rate"]
                for key in key_metrics:
                    before_val = val.metrics_before.get(key)
                    after_val = val.metrics_after.get(key)
                    if before_val is not None and after_val is not None:
                        change = after_val - before_val
                        change_sign = "+" if change > 0 else ""
                        lines.append(f"- {key}: {before_val:.2f} → {after_val:.2f} ({change_sign}{change:.2f})")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        return "\n".join(lines)

