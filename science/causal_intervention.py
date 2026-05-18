"""
Каузальные интервенции для многоагентных диалоговых симуляций.

Реализует Pearl's do-операторный подход к причинному выводу (Pearl 1995, 2009)
применительно к многоагентным LLM-системам. В отличие от наблюдательного дизайна
(`P(Y | X = x)`), оценивается каузальный эффект (`P(Y | do(X = x))`) через активное
вмешательство в один из параметров системы при фиксации всего остального.

ДИЗАЙН: Paired Counterfactual (within-subject)
-----------------------------------------------
Для каждого random seed `s ∈ {1..N}` запускаются ДВЕ симуляции:
   • treatment: `do(X = x_T)` → исход Y_T(s)
   • control:   `do(X = x_C)` → исход Y_C(s)

Поскольку всё, кроме интервенции X, в этих двух мирах идентично (включая seed,
prompts, env_context), различие `Δ(s) = Y_T(s) − Y_C(s)` — это
**Individual Treatment Effect (ITE)** для конкретного seed.

Average Treatment Effect (ATE) = E[Δ(s)] оценивается выборочным средним,
а критерий значимости — paired t-test (Student's t для зависимых выборок),
который обладает значительно большей статистической мощностью, чем независимый
двухвыборочный t-test для того же N, потому что внутрисубъектная вариация
устраняется.

Identifying assumptions (Pearl, Rubin):
  • SUTVA (no interference): запуски разных seed независимы.
  • Consistency: Y_T(s) — это исход, который наблюдается ПРИ интервенции X=x_T.
  • No unmeasured confounders: все источники вариации, кроме `do(X)`, фиксированы
    через seed и общий env_context.

Виды поддерживаемых интервенций:
  • TraitIntervention   — `do(OCEAN_trait[agent] = value)`: фиксация одной
                           черты Big Five у целевого агента.
  • PromptIntervention  — `do(behavioral_directive = directive)`: добавление
                           инструкции к системному промпту агента.

Связь с существующей научной базой проекта:
  • Big Five / OCEAN: Goldberg (1990); тестируем каузальный эффект черты
    на групповую динамику.
  • Wheelan (group development): outcome — переход к стадии storming/norming
    как функция интервенции.
  • Barsade (emotional contagion): outcome — распространение тона/эмоций.
  • Borgatti (SNA): outcome — reciprocity, центральности.
  • Tausczik & Pennebaker (LSM): outcome — длина реплик, языковая согласованность.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Метрики, по которым считается ITE/ATE по умолчанию.
DEFAULT_OUTCOME_METRICS: tuple[str, ...] = (
    "avg_tone",
    "avg_reply_length",
    "actionability_rate",
    "reciprocity",
    "emotion_entropy",
    "question_rate",
)


# ---------------------------------------------------------------------------
# Интервенции
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    """Базовая каузальная интервенция do(X=x).

    Поле `descriptor` используется для документирования каузального графа в отчёте.
    """
    name: str
    target_agent: str           # имя агента, к которому применяется интервенция
    descriptor: str = ""         # человекочитаемое описание do(X=x)

    def apply_to_agent_overrides(self, overrides: dict) -> dict:
        """Возвращает новый словарь agent_overrides с применённой интервенцией."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        return {"type": type(self).__name__, **asdict(self)}


@dataclass
class TraitIntervention(Intervention):
    """do(OCEAN_trait[agent] = value).

    Пример: TraitIntervention("high_A", "Critic", "agreeableness", 0.9)
    зафиксирует agreeableness у агента "Critic" на 0.9, не трогая остальные черты.
    """
    trait: str = "agreeableness"   # one of OCEAN
    value: float = 0.5

    VALID_TRAITS = (
        "openness", "conscientiousness", "extraversion",
        "agreeableness", "neuroticism",
    )

    def __post_init__(self):
        if self.trait not in self.VALID_TRAITS:
            raise ValueError(f"trait must be one of {self.VALID_TRAITS}, got {self.trait!r}")
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(f"value must be in [0,1], got {self.value}")
        if not self.descriptor:
            self.descriptor = f"do({self.trait}[{self.target_agent}] = {self.value:.2f})"

    def apply_to_agent_overrides(self, overrides: dict) -> dict:
        new = copy.deepcopy(overrides) if overrides else {}
        bf = new.setdefault("big_five", {})
        bf[self.trait] = float(self.value)
        return new


# ---------------------------------------------------------------------------
# Результаты
# ---------------------------------------------------------------------------

@dataclass
class CounterfactualPair:
    """Пара (treatment, control) с одинаковым seed.

    Поле `ite` хранит вектор Individual Treatment Effect: для каждой метрики
    разность Y_T - Y_C для этой конкретной пары.
    """
    seed: int
    treatment_outcome: dict
    control_outcome: dict
    ite: dict[str, float] = field(default_factory=dict)
    treatment_error: str | None = None
    control_error: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.treatment_error is None and self.control_error is None

    def compute_ite(self, metrics: Sequence[str]) -> dict[str, float]:
        ite: dict[str, float] = {}
        for m in metrics:
            t = self.treatment_outcome.get(m) if self.treatment_outcome else None
            c = self.control_outcome.get(m) if self.control_outcome else None
            if t is None or c is None:
                continue
            try:
                ite[m] = float(t) - float(c)
            except (TypeError, ValueError):
                continue
        self.ite = ite
        return ite

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "treatment": self.treatment_outcome,
            "control": self.control_outcome,
            "ite": self.ite,
            "valid": self.is_valid,
            "treatment_error": self.treatment_error,
            "control_error": self.control_error,
        }


@dataclass
class CausalExperimentResult:
    """Итог каузального эксперимента: ATE и доверительные интервалы по всем outcome-метрикам."""
    experiment_id: str
    intervention_treatment: dict
    intervention_control: dict
    n_pairs_total: int
    n_pairs_valid: int
    outcome_metrics: list[str]
    ate_by_metric: dict[str, dict]          # metric → {ate, se, ci_low, ci_high, t, df, p, d_z, decision}
    pairs: list[dict]
    timestamp: str
    duration_seconds: float
    config: dict

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Каузальный диспатчер
# ---------------------------------------------------------------------------

OutcomeFn = Callable[[int, dict, dict], dict]


def _build_real_outcome_fn() -> OutcomeFn:
    """Возвращает функцию, запускающую один реальный диалог через DialogueManager.

    Сигнатура: outcome_fn(seed, agent_overrides_for_target, dialogue_cfg) -> dict_with_metrics
    """
    from openai import OpenAI

    from ..config import (
        DEFAULT_AGENTS,
        OPENAI_API_KEY,
        OPENAI_BASE_URL,
        VIZ_EDGE_WINDOW,
    )
    from ..core.agents import Agent, BigFiveProfile
    from ..core.dialogue_manager import DialogueManager
    from ..core.env_context import get_env_context_by_index
    from ..analytics.basic import compute_metrics

    def fn(seed: int, agent_overrides_for_target: dict, dialogue_cfg: dict) -> dict:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot run real causal dialogue")

        random.seed(seed)
        np.random.seed(seed % (2**31 - 1))

        client_kwargs = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        client = OpenAI(**client_kwargs)

        env_context = get_env_context_by_index(dialogue_cfg.get("env_context_index", 0))
        target_agent_name = agent_overrides_for_target.get("__target_agent__")
        big_five_override = agent_overrides_for_target.get("big_five", {}) or {}

        agents = []
        for cfg in DEFAULT_AGENTS:
            cfg = dict(cfg)
            ag = Agent(**cfg)
            if ag.name == target_agent_name and big_five_override:
                # Сохраняем все исходные черты, перезаписываем только указанные
                base = ag.big_five.to_dict() if ag.big_five else {}
                merged = {**base, **{k: float(v) for k, v in big_five_override.items()}}
                ag.big_five = BigFiveProfile(**merged)
            agents.append(ag)

        dm = DialogueManager(
            client=client,
            agents=agents,
            env_context=env_context,
            human_io=None,
            enable_validation=dialogue_cfg.get("enable_validation", True),
            enable_cot=dialogue_cfg.get("enable_cot", False),
            strict_mode=True,
        )

        history = []
        n_turns = dialogue_cfg.get("n_turns", 12)
        for _ in range(n_turns):
            record, _ = dm.step()
            history.append(record)

        agent_names = [a.name for a in agents]
        metrics = compute_metrics(history, agent_names, window_size=VIZ_EDGE_WINDOW)
        summary_all = metrics.get("summary", {}).get("all", {})

        avg_reply_length = (
            sum(len(r.get("reply", "").split()) for r in history) / max(1, len(history))
        )
        negative_tone_rate = (
            sum(1 for r in history if (r.get("tone") or "").lower() == "negative")
            / max(1, len(history))
        )
        positive_tone_rate = (
            sum(1 for r in history if (r.get("tone") or "").lower() == "positive")
            / max(1, len(history))
        )

        return {
            "seed": seed,
            "n_turns": n_turns,
            "avg_reply_length": float(avg_reply_length),
            "avg_tone": float(summary_all.get("avg_tone", 0.0)),
            "actionability_rate": float(summary_all.get("actionability_rate", 0.0)),
            "reciprocity": float(summary_all.get("reciprocity", 0.0)),
            "emotion_entropy": float(summary_all.get("emotion_entropy", 0.0)),
            "question_rate": float(summary_all.get("question_rate", 0.0)),
            "negative_tone_rate": float(negative_tone_rate),
            "positive_tone_rate": float(positive_tone_rate),
            "targeting_rate": float(summary_all.get("targeting_rate", 0.0)),
            "reference_rate": float(summary_all.get("reference_rate", 0.0)),
            "metrics": summary_all,
            "history": history,
        }
    return fn


# ---------------------------------------------------------------------------
# Paired t-test для ATE
# ---------------------------------------------------------------------------

def paired_t_test_ate(ite_values: Sequence[float],
                      alpha: float = 0.05,
                      d_threshold: float = 0.3,
                      alternative: str = "two-sided") -> dict:
    """Paired Student's t-test на H0: E[Δ] = 0 против H1.

    Возвращает ATE, стандартную ошибку, 95% CI, t-статистику, p-value
    и Cohen's d_z (стандартизованный эффект для парного дизайна).

    alternative ∈ {"two-sided", "greater", "less"}.
    Решение: reject_H0 при p < alpha И |d_z| >= d_threshold.
    """
    from scipy import stats as _stats  # noqa: WPS433 (local import — soft dep)

    x = np.asarray([v for v in ite_values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                    dtype=float)
    n = int(x.size)
    if n < 2:
        return {
            "n": n,
            "ate": float("nan"),
            "se": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "t": float("nan"),
            "df": 0,
            "p_value": float("nan"),
            "cohens_d_z": float("nan"),
            "decision": "fail_to_reject_H0",
            "reason": "n<2 paired observations",
        }

    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    df = n - 1

    if sd == 0:
        # Все ITE равны — либо все нули, либо все равны константе
        if mean == 0:
            t_stat, p_value = 0.0, 1.0
        else:
            t_stat, p_value = float("inf"), 0.0
    else:
        t_stat = mean / se
        if alternative == "two-sided":
            p_value = 2.0 * (1.0 - _stats.t.cdf(abs(t_stat), df=df))
        elif alternative == "greater":
            p_value = 1.0 - _stats.t.cdf(t_stat, df=df)
        elif alternative == "less":
            p_value = _stats.t.cdf(t_stat, df=df)
        else:
            raise ValueError(f"unknown alternative: {alternative}")

    # 95% CI на ATE
    if sd > 0:
        t_crit = _stats.t.ppf(1 - alpha / 2, df=df)
        ci_low = mean - t_crit * se
        ci_high = mean + t_crit * se
    else:
        ci_low = ci_high = mean

    d_z = mean / sd if sd > 0 else (0.0 if mean == 0 else float("inf"))

    if math.isnan(p_value):
        decision = "fail_to_reject_H0"
    elif p_value < alpha and abs(d_z) >= d_threshold:
        decision = "reject_H0"
    else:
        decision = "fail_to_reject_H0"

    return {
        "n": n,
        "ate": mean,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "t": float(t_stat),
        "df": df,
        "p_value": float(p_value),
        "cohens_d_z": float(d_z),
        "alpha": alpha,
        "d_threshold": d_threshold,
        "alternative": alternative,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# CausalRunner
# ---------------------------------------------------------------------------

@dataclass
class CausalConfig:
    experiment_id: str
    intervention_treatment: Intervention
    intervention_control: Intervention
    n_pairs: int
    seed_base: int = 7000
    n_turns: int = 12
    env_context_index: int = 0
    parallelism: int = 4
    outcome_metrics: list[str] = field(default_factory=lambda: list(DEFAULT_OUTCOME_METRICS))
    alpha: float = 0.05
    d_threshold: float = 0.3      # |Cohen's d_z| порог; 0.3 = small-to-medium
    alternative: str = "two-sided"
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "intervention_treatment": self.intervention_treatment.to_dict(),
            "intervention_control": self.intervention_control.to_dict(),
            "n_pairs": self.n_pairs,
            "seed_base": self.seed_base,
            "n_turns": self.n_turns,
            "env_context_index": self.env_context_index,
            "parallelism": self.parallelism,
            "outcome_metrics": self.outcome_metrics,
            "alpha": self.alpha,
            "d_threshold": self.d_threshold,
            "alternative": self.alternative,
            "description": self.description,
        }


class CausalRunner:
    """Запускает paired counterfactual эксперимент.

    На каждый seed выполняется treatment- и control-симуляция; затем для каждой
    метрики оценивается ATE через paired_t_test_ate.
    """

    def __init__(self, config: CausalConfig, outcome_fn: OutcomeFn | None = None):
        self.config = config
        self.outcome_fn = outcome_fn or _build_real_outcome_fn()

    def _build_overrides(self, intervention: Intervention) -> dict:
        ov = intervention.apply_to_agent_overrides({})
        ov["__target_agent__"] = intervention.target_agent
        return ov

    def _run_one(self, seed: int, intervention: Intervention) -> tuple[dict | None, str | None]:
        overrides = self._build_overrides(intervention)
        try:
            outcome = self.outcome_fn(
                seed,
                overrides,
                {
                    "n_turns": self.config.n_turns,
                    "env_context_index": self.config.env_context_index,
                    "enable_validation": True,
                    "enable_cot": False,
                },
            )
            return outcome, None
        except Exception as e:
            logger.warning("seed=%s intervention=%s failed: %s", seed, intervention.name, e)
            return None, f"{type(e).__name__}: {e}"

    def _run_pair(self, seed: int) -> CounterfactualPair:
        # Обе симуляции запускаются с одинаковым seed → внутрисубъектная пара
        t_out, t_err = self._run_one(seed, self.config.intervention_treatment)
        c_out, c_err = self._run_one(seed, self.config.intervention_control)
        pair = CounterfactualPair(
            seed=seed,
            treatment_outcome=t_out or {},
            control_outcome=c_out or {},
            treatment_error=t_err,
            control_error=c_err,
        )
        if pair.is_valid:
            pair.compute_ite(self.config.outcome_metrics)
        return pair

    def run(self, output_base: Path = Path("experiments/results")) -> CausalExperimentResult:
        start = time.time()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = output_base / f"{self.config.experiment_id}__{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # сохраняем конфиг
        (out_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        seeds = [self.config.seed_base + i for i in range(self.config.n_pairs)]
        pairs: list[CounterfactualPair] = []
        with ThreadPoolExecutor(max_workers=self.config.parallelism) as ex:
            futures = {ex.submit(self._run_pair, s): s for s in seeds}
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    pair = fut.result()
                except Exception as e:
                    logger.exception("pair seed=%s crashed: %s", seed, e)
                    pair = CounterfactualPair(
                        seed=seed,
                        treatment_outcome={},
                        control_outcome={},
                        treatment_error=f"{type(e).__name__}: {e}",
                        control_error=f"{type(e).__name__}: {e}",
                    )
                pairs.append(pair)
                logger.info(
                    "pair seed=%s done, valid=%s ITE(avg_tone)=%s",
                    seed, pair.is_valid, pair.ite.get("avg_tone"),
                )

        pairs.sort(key=lambda p: p.seed)

        # сохраняем все пары
        (out_dir / "pairs.jsonl").write_text(
            "\n".join(json.dumps(p.to_dict(), ensure_ascii=False, default=str) for p in pairs),
            encoding="utf-8",
        )

        # paired t-test по каждой метрике
        valid_pairs = [p for p in pairs if p.is_valid]
        ate_by_metric: dict[str, dict] = {}
        for metric in self.config.outcome_metrics:
            ite_values = [p.ite.get(metric) for p in valid_pairs if metric in p.ite]
            ite_values = [v for v in ite_values if v is not None]
            test = paired_t_test_ate(
                ite_values,
                alpha=self.config.alpha,
                d_threshold=self.config.d_threshold,
                alternative=self.config.alternative,
            )
            test["mean_treatment"] = float(np.mean([
                p.treatment_outcome.get(metric, float("nan")) for p in valid_pairs
            ])) if valid_pairs else float("nan")
            test["mean_control"] = float(np.mean([
                p.control_outcome.get(metric, float("nan")) for p in valid_pairs
            ])) if valid_pairs else float("nan")
            ate_by_metric[metric] = test

        duration = time.time() - start
        result = CausalExperimentResult(
            experiment_id=self.config.experiment_id,
            intervention_treatment=self.config.intervention_treatment.to_dict(),
            intervention_control=self.config.intervention_control.to_dict(),
            n_pairs_total=len(pairs),
            n_pairs_valid=len(valid_pairs),
            outcome_metrics=list(self.config.outcome_metrics),
            ate_by_metric=ate_by_metric,
            pairs=[p.to_dict() for p in pairs],
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
            config=self.config.to_dict(),
        )

        # summary без полных историй
        summary = {
            "experiment_id": result.experiment_id,
            "timestamp": result.timestamp,
            "duration_seconds": result.duration_seconds,
            "n_pairs_total": result.n_pairs_total,
            "n_pairs_valid": result.n_pairs_valid,
            "intervention_treatment": result.intervention_treatment,
            "intervention_control": result.intervention_control,
            "ate_by_metric": result.ate_by_metric,
            "config": result.config,
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        # пути для отчёта
        result.config["output_dir"] = str(out_dir)
        return result
