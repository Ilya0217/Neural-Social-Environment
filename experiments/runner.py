"""
Оркестратор пакетных экспериментов для проверки гипотез из preregistration.md.

Архитектура:
  ExperimentConfig (YAML) — описание эксперимента:
    arms[] — независимые группы (treatment / control / уровни IV)
    dialogue — общие параметры диалога (n_turns, env_context, ...)
    analysis — какой стат-тест применять, с какими параметрами
  ExperimentRunner — пакетный запуск:
    параллельно через ThreadPoolExecutor
    для каждого диалога — сохранение в JSONL
    после всех arms — применение теста из statistical_tests.py
    summary.json с raw_data, p, эффектом, решением

Два режима dispatcher:
  mock  — синтетические данные с инжектированным эффектом (для unit-тестов и dry-run)
  real  — реальный вызов DialogueManager (требует OPENAI_API_KEY)

Запуск:
  python -m agent_dialogue_sim.experiment_runner --config experiments/configs/h6.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from ..science import statistical_tests as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ArmConfig:
    name: str
    n_dialogues: int
    agent_overrides: dict = field(default_factory=dict)  # OCEAN, prompt mods, ...
    dialogue_overrides: dict = field(default_factory=dict)  # validation, CoT, ...
    seed_offset: int = 0  # base seed for this arm; final seed = seed_base + seed_offset + i


@dataclass
class DialogueConfig:
    n_turns: int = 20
    env_context_index: int = 0
    base_agents: str = "default"  # 'default' uses DEFAULT_AGENTS from config.py


@dataclass
class AnalysisConfig:
    """Описание стат-теста, применяемого к собранным метрикам."""
    test: str  # имя функции из statistical_tests (anova_oneway, welch_t_test, ...)
    metric: str  # ключ метрики из выходного dict диалога (например, 'avg_reply_length')
    params: dict = field(default_factory=dict)  # параметры теста (alpha, alternative, ...)


@dataclass
class ExperimentConfig:
    id: str
    hypothesis: str  # "H1", "H6", ...
    description: str
    arms: list[ArmConfig]
    dialogue: DialogueConfig
    analysis: AnalysisConfig
    seed_base: int = 1000
    parallelism: int = 4

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        arms = [ArmConfig(**a) for a in d["arms"]]
        dialogue = DialogueConfig(**d.get("dialogue", {}))
        analysis = AnalysisConfig(**d["analysis"])
        return cls(
            id=d["id"],
            hypothesis=d["hypothesis"],
            description=d.get("description", ""),
            arms=arms,
            dialogue=dialogue,
            analysis=analysis,
            seed_base=d.get("seed_base", 1000),
            parallelism=d.get("parallelism", 4),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "description": self.description,
            "arms": [asdict(a) for a in self.arms],
            "dialogue": asdict(self.dialogue),
            "analysis": asdict(self.analysis),
            "seed_base": self.seed_base,
            "parallelism": self.parallelism,
        }


# ---------------------------------------------------------------------------
# Dialogue dispatchers
# ---------------------------------------------------------------------------

DialogueResult = dict[str, Any]
DialogueDispatcher = Callable[[int, ArmConfig, DialogueConfig], DialogueResult]


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for dataclasses, sets, и прочих не-serializable объектов."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, set):
        return list(obj)
    # Pydantic v2 models
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    return str(obj)


def mock_dispatcher_factory(
    metric_means: dict[str, float],
    metric_sd: float = 1.0,
    metric_name: str = "metric_value",
) -> DialogueDispatcher:
    """
    Возвращает mock-диспатчер, генерирующий синтетический metric_value
    с распределением N(metric_means[arm.name], metric_sd) для каждой arm.
    Используется в unit-тестах и dry-run.
    """
    def dispatcher(seed: int, arm: ArmConfig, _: DialogueConfig) -> DialogueResult:
        rng = random.Random(seed)
        mean = metric_means.get(arm.name, 0.0)
        value = rng.gauss(mean, metric_sd)
        result: dict[str, Any] = {
            "seed": seed,
            "arm": arm.name,
        }
        # синтетические метрики-наполнители (используются если metric_name не совпадает)
        filler = {
            "avg_reply_length": rng.gauss(15 + mean, 3),
            "actionability_rate": max(0.0, min(1.0, rng.gauss(0.5, 0.1))),
            "toxicity_rate": max(0.0, min(1.0, rng.gauss(0.1, 0.05))),
        }
        for k, v in filler.items():
            if k != metric_name:
                result[k] = v
        # целевая метрика — пишется последней, не перетирается
        result[metric_name] = value
        return result
    return dispatcher


def real_dispatcher(seed: int, arm: ArmConfig, dialogue_config: DialogueConfig) -> DialogueResult:
    """
    Реальный диспатчер: запускает DialogueManager с заданными параметрами,
    собирает метрики через analytics.compute_metrics().

    Применяет arm.agent_overrides к Big Five агентов.
    Применяет arm.dialogue_overrides к DialogueManager (например, enable_validation).
    """
    from openai import OpenAI

    from ..config import (
        DEFAULT_AGENTS, OPENAI_API_KEY, OPENAI_BASE_URL, VIZ_EDGE_WINDOW,
    )
    from ..core.agents import Agent, BigFiveProfile
    from ..core.dialogue_manager import DialogueManager
    from ..core.env_context import get_env_context_by_index
    from ..analytics.basic import compute_metrics

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set; cannot run real dispatcher")

    random.seed(seed)

    client_kwargs = {}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)
    env_context = get_env_context_by_index(dialogue_config.env_context_index)

    # Сборка агентов с override Big Five
    agents = []
    for cfg in DEFAULT_AGENTS:
        cfg = dict(cfg)
        ag = Agent(**cfg)
        ocean_override = arm.agent_overrides.get("big_five", {})
        if ocean_override:
            ag.big_five = BigFiveProfile(**{
                "openness": ocean_override.get("openness", ag.big_five.openness if ag.big_five else 0.5),
                "conscientiousness": ocean_override.get("conscientiousness",
                                                          ag.big_five.conscientiousness if ag.big_five else 0.5),
                "extraversion": ocean_override.get("extraversion",
                                                    ag.big_five.extraversion if ag.big_five else 0.5),
                "agreeableness": ocean_override.get("agreeableness",
                                                     ag.big_five.agreeableness if ag.big_five else 0.5),
                "neuroticism": ocean_override.get("neuroticism",
                                                   ag.big_five.neuroticism if ag.big_five else 0.5),
            })
        agents.append(ag)

    # DialogueManager с применением overrides.
    # strict_mode=True — при API/validation сбое поднимает RuntimeError вместо тихой
    # подмены реплики; experiment_runner._run_arm ловит это как arm.error и
    # отбрасывает диалог из выборки, не контаминируя стат-тест.
    dm_kwargs = {
        "client": client,
        "agents": agents,
        "env_context": env_context,
        "human_io": None,
        "enable_validation": arm.dialogue_overrides.get("enable_validation", True),
        "enable_cot": arm.dialogue_overrides.get("enable_cot", False),
        "strict_mode": True,
    }
    dm = DialogueManager(**dm_kwargs)

    history = []
    for _ in range(dialogue_config.n_turns):
        record, _ = dm.step()
        history.append(record)

    agent_names = [a.name for a in agents]
    metrics = compute_metrics(history, agent_names, window_size=VIZ_EDGE_WINDOW)
    # compute_metrics() возвращает {"summary": {"all": {...}, "window": {...}}, ...}.
    # Для агрегации в эксперименте удобнее метрики по всему диалогу ("all").
    summary_all = metrics.get("summary", {}).get("all", {})

    avg_reply_length = sum(len(r.get("reply", "").split()) for r in history) / max(1, len(history))
    return {
        "seed": seed,
        "arm": arm.name,
        "n_turns": dialogue_config.n_turns,
        "avg_reply_length": avg_reply_length,
        "actionability_rate": float(summary_all.get("actionability_rate", 0.0)),
        "reciprocity": float(summary_all.get("reciprocity", 0.0)),
        "emotion_entropy": float(summary_all.get("emotion_entropy", 0.0)),
        "question_rate": float(summary_all.get("question_rate", 0.0)),
        "avg_tone": float(summary_all.get("avg_tone", 0.0)),
        "metrics": metrics,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class RunOutcome:
    config_id: str
    hypothesis: str
    timestamp: str
    arms: dict[str, list[DialogueResult]]
    test_result: dict
    output_dir: str
    duration_seconds: float


class ExperimentRunner:
    def __init__(
        self,
        config: ExperimentConfig,
        dispatcher: DialogueDispatcher | None = None,
        check_credits: bool = True,
    ):
        """
        check_credits: если True (default) и dispatcher==real_dispatcher,
        перед стартом эксперимента проверяется баланс OpenRouter и сравнивается
        с оценкой стоимости. Mock-режим пропускает проверку.
        """
        self.config = config
        self.dispatcher = dispatcher or real_dispatcher
        self.check_credits = check_credits

    # -- IO ---------------------------------------------------------------

    def _output_dir(self, base: Path) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = base / f"{self.config.id}__{ts}"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _save_jsonl(self, path: Path, records: Sequence[DialogueResult]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                # default=_json_default обрабатывает dataclasses (asdict) и прочие fallback
                f.write(json.dumps(r, ensure_ascii=False, default=_json_default) + "\n")

    # -- Run --------------------------------------------------------------

    def _run_arm(self, arm: ArmConfig) -> list[DialogueResult]:
        results: list[DialogueResult] = []
        seeds = [self.config.seed_base + arm.seed_offset + i for i in range(arm.n_dialogues)]
        with ThreadPoolExecutor(max_workers=self.config.parallelism) as ex:
            futures = {
                ex.submit(self.dispatcher, seed, arm, self.config.dialogue): seed
                for seed in seeds
            }
            for fut in as_completed(futures):
                seed = futures[fut]
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as e:
                    logger.exception("dialogue %s/seed=%s failed: %s", arm.name, seed, e)
                    results.append({"seed": seed, "arm": arm.name, "error": str(e)})
        # Возвращаем отсортированный по seed для воспроизводимости
        return sorted(results, key=lambda r: r.get("seed", 0))

    def _extract_metric(self, results: list[DialogueResult]) -> list[float]:
        metric = self.config.analysis.metric
        out = []
        for r in results:
            if "error" in r:
                continue
            if metric not in r:
                raise KeyError(
                    f"metric '{metric}' not in dialogue result; "
                    f"available keys: {list(r.keys())}"
                )
            out.append(float(r[metric]))
        return out

    def _run_test(self, arm_metrics: dict[str, list[float]]) -> dict:
        """Применяет тест из statistical_tests на основе analysis.test и собранных метрик."""
        test_name = self.config.analysis.test
        params = dict(self.config.analysis.params)
        arm_names = [a.name for a in self.config.arms]
        groups = [arm_metrics[name] for name in arm_names]

        if test_name == "anova_oneway":
            result = st.anova_oneway(groups, **params)
            extra = {"tukey_hsd": st.tukey_hsd_pairs(groups, labels=arm_names,
                                                     alpha=params.get("alpha", 0.05))}
        elif test_name == "welch_t_test":
            if len(arm_names) != 2:
                raise ValueError("welch_t_test requires exactly 2 arms")
            result = st.welch_t_test(groups[0], groups[1], **params)
            extra = {}
        elif test_name == "z_test_two_proportions":
            # для пропорций metric — это 0/1 индикатор; считаем successes
            if len(arm_names) != 2:
                raise ValueError("z_test_two_proportions requires exactly 2 arms")
            sa = sum(1 for v in groups[0] if v > 0.5)
            sb = sum(1 for v in groups[1] if v > 0.5)
            result = st.z_test_two_proportions(
                successes_a=sa, n_a=len(groups[0]),
                successes_b=sb, n_b=len(groups[1]),
                **params,
            )
            extra = {"successes_a": sa, "successes_b": sb}
        elif test_name == "mann_whitney_u":
            if len(arm_names) != 2:
                raise ValueError("mann_whitney_u requires exactly 2 arms")
            result = st.mann_whitney_u(groups[0], groups[1], **params)
            extra = {}
        elif test_name == "tost_equivalence":
            if len(arm_names) != 2:
                raise ValueError("tost_equivalence requires exactly 2 arms")
            result = st.tost_equivalence(groups[0], groups[1], **params)
            extra = {}
        elif test_name == "spearman_correlation":
            # для корреляции metric должна быть кортеж (x, y)
            raise NotImplementedError("spearman через runner: сделать через два metrics_x, metrics_y")
        else:
            raise ValueError(f"unknown test: {test_name}")

        out = result.to_dict()
        out["arm_n"] = {name: len(g) for name, g in zip(arm_names, groups)}
        out["arm_means"] = {name: (sum(g) / len(g) if g else None)
                            for name, g in zip(arm_names, groups)}
        if extra:
            out["extra_runner"] = extra
        return out

    def _preflight_credits(self) -> None:
        """Pre-flight check: убедиться что баланса хватит на оценочную стоимость +50%.
        Запускается только для real dispatcher; mock-режим пропускает."""
        if not self.check_credits or self.dispatcher is not real_dispatcher:
            return
        from ..science.api_credits_check import (
            check_credits, estimate_experiment_cost, CreditsExhaustedError, CreditsCheckError,
        )
        total_dialogues = sum(a.n_dialogues for a in self.config.arms)
        estimated = estimate_experiment_cost(
            n_dialogues=total_dialogues,
            n_turns=self.config.dialogue.n_turns,
        )
        # +50% запас на ретраи валидации и переменность размера контекста
        required = estimated * 1.5
        try:
            info = check_credits()
        except CreditsCheckError as e:
            logger.warning("credits check failed: %s — proceeding without verification", e)
            return
        if info.is_openrouter:
            logger.info("OpenRouter balance: $%.4f remaining (estimate $%.4f, with +50%% buffer $%.4f)",
                        info.remaining_usd, estimated, required)
            if info.remaining_usd < required:
                raise CreditsExhaustedError(
                    f"Insufficient OpenRouter balance. "
                    f"Estimated cost: ${estimated:.4f} ({total_dialogues} dialogues × "
                    f"{self.config.dialogue.n_turns} turns); with 50% buffer: ${required:.4f}. "
                    f"Remaining: ${info.remaining_usd:.4f}. "
                    f"Top up at https://openrouter.ai/settings/credits."
                )

    def run(self, output_base: Path = Path("experiments/results")) -> RunOutcome:
        # Pre-flight: проверка кредитов до создания output dir, чтобы при отказе
        # не оставлять пустых директорий
        self._preflight_credits()

        start = time.time()
        out_dir = self._output_dir(output_base)
        logger.info("output dir: %s", out_dir)

        # сохраняем конфиг
        with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config.to_dict(), f, allow_unicode=True, sort_keys=False)

        arms_data: dict[str, list[DialogueResult]] = {}
        arm_metrics: dict[str, list[float]] = {}
        for arm in self.config.arms:
            logger.info("running arm '%s' (n=%d)", arm.name, arm.n_dialogues)
            results = self._run_arm(arm)
            self._save_jsonl(out_dir / f"arm_{arm.name}.jsonl", results)
            arms_data[arm.name] = results
            arm_metrics[arm.name] = self._extract_metric(results)
            logger.info("arm '%s' done: %d successful, mean(%s)=%.4f",
                        arm.name,
                        len(arm_metrics[arm.name]),
                        self.config.analysis.metric,
                        sum(arm_metrics[arm.name]) / max(1, len(arm_metrics[arm.name])))

        # стат-тест
        test_result = self._run_test(arm_metrics)

        duration = time.time() - start
        outcome = RunOutcome(
            config_id=self.config.id,
            hypothesis=self.config.hypothesis,
            timestamp=datetime.now(timezone.utc).isoformat(),
            arms={k: v for k, v in arms_data.items()},  # включает истории
            test_result=test_result,
            output_dir=str(out_dir),
            duration_seconds=duration,
        )

        # summary.json — без полных диалогов, только метрики и тест
        summary = {
            "config_id": outcome.config_id,
            "hypothesis": outcome.hypothesis,
            "timestamp": outcome.timestamp,
            "duration_seconds": outcome.duration_seconds,
            "arm_metrics": arm_metrics,
            "test_result": test_result,
        }
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        return outcome


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a hypothesis-testing experiment.")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config.")
    parser.add_argument("--output-dir", default="experiments/results",
                        help="Base directory for results.")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock dispatcher (synthetic data) instead of real LLM calls.")
    parser.add_argument("--mock-effect", type=float, default=1.0,
                        help="Mean offset between first and last arm in mock mode.")
    parser.add_argument("--skip-credits-check", action="store_true",
                        help="Skip OpenRouter pre-flight balance check (not recommended).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = ExperimentConfig.from_yaml(args.config)

    if args.mock:
        # spread effect_size linearly across arms
        means = {arm.name: args.mock_effect * i for i, arm in enumerate(config.arms)}
        dispatcher = mock_dispatcher_factory(
            metric_means=means,
            metric_name=config.analysis.metric,
        )
    else:
        dispatcher = real_dispatcher

    runner = ExperimentRunner(config, dispatcher=dispatcher,
                                check_credits=not args.skip_credits_check)
    outcome = runner.run(Path(args.output_dir))

    print("\n=== EXPERIMENT SUMMARY ===")
    print(f"Config:     {outcome.config_id}")
    print(f"Hypothesis: {outcome.hypothesis}")
    print(f"Duration:   {outcome.duration_seconds:.1f}s")
    print(f"Output:     {outcome.output_dir}")
    print(f"Test:       {config.analysis.test}")
    print(f"  statistic = {outcome.test_result.get('statistic'):.4f}")
    print(f"  p-value   = {outcome.test_result.get('p_value'):.4g}")
    if outcome.test_result.get("effect_size") is not None:
        print(f"  effect    = {outcome.test_result.get('effect_size'):.4f} "
              f"({outcome.test_result.get('effect_size_name')})")
    print(f"  decision  = {outcome.test_result.get('decision').upper()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
