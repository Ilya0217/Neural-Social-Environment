"""
Manipulation check для H1 (и других OCEAN-гипотез).

Контекст: H1 утверждает что заданное значение Big Five `extraversion` влияет на
наблюдаемое поведение агента. Прежде чем тестировать DV (длина реплики), нужно
доказать что **манипуляция вообще сработала** — что агент с заданной экстраверсией
0.8 действительно "звучит" более экстравертно, чем агент с 0.2.

Без manipulation check мы не знаем, тестируем ли реальную extraversion или артефакт.

Метод:
  1. Сэмплируем N=20–30 реплик из каждой arm (low/mid/high).
  2. Независимый LLM-судья оценивает каждую реплику по шкале 1–7 на
     (a) экстраверсию, (b) добросовестность, (c) другие OCEAN если нужно.
  3. Считаем Spearman correlation между intended OCEAN value и judge ratings.
  4. Manipulation считается успешной если ρ > 0.4 И p < 0.05.

Применение в pipeline H1:
  - перед запуском stat-теста на DV (длина реплики)
  - блокирует анализ если manipulation провалилась → нечего тестировать
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Sequence

PRINCIPLES = ("openness", "conscientiousness", "extraversion",
              "agreeableness", "neuroticism")


JUDGE_SYSTEM_PROMPT = """\
You are an independent personality assessor. Read a short conversational reply and
rate the speaker on the Big Five (OCEAN) dimensions, using a 7-point scale:

  1 = very low on this trait
  4 = average / unclear from this reply
  7 = very high on this trait

Big Five definitions:
  openness:          curiosity, creativity, openness to new ideas
  conscientiousness: organization, dependability, attention to detail
  extraversion:      sociability, talkativeness, expressiveness, energy in speech
  agreeableness:     cooperation, warmth, conflict avoidance
  neuroticism:       anxiety, emotional reactivity, negativity

Output STRICTLY a JSON object:
{
  "openness": <int 1-7>,
  "conscientiousness": <int 1-7>,
  "extraversion": <int 1-7>,
  "agreeableness": <int 1-7>,
  "neuroticism": <int 1-7>,
  "reasoning": "<one sentence>"
}

Base your rating ONLY on what the reply itself reveals. Do not assume — when in doubt, rate 4.
"""


@dataclass
class TraitRating:
    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int
    reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ManipulationCheckResult:
    """Результат проверки манипуляции для одной OCEAN-черты."""
    trait: str
    n_replies: int
    intended_levels: list[float]      # заданные значения OCEAN
    observed_ratings: list[float]     # 1-7 от судьи
    spearman_rho: float
    p_value: float
    rho_threshold: float = 0.4
    alpha: float = 0.05
    manipulation_successful: bool = False
    interpretation: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Парсинг ответа судьи
# ---------------------------------------------------------------------------

def _parse_judge_response(content: str) -> TraitRating:
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON in judge response: {content[:200]}")
    data = json.loads(m.group(0))
    ratings = {}
    for trait in PRINCIPLES:
        v = data.get(trait, 4)
        try:
            ratings[trait] = max(1, min(7, int(v)))
        except (TypeError, ValueError):
            ratings[trait] = 4
    return TraitRating(
        openness=ratings["openness"],
        conscientiousness=ratings["conscientiousness"],
        extraversion=ratings["extraversion"],
        agreeableness=ratings["agreeableness"],
        neuroticism=ratings["neuroticism"],
        reasoning=str(data.get("reasoning", ""))[:300],
    )


# ---------------------------------------------------------------------------
# Big Five judge
# ---------------------------------------------------------------------------

class BigFiveJudge:
    """LLM-судья, оценивающий Big Five по одной реплике.
    Mock-режим для unit-тестов."""

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "openai/gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 200,
        mock_mapping: Optional[dict] = None,
    ):
        """
        mock_mapping: dict {pattern_in_reply: TraitRating} для тестов.
                      Сопоставляет первую найденную подстроку.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.mock_mapping = mock_mapping or {}
        self.n_calls = 0

    def rate(self, reply: str) -> TraitRating:
        self.n_calls += 1
        if self.client is None:
            return self._mock_rate(reply)
        return self._llm_rate(reply)

    def _mock_rate(self, reply: str) -> TraitRating:
        for pattern, rating in self.mock_mapping.items():
            if pattern in reply:
                return rating
        # Default neutral
        return TraitRating(
            openness=4, conscientiousness=4, extraversion=4,
            agreeableness=4, neuroticism=4, reasoning="default neutral",
        )

    def _llm_rate(self, reply: str) -> TraitRating:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f'Reply:\n"{reply}"\n\nOutput JSON only.'},
            ],
        )
        content = resp.choices[0].message.content or ""
        return _parse_judge_response(content)


# ---------------------------------------------------------------------------
# Расчёт manipulation check
# ---------------------------------------------------------------------------

def check_manipulation(
    replies_by_intended_level: dict[float, list[str]],
    trait: str,
    judge: BigFiveJudge,
    rho_threshold: float = 0.4,
    alpha: float = 0.05,
) -> ManipulationCheckResult:
    """
    Проверка manipulation для одной OCEAN-черты.

    replies_by_intended_level:
      {0.2: ["reply1", "reply2", ...], 0.5: [...], 0.8: [...]}
      — реплики из каждой arm, сгруппированные по заданному уровню.

    Возвращает ManipulationCheckResult с rho, p, и решением.
    """
    if trait not in PRINCIPLES:
        raise ValueError(f"trait must be one of {PRINCIPLES}, got {trait!r}")
    from scipy import stats as scipy_stats

    intended_levels: list[float] = []
    observed_ratings: list[float] = []

    for level, replies in replies_by_intended_level.items():
        for reply in replies:
            rating = judge.rate(reply)
            score = getattr(rating, trait)
            intended_levels.append(level)
            observed_ratings.append(score)

    if len(intended_levels) < 6:
        return ManipulationCheckResult(
            trait=trait,
            n_replies=len(intended_levels),
            intended_levels=intended_levels,
            observed_ratings=observed_ratings,
            spearman_rho=0.0,
            p_value=1.0,
            rho_threshold=rho_threshold,
            alpha=alpha,
            manipulation_successful=False,
            interpretation=f"insufficient data: n={len(intended_levels)} < 6",
        )

    rho, p = scipy_stats.spearmanr(intended_levels, observed_ratings)
    rho = float(rho)
    p = float(p)

    successful = abs(rho) >= rho_threshold and p < alpha

    if successful:
        interp = (
            f"manipulation successful: ρ={rho:.3f}, p={p:.4g} — judge ratings of "
            f"{trait} correlate with intended Big Five level"
        )
    elif p < alpha:
        interp = (
            f"statistically significant correlation (p={p:.4g}) but small effect "
            f"(ρ={rho:.3f} < {rho_threshold}) — manipulation weak"
        )
    elif abs(rho) >= rho_threshold:
        interp = (
            f"correlation in expected direction (ρ={rho:.3f}) but not significant "
            f"(p={p:.4g}) — increase sample size"
        )
    else:
        interp = (
            f"manipulation FAILED: ρ={rho:.3f}, p={p:.4g} — judge ratings of "
            f"{trait} do NOT correlate with intended Big Five level. "
            f"Treat downstream tests with caution."
        )

    return ManipulationCheckResult(
        trait=trait,
        n_replies=len(intended_levels),
        intended_levels=intended_levels,
        observed_ratings=observed_ratings,
        spearman_rho=rho,
        p_value=p,
        rho_threshold=rho_threshold,
        alpha=alpha,
        manipulation_successful=successful,
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Удобный wrapper для интеграции с experiment_runner H1
# ---------------------------------------------------------------------------

def sample_replies_from_arm(
    arm_jsonl_path: str,
    n_per_dialogue: int = 1,
    max_dialogues: Optional[int] = None,
    seed: int = 42,
) -> list[str]:
    """
    Извлекает примеры реплик из arm_*.jsonl (выход experiment_runner) для
    подачи в judge. По умолчанию — 1 случайная реплика на диалог.
    """
    import random
    rng = random.Random(seed)
    out = []
    with open(arm_jsonl_path) as f:
        for line in f:
            if max_dialogues is not None and len(out) >= max_dialogues * n_per_dialogue:
                break
            rec = json.loads(line)
            if "error" in rec:
                continue
            history = rec.get("history", []) or []
            if not history:
                continue
            picks = rng.sample(history, min(n_per_dialogue, len(history)))
            for h in picks:
                reply = h.get("reply")
                if reply:
                    out.append(reply)
    return out
