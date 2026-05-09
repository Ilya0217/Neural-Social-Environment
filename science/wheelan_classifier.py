"""
Классификатор стадий развития группы по Wheelan (2016, 5th ed.;
историко-методологический обзор — Bonebright, 2010).

Используется для гипотезы H3 (см. preregistration.md).

Стадии:
  1. forming      — знакомство, ориентация, мало конфликтов, поверхностное согласие
  2. storming     — конфликт, борьба за роли, негативная аффективность, разногласия
  3. norming      — соглашения, кооперация, положительная социо-эмоциональность
  4. performing   — продуктивная работа, высокая task-доля, взаимность

Признаки (на скользящем окне диалога W=10 ходов, шаг 5):
  - task_ratio        — доля task-ориентированных диалоговых актов (ISO 24617-2)
  - positive_social   — доля positive socio-emotional actов (agreement, support)
  - negative_social   — доля negative socio actов (disagreement, antagonism)
  - reciprocity       — пары A↔B / всего пар
  - cooperation_words — частота слов "agree", "let's", "we should"
  - conflict_words    — частота "but", "disagree", "however", "no"
  - actionability     — доля реплик с конкретными планами

Правила (rule-based, без обучения):
  - storming:    negative_social > 0.25 OR conflict_words > 0.20
  - norming:     positive_social > 0.30 AND cooperation_words > 0.15 AND not storming
  - performing:  reciprocity > 0.5 AND actionability > 0.4 AND task_ratio > 0.6
  - forming:    default (мало данных, низкая task-доля, поверхностное согласие)

Возвращает (stage, confidence ∈ [0, 1]).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Sequence


COOPERATION_PATTERNS = [
    r"\blet'?s\b", r"\bwe should\b", r"\bagree\b", r"\bcooperat", r"\btogether\b",
    r"\bsounds good\b", r"\bgood idea\b", r"\bI('m| am) on board\b",
    r"\bworks for me\b", r"\bperfect\b",
]
CONFLICT_PATTERNS = [
    r"\bbut\b", r"\bhowever\b", r"\bdisagree\b", r"\bno,?\s", r"\bnot sure\b",
    r"\bproblem\b", r"\bissue\b", r"\bwrong\b", r"\bagainst\b", r"\bobjection\b",
]
ACTION_PATTERNS = [
    r"\bshould\b", r"\bwill\b", r"\bplan\b", r"\bnext step\b", r"\bcould\b",
    r"\bI('ll| will)\b", r"\bdeadline\b", r"\bownership\b",
]
QUESTION_TOKENS = re.compile(r"[?]")


@dataclass
class StageFeatures:
    """Численные признаки одного окна диалога."""
    n_turns: int
    task_ratio: float           # Доля task-ориентированных реплик
    positive_social_ratio: float
    negative_social_ratio: float
    reciprocity: float
    cooperation_word_rate: float
    conflict_word_rate: float
    actionability_rate: float
    question_rate: float


@dataclass
class StageClassification:
    """Результат классификации одного окна."""
    stage: str  # forming / storming / norming / performing
    confidence: float  # 0..1, насколько уверены в этой стадии
    features: StageFeatures
    rule_triggered: str  # какое правило сработало (для отладки/логов)


# ---------------------------------------------------------------------------
# Извлечение признаков
# ---------------------------------------------------------------------------

def _count_pattern_hits(text: str, patterns: Sequence[str]) -> int:
    text_l = text.lower()
    return sum(1 for p in patterns if re.search(p, text_l))


def extract_features(window: Sequence[dict]) -> StageFeatures:
    """
    window — список реплик: каждая dict с ключами 'speaker', 'target', 'reply', 'tone', 'emotion'.
    Признаки извлекаются по эвристикам, согласованным с ISO 24617-2 категориями.
    """
    n = len(window)
    if n == 0:
        return StageFeatures(
            n_turns=0, task_ratio=0.0,
            positive_social_ratio=0.0, negative_social_ratio=0.0,
            reciprocity=0.0, cooperation_word_rate=0.0,
            conflict_word_rate=0.0, actionability_rate=0.0,
            question_rate=0.0,
        )

    pos_social = neg_social = task_count = action_count = 0
    coop_hits = conflict_hits = q_hits = 0
    pairs = []

    for r in window:
        reply = r.get("reply", "") or ""
        tone = (r.get("tone") or "neutral").lower()
        speaker = r.get("speaker")
        target = r.get("target")
        if speaker and target:
            pairs.append((speaker, target))

        # Positive/negative socio
        if tone == "positive":
            pos_social += 1
        elif tone == "negative":
            neg_social += 1

        # Task vs socio: task-ориентированные имеют action-маркеры или вопросы про "что/как делать"
        is_task = bool(
            _count_pattern_hits(reply, ACTION_PATTERNS) > 0
            or re.search(r"\b(how|what|when|where|which)\b", reply.lower())
        )
        if is_task:
            task_count += 1

        coop_hits += _count_pattern_hits(reply, COOPERATION_PATTERNS)
        conflict_hits += _count_pattern_hits(reply, CONFLICT_PATTERNS)
        if _count_pattern_hits(reply, ACTION_PATTERNS) > 0:
            action_count += 1
        if QUESTION_TOKENS.search(reply):
            q_hits += 1

    # Reciprocity на уровне окна: доля A↔B пар, имеющих обратную (B↔A)
    set_pairs = set(pairs)
    if set_pairs:
        mutual = sum(1 for (u, v) in set_pairs if (v, u) in set_pairs)
        reciprocity = mutual / len(set_pairs)
    else:
        reciprocity = 0.0

    return StageFeatures(
        n_turns=n,
        task_ratio=task_count / n,
        positive_social_ratio=pos_social / n,
        negative_social_ratio=neg_social / n,
        reciprocity=reciprocity,
        cooperation_word_rate=coop_hits / n,
        conflict_word_rate=conflict_hits / n,
        actionability_rate=action_count / n,
        question_rate=q_hits / n,
    )


# ---------------------------------------------------------------------------
# Правила классификации
# ---------------------------------------------------------------------------

def classify_stage(features: StageFeatures) -> StageClassification:
    """Rule-based классификация (Wheelan 2016)."""
    f = features

    # Storming: явные конфликтные сигналы
    if f.negative_social_ratio > 0.25 or f.conflict_word_rate > 0.20:
        # Уверенность зависит от силы сигнала
        conf = min(1.0, max(f.negative_social_ratio * 2, f.conflict_word_rate * 2.5))
        return StageClassification(
            stage="storming", confidence=conf,
            features=f, rule_triggered="negative_social>0.25 or conflict_words>0.20",
        )

    # Performing: высокий task + reciprocity + действия
    if f.reciprocity > 0.5 and f.actionability_rate > 0.4 and f.task_ratio > 0.6:
        conf = min(1.0, (f.reciprocity + f.actionability_rate + f.task_ratio) / 2.5)
        return StageClassification(
            stage="performing", confidence=conf,
            features=f, rule_triggered="reciprocity>0.5 AND actionability>0.4 AND task>0.6",
        )

    # Norming: положительная социо + кооперация
    if f.positive_social_ratio > 0.30 and f.cooperation_word_rate > 0.15:
        conf = min(1.0, (f.positive_social_ratio + f.cooperation_word_rate) * 1.5)
        return StageClassification(
            stage="norming", confidence=conf,
            features=f, rule_triggered="positive_social>0.30 AND cooperation>0.15",
        )

    # Forming: дефолт. Уверенность низкая если данных мало или нет явных сигналов.
    base_conf = 0.4
    if f.question_rate > 0.4:  # много вопросов = знакомство
        base_conf = min(1.0, base_conf + f.question_rate * 0.4)
    if f.n_turns < 5:  # мало данных
        base_conf = min(base_conf, 0.5)
    return StageClassification(
        stage="forming", confidence=base_conf,
        features=f, rule_triggered="default (no other rule fired)",
    )


# ---------------------------------------------------------------------------
# Окнованная классификация всего диалога
# ---------------------------------------------------------------------------

def classify_dialogue_windows(
    history: Sequence[dict],
    window_size: int = 10,
    step: int = 5,
) -> list[StageClassification]:
    """
    Применить classify_stage ко всем скользящим окнам.

    Используется в H3 для построения последовательности (s_1, s_2, ..., s_K)
    и матрицы переходов P_obs.
    """
    if len(history) < window_size:
        return [classify_stage(extract_features(history))] if history else []

    classifications = []
    for start in range(0, len(history) - window_size + 1, step):
        win = history[start:start + window_size]
        classifications.append(classify_stage(extract_features(win)))
    return classifications


def stage_sequence(history: Sequence[dict],
                    window_size: int = 10,
                    step: int = 5) -> list[str]:
    """Удобный wrapper: вернуть только имена стадий."""
    return [c.stage for c in classify_dialogue_windows(history, window_size, step)]


def transition_matrix(stages: Sequence[str]) -> dict[tuple[str, str], int]:
    """Подсчитать переходы между стадиями для χ²-теста (H3)."""
    transitions: dict[tuple[str, str], int] = {}
    for a, b in zip(stages, stages[1:]):
        transitions[(a, b)] = transitions.get((a, b), 0) + 1
    return transitions
