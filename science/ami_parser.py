"""
Парсер корпуса AMI Meeting Corpus (актуальная версия описана в
Hain et al., 2012, IEEE Trans. ASLP, 20(2), 486–503).
Используется для гипотезы H7 — построение human baseline для коэффициента Джини
распределения реплик в реальных малых группах.

AMI Meeting Corpus: https://groups.inf.ed.ac.uk/ami/corpus/
Open license; 100 часов записанных совещаний по 4 человека (~140 диалогов).

Формат AMI: NXT XML (Nite XML Toolkit). Каждое слово — отдельный <w> элемент
с атрибутами `nite:start`, `nite:end`, `nite:id`. Слова сгруппированы в реплики
через файлы сегментации.

Минимальный путь использования (после загрузки корпуса):
  ami_data/words/EN2001a.A.words.xml
  ami_data/words/EN2001a.B.words.xml
  ...
  ami_data/dialog-acts/EN2001a.A.dialog-act.xml

Этот модуль:
  1. Парсит .words.xml — реплики каждого спикера в одном диалоге
  2. Группирует слова в реплики на основе пауз > 1.5 сек
  3. Строит "history" в формате нашего эксперимента
  4. Считает Gini коэффициент распределения числа реплик по спикерам

Если корпус не загружен — модуль предоставляет синтетические fixtures для тестов
и stubs, документирующие как будет выглядеть реальный pipeline.

Скачивание корпуса (вручную):
  https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/

Запуск парсинга после загрузки:
  python -m agent_dialogue_sim.ami_parser \\
      --ami-dir ./ami_data \\
      --output-jsonl agent_dialogue_sim/experiments/data/ami_human_baseline.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# AMI словные XML используют namespace {http://nite.sourceforge.net/}.
NITE_NS = "{http://nite.sourceforge.net/}"


@dataclass
class AMIWord:
    text: str
    start: float  # seconds
    end: float
    speaker: str  # "A", "B", "C", "D" (или роль из имени файла)


@dataclass
class AMIUtterance:
    """Реплика — последовательность слов одного спикера без больших пауз."""
    speaker: str
    text: str
    start: float
    end: float
    n_words: int


@dataclass
class AMIDialogue:
    meeting_id: str
    n_speakers: int
    utterances: list[AMIUtterance] = field(default_factory=list)

    def to_history_format(self) -> list[dict]:
        """Конвертировать в формат нашего experiment_runner (для удобства совместимости)."""
        return [
            {
                "speaker": u.speaker,
                "target": None,  # AMI не размечает адресата
                "reply": u.text,
                "tone": "neutral",  # без аффективной разметки
                "emotion": "neutral",
                "start": u.start,
                "end": u.end,
                "n_words": u.n_words,
            }
            for u in self.utterances
        ]


# ---------------------------------------------------------------------------
# Парсинг XML
# ---------------------------------------------------------------------------

def _speaker_from_filename(filepath: str | Path) -> str:
    """AMI имена: EN2001a.A.words.xml → spaker = 'A'."""
    name = Path(filepath).name
    parts = name.split(".")
    if len(parts) >= 2:
        return parts[1]
    return "?"


def _meeting_id_from_filename(filepath: str | Path) -> str:
    name = Path(filepath).name
    return name.split(".")[0]


def parse_words_xml(filepath: str | Path) -> list[AMIWord]:
    """
    Парсит один .words.xml файл.
    Каждое слово — элемент <w nite:id=... nite:start=... nite:end=...>текст</w>.
    """
    tree = ET.parse(str(filepath))
    root = tree.getroot()
    speaker = _speaker_from_filename(filepath)
    words: list[AMIWord] = []

    # Слова могут быть как <w> так и <vocalsound> и пр. Берём только <w>.
    for w in root.iter():
        if not w.tag.endswith("}w") and not w.tag == "w":
            continue
        text = (w.text or "").strip()
        if not text:
            continue
        start_str = w.get(NITE_NS + "start") or w.get("nite:start") or w.get("start")
        end_str = w.get(NITE_NS + "end") or w.get("nite:end") or w.get("end")
        try:
            start = float(start_str)
            end = float(end_str)
        except (TypeError, ValueError):
            continue
        words.append(AMIWord(text=text, start=start, end=end, speaker=speaker))
    return words


def group_words_into_utterances(
    words: Sequence[AMIWord],
    pause_threshold: float = 1.5,
) -> list[AMIUtterance]:
    """
    Группирует слова одного спикера в реплики:
      - новая реплика, если пауза с предыдущим словом > pause_threshold секунд
    """
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: w.start)
    utterances: list[AMIUtterance] = []
    current: list[AMIWord] = [sorted_words[0]]
    for w in sorted_words[1:]:
        gap = w.start - current[-1].end
        if gap > pause_threshold:
            utterances.append(_finalize_utterance(current))
            current = [w]
        else:
            current.append(w)
    if current:
        utterances.append(_finalize_utterance(current))
    return utterances


def _finalize_utterance(words: list[AMIWord]) -> AMIUtterance:
    text = " ".join(w.text for w in words)
    return AMIUtterance(
        speaker=words[0].speaker,
        text=text,
        start=words[0].start,
        end=words[-1].end,
        n_words=len(words),
    )


def parse_meeting(meeting_dir: str | Path,
                  meeting_id: str,
                  pause_threshold: float = 1.5) -> AMIDialogue:
    """
    Парсит ОДИН митинг (все спикеры): meeting_dir/words/<meeting_id>.<X>.words.xml
    """
    pattern = str(Path(meeting_dir) / "words" / f"{meeting_id}.*.words.xml")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .words.xml found for meeting {meeting_id} in {meeting_dir}")

    all_utterances: list[AMIUtterance] = []
    speakers = set()
    for f in files:
        speaker = _speaker_from_filename(f)
        speakers.add(speaker)
        words = parse_words_xml(f)
        utterances = group_words_into_utterances(words, pause_threshold=pause_threshold)
        all_utterances.extend(utterances)

    # Сортируем все реплики по времени старта — получаем последовательность диалога
    all_utterances.sort(key=lambda u: u.start)

    return AMIDialogue(
        meeting_id=meeting_id,
        n_speakers=len(speakers),
        utterances=all_utterances,
    )


def list_meetings(meeting_dir: str | Path) -> list[str]:
    """Найти все meeting_id в каталоге AMI."""
    pattern = str(Path(meeting_dir) / "words" / "*.words.xml")
    ids = set()
    for f in glob.glob(pattern):
        ids.add(_meeting_id_from_filename(f))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Gini коэффициент (для H7)
# ---------------------------------------------------------------------------

def gini_coefficient(counts: Sequence[float]) -> float:
    """
    Gini ∈ [0, 1]: 0 — равномерное распределение, 1 — один человек говорит всё.
    Формула Дамгаарда (mean absolute difference / 2*mean).
    """
    if not counts:
        return 0.0
    n = len(counts)
    if n == 1:
        return 0.0
    sorted_counts = sorted(counts)
    s = sum(sorted_counts)
    if s == 0:
        return 0.0
    cum = 0.0
    for i, c in enumerate(sorted_counts):
        cum += (i + 1) * c
    return (2 * cum) / (n * s) - (n + 1) / n


def dialogue_gini(dialogue: AMIDialogue) -> float:
    """Gini для одного AMI диалога — по числу реплик каждого спикера."""
    counter: Counter[str] = Counter()
    for u in dialogue.utterances:
        counter[u.speaker] += 1
    return gini_coefficient(list(counter.values()))


def history_gini(history: Sequence[dict]) -> float:
    """Gini на нашем history-формате (для совместимости с experiment_runner)."""
    counter: Counter[str] = Counter()
    for r in history:
        s = r.get("speaker")
        if s:
            counter[s] += 1
    return gini_coefficient(list(counter.values()))


# ---------------------------------------------------------------------------
# Сохранение в формате human baseline для experiment_runner / TOST H7
# ---------------------------------------------------------------------------

def build_human_baseline(
    meeting_dir: str | Path,
    output_jsonl: str | Path,
    meeting_ids: Optional[Sequence[str]] = None,
    min_speakers: int = 3,
    max_speakers: int = 5,
    max_meetings: Optional[int] = None,
) -> dict:
    """
    Парсит N митингов AMI и сохраняет на каждую строку JSONL запись:
      {"meeting_id": ..., "n_speakers": ..., "n_utterances": ..., "gini": ...}

    Совместимо по схеме с arm_*.jsonl нашего experiment_runner — поле "gini"
    может быть подано прямо в TOST через analysis.metric="gini".

    Возвращает summary dict с агрегированной статистикой.
    """
    if meeting_ids is None:
        meeting_ids = list_meetings(meeting_dir)
    if max_meetings:
        meeting_ids = meeting_ids[:max_meetings]

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    gini_values: list[float] = []
    with out_path.open("w", encoding="utf-8") as f:
        for mid in meeting_ids:
            try:
                dlg = parse_meeting(meeting_dir, mid)
            except FileNotFoundError as e:
                logger.warning("skipping %s: %s", mid, e)
                skipped += 1
                continue
            if not (min_speakers <= dlg.n_speakers <= max_speakers):
                logger.debug("skipping %s: %d speakers outside [%d, %d]",
                             mid, dlg.n_speakers, min_speakers, max_speakers)
                skipped += 1
                continue
            g = dialogue_gini(dlg)
            gini_values.append(g)
            f.write(json.dumps({
                "meeting_id": mid,
                "n_speakers": dlg.n_speakers,
                "n_utterances": len(dlg.utterances),
                "gini": g,
            }, ensure_ascii=False) + "\n")
            written += 1

    summary = {
        "n_meetings_written": written,
        "n_meetings_skipped": skipped,
        "gini_mean": (sum(gini_values) / len(gini_values)) if gini_values else None,
        "gini_min": min(gini_values) if gini_values else None,
        "gini_max": max(gini_values) if gini_values else None,
        "output_path": str(out_path),
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Parse AMI corpus → human baseline JSONL.")
    parser.add_argument("--ami-dir", type=Path, required=True,
                        help="Path to AMI corpus root (must contain words/ subdir).")
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-meetings", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=3)
    parser.add_argument("--max-speakers", type=int, default=5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s: %(message)s")

    summary = build_human_baseline(
        meeting_dir=args.ami_dir,
        output_jsonl=args.output_jsonl,
        max_meetings=args.max_meetings,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["n_meetings_written"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
