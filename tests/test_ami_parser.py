"""
Юнит-тесты для ami_parser.py.

Не требует загруженного AMI корпуса (тяжёлый, ~3GB) — генерирует
синтетические .words.xml fixtures в temp dir и проверяет всю pipeline.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from agent_dialogue_sim.science.ami_parser import (
    AMIWord,
    AMIUtterance,
    AMIDialogue,
    parse_words_xml,
    group_words_into_utterances,
    parse_meeting,
    list_meetings,
    gini_coefficient,
    dialogue_gini,
    history_gini,
    build_human_baseline,
)


def _write_words_xml(path: Path, words: list[tuple[str, float, float]]) -> None:
    """Создать упрощённый AMI .words.xml файл с указанными словами."""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<nite:root xmlns:nite="http://nite.sourceforge.net/">')
    for i, (text, start, end) in enumerate(words):
        lines.append(
            f'  <w nite:id="w{i}" nite:start="{start}" nite:end="{end}">{text}</w>'
        )
    lines.append('</nite:root>')
    path.write_text("\n".join(lines), encoding="utf-8")


class TestGiniCoefficient(unittest.TestCase):
    def test_perfect_equality(self):
        self.assertAlmostEqual(gini_coefficient([10, 10, 10, 10]), 0.0, places=5)

    def test_perfect_inequality(self):
        # один человек говорит всё
        gini = gini_coefficient([100, 0, 0, 0])
        self.assertGreater(gini, 0.7)

    def test_empty_returns_zero(self):
        self.assertEqual(gini_coefficient([]), 0.0)

    def test_single_speaker(self):
        self.assertEqual(gini_coefficient([50]), 0.0)

    def test_known_value(self):
        # 4 спикера: 50, 30, 15, 5 — умеренное неравенство
        gini = gini_coefficient([50, 30, 15, 5])
        self.assertGreater(gini, 0.2)
        self.assertLess(gini, 0.5)


class TestParseWordsXML(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parses_simple_xml(self):
        xml_path = self.tmp / "EN2001a.A.words.xml"
        _write_words_xml(xml_path, [
            ("Hello", 0.0, 0.5),
            ("everyone", 0.6, 1.2),
            ("today", 1.3, 1.7),
        ])
        words = parse_words_xml(xml_path)
        self.assertEqual(len(words), 3)
        self.assertEqual(words[0].text, "Hello")
        self.assertEqual(words[0].speaker, "A")
        self.assertEqual(words[2].speaker, "A")

    def test_skips_words_without_timing(self):
        xml_path = self.tmp / "EN2001a.A.words.xml"
        # Один из слов без атрибутов времени
        xml_path.write_text(
            '<?xml version="1.0"?>'
            '<nite:root xmlns:nite="http://nite.sourceforge.net/">'
            '<w nite:id="w0" nite:start="0.0" nite:end="0.5">Hello</w>'
            '<w nite:id="w1">noTiming</w>'
            '<w nite:id="w2" nite:start="1.0" nite:end="1.5">World</w>'
            '</nite:root>',
            encoding="utf-8",
        )
        words = parse_words_xml(xml_path)
        self.assertEqual(len(words), 2)


class TestGroupIntoUtterances(unittest.TestCase):
    def test_pause_splits_utterances(self):
        # Слова: 0-1, 1.1-2 (~0.1s gap), 5-6 (3s gap → новая реплика)
        words = [
            AMIWord("hi", 0.0, 1.0, "A"),
            AMIWord("there", 1.1, 2.0, "A"),
            AMIWord("ok", 5.0, 6.0, "A"),
        ]
        utts = group_words_into_utterances(words, pause_threshold=1.5)
        self.assertEqual(len(utts), 2)
        self.assertEqual(utts[0].text, "hi there")
        self.assertEqual(utts[1].text, "ok")

    def test_empty_words(self):
        self.assertEqual(group_words_into_utterances([]), [])


class TestParseMeeting(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "words").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_parses_4_speaker_meeting(self):
        # Создаём 4 файла для митинга EN2001a
        _write_words_xml(self.tmp / "words" / "EN2001a.A.words.xml", [
            ("hello", 0.0, 0.5), ("everyone", 0.6, 1.2),
        ])
        _write_words_xml(self.tmp / "words" / "EN2001a.B.words.xml", [
            ("hi", 1.5, 2.0), ("Alex", 2.1, 2.5),
        ])
        _write_words_xml(self.tmp / "words" / "EN2001a.C.words.xml", [
            ("good", 3.0, 3.4), ("morning", 3.5, 4.0),
        ])
        _write_words_xml(self.tmp / "words" / "EN2001a.D.words.xml", [
            ("yes", 5.0, 5.3),
        ])
        dlg = parse_meeting(self.tmp, "EN2001a")
        self.assertEqual(dlg.n_speakers, 4)
        self.assertEqual(dlg.meeting_id, "EN2001a")
        # Все реплики от 4 спикеров
        speakers = {u.speaker for u in dlg.utterances}
        self.assertEqual(speakers, {"A", "B", "C", "D"})

    def test_missing_meeting_raises(self):
        with self.assertRaises(FileNotFoundError):
            parse_meeting(self.tmp, "NONEXISTENT")

    def test_to_history_format(self):
        _write_words_xml(self.tmp / "words" / "M1.A.words.xml", [("hi", 0.0, 0.5)])
        _write_words_xml(self.tmp / "words" / "M1.B.words.xml", [("ok", 1.0, 1.5)])
        dlg = parse_meeting(self.tmp, "M1")
        history = dlg.to_history_format()
        self.assertEqual(len(history), 2)
        self.assertIn("speaker", history[0])
        self.assertIn("reply", history[0])


class TestDialogueGini(unittest.TestCase):
    def test_balanced_dialogue_low_gini(self):
        utts = [
            AMIUtterance("A", "hi", 0, 1, 1),
            AMIUtterance("B", "hi", 1, 2, 1),
            AMIUtterance("C", "hi", 2, 3, 1),
            AMIUtterance("D", "hi", 3, 4, 1),
        ]
        dlg = AMIDialogue(meeting_id="m1", n_speakers=4, utterances=utts)
        self.assertAlmostEqual(dialogue_gini(dlg), 0.0, places=5)

    def test_imbalanced_dialogue_high_gini(self):
        utts = [AMIUtterance("A", "x", i, i + 1, 1) for i in range(20)]
        utts.append(AMIUtterance("B", "x", 21, 22, 1))
        dlg = AMIDialogue(meeting_id="m1", n_speakers=2, utterances=utts)
        self.assertGreater(dialogue_gini(dlg), 0.4)


class TestHistoryGini(unittest.TestCase):
    def test_compatible_with_experiment_runner_format(self):
        history = [
            {"speaker": "Alex", "reply": "x"},
            {"speaker": "Sam", "reply": "y"},
            {"speaker": "Alex", "reply": "z"},
            {"speaker": "Alex", "reply": "w"},
        ]
        # Alex=3, Sam=1 → умеренное неравенство
        gini = history_gini(history)
        self.assertGreater(gini, 0.2)
        self.assertLess(gini, 0.6)


class TestBuildHumanBaseline(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "words").mkdir()
        # Создаём 3 митинга с разной структурой
        # M1: 4 равных спикера
        for sp in ["A", "B", "C", "D"]:
            _write_words_xml(
                self.tmp / "words" / f"M1.{sp}.words.xml",
                [(f"{sp}{i}", float(i * 5 + ord(sp)), float(i * 5 + ord(sp) + 0.3))
                 for i in range(5)],
            )
        # M2: 4 спикера, но один доминирует (A=15, B=2, C=2, D=1)
        for sp, n in [("A", 15), ("B", 2), ("C", 2), ("D", 1)]:
            _write_words_xml(
                self.tmp / "words" / f"M2.{sp}.words.xml",
                [(f"{sp}{i}", float(i * 4 + ord(sp)), float(i * 4 + ord(sp) + 0.3))
                 for i in range(n)],
            )
        # M3: 2 спикера (вне диапазона 3-5)
        for sp in ["A", "B"]:
            _write_words_xml(
                self.tmp / "words" / f"M3.{sp}.words.xml",
                [("hi", 0.0, 0.5)],
            )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_jsonl_filtering_by_speaker_count(self):
        out_path = self.tmp / "human_baseline.jsonl"
        summary = build_human_baseline(
            meeting_dir=self.tmp,
            output_jsonl=out_path,
            min_speakers=3,
            max_speakers=5,
        )
        # M3 имеет 2 спикеров — пропускается
        self.assertEqual(summary["n_meetings_written"], 2)
        self.assertEqual(summary["n_meetings_skipped"], 1)

        # Читаем JSONL
        lines = out_path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        for line in lines:
            rec = json.loads(line)
            self.assertIn("meeting_id", rec)
            self.assertIn("gini", rec)
            self.assertIn("n_speakers", rec)
            self.assertGreaterEqual(rec["n_speakers"], 3)
            self.assertLessEqual(rec["n_speakers"], 5)


class TestListMeetings(unittest.TestCase):
    def test_list(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            (tmp / "words").mkdir()
            for f in ["EN2001a.A.words.xml", "EN2001a.B.words.xml",
                      "EN2002b.A.words.xml"]:
                (tmp / "words" / f).write_text("<root/>")
            ids = list_meetings(tmp)
            self.assertEqual(ids, ["EN2001a", "EN2002b"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
