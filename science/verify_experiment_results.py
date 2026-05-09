"""
Верификация научной валидности результатов эксперимента.

Проверяет, что в собранных данных нет следов сбоев API/валидации, которые
были бы скрытно подменены fallback-строками. Используется как smoke-check
после прогона ДО того, как результаты пойдут в финальный отчёт.

Проверки:
  1. Все диалоги завершились без ошибки (нет ключа `error` в записях)
  2. В history нет fallback-строк (если каким-то образом strict_mode не сработал)
  3. arm_metrics не пустые (минимум 1 успешный диалог в каждой arm)
  4. Соотношение arm sizes соответствует config (нет тихой потери диалогов)

Запуск:
    python -m agent_dialogue_sim.verify_experiment_results <results_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


FALLBACK_MARKERS = [
    "having trouble responding",
    "I'm here, but",
    "let me think about that for a moment",
]


@dataclass
class VerificationReport:
    output_dir: str
    config_id: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    arm_stats: dict[str, dict] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0


def _scan_arm(jsonl_path: Path) -> dict:
    n_total = 0
    n_errors = 0
    n_fallback_dialogues = 0
    n_fallback_replies = 0
    n_real_replies = 0
    error_messages = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        n_total += 1
        if "error" in rec:
            n_errors += 1
            error_messages.append(rec.get("error", ""))
            continue
        had_fallback = False
        for h in rec.get("history", []):
            reply = h.get("reply", "") or ""
            if any(marker in reply for marker in FALLBACK_MARKERS):
                n_fallback_replies += 1
                had_fallback = True
            else:
                n_real_replies += 1
        if had_fallback:
            n_fallback_dialogues += 1
    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "n_fallback_dialogues": n_fallback_dialogues,
        "n_fallback_replies": n_fallback_replies,
        "n_real_replies": n_real_replies,
        "error_messages_sample": error_messages[:3],
    }


def verify(output_dir: Path,
           strict: bool = True,
           min_per_arm: int = 1) -> VerificationReport:
    """
    output_dir — путь к одному результату эксперимента (с config.yaml + arm_*.jsonl).
    strict=True: любой fallback / error → issue (блокирующая ошибка).
    strict=False: warnings вместо issues.
    """
    report = VerificationReport(
        output_dir=str(output_dir),
        config_id="?",
    )
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        report.issues.append(f"missing config.yaml in {output_dir}")
        return report

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    report.config_id = config.get("id", "?")

    expected_arm_sizes = {
        a["name"]: a["n_dialogues"] for a in config.get("arms", [])
    }

    for arm_name, expected_n in expected_arm_sizes.items():
        jsonl_path = output_dir / f"arm_{arm_name}.jsonl"
        if not jsonl_path.exists():
            report.issues.append(f"missing arm_{arm_name}.jsonl")
            continue
        stats = _scan_arm(jsonl_path)
        report.arm_stats[arm_name] = stats

        # Проверка 1: нет лишних/недостающих диалогов
        if stats["n_total"] != expected_n:
            report.issues.append(
                f"arm '{arm_name}': expected {expected_n} dialogues, got {stats['n_total']}"
            )

        # Проверка 2: errors
        if stats["n_errors"] > 0:
            msg = f"arm '{arm_name}': {stats['n_errors']} dialogues failed with error"
            if stats["error_messages_sample"]:
                msg += f" (sample: {stats['error_messages_sample'][0][:120]})"
            (report.issues if strict else report.warnings).append(msg)

        # Проверка 3: fallback-реплики
        if stats["n_fallback_replies"] > 0:
            msg = (
                f"arm '{arm_name}': {stats['n_fallback_replies']} fallback replies "
                f"detected in {stats['n_fallback_dialogues']} dialogues "
                f"(real replies: {stats['n_real_replies']})"
            )
            (report.issues if strict else report.warnings).append(msg)

        # Проверка 4: минимум диалогов
        successful = stats["n_total"] - stats["n_errors"]
        if successful < min_per_arm:
            report.issues.append(
                f"arm '{arm_name}': only {successful} successful dialogues, need ≥ {min_per_arm}"
            )

    return report


def render_text(report: VerificationReport) -> str:
    lines = [
        f"=== Verification report ===",
        f"Output dir: {report.output_dir}",
        f"Config:     {report.config_id}",
        f"Status:     {'✅ VALID' if report.is_valid else '❌ INVALID'}",
        "",
    ]
    if report.arm_stats:
        lines.append("Arm statistics:")
        for arm, stats in report.arm_stats.items():
            lines.append(
                f"  {arm}: total={stats['n_total']}, errors={stats['n_errors']}, "
                f"fallback_replies={stats['n_fallback_replies']}, "
                f"real_replies={stats['n_real_replies']}"
            )
        lines.append("")
    if report.issues:
        lines.append("Issues (BLOCKING):")
        for issue in report.issues:
            lines.append(f"  ❌ {issue}")
        lines.append("")
    if report.warnings:
        lines.append("Warnings:")
        for w in report.warnings:
            lines.append(f"  ⚠ {w}")
    if report.is_valid and not report.warnings:
        lines.append("No issues — results are scientifically valid for analysis.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify scientific validity of experiment results."
    )
    parser.add_argument("results_dir", type=Path,
                        help="Path to experiment results directory (contains config.yaml).")
    parser.add_argument("--strict", action="store_true", default=True,
                        help="Treat fallbacks/errors as blocking issues (default).")
    parser.add_argument("--lax", dest="strict", action="store_false",
                        help="Demote fallbacks/errors to warnings only.")
    parser.add_argument("--min-per-arm", type=int, default=1,
                        help="Minimum successful dialogues per arm.")
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: directory does not exist: {args.results_dir}", file=sys.stderr)
        return 2

    report = verify(args.results_dir, strict=args.strict, min_per_arm=args.min_per_arm)
    print(render_text(report))
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
