"""
Controlled A/B Experiment Framework with Statistical Analysis.

Implements experimental design for comparing dialogue conditions with
proper statistical testing and effect size estimation.

Scientific basis:
- Experimental design in computational social science (Salganik, 2017)
- Effect size estimation (Cohen, 1988; Lakens, 2013)
- Reproducible research practices (Open Science Collaboration, 2015)

Usage:
    python -m agent_dialogue_sim.experiments.ab_experiments \\
        --experiment facilitator_effect --runs 10 --turns 15
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from openai import OpenAI

from ..config import (
    DEFAULT_AGENTS,
    DEFAULT_ENV_CONTEXT_INDEX,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    VIZ_EDGE_WINDOW,
)
from ..core.agents import Agent
from ..core.dialogue_manager import DialogueManager
from ..core.env_context import get_env_context_by_index
from ..analytics.basic import compute_metrics


# =============================================================================
# STATISTICAL FUNCTIONS
# =============================================================================

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """
    Compute Cohen's d effect size.

    Reference: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences.
    See also: Lakens, D. (2013). Calculating and reporting effect sizes.
    Frontiers in Psychology, 4, 863.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    m_a, m_b = mean(group_a), mean(group_b)
    s_a, s_b = std(group_a), std(group_b)
    # Pooled standard deviation
    s_pooled = math.sqrt(((n_a - 1) * s_a ** 2 + (n_b - 1) * s_b ** 2) / (n_a + n_b - 2))
    if s_pooled == 0:
        return 0.0
    return (m_a - m_b) / s_pooled


def welch_t_test(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
    """
    Welch's t-test (unequal variances).
    Returns (t_statistic, approximate_p_value).

    Uses Welch-Satterthwaite approximation for degrees of freedom.
    P-value approximated via normal distribution for simplicity.
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0
    m_a, m_b = mean(group_a), mean(group_b)
    s_a, s_b = std(group_a), std(group_b)
    se = math.sqrt(s_a ** 2 / n_a + s_b ** 2 / n_b)
    if se == 0:
        return 0.0, 1.0
    t = (m_a - m_b) / se
    # Welch-Satterthwaite degrees of freedom
    num = (s_a ** 2 / n_a + s_b ** 2 / n_b) ** 2
    denom = (s_a ** 2 / n_a) ** 2 / (n_a - 1) + (s_b ** 2 / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else 1
    # Approximate p-value using normal approximation (good for df > 30)
    p = 2 * _normal_cdf(-abs(t))
    return t, p


def mann_whitney_u(group_a: List[float], group_b: List[float]) -> Tuple[float, float]:
    """
    Mann-Whitney U test (non-parametric).
    Returns (U_statistic, approximate_p_value).
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    # Combine and rank
    combined = [(v, "a") for v in group_a] + [(v, "b") for v in group_b]
    combined.sort(key=lambda x: x[0])
    # Assign ranks (handle ties by averaging)
    ranks = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[id(combined[k])] = avg_rank
        i = j
    # Sum ranks for group A
    r_a = sum(ranks[id(c)] for c in combined if c[1] == "a")
    u_a = r_a - n_a * (n_a + 1) / 2
    u_b = n_a * n_b - u_a
    u = min(u_a, u_b)
    # Normal approximation for p-value
    mu = n_a * n_b / 2
    sigma = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma == 0:
        return u, 1.0
    z = (u - mu) / sigma
    p = 2 * _normal_cdf(-abs(z))
    return u, p


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def effect_size_label(d: float) -> str:
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def confidence_interval(values: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for the mean."""
    n = len(values)
    if n < 2:
        return (mean(values), mean(values))
    m = mean(values)
    se = std(values) / math.sqrt(n)
    # z-value for 95% CI
    z = 1.96 if confidence == 0.95 else 2.576
    return (m - z * se, m + z * se)


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

@dataclass
class ExperimentCondition:
    """One experimental condition (control or treatment)."""
    name: str
    agents_config: List[Dict[str, Any]]
    description: str = ""


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    condition: str
    run_index: int
    seed: Optional[int]
    metrics: Dict[str, Any]
    history: List[Dict[str, Any]]
    turns: int


@dataclass
class StatisticalComparison:
    """Statistical comparison between two conditions."""
    metric_name: str
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    ci_a: Tuple[float, float]
    ci_b: Tuple[float, float]
    t_stat: float
    p_value: float
    u_stat: float
    u_p_value: float
    cohens_d: float
    effect_label: str
    significant: bool  # p < 0.05


@dataclass
class ExperimentReport:
    """Full experiment report with statistical analysis."""
    experiment_name: str
    hypothesis: str
    conditions: List[str]
    runs_per_condition: int
    turns_per_run: int
    comparisons: List[StatisticalComparison]
    results: List[ExperimentResult]
    timestamp: str = ""


# Predefined experiment configurations
EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    "facilitator_effect": {
        "hypothesis": "The presence of a Facilitator agent improves group cohesion, "
                      "reduces turn inequality, and increases positive tone.",
        "conditions": {
            "control": ExperimentCondition(
                name="without_facilitator",
                description="Two agents (Explorer + Critic) without Facilitator",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                ],
            ),
            "treatment": ExperimentCondition(
                name="with_facilitator",
                description="Three agents (Explorer + Critic + Facilitator)",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                    {"name": "Sam", "nature": "facilitator", "color": "#059669"},
                ],
            ),
        },
    },
    "group_size": {
        "hypothesis": "Larger groups (4 agents) show lower turn equality and higher "
                      "network density compared to smaller groups (3 agents).",
        "conditions": {
            "small": ExperimentCondition(
                name="3_agents",
                description="3 agents",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                    {"name": "Sam", "nature": "facilitator", "color": "#059669"},
                ],
            ),
            "large": ExperimentCondition(
                name="4_agents",
                description="4 agents",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                    {"name": "Sam", "nature": "facilitator", "color": "#059669"},
                    {"name": "Riley", "nature": "explorer", "color": "#7C3AED"},
                ],
            ),
        },
    },
    "environment_effect": {
        "hypothesis": "The type of environment context affects dialogue structure: "
                      "university context produces more questions, startup context "
                      "produces more actionable items.",
        "conditions": {
            "university": ExperimentCondition(
                name="university",
                description="University project planning context",
                agents_config=DEFAULT_AGENTS,
            ),
            "startup": ExperimentCondition(
                name="startup",
                description="Startup MVP prioritization context",
                agents_config=DEFAULT_AGENTS,
            ),
        },
    },
    "bad_apple": {
        "hypothesis": "A single toxic agent (high Neuroticism, low Agreeableness) "
                      "reduces group positivity, increases emotional contagion of "
                      "negative tone, and lowers discourse coherence (Felps et al., 2006).",
        "conditions": {
            "control": ExperimentCondition(
                name="healthy_group",
                description="3 balanced agents without toxic member",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                    {"name": "Sam", "nature": "facilitator", "color": "#059669"},
                ],
            ),
            "treatment": ExperimentCondition(
                name="with_bad_apple",
                description="3 balanced agents + 1 toxic agent",
                agents_config=[
                    {"name": "Alex", "nature": "explorer", "color": "#4F46E5"},
                    {"name": "Jordan", "nature": "critic", "color": "#DC2626"},
                    {"name": "Sam", "nature": "facilitator", "color": "#059669"},
                    {"name": "Viper", "nature": "toxic", "color": "#7F1D1D",
                     "persona": (
                        "You are a deeply frustrated, cynical person. You disagree with almost everything. "
                        "You use sarcasm, dismiss others' ideas, interrupt with 'that won't work', "
                        "'this is a waste of time', 'you clearly don't understand'. "
                        "You rarely offer alternatives — mostly just tear things down. "
                        "You sigh a lot. You say 'whatever' and 'sure, good luck with that'. "
                        "You're not evil — just burned out and bitter. "
                        "Personality: Neuroticism=0.95, Agreeableness=0.1, Extraversion=0.7"
                     )},
                ],
            ),
        },
    },
}


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_condition(
    condition: ExperimentCondition,
    turns: int,
    env_index: int,
    runs: int,
    base_seed: Optional[int] = None,
) -> List[ExperimentResult]:
    """Run multiple sessions for one experimental condition."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    results = []
    for i in range(runs):
        seed = base_seed + i if base_seed is not None else None
        if seed is not None:
            random.seed(seed)

        client = OpenAI(**({"base_url": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}))
        agents = [Agent(**cfg) for cfg in condition.agents_config]
        env_context = get_env_context_by_index(env_index)
        dm = DialogueManager(client=client, agents=agents, env_context=env_context, human_io=None)

        history = []
        for _ in range(turns):
            record, _ = dm.step()
            history.append(record)

        agent_names = [a.name for a in agents]
        metrics = compute_metrics(history, agent_names, window_size=VIZ_EDGE_WINDOW)

        results.append(ExperimentResult(
            condition=condition.name,
            run_index=i,
            seed=seed,
            metrics=metrics,
            history=history,
            turns=dm.turn_no,
        ))
        print(f"  [{condition.name}] Run {i+1}/{runs} done (seed={seed})")

    return results


def extract_metric_values(results: List[ExperimentResult], metric_path: str) -> List[float]:
    """Extract a specific metric from list of results.

    metric_path examples:
        "summary.total.avg_tone"
        "summary.total.reciprocity"
        "summary.total.question_rate"
    """
    values = []
    for r in results:
        obj = r.metrics
        for key in metric_path.split("."):
            if isinstance(obj, dict):
                obj = obj.get(key, None)
            else:
                obj = None
                break
        if obj is not None and isinstance(obj, (int, float)):
            values.append(float(obj))
    return values


def compare_conditions(
    results_a: List[ExperimentResult],
    results_b: List[ExperimentResult],
    condition_a_name: str,
    condition_b_name: str,
    metrics_to_compare: Optional[List[str]] = None,
) -> List[StatisticalComparison]:
    """Compare two conditions across multiple metrics with statistical tests."""
    if metrics_to_compare is None:
        metrics_to_compare = [
            "summary.total.avg_tone",
            "summary.total.reciprocity",
            "summary.total.targeting_rate",
            "summary.total.question_rate",
            "summary.total.emotion_entropy",
            "summary.total.actionability_rate",
        ]

    comparisons = []
    for metric_path in metrics_to_compare:
        vals_a = extract_metric_values(results_a, metric_path)
        vals_b = extract_metric_values(results_b, metric_path)

        if not vals_a or not vals_b:
            continue

        t, p_t = welch_t_test(vals_a, vals_b)
        u, p_u = mann_whitney_u(vals_a, vals_b)
        d = cohens_d(vals_a, vals_b)

        comparisons.append(StatisticalComparison(
            metric_name=metric_path.split(".")[-1],
            condition_a=condition_a_name,
            condition_b=condition_b_name,
            mean_a=mean(vals_a),
            mean_b=mean(vals_b),
            std_a=std(vals_a),
            std_b=std(vals_b),
            ci_a=confidence_interval(vals_a),
            ci_b=confidence_interval(vals_b),
            t_stat=t,
            p_value=p_t,
            u_stat=u,
            u_p_value=p_u,
            cohens_d=d,
            effect_label=effect_size_label(d),
            significant=p_t < 0.05,
        ))

    return comparisons


def render_experiment_report(report: ExperimentReport) -> str:
    """Render experiment report as markdown with statistical tables."""
    lines = [
        f"# 🧪 Experiment Report: {report.experiment_name}",
        "",
        f"**Hypothesis:** {report.hypothesis}",
        "",
        f"**Design:** {report.runs_per_condition} runs × {report.turns_per_run} turns per condition",
        f"**Conditions:** {', '.join(report.conditions)}",
        f"**Generated:** {report.timestamp}",
        "",
        "---",
        "",
        "## Statistical Results",
        "",
        "| Metric | Condition A (M±SD) | Condition B (M±SD) | t | p-value | Cohen's d | Effect | Sig. |",
        "|--------|-------------------|-------------------|---|---------|-----------|--------|------|",
    ]

    for c in report.comparisons:
        sig_mark = "✅" if c.significant else "—"
        lines.append(
            f"| {c.metric_name} "
            f"| {c.condition_a}: {c.mean_a:.3f}±{c.std_a:.3f} "
            f"| {c.condition_b}: {c.mean_b:.3f}±{c.std_b:.3f} "
            f"| {c.t_stat:.2f} "
            f"| {c.p_value:.4f} "
            f"| {c.cohens_d:.2f} "
            f"| {c.effect_label} "
            f"| {sig_mark} |"
        )

    lines.extend([
        "",
        "## Confidence Intervals (95%)",
        "",
        "| Metric | Condition A CI | Condition B CI |",
        "|--------|---------------|---------------|",
    ])

    for c in report.comparisons:
        lines.append(
            f"| {c.metric_name} "
            f"| [{c.ci_a[0]:.3f}, {c.ci_a[1]:.3f}] "
            f"| [{c.ci_b[0]:.3f}, {c.ci_b[1]:.3f}] |"
        )

    lines.extend([
        "",
        "## Mann-Whitney U Test (Non-parametric)",
        "",
        "| Metric | U | p-value | Significant |",
        "|--------|---|---------|-------------|",
    ])
    for c in report.comparisons:
        sig = "✅" if c.u_p_value < 0.05 else "—"
        lines.append(f"| {c.metric_name} | {c.u_stat:.1f} | {c.u_p_value:.4f} | {sig} |")

    # Summary
    sig_count = sum(1 for c in report.comparisons if c.significant)
    large_effects = [c for c in report.comparisons if abs(c.cohens_d) >= 0.8]

    lines.extend([
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Significant results (p<0.05):** {sig_count}/{len(report.comparisons)} metrics",
        f"- **Large effects (|d|≥0.8):** {len(large_effects)} metrics",
    ])

    if large_effects:
        for c in large_effects:
            direction = "higher" if c.mean_a > c.mean_b else "lower"
            lines.append(
                f"  - **{c.metric_name}**: {c.condition_a} is {direction} "
                f"(d={c.cohens_d:.2f}, p={c.p_value:.4f})"
            )

    lines.extend([
        "",
        "---",
        "*Statistical analysis based on Welch's t-test, Mann-Whitney U, and Cohen's d.*",
        "*Reference: Lakens, D. (2013). Calculating and reporting effect sizes. Frontiers in Psychology, 4, 863.*",
    ])

    return "\n".join(lines)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled A/B experiments.")
    parser.add_argument(
        "--experiment", type=str, default="facilitator_effect",
        choices=list(EXPERIMENTS.keys()),
        help="Experiment to run.",
    )
    parser.add_argument("--turns", type=int, default=15, help="Turns per session.")
    parser.add_argument("--runs", type=int, default=5, help="Runs per condition.")
    parser.add_argument("--env", type=int, default=DEFAULT_ENV_CONTEXT_INDEX)
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"))
    args = parser.parse_args()

    exp_def = EXPERIMENTS[args.experiment]
    conditions = exp_def["conditions"]
    cond_names = list(conditions.keys())

    output_dir = args.output_dir.resolve() / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[experiment] Running: {args.experiment}")
    print(f"[experiment] Hypothesis: {exp_def['hypothesis']}")
    print(f"[experiment] Conditions: {cond_names}")
    print(f"[experiment] {args.runs} runs × {args.turns} turns per condition")
    print()

    all_results: Dict[str, List[ExperimentResult]] = {}

    # Handle environment_effect experiment specially
    for cond_key, cond in conditions.items():
        env_idx = args.env
        if args.experiment == "environment_effect":
            env_idx = 0 if cond_key == "university" else 1

        print(f"[condition: {cond.name}] {cond.description}")
        results = run_condition(cond, args.turns, env_idx, args.runs, args.seed)
        all_results[cond_key] = results

    # Compare conditions
    comparisons = compare_conditions(
        all_results[cond_names[0]],
        all_results[cond_names[1]],
        cond_names[0],
        cond_names[1],
    )

    report = ExperimentReport(
        experiment_name=args.experiment,
        hypothesis=exp_def["hypothesis"],
        conditions=cond_names,
        runs_per_condition=args.runs,
        turns_per_run=args.turns,
        comparisons=comparisons,
        results=[r for results in all_results.values() for r in results],
        timestamp=datetime.utcnow().isoformat(),
    )

    # Save report
    report_md = render_experiment_report(report)
    (output_dir / "report.md").write_text(report_md, encoding="utf-8")

    # Save raw data
    raw_data = {
        "experiment": args.experiment,
        "hypothesis": exp_def["hypothesis"],
        "timestamp": report.timestamp,
        "config": {"runs": args.runs, "turns": args.turns, "seed": args.seed},
        "comparisons": [
            {
                "metric": c.metric_name,
                "mean_a": c.mean_a, "mean_b": c.mean_b,
                "std_a": c.std_a, "std_b": c.std_b,
                "t_stat": c.t_stat, "p_value": c.p_value,
                "u_stat": c.u_stat, "u_p_value": c.u_p_value,
                "cohens_d": c.cohens_d, "effect": c.effect_label,
                "significant": c.significant,
            }
            for c in comparisons
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[experiment] Report saved to {output_dir / 'report.md'}")
    print(f"[experiment] Raw data saved to {output_dir / 'results.json'}")

    # Print summary
    print("\n=== RESULTS ===")
    for c in comparisons:
        sig = "***" if c.significant else ""
        print(f"  {c.metric_name}: d={c.cohens_d:.2f} ({c.effect_label}), p={c.p_value:.4f} {sig}")


if __name__ == "__main__":
    main()
