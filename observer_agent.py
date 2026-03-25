"""
Observer Agent — LLM-based qualitative analyst for methodological triangulation.

This module implements a non-participating observer agent that periodically
analyzes dialogue context through an LLM, producing qualitative insights
that are compared with algorithmic metrics for triangulation.

Scientific basis:
- Methodological triangulation (Denzin, 2012; Creswell & Plano Clark, 2017)
- Mixed methods research design (Tashakkori & Teddlie, 2010)
- Computational social science (Lazer et al., 2020)

The observer produces:
1. Qualitative analysis: group dynamics, hidden coalitions, power asymmetry
2. Triangulation report: comparison of LLM-based vs. algorithmic findings
3. Convergence/divergence assessment between the two methods
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ObservationReport:
    """Single observation report from the observer agent."""
    turn: int
    # Qualitative LLM analysis
    group_dynamics: str = ""
    leadership_pattern: str = ""
    coalition_analysis: str = ""
    communication_barriers: str = ""
    emotional_undercurrent: str = ""
    # Triangulation
    convergence_score: float = 0.0  # 0..1, how much LLM agrees with metrics
    agreements: List[str] = field(default_factory=list)
    divergences: List[str] = field(default_factory=list)
    novel_insights: List[str] = field(default_factory=list)
    raw_llm_response: str = ""


@dataclass
class ObserverAgent:
    """
    Non-participating observer that analyzes dialogue via LLM
    and produces qualitative reports for methodological triangulation.

    Reference: Denzin, N. K. (2012). Triangulation 2.0.
    Journal of Mixed Methods Research, 6(2), 80-88.
    """
    client: Any  # OpenAI client
    model: str = ""
    history: List[ObservationReport] = field(default_factory=list)

    def __post_init__(self):
        if not self.model:
            from .config import MODEL
            self.model = MODEL

    def _build_observer_prompt(
        self,
        dialogue_history: List[Dict[str, Any]],
        metrics_summary: Optional[Dict[str, Any]] = None,
        scientific_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the system prompt for the observer agent."""
        # Format dialogue
        dialogue_text = "\n".join(
            f"[Turn {h.get('turn', '?')}] {h['speaker']}→{h.get('target', 'all')}: {h['reply']} "
            f"(tone: {h.get('tone', '?')}, emotion: {h.get('emotion', '?')})"
            for h in dialogue_history[-30:]  # last 30 turns
        )

        # Format algorithmic metrics if available
        metrics_text = ""
        if metrics_summary:
            s = metrics_summary.get("summary", {}).get("window", {})
            metrics_text = f"""
Algorithmic metrics (computed by code):
- Average tone: {s.get('avg_tone', 'N/A')}
- Targeting rate: {s.get('targeting_rate', 'N/A')}
- Talk shares: {s.get('talk_share', 'N/A')}
- Reciprocity: {s.get('reciprocity', 'N/A')}
- Question rate: {s.get('question_rate', 'N/A')}
- Emotion entropy: {s.get('emotion_entropy', 'N/A')}
"""

        scientific_text = ""
        if scientific_summary:
            stage = scientific_summary.get("group_stage", {})
            scientific_text = f"""
Scientific analysis (algorithmic):
- Group stage: {stage.get('stage', 'N/A')} (confidence: {stage.get('confidence', 'N/A')})
- Network density: {scientific_summary.get('network_density', 'N/A')}
- Dominant emotion: {scientific_summary.get('dominant_emotion', 'N/A')}
"""

        return f"""You are a scientific observer analyzing a multi-agent dialogue simulation.
You are NOT a participant — you are an independent qualitative researcher applying
methodological triangulation (Denzin, 2012) to validate computational metrics.

Your task: analyze the dialogue below and produce a structured qualitative assessment.
Compare your observations with the algorithmic metrics provided.

DIALOGUE TRANSCRIPT:
{dialogue_text}

{metrics_text}
{scientific_text}

Respond with a JSON object containing these fields:
- "group_dynamics": (string) Describe the current group dynamics — who leads, who follows, who is marginalized. What phase is the group in?
- "leadership_pattern": (string) Is leadership distributed or centralized? Who exhibits implicit leadership? How is influence exerted?
- "coalition_analysis": (string) Are there visible or hidden coalitions/alliances? Who tends to agree with whom? Are there subgroups?
- "communication_barriers": (string) What communication problems do you observe? Misunderstandings, talking past each other, unaddressed points?
- "emotional_undercurrent": (string) What is the emotional atmosphere beyond surface tone? Hidden tensions, unspoken agreements, group mood shifts?
- "agreements": (list of strings) Where do YOUR qualitative observations AGREE with the algorithmic metrics?
- "divergences": (list of strings) Where do YOUR observations DIVERGE from or ADD to the algorithmic metrics? What do the algorithms miss?
- "novel_insights": (list of strings) Insights that purely computational methods cannot capture — implicit social dynamics, pragmatic meaning, etc.
- "convergence_score": (float 0-1) Overall agreement between your analysis and algorithmic metrics. 1.0 = full agreement.

Be specific, cite actual utterances, and be honest about disagreements with the metrics.
Respond with ONLY a valid JSON object."""

    def observe(
        self,
        dialogue_history: List[Dict[str, Any]],
        turn_no: int,
        metrics_summary: Optional[Dict[str, Any]] = None,
        scientific_summary: Optional[Dict[str, Any]] = None,
    ) -> ObservationReport:
        """
        Perform one observation cycle. Calls LLM to produce qualitative analysis.

        Args:
            dialogue_history: Full dialogue history
            turn_no: Current turn number
            metrics_summary: Output of compute_metrics() if available
            scientific_summary: Output of compute_scientific_analysis() as dict

        Returns:
            ObservationReport with qualitative analysis and triangulation data
        """
        from .config import MODEL, TEMPERATURE, MAX_TOKENS

        prompt = self._build_observer_prompt(
            dialogue_history, metrics_summary, scientific_summary
        )

        report = ObservationReport(turn=turn_no)

        try:
            resp = self.client.chat.completions.create(
                model=self.model or MODEL,
                temperature=max(0.3, TEMPERATURE - 0.3),  # lower temp for analysis
                max_tokens=MAX_TOKENS + 400,  # need more tokens for analysis
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "Analyze the dialogue and produce your observation report as JSON."},
                ],
            )
            raw = resp.choices[0].message.content or ""
            report.raw_llm_response = raw

            # Parse JSON from response
            m = re.search(r"\{.*\}", raw, re.S)
            if m:
                data = json.loads(m.group(0))
                report.group_dynamics = data.get("group_dynamics", "")
                report.leadership_pattern = data.get("leadership_pattern", "")
                report.coalition_analysis = data.get("coalition_analysis", "")
                report.communication_barriers = data.get("communication_barriers", "")
                report.emotional_undercurrent = data.get("emotional_undercurrent", "")
                report.agreements = data.get("agreements", [])
                report.divergences = data.get("divergences", [])
                report.novel_insights = data.get("novel_insights", [])
                report.convergence_score = float(data.get("convergence_score", 0.0))
        except Exception as e:
            import sys
            print(f"[ObserverAgent] Error: {e}", file=sys.stderr)
            report.group_dynamics = f"Observation failed: {str(e)}"

        self.history.append(report)
        return report

    def render_report(self, report: ObservationReport) -> str:
        """Render observation report as markdown."""
        lines = [
            f"# 🔭 Observer Report — Turn {report.turn}",
            "",
            "## Methodological Triangulation",
            f"*Convergence score: {report.convergence_score:.0%}*",
            "",
            "---",
            "",
            "## 1. Group Dynamics",
            report.group_dynamics or "N/A",
            "",
            "## 2. Leadership Pattern",
            report.leadership_pattern or "N/A",
            "",
            "## 3. Coalition Analysis",
            report.coalition_analysis or "N/A",
            "",
            "## 4. Communication Barriers",
            report.communication_barriers or "N/A",
            "",
            "## 5. Emotional Undercurrent",
            report.emotional_undercurrent or "N/A",
            "",
            "---",
            "",
            "## Triangulation Results",
            "",
            "### ✅ Agreements with Algorithmic Metrics",
        ]
        for a in report.agreements:
            lines.append(f"- {a}")
        if not report.agreements:
            lines.append("- No specific agreements noted")

        lines.extend([
            "",
            "### ⚠️ Divergences from Algorithmic Metrics",
        ])
        for d in report.divergences:
            lines.append(f"- {d}")
        if not report.divergences:
            lines.append("- No divergences noted")

        lines.extend([
            "",
            "### 💡 Novel Insights (beyond computational methods)",
        ])
        for n in report.novel_insights:
            lines.append(f"- {n}")
        if not report.novel_insights:
            lines.append("- No novel insights")

        lines.extend([
            "",
            "---",
            "*Observer Agent analysis based on methodological triangulation*",
            "*Reference: Denzin, N. K. (2012). Triangulation 2.0. JMMR, 6(2), 80-88.*",
        ])

        return "\n".join(lines)

    def get_triangulation_summary(self) -> Dict[str, Any]:
        """Get summary of all observations for longitudinal analysis."""
        if not self.history:
            return {"observations": 0}
        scores = [r.convergence_score for r in self.history]
        all_agreements = sum(len(r.agreements) for r in self.history)
        all_divergences = sum(len(r.divergences) for r in self.history)
        all_novel = sum(len(r.novel_insights) for r in self.history)
        return {
            "observations": len(self.history),
            "avg_convergence": sum(scores) / len(scores),
            "min_convergence": min(scores),
            "max_convergence": max(scores),
            "total_agreements": all_agreements,
            "total_divergences": all_divergences,
            "total_novel_insights": all_novel,
            "convergence_trend": scores,
        }
