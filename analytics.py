"""Analytics for agent dialogue.

Computes windowed and global metrics, renders markdown reports, and generates
hypotheses based on both heuristic and modern scientific frameworks (2010-2025).

Scientific foundations integrated from scientific_analytics.py:
- Multidimensional Networks in Teams (Contractor et al., 2012)
- Network Dynamics & Team Performance (Reagans et al., 2016; Leenders et al., 2016)
- ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2020)
- Computational Dialogue Analysis (Jurafsky & Martin, 2024)
- Language Style Matching & LIWC (Tausczik & Pennebaker, 2010; Gonzales et al., 2010)
- High-Dimensional Emotion Model (Cowen & Keltner, 2017; Demszky et al., 2020)
- Team Temporal Dynamics (Shuffler et al., 2018; Mathieu et al., 2017)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
from collections import Counter, defaultdict
from datetime import datetime

from .scientific_analytics import (
    compute_scientific_analysis,
    generate_scientific_hypotheses,
    render_scientific_report,
    ScientificAnalysisResult,
)
from typing import Optional
from .hypothesis_validator import HypothesisValidator, HypothesisValidation

# tone palette
TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
ACTION_KEYWORDS = [
    "plan", "schedule", "deliver", "ship", "prototype", "test",
    "next step", "deadline", "assign", "owner", "task", "todo",
]

def _window(history: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return history[-n:] if n > 0 else history[:]

def _words(s: str) -> int:
    return len((s or "").strip().split())

def _unique_emotions(items: List[str]) -> int:
    return len({(e or "neutral").lower() for e in items})

def _entropy(items: List[str]) -> float:
    # Shannon entropy for emotion diversity
    if not items:
        return 0.0
    c = Counter([ (e or "neutral").lower() for e in items ])
    total = sum(c.values())
    ent = 0.0
    for v in c.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent

def _addressing_delay(history: List[Dict[str, Any]]) -> float:
    """Average turns-to-first-response for addressed messages.

    If a message targets A, measure how many turns until A replies (first reply).
    """
    last_addr = defaultdict(list)  # agent -> turns when addressed
    delays = []
    for rec in history:
        t = rec["turn"]
        speaker = rec["speaker"]
        target = rec.get("target")
        # если спикер сейчас ответил, посчитаем задержку к ближайшей адресации ему
        if last_addr.get(speaker):
            # сколько ходов прошло с последней адресации этому спикеру
            last_t = last_addr[speaker].pop(0)
            delays.append(t - last_t)
        if target:
            last_addr[target].append(t)
    return sum(delays)/len(delays) if delays else 0.0

def compute_metrics(history: List[Dict[str, Any]], agents: List[str], window_size: int) -> Dict[str, Any]:
    win = _window(history, window_size)
    all_msgs = len(history)
    win_msgs = len(win)
    if all_msgs == 0:
        return {"summary": {}, "per_agent": {}, "edges": {}, "window_size": window_size}

    # aggregates over history
    def _aggregate(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        talk = Counter([r["speaker"] for r in sub])
        addr_ok = sum(1 for r in sub if r.get("target"))
        tone_vals = [TONE_SCORE.get((r.get("tone") or "neutral").lower(), 0.0) for r in sub]
        pos_frac = sum(1 for r in sub if (r.get("tone") or "neutral") == "positive") / max(1, len(sub))
        neg_frac = sum(1 for r in sub if (r.get("tone") or "neutral") == "negative") / max(1, len(sub))
        neu_frac = 1.0 - pos_frac - neg_frac
        emotions = [ (r.get("emotion") or "neutral") for r in sub ]
        avg_len = sum(_words(r.get("reply") or "") for r in sub) / max(1, len(sub))
        targets = [ (r["speaker"], r.get("target")) for r in sub if r.get("target") ]
        out_deg = Counter([ s for (s, _) in targets ])
        in_deg  = Counter([ t for (_, t) in targets ])
        reciprocity = _estimate_reciprocity(sub)
        question_ratio = sum(1 for r in sub if "?" in (r.get("reply") or "")) / max(1, len(sub))
        lower_agents = [a.lower() for a in agents]
        reference_hits = 0
        action_hits = 0
        for r in sub:
            reply = (r.get("reply") or "").lower()
            speaker_lower = (r.get("speaker") or "").lower()
            if any(name != speaker_lower and name in reply for name in lower_agents):
                reference_hits += 1
            if any(keyword in reply for keyword in ACTION_KEYWORDS):
                action_hits += 1
        return {
            "messages": len(sub),
            "talk_share": {a: talk.get(a, 0)/max(1, len(sub)) for a in agents},
            "targeting_rate": addr_ok / max(1, len(sub)),
            "avg_tone": sum(tone_vals)/max(1, len(tone_vals)),
            "tone_fracs": {"pos": pos_frac, "neu": neu_frac, "neg": neg_frac},
            "emotion_diversity": _unique_emotions(emotions),
            "emotion_entropy": _entropy(emotions),
            "avg_reply_words": avg_len,
            "out_degree": dict(out_deg),
            "in_degree": dict(in_deg),
            "reciprocity": reciprocity,
            "avg_addressing_delay": _addressing_delay(sub),
            "question_rate": question_ratio,
            "reference_rate": reference_hits / max(1, len(sub)),
            "actionability_rate": action_hits / max(1, len(sub)),
        }

    def _per_agent(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        for a in agents:
            msgs = [r for r in sub if r["speaker"] == a]
            res[a] = {
                "msgs": len(msgs),
                "avg_words": sum(_words(r.get("reply") or "") for r in msgs)/max(1, len(msgs)),
                "avg_tone": sum(TONE_SCORE.get((r.get("tone") or "neutral").lower(), 0.0) for r in msgs)/max(1, len(msgs)),
                "last_emotion": (msgs[-1].get("emotion") if msgs else "neutral"),
                "targets_diversity": len({r.get("target") for r in msgs if r.get("target")}),
            }
        return res

    summary_all = _aggregate(history)
    summary_win = _aggregate(win)
    per_agent_win = _per_agent(win)
    
    # Compute scientific analysis
    scientific = compute_scientific_analysis(history, agents, window_size)

    return {
        "window_size": window_size,
        "summary": {"all": summary_all, "window": summary_win},
        "per_agent": per_agent_win,
        "scientific": scientific,
    }

def _estimate_reciprocity(sub: List[Dict[str, Any]]) -> float:
    """
    Доля «взаимных» обменов: (A->B) за окном сопровождалось хотя бы одним (B->A).
    """
    pairs = [(r["speaker"], r.get("target")) for r in sub if r.get("target")]
    if not pairs:
        return 0.0
    set_pairs = set(pairs)
    mutual = sum(1 for (u, v) in set_pairs if (v, u) in set_pairs)
    return mutual / max(1, len(set_pairs))

def hypotheses_from_metrics(metrics: Dict[str, Any], user_name: str | None = "User") -> List[str]:
    """
    Generate hypotheses combining heuristic rules and scientific frameworks.
    
    Returns list of hypothesis strings for display.
    """
    s = metrics["summary"]["window"]
    hyp = []
    
    # === HEURISTIC HYPOTHESES ===
    # dominance
    talk = s["talk_share"]
    top_speaker, top_share = max(talk.items(), key=lambda kv: kv[1]) if talk else ("-", 0)
    if top_share > 0.45:
        hyp.append(f"⚠️ Agent '{top_speaker}' dominates (~{int(top_share*100)}%) — check role balance")
    # mood / tone
    if s["avg_tone"] < -0.25:
        hyp.append("⚠️ Negative tone detected — risk of conflict")
    elif s["avg_tone"] > 0.35:
        hyp.append("✅ Positive atmosphere — good for decisions")
    # addressing
    if s["targeting_rate"] < 0.7:
        hyp.append("⚠️ Low addressing (<70%) — possible miscommunication")
    # delays
    if s["avg_addressing_delay"] > 2.5:
        hyp.append("⚠️ High response delay (>2.5 turns)")
    # questions
    if s.get("question_rate", 0) < 0.15:
        hyp.append("💡 Few questions (<15%) — encourage exploration")
    # attention to user
    if user_name and s["in_degree"].get(user_name, 0) == 0:
        hyp.append(f"💡 '{user_name}' hasn't been addressed")
    
    # === SCIENTIFIC HYPOTHESES ===
    scientific = metrics.get("scientific")
    if scientific:
        sci_hyps = generate_scientific_hypotheses(scientific, list(talk.keys()))
        for sh in sci_hyps[:3]:  # Top 3 scientific hypotheses
            finding = sh.get("finding", "")
            framework = sh.get("framework", "")
            if finding:
                hyp.append(f"📚 [{framework}] {finding}")

    return hyp or ["✅ Communication is stable — no significant deviations"]


def get_scientific_hypotheses_full(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get full scientific hypotheses with references for detailed display.
    """
    scientific = metrics.get("scientific")
    if not scientific:
        return []
    
    agents = list(metrics["summary"]["window"]["talk_share"].keys())
    return generate_scientific_hypotheses(scientific, agents)

def render_markdown_report(metrics: Dict[str, Any], turn: int, title: str) -> str:
    s_all = metrics["summary"]["all"]
    s_win = metrics["summary"]["window"]
    pa = metrics["per_agent"]
    scientific = metrics.get("scientific")
    ts = datetime.utcnow().strftime("%H:%M:%S UTC")

    def pct(x): return f"{x*100:.0f}%"
    def tone_emoji(x):
        return "🟢" if x > 0.2 else ("🟡" if abs(x) <= 0.2 else "🔴")

    # top block
    md = []
    md.append(f"# 📈 Dialogue Metrics — turn {turn} ({ts})")
    md.append("")
    md.append("| Window messages | Total messages |")
    md.append("|---:|---:|")
    md.append("| {win} | {all_} |".format(win=s_win["messages"], all_=s_all["messages"]))
    md.append("")
    md.append("| Metric | Window | Total |")
    md.append("|---|---:|---:|")
    md.append(f"| Avg tone | {s_win['avg_tone']:+.2f} {tone_emoji(s_win['avg_tone'])} | {s_all['avg_tone']:+.2f} |")
    md.append(f"| Addressing, % | {pct(s_win['targeting_rate'])} | {pct(s_all['targeting_rate'])} |")
    md.append(f"| Emotion entropy | {s_win['emotion_entropy']:.2f} | {s_all['emotion_entropy']:.2f} |")
    md.append(f"| Avg reply words | {s_win['avg_reply_words']:.1f} | {s_all['avg_reply_words']:.1f} |")
    md.append(f"| Reciprocity | {s_win['reciprocity']:.2f} | {s_all['reciprocity']:.2f} |")
    md.append(f"| Response delay (turns) | {s_win['avg_addressing_delay']:.2f} | {s_all['avg_addressing_delay']:.2f} |")
    md.append(f"| Questions, % | {pct(s_win.get('question_rate', 0))} | {pct(s_all.get('question_rate', 0))} |")
    md.append(f"| Named references, % | {pct(s_win.get('reference_rate', 0))} | {pct(s_all.get('reference_rate', 0))} |")
    md.append(f"| Actionable cues, % | {pct(s_win.get('actionability_rate', 0))} | {pct(s_all.get('actionability_rate', 0))} |")
    md.append("")
    
    # === SCIENTIFIC ANALYSIS SECTION (Modern Frameworks 2010-2025) ===
    if scientific:
        md.append("---")
        md.append("## 🔬 Scientific Analysis")
        md.append("")
        
        # Team Temporal Dynamics (Shuffler et al., 2018)
        stage_data = scientific.group_stage
        if stage_data:
            stage = stage_data.get("stage", "unknown").upper()
            confidence = stage_data.get("confidence", 0)
            md.append(f"### Team Phase (Shuffler et al., 2018): **{stage}** ({pct(confidence)} confidence)")
            md.append("")
        
        # Network Analysis (Contractor et al., 2012)
        md.append("### Multidimensional Networks (Contractor et al., 2012)")
        md.append(f"- **Network Density:** {pct(scientific.network_density)}")
        md.append(f"- **Clustering:** {scientific.clustering_coefficient:.2f}")
        md.append("")
        
        # Network Dynamics (Reagans et al., 2016)
        social_cap = scientific.social_capital
        if social_cap:
            md.append(f"### Network Dynamics (Reagans et al., 2016): {pct(social_cap.get('group_cohesion', 0))}")
            md.append("")
        
        # Dialogue Act Profile (ISO 24617-2, Bunt et al., 2020)
        dialogue_acts = scientific.dialogue_act_profile
        if dialogue_acts:
            md.append("### Dialogue Acts (ISO 24617-2, Bunt et al., 2020)")
            md.append(f"- **Task-oriented:** {pct(dialogue_acts.get('task_ratio', 0))}")
            md.append(f"- **Socio-emotional:** {pct(dialogue_acts.get('socio_ratio', 0))}")
            md.append(f"- **Positive/Negative ratio:** {dialogue_acts.get('positive_negative_ratio', 0):.2f}")
            md.append("")
        
        # Emotional Climate (Cowen & Keltner, 2017)
        md.append(f"### Emotional Climate (Cowen & Keltner, 2017): **{scientific.dominant_emotion.title()}**")
        emotions = scientific.emotion_distribution
        if emotions:
            top_emotions = sorted(emotions.items(), key=lambda x: -x[1])[:3]
            emotions_str = ", ".join(f"{e}: {pct(v)}" for e, v in top_emotions if v > 0)
            if emotions_str:
                md.append(f"Top emotions: {emotions_str}")
        md.append("")
        
        # Turn-Taking (Gonzales et al., 2010; Tausczik & Pennebaker, 2010)
        turn_data = scientific.turn_taking
        if turn_data:
            md.append("### Turn-Taking (Gonzales et al., 2010; Tausczik & Pennebaker, 2010)")
            md.append(f"- **Inequality (Gini):** {turn_data.get('turn_inequality_gini', 0):.2f}")
            md.append(f"- **Adjacency completion:** {pct(turn_data.get('adjacency_completion_rate', 0))}")
            md.append("")
    
    md.append("---")
    
    # per-agent stats
    md.append("## Per agent")
    md.append("| Agent | Msgs | Avg words | Avg tone | Last emotion | Target diversity |")
    md.append("|---|---:|---:|---:|---|---:|")
    for a, st in sorted(pa.items(), key=lambda kv: kv[1]["msgs"], reverse=True):
        md.append(f"| {a} | {st['msgs']} | {st['avg_words']:.1f} | {st['avg_tone']:+.2f} | {st['last_emotion']} | {st['targets_diversity']} |")
    md.append("")
    
    # Centrality (if available)
    if scientific and scientific.centrality:
        md.append("## Centrality Analysis")
        md.append("| Agent | In-Centrality | Out-Centrality | Betweenness |")
        md.append("|---|---:|---:|---:|")
        for a in sorted(scientific.centrality.keys()):
            c = scientific.centrality[a]
            b = scientific.betweenness.get(a, 0)
            md.append(f"| {a} | {pct(c.get('in_centrality', 0))} | {pct(c.get('out_centrality', 0))} | {pct(b)} |")
        md.append("")
    
    # speaking shares (as table)
    md.append("## Speaking shares")
    tt = s_win["talk_share"]
    md.append("| Agent | Share |")
    md.append("|---|---:|")
    for a, v in sorted(tt.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"| {a} | {pct(v)} |")
    md.append("")
    return "\n".join(md)

def save_report_md(md_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text, encoding="utf-8")


# ========== HYPOTHESIS VALIDATION ==========

try:
    from .hypothesis_validator import HypothesisValidator, HypothesisValidation
    
    # Глобальный валидатор гипотез
    _hypothesis_validator = HypothesisValidator()
    
    def validate_hypotheses(
        hypotheses: List[Dict[str, Any]],
        current_metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[HypothesisValidation]:
        """
        Проверить гипотезы на основе метрик.
        
        Args:
            hypotheses: Список гипотез из generate_scientific_hypotheses
            current_metrics: Текущие метрики
            previous_metrics: Предыдущие метрики для сравнения
            history: История диалога для извлечения примеров
        
        Returns:
            Список результатов проверки
        """
        _hypothesis_validator.add_metrics_snapshot(
            len(_hypothesis_validator.metrics_history),
            current_metrics
        )
        return _hypothesis_validator.validate_all_hypotheses(
            hypotheses, current_metrics, previous_metrics, history
        )
    
    def get_hypothesis_validation_report(
        hypotheses: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        previous_metrics: Optional[Dict[str, Any]] = None,
        turn: int = 0,
        history: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Получить отчёт о проверке гипотез.
        
        Args:
            hypotheses: Список гипотез
            metrics: Текущие метрики
            previous_metrics: Предыдущие метрики
            turn: Номер хода
            history: История диалога для извлечения примеров
        
        Returns:
            Markdown отчёт
        """
        validations = validate_hypotheses(hypotheses, metrics, previous_metrics, history)
        return _hypothesis_validator.render_validation_report(validations, turn)
    
except ImportError:
    # Если модуль hypothesis_validator не найден, создаём заглушки
    def validate_hypotheses(*args, **kwargs):
        return []
    
    def get_hypothesis_validation_report(*args, **kwargs):
        return "# Проверка гипотез\n\nМодуль проверки гипотез недоступен."
