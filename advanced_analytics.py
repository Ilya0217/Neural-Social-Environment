"""
Advanced Analytics Module — Emotional Contagion, LSM, Discourse Coherence.

Scientific foundations:
1. Emotional Contagion (Barsade, 2002; Hatfield et al., 1994)
2. Language Style Matching (Gonzales et al., 2010; Tausczik & Pennebaker, 2010)
3. Discourse Coherence (Graesser et al., 2014; McNamara et al., 2014)
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# 1. EMOTIONAL CONTAGION TRACKER (Barsade, 2002; Hatfield et al., 1994)
# =============================================================================

TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


@dataclass
class ContagionEvent:
    """A single emotional contagion event."""
    turn: int
    source: str
    target: str
    source_tone: str
    target_tone: str
    lag: int  # turns between source and target adopting similar tone
    contagion_type: str  # "amplification", "dampening", "infection", "resistance"


@dataclass
class EmotionalContagionResult:
    """Full emotional contagion analysis."""
    events: List[ContagionEvent] = field(default_factory=list)
    contagion_rate: float = 0.0  # % of interactions where tone spread
    avg_lag: float = 0.0  # avg turns for emotion to spread
    most_contagious_agent: str = ""
    most_susceptible_agent: str = ""
    tone_trajectories: Dict[str, List[float]] = field(default_factory=dict)
    infection_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)


def compute_emotional_contagion(
    history: List[Dict[str, Any]],
    agents: List[str],
) -> EmotionalContagionResult:
    """
    Track how emotions spread through the agent network.

    Reference: Barsade, S. G. (2002). The ripple effect: Emotional contagion
    and its influence on group behavior. Administrative Science Quarterly, 47(4), 644-675.
    """
    if len(history) < 3:
        return EmotionalContagionResult()

    # Build tone trajectories per agent
    trajectories: Dict[str, List[float]] = {a: [] for a in agents}
    for h in history:
        speaker = h.get("speaker", "")
        tone = TONE_SCORE.get(h.get("tone", "neutral"), 0.0)
        if speaker in trajectories:
            trajectories[speaker].append(tone)

    # Detect contagion events: when agent B responds to A and adopts A's tone
    events: List[ContagionEvent] = []
    infection_counts: Dict[str, int] = defaultdict(int)  # how many times agent infected others
    susceptible_counts: Dict[str, int] = defaultdict(int)  # how many times agent got infected

    for i in range(1, len(history)):
        curr = history[i]
        curr_speaker = curr.get("speaker", "")
        curr_tone = curr.get("tone", "neutral")
        curr_target = curr.get("target")

        # Look back for who this agent is responding to
        for j in range(max(0, i - 5), i):
            prev = history[j]
            prev_speaker = prev.get("speaker", "")
            prev_tone = prev.get("tone", "neutral")

            if prev_speaker == curr_speaker:
                continue
            if curr_target and curr_target != prev_speaker:
                continue

            lag = i - j

            # Determine contagion type
            if curr_tone == prev_tone and prev_tone != "neutral":
                ctype = "infection"
                infection_counts[prev_speaker] += 1
                susceptible_counts[curr_speaker] += 1
                events.append(ContagionEvent(
                    turn=i, source=prev_speaker, target=curr_speaker,
                    source_tone=prev_tone, target_tone=curr_tone,
                    lag=lag, contagion_type=ctype,
                ))
            elif prev_tone == "negative" and curr_tone == "positive":
                events.append(ContagionEvent(
                    turn=i, source=prev_speaker, target=curr_speaker,
                    source_tone=prev_tone, target_tone=curr_tone,
                    lag=lag, contagion_type="resistance",
                ))
            break  # only match closest prior speaker

    # Compute infection matrix (pairwise contagion rates)
    pair_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    pair_match: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(history)):
        curr = history[i]
        for j in range(max(0, i - 3), i):
            prev = history[j]
            if prev["speaker"] != curr["speaker"]:
                pair_total[prev["speaker"]][curr["speaker"]] += 1
                if prev.get("tone") == curr.get("tone") and prev.get("tone") != "neutral":
                    pair_match[prev["speaker"]][curr["speaker"]] += 1
                break

    infection_matrix: Dict[str, Dict[str, float]] = {}
    for src in agents:
        infection_matrix[src] = {}
        for tgt in agents:
            if src == tgt:
                infection_matrix[src][tgt] = 0.0
            else:
                total = pair_total[src][tgt]
                infection_matrix[src][tgt] = pair_match[src][tgt] / total if total > 0 else 0.0

    contagion_rate = len([e for e in events if e.contagion_type == "infection"]) / max(1, len(history) - 1)
    avg_lag = (sum(e.lag for e in events) / len(events)) if events else 0.0
    most_contagious = max(infection_counts, key=infection_counts.get, default="") if infection_counts else ""
    most_susceptible = max(susceptible_counts, key=susceptible_counts.get, default="") if susceptible_counts else ""

    return EmotionalContagionResult(
        events=events,
        contagion_rate=contagion_rate,
        avg_lag=avg_lag,
        most_contagious_agent=most_contagious,
        most_susceptible_agent=most_susceptible,
        tone_trajectories=trajectories,
        infection_matrix=infection_matrix,
    )


# =============================================================================
# 2. LANGUAGE STYLE MATCHING (Gonzales et al., 2010; Tausczik & Pennebaker, 2010)
# =============================================================================

# Function word categories (simplified LIWC-like categories)
FUNCTION_WORD_CATEGORIES = {
    "pronouns": {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
                 "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
                 "it", "its", "they", "them", "their", "theirs"},
    "articles": {"a", "an", "the"},
    "prepositions": {"in", "on", "at", "to", "for", "with", "from", "by", "about",
                     "into", "through", "during", "before", "after", "above", "below",
                     "between", "under", "over", "of"},
    "auxiliaries": {"is", "am", "are", "was", "were", "be", "been", "being",
                    "have", "has", "had", "do", "does", "did", "will", "would",
                    "could", "should", "may", "might", "can", "shall", "must"},
    "negations": {"not", "no", "never", "neither", "nor", "nobody", "nothing",
                  "nowhere", "none", "don't", "doesn't", "didn't", "won't",
                  "can't", "couldn't", "shouldn't", "wouldn't", "isn't", "aren't"},
    "conjunctions": {"and", "but", "or", "so", "yet", "because", "although",
                     "while", "if", "when", "since", "unless", "though"},
    "quantifiers": {"all", "some", "any", "every", "each", "few", "many", "much",
                    "most", "several", "both", "more", "less", "enough"},
    "fillers": {"like", "well", "actually", "basically", "honestly", "literally",
                "obviously", "definitely", "probably", "maybe", "perhaps",
                "anyway", "right", "okay", "ok", "yeah", "hmm", "oh", "ah"},
}


@dataclass
class LSMResult:
    """Language Style Matching analysis result."""
    pairwise_lsm: Dict[str, Dict[str, float]] = field(default_factory=dict)
    group_lsm: float = 0.0
    category_scores: Dict[str, float] = field(default_factory=dict)
    agent_profiles: Dict[str, Dict[str, float]] = field(default_factory=dict)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z']+", text.lower())


def _compute_category_rates(tokens: List[str]) -> Dict[str, float]:
    """Compute usage rate for each function word category."""
    total = max(1, len(tokens))
    rates = {}
    for cat, words in FUNCTION_WORD_CATEGORIES.items():
        count = sum(1 for t in tokens if t in words)
        rates[cat] = count / total
    return rates


def _lsm_score(rates_a: Dict[str, float], rates_b: Dict[str, float]) -> float:
    """
    Compute LSM between two speakers across all categories.
    LSM = 1 - |rate_a - rate_b| / (rate_a + rate_b + 0.0001)
    Averaged across all categories.

    Reference: Ireland, M. E., & Pennebaker, J. W. (2010). Language style matching
    in writing. Psychological Science, 21(10), 1547-1553.
    """
    scores = []
    for cat in FUNCTION_WORD_CATEGORIES:
        a = rates_a.get(cat, 0)
        b = rates_b.get(cat, 0)
        denom = a + b + 0.0001
        scores.append(1 - abs(a - b) / denom)
    return sum(scores) / len(scores) if scores else 0.0


def compute_language_style_matching(
    history: List[Dict[str, Any]],
    agents: List[str],
) -> LSMResult:
    """
    Compute Language Style Matching (LSM) for all agent pairs.

    LSM measures similarity in function word usage — a predictor of group cohesion.

    Reference: Gonzales, A. L., Hancock, J. T., & Pennebaker, J. W. (2010).
    Language style matching as a predictor of social dynamics in small groups.
    Communication Research, 37(1), 3-19.
    """
    if len(history) < 3:
        return LSMResult()

    # Collect all text per agent
    agent_texts: Dict[str, List[str]] = defaultdict(list)
    for h in history:
        speaker = h.get("speaker", "")
        reply = h.get("reply", "")
        if speaker in agents:
            agent_texts[speaker].append(reply)

    # Compute per-agent function word profiles
    agent_profiles: Dict[str, Dict[str, float]] = {}
    for agent in agents:
        all_tokens = []
        for text in agent_texts.get(agent, []):
            all_tokens.extend(_tokenize(text))
        agent_profiles[agent] = _compute_category_rates(all_tokens)

    # Pairwise LSM
    pairwise: Dict[str, Dict[str, float]] = {}
    all_scores = []
    for a in agents:
        pairwise[a] = {}
        for b in agents:
            if a == b:
                pairwise[a][b] = 1.0
            else:
                score = _lsm_score(agent_profiles[a], agent_profiles[b])
                pairwise[a][b] = round(score, 3)
                all_scores.append(score)

    group_lsm = sum(all_scores) / len(all_scores) if all_scores else 0.0

    # Per-category group average
    category_scores = {}
    for cat in FUNCTION_WORD_CATEGORIES:
        cat_scores = []
        for i, a in enumerate(agents):
            for b in agents[i + 1:]:
                ra = agent_profiles[a].get(cat, 0)
                rb = agent_profiles[b].get(cat, 0)
                cat_scores.append(1 - abs(ra - rb) / (ra + rb + 0.0001))
        category_scores[cat] = sum(cat_scores) / len(cat_scores) if cat_scores else 0.0

    return LSMResult(
        pairwise_lsm=pairwise,
        group_lsm=round(group_lsm, 3),
        category_scores={k: round(v, 3) for k, v in category_scores.items()},
        agent_profiles={k: {c: round(r, 4) for c, r in v.items()} for k, v in agent_profiles.items()},
    )


# =============================================================================
# 3. DISCOURSE COHERENCE ANALYSIS (Graesser et al., 2014)
# =============================================================================

def _word_set(text: str) -> set:
    """Extract unique content words (excluding very common function words)."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "must", "i", "you", "he",
            "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "this", "that", "these", "those",
            "in", "on", "at", "to", "for", "with", "from", "by", "of", "and",
            "but", "or", "not", "no", "so", "if", "as"}
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in stop and len(t) > 2}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


@dataclass
class CoherenceResult:
    """Discourse coherence analysis result."""
    overall_coherence: float = 0.0  # avg adjacent-pair similarity
    coherence_trajectory: List[float] = field(default_factory=list)
    topic_drift_points: List[int] = field(default_factory=list)  # turns where coherence drops sharply
    agent_coherence: Dict[str, float] = field(default_factory=dict)  # how coherent each agent is with prior context
    thread_continuity: float = 0.0  # % of turns that reference prior speaker's content


def compute_discourse_coherence(
    history: List[Dict[str, Any]],
    agents: List[str],
) -> CoherenceResult:
    """
    Measure discourse coherence using lexical overlap between adjacent turns.

    High coherence = agents are building on each other's ideas.
    Low coherence = agents are talking past each other.

    Reference: Graesser, A. C., McNamara, D. S., Cai, Z., Conley, M., Li, H., & Pennebaker, J. (2014).
    Coh-Metrix measures text characteristics at multiple levels of language and discourse.
    Elementary School Journal, 115(2), 210-229.
    """
    if len(history) < 3:
        return CoherenceResult()

    # Compute adjacent-pair coherence
    coherence_scores: List[float] = []
    for i in range(1, len(history)):
        words_prev = _word_set(history[i - 1].get("reply", ""))
        words_curr = _word_set(history[i].get("reply", ""))
        sim = _jaccard_similarity(words_prev, words_curr)
        coherence_scores.append(sim)

    # Detect topic drift: where coherence drops below mean - 1.5*std
    mean_c = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    std_c = math.sqrt(sum((x - mean_c) ** 2 for x in coherence_scores) / max(1, len(coherence_scores)))
    threshold = max(0, mean_c - 1.5 * std_c)
    drift_points = [i + 1 for i, c in enumerate(coherence_scores) if c < threshold]

    # Per-agent coherence: how well each agent connects to prior context
    agent_scores: Dict[str, List[float]] = defaultdict(list)
    for i in range(1, len(history)):
        speaker = history[i].get("speaker", "")
        words_curr = _word_set(history[i].get("reply", ""))
        # Compare with last 3 turns (context window)
        context_words = set()
        for j in range(max(0, i - 3), i):
            context_words |= _word_set(history[j].get("reply", ""))
        if context_words and speaker in agents:
            agent_scores[speaker].append(_jaccard_similarity(context_words, words_curr))

    agent_coherence = {
        a: round(sum(scores) / len(scores), 3) if scores else 0.0
        for a, scores in agent_scores.items()
    }

    # Thread continuity: % of turns where agent uses words from the person they're responding to
    thread_hits = 0
    thread_total = 0
    for i in range(1, len(history)):
        target = history[i].get("target")
        if not target:
            continue
        # Find last message from target
        for j in range(i - 1, max(0, i - 6), -1):
            if history[j].get("speaker") == target:
                words_ref = _word_set(history[j].get("reply", ""))
                words_curr = _word_set(history[i].get("reply", ""))
                thread_total += 1
                if words_ref & words_curr:
                    thread_hits += 1
                break

    return CoherenceResult(
        overall_coherence=round(mean_c, 3),
        coherence_trajectory=[round(c, 3) for c in coherence_scores],
        topic_drift_points=drift_points,
        agent_coherence=agent_coherence,
        thread_continuity=round(thread_hits / max(1, thread_total), 3),
    )


# =============================================================================
# COMBINED ADVANCED ANALYSIS
# =============================================================================

@dataclass
class AdvancedAnalysisResult:
    """Container for all advanced analytics."""
    emotional_contagion: EmotionalContagionResult = field(default_factory=EmotionalContagionResult)
    language_style_matching: LSMResult = field(default_factory=LSMResult)
    discourse_coherence: CoherenceResult = field(default_factory=CoherenceResult)


def compute_advanced_analysis(
    history: List[Dict[str, Any]],
    agents: List[str],
) -> AdvancedAnalysisResult:
    """Run all advanced analytics."""
    return AdvancedAnalysisResult(
        emotional_contagion=compute_emotional_contagion(history, agents),
        language_style_matching=compute_language_style_matching(history, agents),
        discourse_coherence=compute_discourse_coherence(history, agents),
    )


def render_advanced_report(result: AdvancedAnalysisResult, agents: List[str]) -> str:
    """Render advanced analytics as markdown."""
    lines = ["# Advanced Analytics Report", ""]

    # --- Emotional Contagion ---
    ec = result.emotional_contagion
    lines.extend([
        "## 🔥 Emotional Contagion (Barsade, 2002)",
        "",
        f"- **Contagion rate:** {ec.contagion_rate:.0%} of interactions",
        f"- **Average lag:** {ec.avg_lag:.1f} turns",
        f"- **Most contagious agent:** {ec.most_contagious_agent or 'N/A'}",
        f"- **Most susceptible agent:** {ec.most_susceptible_agent or 'N/A'}",
        "",
    ])
    if ec.infection_matrix:
        lines.append("**Infection Matrix** (probability of tone spreading A→B):")
        lines.append("")
        header = "| | " + " | ".join(agents) + " |"
        sep = "|---|" + "|".join(["---:"] * len(agents)) + "|"
        lines.extend([header, sep])
        for src in agents:
            row = f"| **{src}** |"
            for tgt in agents:
                val = ec.infection_matrix.get(src, {}).get(tgt, 0)
                row += f" {val:.0%} |" if src != tgt else " — |"
            lines.append(row)
        lines.append("")

    # --- LSM ---
    lsm = result.language_style_matching
    lines.extend([
        "---",
        "",
        "## 🔗 Language Style Matching (Gonzales et al., 2010)",
        "",
        f"- **Group LSM:** {lsm.group_lsm:.3f} (>0.8 = high cohesion)",
        "",
    ])
    if lsm.pairwise_lsm:
        lines.append("**Pairwise LSM:**")
        lines.append("")
        header = "| | " + " | ".join(agents) + " |"
        sep = "|---|" + "|".join(["---:"] * len(agents)) + "|"
        lines.extend([header, sep])
        for a in agents:
            row = f"| **{a}** |"
            for b in agents:
                val = lsm.pairwise_lsm.get(a, {}).get(b, 0)
                row += f" {val:.3f} |"
            lines.append(row)
        lines.append("")
    if lsm.category_scores:
        lines.append("**Category-level matching:**")
        for cat, score in sorted(lsm.category_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            lines.append(f"- {cat}: {bar} {score:.3f}")
        lines.append("")

    # --- Discourse Coherence ---
    dc = result.discourse_coherence
    lines.extend([
        "---",
        "",
        "## 🧩 Discourse Coherence (Graesser et al., 2014)",
        "",
        f"- **Overall coherence:** {dc.overall_coherence:.3f}",
        f"- **Thread continuity:** {dc.thread_continuity:.0%}",
        f"- **Topic drift points:** {dc.topic_drift_points if dc.topic_drift_points else 'None detected'}",
        "",
    ])
    if dc.agent_coherence:
        lines.append("**Per-agent coherence** (how well each connects to prior context):")
        for agent, score in sorted(dc.agent_coherence.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
            lines.append(f"- {agent}: {bar} {score:.3f}")
        lines.append("")

    lines.extend([
        "---",
        "*Advanced analytics based on peer-reviewed frameworks.*",
    ])
    return "\n".join(lines)
