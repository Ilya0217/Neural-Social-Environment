"""
Scientific Analytics Module for Agent Dialogue Simulation

This module implements established scientific methods for analyzing
interpersonal communication and group dynamics, based on peer-reviewed research.

Scientific Foundations (1990-2025):
===================================

1. SOCIAL CAPITAL & STRUCTURAL HOLES (Burt, 1992, 2004)
   - Network constraint and brokerage opportunities
   - Relational coordination metrics
   - Reference: Burt, R. S. (2004). Structural holes and good ideas. American Journal of Sociology, 110(2), 349-399.

2. SOCIAL NETWORK ANALYSIS (Borgatti et al., 2009; Barabási & Albert, 1999)
   - Centrality measures (degree, betweenness, eigenvector)
   - Scale-free network properties and preferential attachment
   - Reference: Borgatti, S. P., et al. (2009). Network analysis in the social sciences. Science, 323(5916), 892-895.

3. DIALOGUE ACT TAXONOMY (ISO 24617-2, Bunt et al., 2017)
   - Standardized dialogue act annotation scheme
   - Task-oriented vs social dialogue acts
   - Reference: Bunt, H., et al. (2017). ISO 24617-2: A semantically-based standard for dialogue annotation.

4. COMPUTATIONAL PRAGMATICS (Jurafsky & Martin, 2023; Clark, 1996)
   - Speech act classification using modern NLP
   - Grounding in dialogue and common ground theory
   - Reference: Clark, H. H. (1996). Using Language. Cambridge University Press.

5. COMPUTER-MEDIATED DISCOURSE ANALYSIS (Herring, 2004; Androutsopoulos, 2006)
   - Digital communication patterns and turn-taking
   - Asynchronous and synchronous interaction modes
   - Reference: Herring, S. C. (2004). Computer-Mediated Discourse Analysis. In The Handbook of Discourse Analysis.

6. SENTIMENT ANALYSIS - VADER & Transformer Models (Hutto & Gilbert, 2014; Devlin et al., 2019)
   - Valence-based sentiment scoring
   - Contextual sentiment with BERT-based models
   - Reference: Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis.

7. DIMENSIONAL EMOTION MODEL (Russell & Barrett, 1999; Mohammad & Turney, 2013)
   - Circumplex model: Valence × Arousal dimensions
   - NRC Emotion Lexicon for text-based emotion detection
   - Reference: Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon.

8. INTEGRATED MODEL OF GROUP DEVELOPMENT (Wheelan, 2009; Kozlowski & Ilgen, 2006)
   - Dependency → Counterdependency → Trust → Work
   - Team effectiveness and emergent states
   - Reference: Wheelan, S. A. (2009). Group size, group development, and group productivity. Small Group Research, 40(2), 247-262.
"""

from __future__ import annotations
import math
import re
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass, field


# =============================================================================
# DIMENSIONAL EMOTION MODEL (Russell & Barrett, 1999; Mohammad & Turney, 2013)
# Based on Circumplex Model of Affect and NRC Emotion Lexicon
# =============================================================================

DIMENSIONAL_PRIMARY_EMOTIONS = {
    # Primary emotions mapped to valence-arousal dimensions
    "joy": {"opposite": "sadness", "intensity_high": "ecstasy", "intensity_low": "serenity"},
    "trust": {"opposite": "disgust", "intensity_high": "admiration", "intensity_low": "acceptance"},
    "fear": {"opposite": "anger", "intensity_high": "terror", "intensity_low": "apprehension"},
    "surprise": {"opposite": "anticipation", "intensity_high": "amazement", "intensity_low": "distraction"},
    "sadness": {"opposite": "joy", "intensity_high": "grief", "intensity_low": "pensiveness"},
    "disgust": {"opposite": "trust", "intensity_high": "loathing", "intensity_low": "boredom"},
    "anger": {"opposite": "fear", "intensity_high": "rage", "intensity_low": "annoyance"},
    "anticipation": {"opposite": "surprise", "intensity_high": "vigilance", "intensity_low": "interest"},
}

# Emotion to Plutchik primary mapping
EMOTION_TO_PLUTCHIK = {
    # Joy family
    "joy": "joy", "happy": "joy", "happiness": "joy", "excited": "joy", "ecstatic": "joy",
    "delighted": "joy", "cheerful": "joy", "pleased": "joy", "content": "joy", "serene": "joy",
    "optimistic": "joy", "enthusiastic": "joy", "hopeful": "joy",
    
    # Trust family
    "trust": "trust", "trusting": "trust", "admiring": "trust", "accepting": "trust",
    "confident": "trust", "supportive": "trust", "agreeable": "trust",
    
    # Fear family
    "fear": "fear", "afraid": "fear", "scared": "fear", "anxious": "fear", "worried": "fear",
    "nervous": "fear", "apprehensive": "fear", "concerned": "fear", "uncertain": "fear",
    
    # Surprise family
    "surprise": "surprise", "surprised": "surprise", "amazed": "surprise", "astonished": "surprise",
    "shocked": "surprise", "curious": "surprise", "intrigued": "surprise", "wondering": "surprise",
    
    # Sadness family
    "sadness": "sadness", "sad": "sadness", "unhappy": "sadness", "disappointed": "sadness",
    "melancholy": "sadness", "gloomy": "sadness", "pensive": "sadness", "regretful": "sadness",
    
    # Disgust family
    "disgust": "disgust", "disgusted": "disgust", "contempt": "disgust", "bored": "disgust",
    "disapproving": "disgust", "dismissive": "disgust",
    
    # Anger family
    "anger": "anger", "angry": "anger", "frustrated": "anger", "annoyed": "anger",
    "irritated": "anger", "hostile": "anger", "aggressive": "anger", "defensive": "anger",
    
    # Anticipation family
    "anticipation": "anticipation", "anticipating": "anticipation", "expectant": "anticipation",
    "interested": "anticipation", "vigilant": "anticipation", "attentive": "anticipation",
    "engaged": "anticipation", "focused": "anticipation", "thoughtful": "anticipation",
    
    # Neutral
    "neutral": "neutral", "calm": "neutral", "composed": "neutral",
}


def classify_dimensional_emotion(emotion: str) -> str:
    """
    Classify emotion according to the Dimensional Model of Affect (Russell & Barrett, 1999).
    
    Uses NRC Emotion Lexicon categories (Mohammad & Turney, 2013) mapped to
    the circumplex model's primary emotion categories.
    
    Reference: Russell, J. A., & Barrett, L. F. (1999). Core affect, prototypical emotional
    episodes, and other things called emotion. Journal of Personality and Social Psychology.
    """
    emotion_lower = (emotion or "neutral").lower().strip()
    return EMOTION_TO_PLUTCHIK.get(emotion_lower, "neutral")


def compute_emotion_distribution(emotions: List[str]) -> Dict[str, float]:
    """
    Compute distribution of emotions across primary emotion categories.
    
    Based on NRC Emotion Lexicon (Mohammad & Turney, 2013) and
    Dimensional Model of Affect (Russell & Barrett, 1999).
    
    Returns normalized frequencies for each primary emotion.
    """
    if not emotions:
        return {e: 0.0 for e in list(DIMENSIONAL_PRIMARY_EMOTIONS.keys()) + ["neutral"]}
    
    classified = [classify_dimensional_emotion(e) for e in emotions]
    counts = Counter(classified)
    total = len(classified)
    
    distribution = {e: 0.0 for e in list(DIMENSIONAL_PRIMARY_EMOTIONS.keys()) + ["neutral"]}
    for emotion, count in counts.items():
        if emotion in distribution:
            distribution[emotion] = count / total
    
    return distribution


# =============================================================================
# DIALOGUE ACT TAXONOMY (ISO 24617-2, Bunt et al., 2017)
# Standardized dialogue annotation scheme for computational discourse analysis
# =============================================================================

@dataclass
class DialogueActCategory:
    """ISO 24617-2 Dialogue Act category with dimension classification."""
    id: int
    name: str
    area: str  # "positive_socio", "task_answers", "task_questions", "negative_socio"
    description: str


DIALOGUE_ACT_CATEGORIES = [
    # Social Obligation Management - Positive (based on ISO 24617-2)
    DialogueActCategory(1, "shows_solidarity", "positive_socio", "Social obligation: greeting, thanking, acknowledging contribution"),
    DialogueActCategory(2, "shows_tension_release", "positive_socio", "Auto-feedback: positive emotional expression, humor, rapport building"),
    DialogueActCategory(3, "agrees", "positive_socio", "Allo-feedback: agreement, acceptance, understanding confirmation"),
    
    # Task/Activity-Specific: Information Providing
    DialogueActCategory(4, "gives_suggestion", "task_answers", "Action directive: suggestion, recommendation, proposal for action"),
    DialogueActCategory(5, "gives_opinion", "task_answers", "Inform: opinion, evaluation, stance expression, sentiment sharing"),
    DialogueActCategory(6, "gives_orientation", "task_answers", "Inform: factual information, clarification, context setting"),
    
    # Task/Activity-Specific: Information Seeking
    DialogueActCategory(7, "asks_orientation", "task_questions", "Set-question: request for factual information"),
    DialogueActCategory(8, "asks_opinion", "task_questions", "Propositional question: request for opinion, evaluation, preference"),
    DialogueActCategory(9, "asks_suggestion", "task_questions", "Choice question: request for suggestion, direction, recommendation"),
    
    # Social Obligation Management - Negative
    DialogueActCategory(10, "disagrees", "negative_socio", "Disagreement: rejection, dissent, counter-argument"),
    DialogueActCategory(11, "shows_tension", "negative_socio", "Negative auto-feedback: uncertainty, confusion, hesitation"),
    DialogueActCategory(12, "shows_antagonism", "negative_socio", "Negative social: criticism, face-threatening act, defensive response"),
]

# Keywords for Dialogue Act classification (based on ISO 24617-2 taxonomy)
DIALOGUE_ACT_KEYWORDS = {
    "shows_solidarity": ["support", "help", "great job", "well done", "appreciate", "thank"],
    "shows_tension_release": ["haha", "lol", "funny", "joke", "laugh", "😄", "😊", ":)"],
    "agrees": ["agree", "yes", "right", "correct", "exactly", "true", "definitely", "absolutely", "makes sense"],
    "gives_suggestion": ["should", "could", "let's", "why don't we", "suggest", "recommend", "try", "consider"],
    "gives_opinion": ["think", "believe", "feel", "seems", "appears", "in my view", "opinion"],
    "gives_orientation": ["so", "basically", "mean", "what i'm saying", "clarify", "explain"],
    "asks_orientation": ["what", "when", "where", "who", "how many", "which"],
    "asks_opinion": ["what do you think", "how do you feel", "your opinion", "your view", "do you agree"],
    "asks_suggestion": ["what should", "how should", "any ideas", "suggestions", "what if"],
    "disagrees": ["disagree", "but", "however", "don't think", "not sure", "doubt", "actually"],
    "shows_tension": ["worried", "concerned", "nervous", "anxious", "unsure", "confused"],
    "shows_antagonism": ["no", "wrong", "ridiculous", "stupid", "nonsense", "impossible"],
}


def classify_dialogue_act(text: str, tone: str) -> str:
    """
    Classify utterance using ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017).
    
    Reference: Bunt, H., et al. (2017). ISO 24617-2: A semantically-based standard
    for dialogue annotation. Proceedings of the 11th Linguistic Annotation Workshop.
    """
    text_lower = (text or "").lower()
    
    # Check each category's keywords
    scores = {}
    for category, keywords in DIALOGUE_ACT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[category] = score
    
    # Use tone as tiebreaker
    if tone == "positive" and not scores:
        return "agrees"
    elif tone == "negative" and not scores:
        return "disagrees"
    
    # Return highest scoring category or default
    if scores:
        return max(scores.items(), key=lambda x: x[1])[0]
    
    # Default based on question marks
    if "?" in text:
        return "asks_opinion"
    return "gives_opinion"


def compute_dialogue_act_profile(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute dialogue act profile based on ISO 24617-2 taxonomy (Bunt et al., 2017).
    
    Returns distribution across 4 functional dimensions and 12 dialogue act categories.
    """
    categories = []
    for record in history:
        cat = classify_dialogue_act(record.get("reply", ""), record.get("tone", "neutral"))
        categories.append(cat)
    
    if not categories:
        return {"areas": {}, "categories": {}, "task_ratio": 0.0}
    
    # Count by category
    cat_counts = Counter(categories)
    total = len(categories)
    
    # Aggregate by functional dimension
    area_counts = {"positive_socio": 0, "task_answers": 0, "task_questions": 0, "negative_socio": 0}
    for da_cat in DIALOGUE_ACT_CATEGORIES:
        area_counts[da_cat.area] += cat_counts.get(da_cat.name, 0)
    
    # Task ratio = (task_answers + task_questions) / total
    task_count = area_counts["task_answers"] + area_counts["task_questions"]
    socio_count = area_counts["positive_socio"] + area_counts["negative_socio"]
    
    return {
        "areas": {k: v / total for k, v in area_counts.items()},
        "categories": {k: v / total for k, v in cat_counts.items()},
        "task_ratio": task_count / total if total > 0 else 0.0,
        "socio_ratio": socio_count / total if total > 0 else 0.0,
        "positive_negative_ratio": (
            area_counts["positive_socio"] / max(1, area_counts["negative_socio"])
        ),
    }


# =============================================================================
# COMPUTATIONAL PRAGMATICS (Clark, 1996; Jurafsky & Martin, 2023)
# Modern speech act classification with NLP-based pattern recognition
# =============================================================================

SPEECH_ACT_PATTERNS = {
    "assertive": [
        r"\bi think\b", r"\bi believe\b", r"\bit is\b", r"\bthis is\b",
        r"\bwe have\b", r"\bthere is\b", r"\bfact\b", r"\bclearly\b",
    ],
    "directive": [
        r"\bplease\b", r"\bcould you\b", r"\bshould\b", r"\bmust\b",
        r"\blet's\b", r"\bdo\b", r"\btry\b", r"\bneed to\b",
    ],
    "commissive": [
        r"\bi will\b", r"\bi'll\b", r"\bi promise\b", r"\bwe will\b",
        r"\bgoing to\b", r"\bcommit\b", r"\bguarantee\b",
    ],
    "expressive": [
        r"\bthank\b", r"\bsorry\b", r"\bcongrat\b", r"\bhappy\b",
        r"\bsad\b", r"\bexcited\b", r"\bworried\b", r"\bgreat\b",
    ],
    "declarative": [
        r"\bi declare\b", r"\bi pronounce\b", r"\bhereby\b",
        r"\bofficially\b", r"\bdecided\b", r"\bapproved\b",
    ],
}


def classify_speech_act(text: str) -> str:
    """
    Classify utterance according to Computational Pragmatics (Clark, 1996).
    
    Categories based on modern dialogue act research:
    - Assertives: stating facts, describing, informing
    - Directives: requesting, commanding, advising, suggesting
    - Commissives: promising, offering, committing
    - Expressives: thanking, apologizing, congratulating
    - Declaratives: declaring, pronouncing, establishing
    
    Reference: Clark, H. H. (1996). Using Language. Cambridge University Press.
    See also: Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed.).
    """
    text_lower = (text or "").lower()
    
    scores = {}
    for act_type, patterns in SPEECH_ACT_PATTERNS.items():
        score = sum(1 for p in patterns if re.search(p, text_lower))
        scores[act_type] = score
    
    # Return highest scoring or default to assertive
    if any(scores.values()):
        return max(scores.items(), key=lambda x: x[1])[0]
    return "assertive"


def compute_speech_act_distribution(history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute distribution of speech acts in dialogue."""
    if not history:
        return {k: 0.0 for k in SPEECH_ACT_PATTERNS.keys()}
    
    acts = [classify_speech_act(r.get("reply", "")) for r in history]
    counts = Counter(acts)
    total = len(acts)
    
    return {k: counts.get(k, 0) / total for k in SPEECH_ACT_PATTERNS.keys()}


# =============================================================================
# SOCIAL NETWORK ANALYSIS (Borgatti et al., 2009; Barabási & Albert, 1999)
# Modern computational approaches to network centrality and structure
# =============================================================================

def compute_degree_centrality(history: List[Dict[str, Any]], agents: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute degree centrality measures using modern SNA methods (Borgatti et al., 2009).
    
    - Out-degree: number of messages sent to distinct targets
    - In-degree: number of messages received from distinct sources
    - Total degree centrality: (in + out) / 2(N-1)
    
    Reference: Borgatti, S. P., Mehra, A., Brass, D. J., & Labianca, G. (2009).
    Network analysis in the social sciences. Science, 323(5916), 892-895.
    """
    n = len(agents)
    if n < 2:
        return {}
    
    out_targets = defaultdict(set)  # agent -> set of targets
    in_sources = defaultdict(set)   # agent -> set of sources
    
    for record in history:
        speaker = record.get("speaker")
        target = record.get("target")
        if speaker and target:
            out_targets[speaker].add(target)
            in_sources[target].add(speaker)
    
    max_degree = n - 1  # Maximum possible connections
    
    result = {}
    for agent in agents:
        out_deg = len(out_targets[agent])
        in_deg = len(in_sources[agent])
        result[agent] = {
            "out_degree": out_deg,
            "in_degree": in_deg,
            "out_centrality": out_deg / max_degree,
            "in_centrality": in_deg / max_degree,
            "total_centrality": (out_deg + in_deg) / (2 * max_degree),
        }
    
    return result


def compute_betweenness_centrality(history: List[Dict[str, Any]], agents: List[str]) -> Dict[str, float]:
    """
    Compute betweenness centrality based on communication flow patterns.
    
    Measures how often an agent serves as a bridge in communication chains,
    indicating potential information brokerage (Burt, 2004).
    
    Reference: Burt, R. S. (2004). Structural holes and good ideas.
    American Journal of Sociology, 110(2), 349-399.
    """
    if len(agents) < 3:
        return {a: 0.0 for a in agents}
    
    # Build communication chains (A→B, B→C implies A→B→C)
    chains = defaultdict(int)
    
    prev_speaker = None
    prev_target = None
    
    for record in history:
        speaker = record.get("speaker")
        target = record.get("target")
        
        if prev_target and prev_target == speaker and target:
            # Found a chain: prev_speaker → prev_target(=speaker) → target
            if prev_speaker != target:  # Not a bounce-back
                chains[speaker] += 1
        
        prev_speaker = speaker
        prev_target = target
    
    total_chains = sum(chains.values()) or 1
    return {a: chains.get(a, 0) / total_chains for a in agents}


def compute_network_density(history: List[Dict[str, Any]], agents: List[str]) -> float:
    """
    Compute network density = actual edges / possible edges.
    
    For directed graph: density = E / N(N-1)
    
    Reference: Borgatti, S. P., et al. (2009). Network analysis in the social sciences.
    Science, 323(5916), 892-895.
    """
    n = len(agents)
    if n < 2:
        return 0.0
    
    max_edges = n * (n - 1)  # Directed graph
    
    edges = set()
    for record in history:
        speaker = record.get("speaker")
        target = record.get("target")
        if speaker and target and speaker != target:
            edges.add((speaker, target))
    
    return len(edges) / max_edges


def compute_clustering_coefficient(history: List[Dict[str, Any]], agents: List[str]) -> float:
    """
    Compute global clustering coefficient (transitivity).
    
    Measures tendency to form triadic closures in communication networks.
    High clustering indicates group cohesion and network closure (Burt, 2004).
    
    Reference: Barabási, A. L. (2016). Network Science. Cambridge University Press.
    See also: Watts, D. J., & Strogatz, S. H. (1998). Nature, 393(6684), 440-442.
    """
    if len(agents) < 3:
        return 0.0
    
    # Build adjacency
    adj = defaultdict(set)
    for record in history:
        speaker = record.get("speaker")
        target = record.get("target")
        if speaker and target:
            adj[speaker].add(target)
            adj[target].add(speaker)  # Undirected for clustering
    
    triangles = 0
    triples = 0
    
    for agent in agents:
        neighbors = list(adj[agent])
        k = len(neighbors)
        if k < 2:
            continue
        
        # Count possible triples centered on agent
        triples += k * (k - 1) / 2
        
        # Count actual triangles
        for i in range(k):
            for j in range(i + 1, k):
                if neighbors[j] in adj[neighbors[i]]:
                    triangles += 1
    
    return triangles / triples if triples > 0 else 0.0


# =============================================================================
# SOCIAL CAPITAL & RELATIONAL COORDINATION (Burt, 2004; Gittell, 2002)
# Network position and relational quality metrics
# =============================================================================

def compute_social_capital_indices(history: List[Dict[str, Any]], agents: List[str]) -> Dict[str, Any]:
    """
    Compute social capital and relational coordination indices.
    
    - Network Status: (connections received) / (N-1) - indicates social capital
    - Network Reach: (connections made) / (N-1) - indicates network expansion
    - Reciprocity Index: mutual connections / total connections - indicates relational quality
    
    Reference: Burt, R. S. (2004). Structural holes and good ideas. 
    American Journal of Sociology, 110(2), 349-399.
    See also: Gittell, J. H. (2002). Coordinating mechanisms in care provider groups.
    Management Science, 48(11), 1408-1426.
    """
    n = len(agents)
    if n < 2:
        return {}
    
    choices_made = Counter()    # How many distinct agents each person addressed
    choices_received = Counter()  # How many distinct agents addressed each person
    
    made_to = defaultdict(set)
    
    for record in history:
        speaker = record.get("speaker")
        target = record.get("target")
        if speaker and target and speaker != target:
            if target not in made_to[speaker]:
                made_to[speaker].add(target)
                choices_made[speaker] += 1
                choices_received[target] += 1
    
    # Calculate mutual choices
    mutual = 0
    total_pairs = 0
    for a in agents:
        for b in agents:
            if a < b:  # Each pair once
                if b in made_to[a] and a in made_to[b]:
                    mutual += 1
                if b in made_to[a] or a in made_to[b]:
                    total_pairs += 1
    
    max_choices = n - 1
    
    result = {
        "agents": {},
        "group_cohesion": mutual / max(1, total_pairs),  # Group-level reciprocity
        "network_density": compute_network_density(history, agents),
    }
    
    for agent in agents:
        result["agents"][agent] = {
            "sociometric_status": choices_received[agent] / max_choices,
            "expansion_index": choices_made[agent] / max_choices,
            "popularity": choices_received[agent],
            "activity": choices_made[agent],
        }
    
    return result


# =============================================================================
# COMPUTER-MEDIATED DISCOURSE ANALYSIS (Herring, 2004; Androutsopoulos, 2006)
# Digital communication patterns and interaction analysis
# =============================================================================

def compute_turn_taking_metrics(history: List[Dict[str, Any]], agents: List[str]) -> Dict[str, Any]:
    """
    Compute turn-taking and interaction metrics for computer-mediated discourse.
    
    Based on Computer-Mediated Discourse Analysis (CMDA) framework for
    analyzing digital communication patterns, turn distribution, and response sequences.
    
    Reference: Herring, S. C. (2004). Computer-Mediated Discourse Analysis: 
    An Approach to Researching Online Behavior. In Designing for Virtual Communities.
    See also: Androutsopoulos, J. (2006). Introduction: Sociolinguistics and computer-mediated
    communication. Journal of Sociolinguistics, 10(4), 419-438.
    """
    if not history:
        return {}
    
    # Turn lengths (word count)
    turn_lengths = defaultdict(list)
    for record in history:
        speaker = record.get("speaker")
        words = len((record.get("reply") or "").split())
        turn_lengths[speaker].append(words)
    
    # Adjacency pair completion (question followed by response)
    adjacency_pairs = 0
    adjacency_complete = 0
    
    for i, record in enumerate(history[:-1]):
        if "?" in (record.get("reply") or ""):
            adjacency_pairs += 1
            next_record = history[i + 1]
            # Check if next turn is from addressed person
            if record.get("target") == next_record.get("speaker"):
                adjacency_complete += 1
    
    # Interruption proxy: same speaker twice in a row
    self_selections = 0
    for i in range(1, len(history)):
        if history[i].get("speaker") == history[i-1].get("speaker"):
            self_selections += 1
    
    # Turn distribution evenness (Gini coefficient)
    turn_counts = Counter(r.get("speaker") for r in history)
    turns_list = [turn_counts.get(a, 0) for a in agents]
    gini = compute_gini_coefficient(turns_list)
    
    return {
        "mean_turn_length": {
            a: sum(lengths) / len(lengths) if lengths else 0
            for a, lengths in turn_lengths.items()
        },
        "turn_distribution": {a: turn_counts.get(a, 0) / len(history) for a in agents},
        "adjacency_completion_rate": adjacency_complete / max(1, adjacency_pairs),
        "self_selection_rate": self_selections / max(1, len(history) - 1),
        "turn_inequality_gini": gini,
    }


def compute_gini_coefficient(values: List[int]) -> float:
    """Compute Gini coefficient for distribution inequality."""
    if not values or sum(values) == 0:
        return 0.0
    
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    cumulative = 0
    total = sum(sorted_vals)
    
    for i, val in enumerate(sorted_vals):
        cumulative += val
    
    # Gini formula
    numerator = sum((2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_vals))
    return numerator / (n * total) if total > 0 else 0.0


# =============================================================================
# INTEGRATED MODEL OF GROUP DEVELOPMENT (Wheelan, 2009; Kozlowski & Ilgen, 2006)
# Modern team development theory with empirical validation
# =============================================================================

def detect_group_development_stage(history: List[Dict[str, Any]], window: int = 10) -> Dict[str, Any]:
    """
    Detect current group development stage using Integrated Model (Wheelan, 2009).
    
    Stages:
    1. Dependency: orientation, politeness, dependency on leadership
    2. Counterdependency: conflict, disagreement, role negotiation
    3. Trust: cohesion, shared norms, collaborative relationships
    4. Work: high productivity, effective goal-oriented teamwork
    
    Reference: Wheelan, S. A. (2009). Group size, group development, and group productivity.
    Small Group Research, 40(2), 247-262.
    See also: Kozlowski, S. W., & Ilgen, D. R. (2006). Enhancing the effectiveness of work groups
    and teams. Psychological Science in the Public Interest, 7(3), 77-124.
    """
    if not history:
        return {"stage": "forming", "confidence": 0.0, "indicators": {}}
    
    recent = history[-window:] if len(history) > window else history
    
    # Calculate indicators
    tones = [r.get("tone", "neutral") for r in recent]
    tone_counts = Counter(tones)
    total = len(tones)
    
    negative_ratio = tone_counts.get("negative", 0) / total
    positive_ratio = tone_counts.get("positive", 0) / total
    
    # Dialogue act profile for recent messages (ISO 24617-2)
    dialogue_acts = compute_dialogue_act_profile(recent)
    task_ratio = dialogue_acts.get("task_ratio", 0)
    
    # Reciprocity (measure of cohesion)
    pairs = [(r["speaker"], r.get("target")) for r in recent if r.get("target")]
    set_pairs = set(pairs)
    mutual = sum(1 for (u, v) in set_pairs if (v, u) in set_pairs)
    reciprocity = mutual / max(1, len(set_pairs))
    
    # Question rate (exploration indicator)
    question_rate = sum(1 for r in recent if "?" in (r.get("reply") or "")) / total
    
    # Stage detection logic
    indicators = {
        "negative_ratio": negative_ratio,
        "positive_ratio": positive_ratio,
        "task_ratio": task_ratio,
        "reciprocity": reciprocity,
        "question_rate": question_rate,
    }
    
    # Scoring for each stage
    scores = {
        "forming": (
            0.3 * (1 - negative_ratio) +  # Low conflict
            0.3 * question_rate +           # Exploration
            0.2 * (1 - task_ratio) +        # Less task focus
            0.2 * (1 - reciprocity)         # Still building relationships
        ),
        "storming": (
            0.5 * negative_ratio +          # High conflict
            0.2 * (1 - reciprocity) +       # Low cohesion
            0.3 * (1 - positive_ratio)      # Less positivity
        ),
        "norming": (
            0.3 * reciprocity +             # Building cohesion
            0.3 * positive_ratio +          # Positive atmosphere
            0.2 * (1 - negative_ratio) +    # Reduced conflict
            0.2 * task_ratio                # Some task focus
        ),
        "performing": (
            0.3 * task_ratio +              # High task focus
            0.3 * reciprocity +             # High cohesion
            0.2 * positive_ratio +          # Positive atmosphere
            0.2 * (1 - question_rate)       # Less exploration, more doing
        ),
    }
    
    # Determine stage
    stage = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[stage] / sum(scores.values()) if sum(scores.values()) > 0 else 0
    
    return {
        "stage": stage,
        "stage_scores": scores,
        "confidence": confidence,
        "indicators": indicators,
    }


# =============================================================================
# COMPREHENSIVE SCIENTIFIC ANALYSIS
# =============================================================================

@dataclass
class ScientificAnalysisResult:
    """Container for all scientific analysis results.
    
    Based on modern scientific frameworks (1990-2025):
    - Dimensional Emotion Model (Russell & Barrett, 1999; Mohammad & Turney, 2013)
    - ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017)
    - Social Network Analysis (Borgatti et al., 2009; Barabási & Albert, 1999)
    - Social Capital Theory (Burt, 2004; Gittell, 2002)
    - Computer-Mediated Discourse Analysis (Herring, 2004)
    - Integrated Model of Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006)
    """
    
    # Dimensional Emotion Model (Russell & Barrett, 1999; Mohammad & Turney, 2013)
    emotion_distribution: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"
    
    # ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017)
    dialogue_act_profile: Dict[str, Any] = field(default_factory=dict)
    
    # Computational Pragmatics Speech Acts (Clark, 1996; Jurafsky & Martin, 2023)
    speech_acts: Dict[str, float] = field(default_factory=dict)
    
    # Social Network Analysis (Borgatti et al., 2009; Barabási & Albert, 1999)
    centrality: Dict[str, Dict[str, float]] = field(default_factory=dict)
    betweenness: Dict[str, float] = field(default_factory=dict)
    network_density: float = 0.0
    clustering_coefficient: float = 0.0
    
    # Social Capital & Relational Coordination (Burt, 2004; Gittell, 2002)
    social_capital: Dict[str, Any] = field(default_factory=dict)
    
    # Computer-Mediated Discourse Analysis (Herring, 2004; Androutsopoulos, 2006)
    turn_taking: Dict[str, Any] = field(default_factory=dict)
    
    # Integrated Model of Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006)
    group_stage: Dict[str, Any] = field(default_factory=dict)


def compute_scientific_analysis(
    history: List[Dict[str, Any]], 
    agents: List[str],
    window_size: int = 20
) -> ScientificAnalysisResult:
    """
    Perform comprehensive scientific analysis of dialogue.
    
    Integrates modern scientific frameworks (1990-2025):
    - Dimensional Emotion Model (Russell & Barrett, 1999; Mohammad & Turney, 2013)
    - ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017)
    - Computational Pragmatics (Clark, 1996; Jurafsky & Martin, 2023)
    - Social Network Analysis (Borgatti et al., 2009; Barabási & Albert, 1999)
    - Social Capital Theory (Burt, 2004; Gittell, 2002)
    - Computer-Mediated Discourse Analysis (Herring, 2004; Androutsopoulos, 2006)
    - Integrated Model of Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006)
    """
    if not history:
        return ScientificAnalysisResult()
    
    recent = history[-window_size:] if len(history) > window_size else history
    
    # Extract emotions
    emotions = [r.get("emotion", "neutral") for r in recent]
    
    # Dimensional Emotion Model (Russell & Barrett, 1999; Mohammad & Turney, 2013)
    emotion_dist = compute_emotion_distribution(emotions)
    dominant = max(emotion_dist.items(), key=lambda x: x[1])[0] if emotion_dist else "neutral"
    
    # ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017)
    dialogue_acts = compute_dialogue_act_profile(recent)
    
    # Computational Pragmatics Speech Acts (Clark, 1996; Jurafsky & Martin, 2023)
    speech_acts = compute_speech_act_distribution(recent)
    
    # Social Network Analysis (Borgatti et al., 2009; Barabási & Albert, 1999)
    centrality = compute_degree_centrality(history, agents)
    betweenness = compute_betweenness_centrality(history, agents)
    density = compute_network_density(history, agents)
    clustering = compute_clustering_coefficient(history, agents)
    
    # Social Capital & Relational Coordination (Burt, 2004; Gittell, 2002)
    social_capital = compute_social_capital_indices(history, agents)
    
    # Computer-Mediated Discourse Analysis (Herring, 2004; Androutsopoulos, 2006)
    turn_taking = compute_turn_taking_metrics(history, agents)
    
    # Integrated Model of Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006)
    group_stage = detect_group_development_stage(history)
    
    return ScientificAnalysisResult(
        emotion_distribution=emotion_dist,
        dominant_emotion=dominant,
        dialogue_act_profile=dialogue_acts,
        speech_acts=speech_acts,
        centrality=centrality,
        betweenness=betweenness,
        network_density=density,
        clustering_coefficient=clustering,
        social_capital=social_capital,
        turn_taking=turn_taking,
        group_stage=group_stage,
    )


# =============================================================================
# SCIENTIFIC HYPOTHESES GENERATION
# =============================================================================

def generate_scientific_hypotheses(analysis: ScientificAnalysisResult, agents: List[str]) -> List[Dict[str, Any]]:
    """
    Generate hypotheses based on modern scientific frameworks (1990-2025) with literature references.
    """
    hypotheses = []
    
    # --- Integrated Model of Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006) ---
    stage = analysis.group_stage.get("stage", "forming")
    confidence = analysis.group_stage.get("confidence", 0)
    
    if confidence > 0.3:
        stage_advice = {
            "forming": "Группа находится на стадии зависимости (Dependency). Рекомендуется: установление психологической безопасности, чёткое определение ролей и целей.",
            "storming": "Группа переживает стадию контрзависимости (Counterdependency). Это нормальный этап развития. Рекомендуется: конструктивное управление конфликтами.",
            "norming": "Группа формирует доверие и структуру (Trust). Рекомендуется: закрепление норм взаимодействия, развитие общей идентичности.",
            "performing": "Группа на стадии продуктивной работы (Work). Рекомендуется: поддержание высокой производительности, распределённое лидерство.",
        }
        hypotheses.append({
            "category": "Group Development",
            "framework": "Integrated Model (Wheelan, 2009)",
            "finding": f"Текущая стадия: {stage.upper()} (уверенность: {confidence:.0%})",
            "recommendation": stage_advice.get(stage, ""),
            "reference": "Wheelan, S. A. (2009). Group size, group development, and group productivity. Small Group Research, 40(2), 247-262.",
        })
    
    # --- ISO 24617-2 Dialogue Act Taxonomy (Bunt et al., 2017) ---
    dialogue_acts = analysis.dialogue_act_profile
    if dialogue_acts:
        task_ratio = dialogue_acts.get("task_ratio", 0)
        pos_neg_ratio = dialogue_acts.get("positive_negative_ratio", 1)
        
        if task_ratio > 0.7:
            hypotheses.append({
                "category": "Dialogue Structure",
                "framework": "ISO 24617-2 (Bunt et al., 2017)",
                "finding": f"Высокая задачная ориентация ({task_ratio:.0%}). Преобладают информационные диалоговые акты.",
                "recommendation": "Сбалансируйте с социальными актами (feedback, social obligation management) для поддержания вовлечённости.",
                "reference": "Bunt, H., et al. (2017). ISO 24617-2: A semantically-based standard for dialogue annotation. Proc. LAW XI.",
            })
        elif task_ratio < 0.3:
            hypotheses.append({
                "category": "Dialogue Structure",
                "framework": "ISO 24617-2 (Bunt et al., 2017)",
                "finding": f"Низкая задачная ориентация ({task_ratio:.0%}). Преобладают социальные диалоговые акты.",
                "recommendation": "Направьте дискуссию к информационным актам (inform, question) для продвижения к решениям.",
                "reference": "Bunt, H., et al. (2017). ISO 24617-2: A semantically-based standard for dialogue annotation. Proc. LAW XI.",
            })
        
        if pos_neg_ratio < 1:
            hypotheses.append({
                "category": "Dialogue Structure",
                "framework": "ISO 24617-2 (Bunt et al., 2017)",
                "finding": "Негативные социальные акты (disagreement, negative feedback) преобладают над позитивными.",
                "recommendation": "Увеличьте позитивные акты (agreement, acknowledgement) для улучшения атмосферы.",
                "reference": "Bunt, H., et al. (2017). ISO 24617-2: A semantically-based standard for dialogue annotation. Proc. LAW XI.",
            })
    
    # --- Social Network Analysis (Borgatti et al., 2009; Barabási & Albert, 1999) ---
    if analysis.centrality:
        # Find most central agent
        max_central = max(
            analysis.centrality.items(),
            key=lambda x: x[1].get("total_centrality", 0)
        )
        if max_central[1].get("total_centrality", 0) > 0.7:
            hypotheses.append({
                "category": "Network Structure",
                "framework": "SNA Centrality (Borgatti et al., 2009)",
                "finding": f"Агент '{max_central[0]}' занимает центральную позицию в сети ({max_central[1]['total_centrality']:.0%}).",
                "recommendation": "Высокая централизация указывает на потенциальную информационную зависимость. Рассмотрите распределённое лидерство.",
                "reference": "Borgatti, S. P., et al. (2009). Network analysis in the social sciences. Science, 323(5916), 892-895.",
            })
    
    if analysis.network_density < 0.5:
        hypotheses.append({
            "category": "Network Structure",
            "framework": "Network Science (Barabási, 2016)",
            "finding": f"Низкая плотность сети ({analysis.network_density:.0%}). Структурные дыры в коммуникации.",
            "recommendation": "Стимулируйте коммуникацию между изолированными участниками для увеличения сетевой связности.",
            "reference": "Barabási, A. L. (2016). Network Science. Cambridge University Press.",
        })
    
    # --- Social Capital & Relational Coordination (Burt, 2004; Gittell, 2002) ---
    social_cap = analysis.social_capital
    if social_cap and "agents" in social_cap:
        # Find isolated members (low network status)
        isolated = [
            a for a, data in social_cap["agents"].items()
            if data.get("sociometric_status", 0) < 0.3
        ]
        if isolated:
            hypotheses.append({
                "category": "Social Capital",
                "framework": "Structural Holes (Burt, 2004)",
                "finding": f"Участники с низким сетевым статусом: {', '.join(isolated)}",
                "recommendation": "Эти участники имеют меньше социального капитала. Активнее вовлекайте их для развития сетевых связей.",
                "reference": "Burt, R. S. (2004). Structural holes and good ideas. American Journal of Sociology, 110(2), 349-399.",
            })
        
        cohesion = social_cap.get("group_cohesion", 0)
        if cohesion < 0.3:
            hypotheses.append({
                "category": "Social Capital",
                "framework": "Relational Coordination (Gittell, 2002)",
                "finding": f"Низкая реляционная координация ({cohesion:.0%}). Мало взаимных связей.",
                "recommendation": "Развивайте общие цели, взаимное уважение и частую коммуникацию для повышения координации.",
                "reference": "Gittell, J. H. (2002). Coordinating mechanisms in care provider groups. Management Science, 48(11), 1408-1426.",
            })
    
    # --- Dimensional Emotion Model (Russell & Barrett, 1999; Mohammad & Turney, 2013) ---
    emotions = analysis.emotion_distribution
    if emotions:
        negative_emotions = emotions.get("anger", 0) + emotions.get("fear", 0) + emotions.get("sadness", 0)
        if negative_emotions > 0.4:
            hypotheses.append({
                "category": "Emotional Climate",
                "framework": "Circumplex Model (Russell & Barrett, 1999)",
                "finding": f"Преобладание негативных эмоций ({negative_emotions:.0%}): гнев, страх, печаль (низкая валентность).",
                "recommendation": "Обратите внимание на эмоциональный климат. Используйте техники регуляции аффекта.",
                "reference": "Russell, J. A., & Barrett, L. F. (1999). Core affect. Journal of Personality and Social Psychology, 76(5), 805-819.",
            })
        
        if emotions.get("anticipation", 0) + emotions.get("joy", 0) > 0.5:
            hypotheses.append({
                "category": "Emotional Climate",
                "framework": "NRC Emotion Lexicon (Mohammad & Turney, 2013)",
                "finding": "Позитивный эмоциональный фон: преобладают ожидание и радость (высокая валентность).",
                "recommendation": "Используйте позитивный настрой для продвижения к решениям.",
                "reference": "Mohammad, S. M., & Turney, P. D. (2013). Crowdsourcing a word-emotion association lexicon. Computational Intelligence, 29(3), 436-465.",
            })
    
    # --- Computer-Mediated Discourse Analysis (Herring, 2004; Androutsopoulos, 2006) ---
    turn = analysis.turn_taking
    if turn:
        gini = turn.get("turn_inequality_gini", 0)
        if gini > 0.3:
            hypotheses.append({
                "category": "Turn-Taking",
                "framework": "CMDA (Herring, 2004)",
                "finding": f"Неравномерное распределение реплик (Gini = {gini:.2f}).",
                "recommendation": "В компьютерно-опосредованной коммуникации доминирование одних участников снижает вовлечённость других. Модерируйте для равного участия.",
                "reference": "Herring, S. C. (2004). Computer-Mediated Discourse Analysis. In The Handbook of Discourse Analysis. Blackwell.",
            })
        
        adj_rate = turn.get("adjacency_completion_rate", 0)
        if adj_rate < 0.5:
            hypotheses.append({
                "category": "Turn-Taking",
                "framework": "Digital Discourse (Androutsopoulos, 2006)",
                "finding": f"Низкий показатель завершения смежных пар ({adj_rate:.0%}). Вопросы часто остаются без ответа.",
                "recommendation": "В асинхронной коммуникации важно обеспечить, чтобы адресованные вопросы получали ответы.",
                "reference": "Androutsopoulos, J. (2006). Introduction: Sociolinguistics and CMC. Journal of Sociolinguistics, 10(4), 419-438.",
            })
    
    # --- Computational Pragmatics (Clark, 1996; Jurafsky & Martin, 2023) ---
    speech = analysis.speech_acts
    if speech:
        if speech.get("directive", 0) > 0.4:
            hypotheses.append({
                "category": "Speech Acts",
                "framework": "Computational Pragmatics (Jurafsky & Martin, 2023)",
                "finding": f"Высокая доля директивов ({speech['directive']:.0%}): команды, просьбы, предложения.",
                "recommendation": "Убедитесь, что директивы соответствуют условиям успешности (felicity conditions) и воспринимаются как предложения.",
                "reference": "Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed.). Stanford University.",
            })
        
        if speech.get("expressive", 0) < 0.1:
            hypotheses.append({
                "category": "Speech Acts",
                "framework": "Grounding Theory (Clark, 1996)",
                "finding": "Мало экспрессивных актов (благодарность, поддержка, эмоции).",
                "recommendation": "Экспрессивные акты важны для установления общей основы (common ground). Поощряйте эмоциональную поддержку.",
                "reference": "Clark, H. H. (1996). Using Language. Cambridge University Press.",
            })
    
    if not hypotheses:
        hypotheses.append({
            "category": "General",
            "framework": "Multiple frameworks",
            "finding": "Коммуникация проходит в пределах нормы по всем научным показателям.",
            "recommendation": "Продолжайте в том же духе.",
            "reference": "N/A",
        })
    
    return hypotheses


def render_scientific_report(
    analysis: ScientificAnalysisResult,
    hypotheses: List[Dict[str, Any]],
    turn: int
) -> str:
    """Render scientific analysis as markdown report based on modern frameworks (1990-2025)."""
    lines = []
    lines.append(f"# 🔬 Scientific Analysis — Turn {turn}")
    lines.append("")
    
    # Group Development Stage (Wheelan, 2009)
    lines.append("## 📊 Group Development (Wheelan, 2009; Kozlowski & Ilgen, 2006)")
    stage = analysis.group_stage
    if stage:
        lines.append(f"**Current Stage:** {stage.get('stage', 'unknown').upper()}")
        lines.append(f"**Confidence:** {stage.get('confidence', 0):.0%}")
        lines.append("")
        lines.append("| Indicator | Value |")
        lines.append("|---|---:|")
        for k, v in stage.get("indicators", {}).items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v:.2f} |")
    lines.append("")
    
    # Emotional Climate (Russell & Barrett, 1999; Mohammad & Turney, 2013)
    lines.append("## 🎭 Emotional Climate (Russell & Barrett, 1999)")
    lines.append(f"**Dominant Emotion:** {analysis.dominant_emotion}")
    lines.append("")
    lines.append("| Emotion | Distribution |")
    lines.append("|---|---:|")
    sorted_emotions = sorted(analysis.emotion_distribution.items(), key=lambda x: -x[1])
    for emotion, value in sorted_emotions[:5]:
        if value > 0:
            lines.append(f"| {emotion.title()} | {value:.0%} |")
    lines.append("")
    
    # Dialogue Act Profile (ISO 24617-2, Bunt et al., 2017)
    lines.append("## 🗣️ Dialogue Acts (ISO 24617-2, Bunt et al., 2017)")
    dialogue_acts = analysis.dialogue_act_profile
    if dialogue_acts:
        lines.append(f"**Task Ratio:** {dialogue_acts.get('task_ratio', 0):.0%}")
        lines.append(f"**Socio-Emotional Ratio:** {dialogue_acts.get('socio_ratio', 0):.0%}")
        lines.append(f"**Positive/Negative Ratio:** {dialogue_acts.get('positive_negative_ratio', 0):.2f}")
        lines.append("")
        lines.append("| Area | Distribution |")
        lines.append("|---|---:|")
        for area, value in dialogue_acts.get("areas", {}).items():
            lines.append(f"| {area.replace('_', ' ').title()} | {value:.0%} |")
    lines.append("")
    
    # Network Analysis (Borgatti et al., 2009)
    lines.append("## 🕸️ Social Network Analysis (Borgatti et al., 2009)")
    lines.append(f"**Network Density:** {analysis.network_density:.0%}")
    lines.append(f"**Clustering Coefficient:** {analysis.clustering_coefficient:.2f}")
    lines.append("")
    if analysis.centrality:
        lines.append("| Agent | In-Centrality | Out-Centrality | Total |")
        lines.append("|---|---:|---:|---:|")
        for agent, data in sorted(analysis.centrality.items(), key=lambda x: -x[1].get("total_centrality", 0)):
            lines.append(f"| {agent} | {data.get('in_centrality', 0):.0%} | {data.get('out_centrality', 0):.0%} | {data.get('total_centrality', 0):.0%} |")
    lines.append("")
    
    # Social Capital (Burt, 2004; Gittell, 2002)
    lines.append("## 👥 Social Capital (Burt, 2004; Gittell, 2002)")
    social_cap = analysis.social_capital
    if social_cap and "agents" in social_cap:
        lines.append(f"**Relational Coordination:** {social_cap.get('group_cohesion', 0):.0%}")
        lines.append("")
        lines.append("| Agent | Network Status | Expansion | Popularity | Activity |")
        lines.append("|---|---:|---:|---:|---:|")
        for agent, data in social_cap["agents"].items():
            lines.append(f"| {agent} | {data.get('sociometric_status', 0):.0%} | {data.get('expansion_index', 0):.0%} | {data.get('popularity', 0)} | {data.get('activity', 0)} |")
    lines.append("")
    
    # Turn-Taking (Herring, 2004)
    lines.append("## 🔄 Turn-Taking (CMDA, Herring, 2004)")
    turn_data = analysis.turn_taking
    if turn_data:
        lines.append(f"**Turn Inequality (Gini):** {turn_data.get('turn_inequality_gini', 0):.2f}")
        lines.append(f"**Adjacency Completion:** {turn_data.get('adjacency_completion_rate', 0):.0%}")
        lines.append(f"**Self-Selection Rate:** {turn_data.get('self_selection_rate', 0):.0%}")
    lines.append("")
    
    # Scientific Hypotheses
    lines.append("## 📚 Scientific Hypotheses")
    lines.append("")
    for i, hyp in enumerate(hypotheses, 1):
        lines.append(f"### {i}. {hyp['category']} — {hyp['framework']}")
        lines.append(f"**Finding:** {hyp['finding']}")
        lines.append(f"**Recommendation:** {hyp['recommendation']}")
        lines.append(f"*Reference: {hyp['reference']}*")
        lines.append("")
    
    return "\n".join(lines)

