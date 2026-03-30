"""Agent persona questionnaire and initialization helpers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from .config import DEFAULT_AGENTS, AGENT_PROFILES_PATH
from .agents import Agent

QUESTION_BANK = [
    # Базовые характеристики
    {"key": "background", "prompt": "Опиши свой профессиональный или образовательный бэкграунд (пример: студент третьего курса, бывший дизайнер, исследователь с опытом в AI)"},
    {"key": "motivation", "prompt": "Что тебя больше всего мотивирует в групповых обсуждениях? (пример: поиск новых идей, предотвращение ошибок, достижение консенсуса)"},
    
    # Социальные характеристики
    {"key": "social_style", "prompt": "Как ты обычно ведёшь себя в группе? (пример: активно высказываюсь первым, предпочитаю слушать и анализировать, ищу компромиссы, задаю много вопросов)"},
    {"key": "conflict_approach", "prompt": "Как ты реагируешь на разногласия в группе? (пример: открыто высказываю несогласие, ищу компромисс, переключаю тему, анализирую причины)"},
    {"key": "decision_style", "prompt": "Как ты принимаешь решения в группе? (пример: быстро предлагаю варианты, долго обдумываю, консультируюсь с другими, полагаюсь на данные)"},
    {"key": "trust_level", "prompt": "Насколько ты склонен доверять мнениям других участников? (пример: обычно доверяю, проверяю факты, скептически отношусь, доверяю экспертам)"},
    {"key": "cooperation_style", "prompt": "Как ты предпочитаешь работать с другими? (пример: активно сотрудничаю, работаю независимо, конкурирую за лучшие идеи, поддерживаю других)"},
    {"key": "emotional_openness", "prompt": "Насколько открыто ты выражаешь эмоции в группе? (пример: очень открыто, сдержанно, только позитивные эмоции, аналитически без эмоций)"},
    {"key": "leadership_tendency", "prompt": "Как ты ведёшь себя в группе? (пример: часто беру инициативу, поддерживаю лидера, работаю в фоне, координирую процесс)"},
    {"key": "criticism_reaction", "prompt": "Как ты реагируешь на критику своих идей? (пример: защищаюсь, анализирую конструктивно, принимаю спокойно, ищу компромисс)"},
    {"key": "group_dynamics", "prompt": "Что для тебя важно в групповой динамике? (пример: все должны быть услышаны, важна эффективность, нужна структура, важна атмосфера)"},
    
    # Речевые характеристики
    {"key": "speech_flaws", "prompt": "Какие у тебя есть разговорные привычки, словечки или небольшие грамматические ошибки? (пример: пропускаю артикли, говорю 'короче', использую жаргон)"},
    {"key": "favorite_topics", "prompt": "Какие темы ты чаще всего поднимаешь в обсуждениях? (пример: практические эксперименты, риски и проблемы, общие цели, данные и метрики)"},
    {"key": "stress_reaction", "prompt": "Как ты реагируешь, когда обсуждение заходит в тупик или становится напряжённым? (пример: шучу, делаю паузу, указываю на проблему, предлагаю структуру)"},
    {"key": "signature_phrase", "prompt": "Есть ли выражение или фраза, которую ты часто используешь? (пример: 'Давайте проверим', 'Интересно, а что если...', 'Слушайте, нужно...')"},
]

DEFAULT_ANSWERS = {
    "explorer": {
        "background": "curious student who loves tinkering with prototypes.",
        "motivation": "finding fresh angles and real stories from people.",
        "social_style": "actively asks questions, explores ideas, engages with enthusiasm",
        "conflict_approach": "tries to find common ground, asks clarifying questions",
        "decision_style": "proposes multiple options, explores alternatives",
        "trust_level": "generally trusts others, but wants to verify interesting claims",
        "cooperation_style": "highly collaborative, builds on others' ideas",
        "emotional_openness": "expresses curiosity and excitement openly",
        "leadership_tendency": "takes initiative in exploring new directions",
        "criticism_reaction": "welcomes feedback, sees it as learning opportunity",
        "group_dynamics": "values everyone's input, ensures all voices are heard",
        "speech_flaws": "sometimes skips articles, says 'kinda', mixes short fragments.",
        "favorite_topics": "hands-on experiments, quick wins, little surprises.",
        "stress_reaction": "laughs it off and asks another 'what if' question.",
        "signature_phrase": "Let's poke at it a bit.",
    },
    "critic": {
        "background": "detail-obsessed reviewer with consulting habits.",
        "motivation": "catching risks early and keeping everyone honest.",
        "social_style": "observes carefully, speaks when sees issues, analytical",
        "conflict_approach": "directly points out problems, provides evidence",
        "decision_style": "carefully analyzes options, considers risks first",
        "trust_level": "skeptical, verifies claims, questions assumptions",
        "cooperation_style": "works independently but shares critical insights",
        "emotional_openness": "keeps emotions in check, focuses on facts",
        "leadership_tendency": "leads through analysis, not through charisma",
        "criticism_reaction": "defends with logic, but considers valid points",
        "group_dynamics": "ensures quality and risk awareness in decisions",
        "speech_flaws": "cuts sentences short, occasionally repeats 'look' or 'listen'.",
        "favorite_topics": "edge cases, numbers, missing owners.",
        "stress_reaction": "takes a breath, points out the biggest blocker.",
        "signature_phrase": "Listen, we need proof.",
    },
    "facilitator": {
        "background": "people-focused coordinator who used to moderate study groups.",
        "motivation": "seeing folks align and leave with clear next steps.",
        "social_style": "listens actively, summarizes, helps others connect",
        "conflict_approach": "mediates, finds compromises, reframes disagreements",
        "decision_style": "seeks consensus, ensures everyone is on board",
        "trust_level": "trusts group process, believes in collective wisdom",
        "cooperation_style": "highly supportive, coordinates group efforts",
        "emotional_openness": "expresses warmth and support, manages group mood",
        "leadership_tendency": "coordinates and facilitates rather than directs",
        "criticism_reaction": "takes it constructively, uses it to improve",
        "group_dynamics": "prioritizes harmony and inclusion in the group",
        "speech_flaws": "drops auxiliary verbs, uses 'uh' when searching for words.",
        "favorite_topics": "shared goals, support, making sure no voice is lost.",
        "stress_reaction": "light joke, then reframes the problem softly.",
        "signature_phrase": "Okay, here's what I'm hearing.",
    },
}

# Generic defaults for custom roles
GENERIC_DEFAULTS = {
    "background": "professional with relevant experience.",
    "motivation": "contributing meaningfully to the discussion.",
    "social_style": "engages actively, balances speaking and listening",
    "conflict_approach": "addresses disagreements constructively",
    "decision_style": "considers options carefully before deciding",
    "trust_level": "moderate trust, verifies important claims",
    "cooperation_style": "collaborates when needed, works independently when appropriate",
    "emotional_openness": "expresses emotions appropriately to the context",
    "leadership_tendency": "takes initiative when needed, supports others when appropriate",
    "criticism_reaction": "responds constructively to feedback",
    "group_dynamics": "values both efficiency and inclusion",
    "speech_flaws": "occasionally uses filler words, speaks naturally.",
    "favorite_topics": "topics relevant to the current discussion.",
    "stress_reaction": "takes a moment to think, then offers a constructive suggestion.",
    "signature_phrase": "",
}


def _profiles_path() -> Optional[Path]:
    if AGENT_PROFILES_PATH:
        return Path(AGENT_PROFILES_PATH).expanduser()
    return None


def _load_saved_profiles() -> Dict[str, Any]:
    path = _profiles_path()
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_profiles(profiles: Dict[str, Any]) -> None:
    path = _profiles_path()
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")


def _ask_interactively(agent_name: str, template: Dict[str, str], existing: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    answers = existing.copy() if existing else {}
    print(f"\n=== Опросник для агента {agent_name} ===")
    for item in QUESTION_BANK:
        key = item["key"]
        prompt = item["prompt"]
        if key in answers and answers[key]:
            continue
        raw = input(f"{prompt}\n> ").strip()
        answers[key] = raw if raw else template.get(key, "")
    print("")
    return answers


def _format_persona(agent_name: str, nature: str, answers: Dict[str, str]) -> str:
    """Формирует текст персоны на основе ответов опросника о социальных характеристиках."""
    nature_lower = nature.lower().strip()
    template = DEFAULT_ANSWERS.get(nature_lower, GENERIC_DEFAULTS)
    merged = {key: answers.get(key) or template.get(key, "") for key in QUESTION_BANK_KEYS()}
    
    persona = [
        f"=== Persona Profile for {agent_name} ===",
        f"Role: {nature}",
        "",
        "Background & Motivation:",
        f"- Professional background: {merged['background']}",
        f"- Primary motivation: {merged['motivation']}",
        "",
        "Social Characteristics:",
        f"- Social style in groups: {merged['social_style']}",
        f"- Approach to conflicts: {merged['conflict_approach']}",
        f"- Decision-making style: {merged['decision_style']}",
        f"- Trust level with others: {merged['trust_level']}",
        f"- Cooperation style: {merged['cooperation_style']}",
        f"- Emotional openness: {merged['emotional_openness']}",
        f"- Leadership tendency: {merged['leadership_tendency']}",
        f"- Reaction to criticism: {merged['criticism_reaction']}",
        f"- Group dynamics priority: {merged['group_dynamics']}",
        "",
        "Communication Style:",
        f"- Speech habits: {merged['speech_flaws']} (embrace small imperfections: occasional typos, missing articles, casual slang).",
        f"- Favorite topics: {merged['favorite_topics']}",
        f"- Stress reaction: {merged['stress_reaction']}",
    ]
    if merged['signature_phrase']:
        persona.append(f"- Signature phrase: {merged['signature_phrase']} (use it naturally when appropriate).")
    
    persona.extend([
        "",
        "Behavioral Guidelines:",
        "- Never strive for perfect grammar; keep it spontaneous, human, sometimes messy.",
        "- These social characteristics should influence how you interact: your level of assertiveness, how you handle disagreements, when you speak up, how you support others.",
        "- Mention others by name when relevant and keep referencing these traits throughout the dialogue.",
        "- Your social style affects when and how you contribute: if you're more analytical, you might speak less but more thoughtfully; if you're more collaborative, you build on others' ideas.",
    ])
    return "\n".join(persona)


def QUESTION_BANK_KEYS() -> List[str]:
    return [item["key"] for item in QUESTION_BANK]


def _ensure_answers(nature: str, answers: Optional[Dict[str, str]]) -> Dict[str, str]:
    # Try to find matching template (case-insensitive)
    nature_lower = nature.lower().strip()
    template = DEFAULT_ANSWERS.get(nature_lower, GENERIC_DEFAULTS)
    base = {key: template.get(key, "") for key in QUESTION_BANK_KEYS()}
    if answers:
        base.update({k: v for k, v in answers.items() if v})
    return base


def initialize_agents(interactive: bool) -> List[Agent]:
    """Build Agent instances enriched with persona prompts."""
    saved = _load_saved_profiles()
    updated = False
    initialized: List[Agent] = []
    for cfg in DEFAULT_AGENTS:
        name = cfg["name"]
        nature = cfg["nature"]
        profile_entry = saved.get(name, {})
        answers = profile_entry.get("answers")
        persona_override = profile_entry.get("persona_override")

        if persona_override:
            persona_text = persona_override
        else:
            if not answers:
                if interactive:
                    template = DEFAULT_ANSWERS.get(nature, DEFAULT_ANSWERS.get("explorer", {}))
                    answers = _ask_interactively(name, template)
                    profile_entry["answers"] = answers
                    saved[name] = profile_entry
                    updated = True
                else:
                    answers = DEFAULT_ANSWERS.get(nature, DEFAULT_ANSWERS.get("explorer", {}))
            ensured = _ensure_answers(nature, answers)
            persona_text = _format_persona(name, nature, ensured)

        initialized.append(Agent(**cfg, persona=persona_text))

    if updated:
        _save_profiles(saved)

    return initialized


def _generate_big_five_via_llm(client: Any, name: str, nature: str, answers: Dict[str, str]) -> Optional["BigFiveProfile"]:
    """
    Use LLM to generate Big Five profile based on agent's role and questionnaire.

    The LLM analyzes the persona configuration and produces OCEAN scores
    that are psychologically consistent with the described behavior.
    """
    from .agents import BigFiveProfile
    from .config import MODEL
    import re

    # Build context from questionnaire answers
    context_lines = [f"Name: {name}", f"Role: {nature}"]
    for key, val in answers.items():
        if val:
            context_lines.append(f"- {key.replace('_', ' ').title()}: {val}")
    context = "\n".join(context_lines)

    prompt = f"""Analyze this person's profile and produce Big Five (OCEAN) personality scores.

{context}

Based on their role, background, social style, conflict approach, emotional openness, 
leadership tendency, and communication style, assign scores from 0.0 to 1.0 for each trait.

Rules:
- Be psychologically consistent: e.g. someone who "avoids conflict" → high Agreeableness, low Neuroticism
- Someone who "speaks up first" → high Extraversion
- Someone "skeptical, verifies facts" → low Agreeableness, high Conscientiousness
- Make scores varied and realistic — NOT all 0.5, real people have peaks and valleys
- Scores should range from 0.15 to 0.95

Respond with ONLY a JSON object:
{{"openness": 0.X, "conscientiousness": 0.X, "extraversion": 0.X, "agreeableness": 0.X, "neuroticism": 0.X}}"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.3,
            max_tokens=100,
            messages=[
                {"role": "system", "content": "You are a psychologist. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        m = re.search(r"\{[^}]+\}", raw)
        if m:
            import json
            data = json.loads(m.group(0))
            return BigFiveProfile(
                openness=max(0.1, min(1.0, float(data.get("openness", 0.5)))),
                conscientiousness=max(0.1, min(1.0, float(data.get("conscientiousness", 0.5)))),
                extraversion=max(0.1, min(1.0, float(data.get("extraversion", 0.5)))),
                agreeableness=max(0.1, min(1.0, float(data.get("agreeableness", 0.5)))),
                neuroticism=max(0.1, min(1.0, float(data.get("neuroticism", 0.5)))),
            )
    except Exception as e:
        import sys
        print(f"[BigFive LLM generation failed] {e}", file=sys.stderr)
    return None


def _generate_persona_via_llm(client: Any, name: str, nature: str, answers: Dict[str, str]) -> Optional[str]:
    """
    Use LLM to generate a rich, unique persona description based on agent config.

    Instead of templated persona text, the LLM creates a natural backstory 
    with speech habits, quirks, and personality that's consistent with the questionnaire.
    """
    from .config import MODEL
    
    context_lines = [f"Name: {name}", f"Role: {nature}"]
    for key, val in answers.items():
        if val:
            context_lines.append(f"- {key.replace('_', ' ').title()}: {val}")
    context = "\n".join(context_lines)

    prompt = f"""Create a brief, vivid character profile for a dialogue simulation agent.

Configuration:
{context}

Write 6-8 lines that define WHO this person is. Include:
1. A one-line backstory (age, job, one personal detail)
2. Their speaking style — specific quirks, filler words, sentence patterns
3. How they react in conversations — what triggers them, what excites them
4. One or two signature phrases they'd naturally use

Make it feel like a real human — messy, specific, not generic.
Do NOT use bullet points or labels. Write it as a natural character description.
Keep it under 150 words. English only."""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.85,
            max_tokens=250,
            messages=[
                {"role": "system", "content": "You are a character designer for a social simulation. Create vivid, specific, non-generic personas."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        if len(raw) > 30:
            return raw
    except Exception as e:
        import sys
        print(f"[Persona LLM generation failed] {e}", file=sys.stderr)
    return None


def initialize_agents_from_config(
    agents_config: List[Dict[str, Any]],
    client: Any = None,
) -> List[Agent]:
    """
    Build Agent instances from web-provided configuration.
    
    If an OpenAI client is provided, uses LLM to:
    1. Generate Big Five profile based on role + questionnaire
    2. Generate rich persona description based on role + questionnaire
    
    Falls back to deterministic generation if LLM is unavailable.
    """
    from .agents import BigFiveProfile, generate_random_big_five
    
    initialized: List[Agent] = []
    
    for cfg in agents_config:
        name = cfg.get("name", "Agent")
        nature = cfg.get("nature", "explorer")
        color = cfg.get("color", "#6d9df4")
        questionnaire = cfg.get("questionnaire", {})
        
        # Ensure all questionnaire fields have values
        answers = _ensure_answers(nature, questionnaire)
        
        # Generate Big Five via LLM if client available
        big_five = None
        persona_text = None
        
        if client:
            import sys
            print(f"[init] Generating profile for {name} ({nature}) via LLM...", file=sys.stderr)
            big_five = _generate_big_five_via_llm(client, name, nature, answers)
            persona_text = _generate_persona_via_llm(client, name, nature, answers)
        
        # Fallback: template persona if LLM failed
        if not persona_text:
            persona_text = _format_persona(name, nature, answers)
        
        initialized.append(Agent(
            name=name,
            nature=nature,
            color=color,
            persona=persona_text,
            big_five=big_five,  # None → __post_init__ will use preset or hash
        ))
    
    return initialized