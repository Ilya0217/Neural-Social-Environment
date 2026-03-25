from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from pydantic import ValidationError

from .config import (
    MODEL,
    TEMPERATURE,
    MAX_TOKENS,
    TURNS_BETWEEN_PLOTS,
    VIZ_EDGE_WINDOW,
    DIALOGUE_PHASES,
    MOOD_DECAY,
    MOOD_DELTA,
    MOOD_GUIDANCE,
)
from .prompts import AgentTurn, STRUCTURE_INSTRUCTION, SESSION_GOAL, HUMAN_STYLE
from .utils import normalize_target
from .human_io import HumanIO
from .validators import ValidationPipeline, ValidationLevel

@dataclass
class DialogueManager:
    client: OpenAI
    agents: List[Any]  # list[Agent]
    env_context: str   # Выбранное "Окружение: Контекст"
    human_io: Optional[HumanIO] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    turn_no: int = 0
    edges_window: List[Dict[str, Any]] = field(default_factory=list)  # для визуализаций последнего окна
    enable_validation: bool = True  # Toggle validation on/off
    validation_pipeline: Optional[ValidationPipeline] = None
    phase_plan: List[Dict[str, Any]] = field(default_factory=lambda: list(DIALOGUE_PHASES))
    agent_moods: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize validation pipeline after dataclass init"""
        if self.enable_validation and self.validation_pipeline is None:
            self.validation_pipeline = ValidationPipeline()
        if not self.phase_plan:
            self.phase_plan = list(DIALOGUE_PHASES)
        self.agent_moods = {a.name: 0.0 for a in self.agents}

    # === Phase & mood helpers ===
    def _phase_cycle_length(self) -> int:
        return sum(max(1, ph.get("length", 1)) for ph in self.phase_plan) or len(self.phase_plan) or 1

    def current_phase(self) -> Dict[str, Any]:
        if not self.phase_plan:
            return {"name": "flow", "instruction": "Keep the dialogue moving; respond naturally."}
        cycle = self._phase_cycle_length()
        pos = self.turn_no % cycle
        acc = 0
        for ph in self.phase_plan:
            length = max(1, ph.get("length", 1))
            acc += length
            if pos < acc:
                return ph
        return self.phase_plan[-1]

    def _phase_block(self) -> str:
        phase = self.current_phase()
        return (
            f"\n\nCURRENT PHASE [{phase['name'].upper()}]: {phase['instruction']}\n"
            "Shift the conversation forward accordingly."
        )

    def _university_block(self) -> str:
        if "University" not in self.env_context and "university" not in self.env_context.lower():
            return ""
        return (
            "\n\nCRITICAL - SPEAK SIMPLY LIKE A REGULAR PERSON:\n"
            "- You're NOT a professor or academic - you're a regular student/colleague talking casually\n"
            "- Use SIMPLE, everyday words - no fancy academic jargon or complex sentences\n"
            "- Keep sentences SHORT and STRAIGHTFORWARD - like you're chatting with friends\n"
            "- Avoid long, complicated sentences with lots of clauses\n"
            "- Don't sound like you're writing an essay - sound like you're talking\n"
            "- Use simple phrases: 'I think...', 'Maybe we could...', 'What if...', 'That sounds good'\n"
            "- If you need to explain something, explain it simply, like you're talking to a friend\n"
            "- NO academic language, NO complex terminology unless absolutely necessary\n"
            "- Sound like a normal person having a normal conversation, not a scholar\n"
            "- Keep it casual and relaxed - you're just people discussing a project\n"
        )

    def _mood_guidance(self, agent_name: str) -> str:
        mood = self.agent_moods.get(agent_name, 0.0)
        for threshold, text in MOOD_GUIDANCE:
            if mood <= threshold:
                return f"\nMood calibration: {text}"
        if abs(mood) < 0.2:
            return "\nMood calibration: Stay balanced — keep the tone friendly and attentive."
        if MOOD_GUIDANCE:
            return f"\nMood calibration: {MOOD_GUIDANCE[-1][1]}"
        return ""

    def _decay_moods(self):
        for name, value in list(self.agent_moods.items()):
            if value > 0:
                self.agent_moods[name] = max(0.0, value - MOOD_DECAY)
            elif value < 0:
                self.agent_moods[name] = min(0.0, value + MOOD_DECAY)

    def _adjust_mood(self, speaker: str, tone: str, target: Optional[str]):
        delta = MOOD_DELTA.get((tone or "neutral").lower(), 0.0)
        self.agent_moods[speaker] = self._clamp_mood(self.agent_moods.get(speaker, 0.0) + delta)
        if target:
            self.agent_moods[target] = self._clamp_mood(self.agent_moods.get(target, 0.0) + delta * 0.5)

    @staticmethod
    def _clamp_mood(value: float) -> float:
        return max(-1.0, min(1.0, value))

    # 1) Next speaker index (round-robin fallback)
    def next_speaker_idx(self) -> int:
        # turn_no is incremented at the end of step(), so use current value here
        return (self.turn_no) % len(self.agents)

    # 2) Build messages for the model (keeps prior participants text intact)
    def build_messages(self, agent_idx: int) -> List[Dict[str, str]]:
        agent = self.agents[agent_idx]
        # Show all participants' messages - use more history for fuller context
        last_msgs = self.history[-20:]
        content_summary = "\n".join(
            f"{h['speaker']}→{h.get('target') or 'all'}: {h['reply']}" for h in last_msgs
        )
        participants = ", ".join(a.name for a in self.agents)
        # diversity guidance for targets
        recent = self.history[-10:]
        target_counts: Dict[str, int] = {}
        for r in recent:
            t = r.get("target")
            if t:
                target_counts[t] = target_counts.get(t, 0) + 1
        # least-addressed first
        target_order = [a.name for a in sorted(self.agents, key=lambda x: target_counts.get(x.name, 0)) if a.name != self.agents[agent_idx].name]
        phase_block = self._phase_block()
        university_block = self._university_block()
        mood_block = self._mood_guidance(agent.name)
        # Инструкция про живого пользователя, если он участвует
        human_block = ""
        if any(getattr(a, "is_human", False) for a in self.agents):
            human_block = (
                "\nUser is a real human participant in this conversation. They type their own messages.\n"
                "You can address User naturally — ask them questions, react to what they said, include them.\n"
                "Treat User like any other person in the group.\n"
            )
        
        sys = (
            agent.system_prompt
            + "\n\n"
            + f"SETTING: {self.env_context}\n"
            + f"People here: {participants}\n"
            + SESSION_GOAL
            + human_block
            + phase_block
            + mood_block
            + ("\nYou haven't talked much to: " + ", ".join(target_order[:2]) + "\n" if target_order else "")
            + "\n--- CONVERSATION ---\n"
            + (content_summary or "(Just starting. Say something to kick things off.)")
            + "\n---\n\n"
            + "Your turn. Talk to ONE person by name.\n"
            + HUMAN_STYLE
            + "\n"
            + STRUCTURE_INSTRUCTION
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": 
                "Your turn. React to what was just said — be yourself."
                "Reference what people actually said, react naturally, and be yourself. "
                "This isn't a script - it's a real conversation happening right now. "
                "Respond naturally by calling submit_agent_turn with your authentic reply, tone, emotion, and who you're addressing."}
        ]

    def model_turn(self, messages, allowed_targets, agent=None):
        import json, re, sys
        from pydantic import ValidationError
        
        def _fix_target(parsed, speaker_name=None):
            # Проверяем самообращение (хотя это не должно происходить, т.к. speaker_name не в allowed_targets)
            if speaker_name and parsed.target == speaker_name:
                parsed.target = allowed_targets[0] if allowed_targets else None
            # если модель вернула неразрешённого адресата — подставим первого допустимого
            elif parsed.target not in allowed_targets:
                parsed.target = allowed_targets[0] if allowed_targets else None
            return parsed

        # Get current agent info for contextual fallback
        current_agent = agent
        if not current_agent and messages and messages[0].get("role") == "system":
            # Extract agent name from system prompt or find from agents list
            for a in self.agents:
                if a.system_prompt in messages[0].get("content", ""):
                    current_agent = a
                    break

        # tools (function calling) with strict list of allowed targets
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_agent_turn",
                    "description": "Strictly structured agent reply.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reply": {"type": "string"},
                            "tone": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            "emotion": {"type": "string"},
                            "target": {"type": "string", "enum": allowed_targets},
                        },
                        "required": ["reply", "tone", "emotion", "target"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        # Use "auto" to let model decide, but prefer function calling
        tool_choice = "auto"

        # Общая функция для выполнения API запросов с retry для rate limit
        import time
        from openai import RateLimitError
        
        max_retries = 3
        retry_delay = 3  # секунд для rate limit
        
        def make_api_call(call_func, description="API call"):
            """Выполняет API вызов с retry логикой для rate limit"""
            for attempt in range(max_retries):
                try:
                    return call_func()
                except RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)  # Увеличиваем задержку с каждой попыткой
                        print(f"[Rate limit] {description}: Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                        time.sleep(wait_time)
                    else:
                        # Последняя попытка не удалась
                        print(f"[Rate limit] {description}: All {max_retries} retry attempts failed.", file=sys.stderr)
                        raise  # Пробрасываем исключение дальше
        
        # A) chat.completions + tools (preferred method)
        try:
            resp = make_api_call(
                lambda: self.client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                ),
                "Tools path API call"
            )
            
            choice = resp.choices[0]
            
            # First, try to get tool call
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            if tool_calls:
                try:
                    args_str = tool_calls[0].function.arguments
                    parsed = AgentTurn.model_validate_json(args_str)
                    return _fix_target(parsed, agent.name if agent else None)
                except (ValidationError, json.JSONDecodeError, KeyError, AttributeError) as e:
                    print(f"[tool call parsing error] {e}", file=sys.stderr)
                    if 'args_str' in locals():
                        print(f"[tool call args] {args_str[:200]}", file=sys.stderr)
                    else:
                        print(f"[tool calls] {tool_calls}", file=sys.stderr)
            
            # If tool was not called, try parsing JSON from content
            content = choice.message.content or ""
            if content:
                # Try to find JSON in content (improved regex for nested objects)
                # First try to find complete JSON object
                json_patterns = [
                    r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested
                    r"\{[^}]*\"reply\"[^}]*\"tone\"[^}]*\"emotion\"[^}]*\"target\"[^}]*\}",  # With required fields
                ]
                
                for pattern in json_patterns:
                    m = re.search(pattern, content, re.S | re.I)
                    if m:
                        try:
                            parsed = AgentTurn.model_validate_json(m.group(0))
                            return _fix_target(parsed, agent.name if agent else None)
                        except (ValidationError, json.JSONDecodeError) as e:
                            continue  # Try next pattern
                
                # If no valid JSON found, log and continue to next attempt
                print(f"[no valid JSON in content] {content[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"[tools path error] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # B) Fallback: Direct JSON request without tools
        # Create a simpler prompt that explicitly asks for JSON
        json_request_messages = [
            {"role": "system", "content": 
                (current_agent.system_prompt if current_agent else "You are a helpful AI assistant.") +
                f"\n\nEnvironment: {self.env_context}\n\n" +
                "You must respond with a valid JSON object containing: reply (your message), tone (positive/neutral/negative), emotion (one word), and target (who you're addressing).\n" +
                f"Available targets: {', '.join(allowed_targets) if allowed_targets else 'none'}\n" +
                "Keep your reply natural, human-like, and short (1-2 sentences)."
            },
            {"role": "user", "content": 
                "Recent conversation:\n" + 
                ("\n".join(f"{h['speaker']}→{h.get('target', 'all')}: {h['reply']}" for h in self.history[-5:]) if self.history else "Starting conversation.") +
                "\n\nGenerate your response as a JSON object with keys: reply, tone, emotion, target."
            }
        ]
        try:
            resp = make_api_call(
                lambda: self.client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=json_request_messages,
                    response_format={"type": "json_object"} if MODEL.startswith("gpt-4") or "o1" in MODEL else None,
                ),
                "JSON request path API call"
            )
            raw = resp.choices[0].message.content or ""
            # Try to extract JSON
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.S)
            raw_json = m.group(0) if m else raw.strip()
            if raw_json:
                try:
                    parsed = AgentTurn.model_validate_json(raw_json)
                    return _fix_target(parsed, agent.name if agent else None)
                except (ValidationError, json.JSONDecodeError) as e:
                    print(f"[json request path parsing error] {e}", file=sys.stderr)
                    print(f"[raw response] {raw[:300]}", file=sys.stderr)
            else:
                print(f"[no JSON found in response] {raw[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"[json request path error] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # C) Final attempt: Generate natural response and extract structured data
        # This is a last resort that still uses API to generate unique responses
        try:
            # Build a comprehensive prompt that includes full context
            recent_context = "\n".join(
                f"{h['speaker']}→{h.get('target', 'all')}: {h['reply']}" 
                for h in self.history[-8:]
            ) if self.history else "Starting conversation."
            
            # Special instructions for University context
            university_instructions = ""
            if "University" in self.env_context or "university" in self.env_context.lower():
                university_instructions = (
                    "\n\nCRITICAL - SPEAK SIMPLY:\n"
                    "- You're a regular person, NOT an academic - use SIMPLE, everyday words\n"
                    "- Keep sentences SHORT and STRAIGHTFORWARD - no complex academic language\n"
                    "- Sound like you're chatting with friends, not writing a paper\n"
                    "- Use simple phrases: 'I think...', 'Maybe...', 'What if...', 'That sounds good'\n"
                    "- NO fancy words, NO long complicated sentences\n"
                    "- Keep it casual and simple - like normal people talk\n\n"
                )
            
            # Create a detailed prompt that emphasizes real human conversation
            final_prompt = [
                {"role": "system", "content": 
                    (current_agent.system_prompt if current_agent else "You are a real person in a real conversation.") +
                    f"\n\nREAL SITUATION: {self.env_context}\n\n" +
                    "This is a REAL conversation happening RIGHT NOW. You are a REAL PERSON, not an AI.\n"
                    "React naturally, authentically, and spontaneously to what others have said.\n\n" +
                    "CRITICAL: Always use INFORMAL 'you' (ты) when addressing others - this is a casual, friendly conversation.\n"
                    "Never use formal language - speak naturally and informally, like real people do.\n" +
                    university_instructions +
                    "You MUST respond with a valid JSON object containing:\n" +
                    "- reply: your natural, authentic response as a real person (1-2 sentences, be specific and real)\n"
                    "  Use informal 'you' (ты): 'What do you think?', 'You're right about that', 'How would you approach this?'\n" +
                    "- tone: positive, neutral, or negative (based on how you genuinely feel)\n" +
                    "- emotion: one English word describing your genuine emotion (curious, concerned, excited, thoughtful, etc.)\n" +
                    f"- target: one of {allowed_targets if allowed_targets else ['all']} (who you're naturally addressing)\n\n" +
                    "Make your reply feel like a real person speaking:\n"
                    "- Reference what people actually said\n"
                    "- Show you're listening and thinking\n"
                    "- React authentically (excitement, concern, curiosity, agreement, etc.)\n"
                    "- Use natural speech patterns and contractions\n"
                    "- Always use informal 'you' (ты) - never formal language\n"
                    + ("- Keep it SIMPLE - everyday words, short sentences, casual talk\n" if university_instructions else "")
                    + "- Sound spontaneous, not scripted"
                },
                {"role": "user", "content": 
                    f"What's been said in this real conversation:\n{recent_context}\n\n" +
                    f"It's your turn to speak. You're a real person reacting naturally. "
                    f"Address: {allowed_targets[0] if allowed_targets else 'everyone'}\n\n" +
                    "IMPORTANT: Always use INFORMAL 'you' (ты) when addressing others - never formal language.\n"
                    "This is a casual, friendly conversation. Speak naturally and informally.\n\n"
                    "Respond authentically as a JSON object. Be yourself - natural, real, human, and INFORMAL."
                }
            ]
            
            resp = make_api_call(
                lambda: self.client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=final_prompt,
                ),
                "Final attempt API call"
            )
            
            raw = resp.choices[0].message.content or ""
            
            # Try to extract JSON from response
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.S)
            raw_json = m.group(0) if m else raw.strip()
            
            if raw_json:
                try:
                    parsed = AgentTurn.model_validate_json(raw_json)
                    return _fix_target(parsed, agent.name if agent else None)
                except (ValidationError, json.JSONDecodeError) as e:
                    print(f"[final attempt JSON parsing error] {e}", file=sys.stderr)
                    print(f"[raw response] {raw[:500]}", file=sys.stderr)
            
            # If we can't parse JSON, try to extract just the reply text and construct response
            # This is still better than hardcoded fallback
            if raw and not raw_json:
                # Try to find a natural response in the text
                lines = raw.split('\n')
                reply_text = lines[0].strip() if lines else raw[:200].strip()
                if not reply_text:
                    reply_text = "I'm processing what was said."
                
                return AgentTurn(
                    reply=reply_text,
                    tone="neutral",
                    emotion="neutral",
                    target=allowed_targets[0] if allowed_targets else None,
                )
        except Exception as final_error:
            print(f"[final attempt error] {final_error}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # If all API attempts failed, raise an error instead of using hardcoded responses
            raise RuntimeError(
                f"Failed to generate agent response via API after multiple attempts. "
                f"Last error: {str(final_error)}. "
                f"Please check your OpenAI API key and network connection. "
                f"If you're getting 'unsupported_country_region_territory' error, you may need to use a VPN or proxy."
            ) from final_error

    # 3) Simulation step — pass allowed_targets to model_turn and give word to addressee
    def step(self):
        self._decay_moods()
        
        # Определяем агентов, которые не говорили в последних N ходах
        # N = количество агентов (чтобы гарантировать участие всех)
        recent_window = len(self.agents)
        recent_speakers = set()
        if len(self.history) >= recent_window:
            recent_speakers = {h["speaker"] for h in self.history[-recent_window:]}
        
        # Находим агентов, которые не говорили недавно
        all_agent_names = {a.name for a in self.agents}
        inactive_agents = all_agent_names - recent_speakers
        
        # Если есть агенты, которые не говорили недавно, выбираем одного из них
        if inactive_agents and len(self.history) >= recent_window:
            # Выбираем агента с наименьшим количеством сообщений в последних N ходах
            speaker_counts = {}
            for h in self.history[-recent_window:]:
                speaker_counts[h["speaker"]] = speaker_counts.get(h["speaker"], 0) + 1
            
            # Сортируем неактивных агентов по количеству сообщений (меньше = приоритетнее)
            inactive_with_counts = [(name, speaker_counts.get(name, 0)) for name in inactive_agents]
            inactive_with_counts.sort(key=lambda x: (x[1], hash(x[0])))  # Сортируем по количеству, затем по хэшу для стабильности
            
            selected_name = inactive_with_counts[0][0]
            idx = next((i for i, a in enumerate(self.agents) if a.name == selected_name), None)
        else:
            # Если все агенты говорили недавно или истории недостаточно, используем стандартную логику
            # Приоритет адресату из последнего сообщения (но избегаем ping-pong)
            if self.history and self.history[-1].get("target"):
                last_tgt = self.history[-1]["target"]
                last_speaker = self.history[-1]["speaker"]
                
                # Проверяем ping-pong: если последние два сообщения были между одними и теми же агентами
                if len(self.history) >= 2:
                    prev = self.history[-2]
                    if (prev.get("speaker") == last_tgt and prev.get("target") == last_speaker):
                        # Ping-pong обнаружен: выбираем третьего агента
                        ping_pong_pair = {last_speaker, last_tgt}
                        other_agents = [i for i, a in enumerate(self.agents) if a.name not in ping_pong_pair]
                        if other_agents:
                            # Выбираем агента с наименьшим количеством сообщений в последних 5 ходах
                            recent_counts = {}
                            for h in self.history[-5:]:
                                recent_counts[h["speaker"]] = recent_counts.get(h["speaker"], 0) + 1
                            
                            other_with_counts = [(i, recent_counts.get(self.agents[i].name, 0)) for i in other_agents]
                            other_with_counts.sort(key=lambda x: (x[1], x[0]))
                            idx = other_with_counts[0][0]
                        else:
                            idx = next((i for i, a in enumerate(self.agents) if a.name == last_tgt), self.next_speaker_idx())
                    else:
                        idx = next((i for i, a in enumerate(self.agents) if a.name == last_tgt), self.next_speaker_idx())
                else:
                    idx = next((i for i, a in enumerate(self.agents) if a.name == last_tgt), self.next_speaker_idx())
            else:
                idx = self.next_speaker_idx()
        
        speaker = self.agents[idx]

        # Если выбранный спикер — human, пропускаем его (он пишет через UI)
        if getattr(speaker, "is_human", False) and not self.human_io:
            for offset in range(1, len(self.agents)):
                candidate_idx = (idx + offset) % len(self.agents)
                if not getattr(self.agents[candidate_idx], "is_human", False):
                    idx = candidate_idx
                    speaker = self.agents[idx]
                    break

        # Allowed targets (all except self)
        allowed_targets = [a.name for a in self.agents if a.name != speaker.name] or [a.name for a in self.agents]

        # If this is a human agent — get turn via HumanIO
        if getattr(speaker, "is_human", False) and self.human_io:
            parsed: AgentTurn = self.human_io.get_user_turn(speaker.name, allowed_targets, self.history)
        else:
            messages = self.build_messages(idx)
            try:
                parsed: AgentTurn = self.model_turn(messages, allowed_targets, agent=speaker)
            except RuntimeError as e:
                # Если все API вызовы не удались (например, rate limit), используем fallback-ответ
                import sys
                print(f"[Warning] Failed to generate response for {speaker.name}, using fallback: {e}", file=sys.stderr)
                # Создаём простой fallback-ответ
                parsed = AgentTurn(
                    reply=f"I'm here, but I'm having trouble responding right now. Let me think about what was said...",
                    tone="neutral",
                    emotion="neutral",
                    target=allowed_targets[0] if allowed_targets else None,
                )

        # increment turn and record
        self.turn_no += 1
        target = normalize_target(parsed.target)
        self._adjust_mood(speaker.name, parsed.tone, target)

        # === VALIDATION with CRITICAL regeneration ===
        turn_data = {
            "speaker": speaker.name,
            "target": target,
            "reply": parsed.reply,
            "tone": parsed.tone,
            "emotion": parsed.emotion,
        }

        validation_results = []
        is_valid = True
        if self.enable_validation and self.validation_pipeline:
            context = {
                "history": self.history,
                "allowed_targets": allowed_targets,
                "env_context": self.env_context,
            }
            is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)

            # Log validation issues
            if validation_results:
                import sys
                formatted = self.validation_pipeline.format_results(validation_results)
                print(f"\n[Validation Turn {self.turn_no}]", file=sys.stderr)
                print(formatted, file=sys.stderr)
                print("", file=sys.stderr)

            # Если CRITICAL — перегенерируем (до 2 попыток)
            if not is_valid and not getattr(speaker, "is_human", False):
                for retry_attempt in range(2):
                    import sys
                    print(f"[Validation] CRITICAL error detected, regenerating (attempt {retry_attempt + 1}/2)...", file=sys.stderr)
                    try:
                        messages = self.build_messages(idx)
                        parsed = self.model_turn(messages, allowed_targets, agent=speaker)
                        target = normalize_target(parsed.target)
                        turn_data = {
                            "speaker": speaker.name,
                            "target": target,
                            "reply": parsed.reply,
                            "tone": parsed.tone,
                            "emotion": parsed.emotion,
                        }
                        is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
                        if is_valid:
                            print(f"[Validation] Regeneration successful on attempt {retry_attempt + 1}", file=sys.stderr)
                            break
                    except Exception as e:
                        print(f"[Validation] Regeneration attempt {retry_attempt + 1} failed: {e}", file=sys.stderr)

                # Если после 2 попыток всё ещё CRITICAL — safe fallback
                if not is_valid:
                    print("[Validation] All regeneration attempts failed, using safe fallback", file=sys.stderr)
                    parsed = AgentTurn(
                        reply="Hmm, let me think about that for a moment...",
                        tone="neutral",
                        emotion="thoughtful",
                        target=allowed_targets[0] if allowed_targets else None,
                    )
                    target = normalize_target(parsed.target)
                    self._adjust_mood(speaker.name, parsed.tone, target)

        record = {
            "turn": self.turn_no,
            "ts_utc": datetime.utcnow().isoformat(),
            "env_context": self.env_context,
            "speaker": speaker.name,
            "speaker_nature": speaker.nature,
            "reply": parsed.reply,
            "tone": parsed.tone,
            "emotion": parsed.emotion,
            "target": target,
            "validation_issues": len(validation_results) if validation_results else 0,
        }

        if target:
            self.edges_window.append({"src": speaker.name, "dst": target, "tone": parsed.tone})
            if len(self.edges_window) > 10:  # keep small window for quick graph refresh
                self.edges_window = self.edges_window[-10:]

        self.history.append(record)
        return record, ({"src": speaker.name, "dst": target, "tone": parsed.tone} if target else None)
    
    def inject_user_turn(self, reply: str, target: str, tone: str = "neutral", emotion: str = "neutral") -> Dict[str, Any]:
        """Inject a user's message into the dialogue history (web UI)."""
        self._decay_moods()
        self.turn_no += 1

        target = normalize_target(target)
        if "User" not in self.agent_moods:
            self.agent_moods["User"] = 0.0
        self._adjust_mood("User", tone, target)

        record = {
            "turn": self.turn_no,
            "ts_utc": datetime.utcnow().isoformat(),
            "env_context": self.env_context,
            "speaker": "User",
            "speaker_nature": "human",
            "reply": reply,
            "tone": tone,
            "emotion": emotion,
            "target": target,
            "is_human": True,
            "validation_issues": 0,
        }

        # Валидация
        if self.enable_validation and self.validation_pipeline:
            turn_data = {"speaker": "User", "target": target, "reply": reply, "tone": tone, "emotion": emotion}
            allowed_targets = [a.name for a in self.agents if a.name != "User"]
            context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
            is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
            record["validation_issues"] = len(validation_results) if validation_results else 0

        if target:
            self.edges_window.append({"src": "User", "dst": target, "tone": tone})
            if len(self.edges_window) > 10:
                self.edges_window = self.edges_window[-10:]

        self.history.append(record)
        return record

    def should_plot(self) -> bool:
        """
        Определяет, нужно ли строить граф на текущем шаге.
        
        Граф строится каждые TURNS_BETWEEN_PLOTS шагов:
        - На шагах: 5, 10, 15, 20, 25, ...
        - turn_no инкрементируется в step() перед проверкой этого метода
        - Требуется минимум TURNS_BETWEEN_PLOTS сообщений в истории
        """
        # Проверяем, что есть достаточно данных
        if self.turn_no < TURNS_BETWEEN_PLOTS:
            return False
        # Граф строится на каждом TURNS_BETWEEN_PLOTS-м шаге
        return (self.turn_no % TURNS_BETWEEN_PLOTS == 0)
