from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import anthropic
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
    client: anthropic.Anthropic
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
        
        sys = (
            f"REAL SITUATION: {self.env_context}\n\n"
            + "You are in a REAL meeting/discussion right now. This is happening in real-time. "
            + "You are a REAL PERSON, not an AI. Act naturally, react authentically, think out loud.\n\n"
            + agent.system_prompt
            + "\n\n"
            + SESSION_GOAL
            + phase_block
            + university_block
            + mood_block
            + "\n\n"
            + f"Other people in this conversation: {participants}\n"
            + "You're all sitting together (or in a video call) having this discussion right now.\n"
            + "When you speak, address one specific person by name - pick who you're naturally responding to or who you want to engage with.\n"
            + "Don't address everyone at once - talk to one person, like in a real conversation.\n"
            + "IMPORTANT: Always use INFORMAL 'you' (ты) when addressing others - this is a casual, friendly conversation.\n"
            + "Never use formal language - speak naturally and informally, like real people do in real meetings.\n"
            + ("\nPeople you haven't talked to much recently: " + ", ".join(target_order[:3]) + " - consider engaging with them.\n" if target_order else "\n")
            + "\nWhat's been said so far (this is the actual conversation happening right now):\n"
            + (content_summary or "(This is the start of the conversation - you're all just beginning to discuss this.)")
            + "\n\n"
            + "YOUR TURN TO SPEAK:\n"
            + "You're a real person in a real conversation. React naturally to what's been said.\n"
            + "- Always use INFORMAL 'you' (ты) when addressing others - never formal language\n"
            + "- If someone said something interesting, react to it naturally\n"
            + "- If you have a question, ask it like a real person would: 'What do you think?', 'How would you approach this?'\n"
            + "- If you're thinking about something, share your thoughts naturally\n"
            + "- If you agree or disagree, say so authentically: 'You're right about that', 'I see what you mean'\n"
            + "- Reference specific things people said - show you're actually listening\n"
            + "- Sound like you're really there, really engaged, really thinking\n"
            + "- Don't sound scripted or robotic - be spontaneous and natural\n"
            + "- Speak informally and casually, like friends/colleagues do\n"
            + ("- REMEMBER: Keep it SIMPLE - use everyday words, short sentences, casual language\n" if "University" in self.env_context or "university" in self.env_context.lower() else "")
            + "\n"
            + STRUCTURE_INSTRUCTION
            + "\n\n"
            + HUMAN_STYLE
            + "\n\n"
            + "Respond by calling submit_agent_turn with your natural, human reply."
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": 
                "It's your turn to speak in this real conversation. "
                "You're a real person, reacting naturally to what others just said. "
                "Speak authentically - show you're listening, thinking, and genuinely engaged. "
                "Reference what people actually said, react naturally, and be yourself. "
                "This isn't a script - it's a real conversation happening right now. "
                "Respond naturally by calling submit_agent_turn with your authentic reply, tone, emotion, and who you're addressing."}
        ]

    def model_turn(self, messages, allowed_targets, agent=None):
        import json, re, sys, time
        from pydantic import ValidationError
        
        def _fix_target(parsed, speaker_name=None):
            if speaker_name and parsed.target == speaker_name:
                parsed.target = allowed_targets[0] if allowed_targets else None
            elif parsed.target not in allowed_targets:
                parsed.target = allowed_targets[0] if allowed_targets else None
            return parsed

        current_agent = agent
        if not current_agent and messages and messages[0].get("role") == "system":
            for a in self.agents:
                if a.system_prompt in messages[0].get("content", ""):
                    current_agent = a
                    break

        # Anthropic tool definition
        tools = [
            {
                "name": "submit_agent_turn",
                "description": "Strictly structured agent reply.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "reply": {"type": "string"},
                        "tone": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                        "emotion": {"type": "string"},
                        "target": {"type": "string", "enum": allowed_targets},
                    },
                    "required": ["reply", "tone", "emotion", "target"],
                },
            }
        ]

        max_retries = 3
        retry_delay = 20

        def make_api_call(call_func, description="API call"):
            for attempt in range(max_retries):
                try:
                    return call_func()
                except anthropic.RateLimitError as e:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (attempt + 1)
                        print(f"[Rate limit] {description}: Waiting {wait_time}s before retry (attempt {attempt + 1}/{max_retries})...", file=sys.stderr)
                        time.sleep(wait_time)
                    else:
                        print(f"[Rate limit] {description}: All {max_retries} retry attempts failed.", file=sys.stderr)
                        raise

        # Extract system prompt and user messages for Anthropic format
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append(msg)
        if not user_messages:
            user_messages = [{"role": "user", "content": "It's your turn to speak."}]

        # A) Anthropic messages + tools (preferred)
        try:
            resp = make_api_call(
                lambda: self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=system_prompt,
                    messages=user_messages,
                    tools=tools,
                    tool_choice={"type": "auto"},
                ),
                "Tools path API call"
            )

            # Check for tool_use blocks
            for block in resp.content:
                if block.type == "tool_use" and block.name == "submit_agent_turn":
                    try:
                        parsed = AgentTurn.model_validate(block.input)
                        return _fix_target(parsed, agent.name if agent else None)
                    except (ValidationError, KeyError, AttributeError) as e:
                        print(f"[tool call parsing error] {e}", file=sys.stderr)

            # If no tool_use, try parsing JSON from text blocks
            for block in resp.content:
                if block.type == "text":
                    content = block.text
                    json_patterns = [
                        r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",
                        r"\{[^}]*\"reply\"[^}]*\"tone\"[^}]*\"emotion\"[^}]*\"target\"[^}]*\}",
                    ]
                    for pattern in json_patterns:
                        m = re.search(pattern, content, re.S | re.I)
                        if m:
                            try:
                                parsed = AgentTurn.model_validate_json(m.group(0))
                                return _fix_target(parsed, agent.name if agent else None)
                            except (ValidationError, json.JSONDecodeError):
                                continue
                    print(f"[no valid JSON in content] {content[:300]}", file=sys.stderr)
        except Exception as e:
            print(f"[tools path error] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # B) Fallback: Direct JSON request without tools
        fallback_system = (
            (current_agent.system_prompt if current_agent else "You are a helpful AI assistant.") +
            f"\n\nEnvironment: {self.env_context}\n\n" +
            "You must respond with ONLY a valid JSON object containing: reply (your message), tone (positive/neutral/negative), emotion (one word), and target (who you're addressing).\n" +
            f"Available targets: {', '.join(allowed_targets) if allowed_targets else 'none'}\n" +
            "Keep your reply natural, human-like, and short (1-2 sentences)."
        )
        recent_lines = "\n".join(
            f"{h['speaker']}→{h.get('target', 'all')}: {h['reply']}" for h in self.history[-5:]
        ) if self.history else "Starting conversation."
        fallback_user = (
            f"Recent conversation:\n{recent_lines}"
            "\n\nGenerate your response as a JSON object with keys: reply, tone, emotion, target."
        )
        try:
            resp = make_api_call(
                lambda: self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=fallback_system,
                    messages=[{"role": "user", "content": fallback_user}],
                ),
                "JSON request path API call"
            )
            raw = resp.content[0].text if resp.content else ""
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.S)
            raw_json = m.group(0) if m else raw.strip()
            if raw_json:
                try:
                    parsed = AgentTurn.model_validate_json(raw_json)
                    return _fix_target(parsed, agent.name if agent else None)
                except (ValidationError, json.JSONDecodeError) as e:
                    print(f"[json request path parsing error] {e}", file=sys.stderr)
        except Exception as e:
            print(f"[json request path error] {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        # C) Final attempt
        try:
            recent_context = "\n".join(
                f"{h['speaker']}→{h.get('target', 'all')}: {h['reply']}"
                for h in self.history[-8:]
            ) if self.history else "Starting conversation."

            final_system = (
                (current_agent.system_prompt if current_agent else "You are a real person in a real conversation.") +
                f"\n\nREAL SITUATION: {self.env_context}\n\n" +
                "You MUST respond with ONLY a valid JSON object containing:\n"
                "- reply: your natural response (1-2 sentences)\n"
                "- tone: positive, neutral, or negative\n"
                "- emotion: one English word\n"
                f"- target: one of {allowed_targets if allowed_targets else ['all']}\n"
            )
            resp = make_api_call(
                lambda: self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    system=final_system,
                    messages=[{"role": "user", "content":
                        f"Conversation so far:\n{recent_context}\n\nYour turn. Respond as JSON only."}],
                ),
                "Final attempt API call"
            )
            raw = resp.content[0].text if resp.content else ""
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.S)
            raw_json = m.group(0) if m else raw.strip()
            if raw_json:
                try:
                    parsed = AgentTurn.model_validate_json(raw_json)
                    return _fix_target(parsed, agent.name if agent else None)
                except (ValidationError, json.JSONDecodeError):
                    pass

            if raw:
                lines = raw.split('\n')
                reply_text = lines[0].strip() if lines else raw[:200].strip()
                return AgentTurn(
                    reply=reply_text or "I'm processing what was said.",
                    tone="neutral",
                    emotion="neutral",
                    target=allowed_targets[0] if allowed_targets else None,
                )
        except Exception as final_error:
            print(f"[final attempt error] {final_error}", file=sys.stderr)
            raise RuntimeError(
                f"Failed to generate agent response via API after multiple attempts. "
                f"Last error: {str(final_error)}. "
                f"Please check your ANTHROPIC_API_KEY and network connection."
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

        # === VALIDATION ===
        turn_data = {
            "speaker": speaker.name,
            "target": target,
            "reply": parsed.reply,
            "tone": parsed.tone,
            "emotion": parsed.emotion,
        }
        
        validation_results = []
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
