import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from pydantic import ValidationError

from ..config import (
    DIALOGUE_PHASES,
    HUMAN_TO_AGENT_RATIO,
    HUMAN_TURN_TIMEOUT,
    MAX_TOKENS,
    MODEL,
    MOOD_DECAY,
    MOOD_DELTA,
    MOOD_GUIDANCE,
    TEMPERATURE,
    TURNS_BETWEEN_PLOTS,
    VIZ_EDGE_WINDOW,
)
from ..utils import normalize_target
from .human_io import HumanIO
from .prompts import HUMAN_STYLE, SESSION_GOAL, STRUCTURE_INSTRUCTION, AgentTurn
from .validators import ValidationLevel, ValidationPipeline


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
    enable_cot: bool = False  # Inject Chain-of-Thought instruction into system prompt (for H6 experiments)
    strict_mode: bool = False  # If True, raise RuntimeError on API or persistent validation failure
                                # instead of silently substituting fallback replies. Used in scientific
                                # experiments where contaminated data destroys statistical inference.
    validation_pipeline: Optional[ValidationPipeline] = None
    phase_plan: List[Dict[str, Any]] = field(default_factory=lambda: list(DIALOGUE_PHASES))
    agent_moods: Dict[str, float] = field(default_factory=dict)
    # Multi-user experiment fields
    multi_user_mode: bool = False
    pending_human: Optional[str] = None
    pending_human_since: Optional[float] = None
    _human_turn_count: int = 0
    _human_to_agent_ratio: int = 2
    active_human_names: set[str] = field(default_factory=set)

    def __post_init__(self):
        """Initialize validation pipeline after dataclass init"""
        if self.enable_validation and self.validation_pipeline is None:
            self.validation_pipeline = ValidationPipeline()
        if not self.phase_plan:
            self.phase_plan = list(DIALOGUE_PHASES)
        self.agent_moods = {a.name: 0.0 for a in self.agents}
        # Multi-user setup
        humans = [a for a in self.agents if getattr(a, "is_human", False)]
        self.multi_user_mode = len(humans) > 1
        self._human_to_agent_ratio = HUMAN_TO_AGENT_RATIO

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

    def _cot_block(self) -> str:
        """Chain-of-Thought instruction (Wei et al. 2022). Inserted only when enable_cot=True.
        Used for H6 experiment: testing whether CoT prompting raises actionability_rate."""
        if not self.enable_cot:
            return ""
        return (
            "\n\nREASONING PROTOCOL (think step by step, internally, BEFORE you reply):\n"
            "1. Identify the most important point or open question in the latest message.\n"
            "2. Decide what is missing: a concrete next step, a constraint, an example, a risk, "
            "a counter-argument, or an assignment of responsibility.\n"
            "3. Compose a reply that adds exactly that missing piece — do not just acknowledge.\n"
            "Do not output the reasoning steps themselves — only the final reply.\n"
        )

    def _context_style_block(self) -> str:
        context = (self.env_context or "").lower()
        if "research seminar" in context or "planning experiments" in context:
            return (
                "\n\nSTYLE CALIBRATION FOR RESEARCH DISCUSSION:\n"
                "- Speak naturally, but think like a serious research collaborator.\n"
                "- Prefer concrete claims, variables, metrics, datasets, baselines, controls, or experimental next steps.\n"
                "- Avoid filler agreement unless you add evidence, a counterpoint, or a methodological action.\n"
            )
        if "university" in context or "coursework" in context:
            return (
                "\n\nSTYLE CALIBRATION FOR PROJECT WORK:\n"
                "- Sound like capable students or colleagues discussing a real project.\n"
                "- Be clear and natural, but not sloppy, random, or overdramatic.\n"
                "- Prefer concrete constraints, examples, trade-offs, and next steps.\n"
            )
        if "startup" in context or "mvp" in context:
            return (
                "\n\nSTYLE CALIBRATION FOR PRODUCT DECISIONS:\n"
                "- Keep the discussion practical: trade-offs, user impact, scope, sequencing, and risks.\n"
                "- Avoid vague brainstorming that never lands on a decision.\n"
            )
        return ""

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

    def _select_balanced_speaker_idx(self) -> int:
        """Pick the next speaker with soft address-following and strong anti-dominance bias."""
        if not self.agents:
            return 0

        candidate_indices = [
            i for i, agent in enumerate(self.agents)
            if not getattr(agent, "is_human", False) or self.human_io
        ]
        if not candidate_indices:
            return 0

        history_window = self.history[-max(len(self.agents) * 2, 8):]
        recent_speakers = [h["speaker"] for h in history_window]
        last_record = self.history[-1] if self.history else {}
        last_speaker = last_record.get("speaker")
        last_target = last_record.get("target")

        scores = []
        for idx in candidate_indices:
            speaker = self.agents[idx]
            name = speaker.name
            score = 0

            # Strong fairness bias: agents who were quiet recently go first.
            recent_count = recent_speakers.count(name)
            score += recent_count * 5

            # Penalize immediate repeats heavily.
            if recent_speakers and recent_speakers[-1] == name:
                score += 100
            if len(recent_speakers) >= 2 and recent_speakers[-2] == name:
                score += 25

            # Break persistent ping-pong pairs.
            if len(self.history) >= 2:
                prev = self.history[-2]
                if prev.get("speaker") == name and prev.get("target") == last_speaker:
                    score += 35
                if last_target == name and prev.get("speaker") == last_target and prev.get("target") == last_speaker:
                    score += 40

            # Small bonus, not a hard rule, for replying to the last addressee.
            if last_target == name:
                score -= 3

            # Prefer whoever has been silent the longest.
            last_spoke_idx = -1
            for hist_idx in range(len(self.history) - 1, -1, -1):
                if self.history[hist_idx]["speaker"] == name:
                    last_spoke_idx = hist_idx
                    break
            score += last_spoke_idx / 1000 if last_spoke_idx >= 0 else -10

            scores.append((score, idx))

        scores.sort(key=lambda item: (item[0], item[1]))
        return scores[0][1]

    def _align_reply_with_target(self, reply: str, speaker_name: str, target: Optional[str]) -> str:
        """Ensure the textual addressee matches the structured target."""
        if not reply or not target:
            return reply

        participant_names = [
            a.name for a in self.agents
            if a.name not in {speaker_name, target}
        ]
        if not participant_names:
            return reply

        escaped_names = "|".join(sorted((re.escape(name) for name in participant_names), key=len, reverse=True))
        if not escaped_names:
            return reply

        patterns = [
            rf"^(\s*)([{chr(34)}'“”‘’]?)(?:{escaped_names})(\b[\s,!:.-]*)",
            rf"^(\s*)(?:great point|good point|fair point|exactly|right|okay|well|so)\s*,\s*(?:{escaped_names})(\b[\s,!:.-]*)",
        ]

        for pattern in patterns:
            match = re.match(pattern, reply, flags=re.IGNORECASE)
            if not match:
                continue

            prefix = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
            suffix = match.group(match.lastindex) if match.lastindex else ""
            return f"{prefix}{target}{suffix}{reply[match.end():]}"

        return reply

    def _tokenize_reply(self, text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"\b[a-zA-Zа-яА-Я0-9]{4,}\b", text or "")
        }

    def _choose_fallback_target(
        self, speaker_name: str, allowed_targets: List[str], reply: str = ""
    ) -> Optional[str]:
        if not allowed_targets:
            return None

        for candidate in allowed_targets:
            if re.search(rf"\b{re.escape(candidate)}\b", reply or "", flags=re.IGNORECASE):
                return candidate

        target_counts: Dict[str, int] = {name: 0 for name in allowed_targets}
        for record in self.history[-12:]:
            target = record.get("target")
            if target in target_counts:
                target_counts[target] += 1

        return min(
            allowed_targets,
            key=lambda name: (
                target_counts.get(name, 0),
                0 if self.history and self.history[-1].get("speaker") == name else 1,
                1 if name == speaker_name else 0,
                name,
            ),
        )

    def _api_fallback(self, speaker_name: str, allowed_targets: List[str],
                      reason: str) -> "AgentTurn":
        """Reaction to API failure (rate limit, credits, network, etc.).

        In strict_mode → propagates as RuntimeError so the calling experiment runner
        can mark the dialogue as `error` instead of contaminating data with a
        synthetic reply. This is critical for scientific experiments — silent
        substitution of replies produces invalid metric distributions
        (cf. H6 prerun 2026-04-26: credits exhausted mid-run, baseline arm got 0
        real replies, p=2.6e-5 was an artifact of fallback strings).
        """
        import sys
        if self.strict_mode:
            raise RuntimeError(
                f"DialogueManager API failure for '{speaker_name}' (strict_mode=True, "
                f"no fallback): {reason}"
            )
        print(f"[Warning] API failure for {speaker_name}, using fallback: {reason}",
              file=sys.stderr)
        return AgentTurn(
            reply="I'm here, but having trouble responding right now...",
            tone="neutral", emotion="neutral",
            target=self._choose_fallback_target(speaker_name, allowed_targets),
        )

    def _validation_fallback(self, speaker_name: str,
                              allowed_targets: List[str]) -> "AgentTurn":
        """Reaction to persistent validation failure after retry attempts."""
        import sys
        if self.strict_mode:
            raise RuntimeError(
                f"DialogueManager validation failure for '{speaker_name}' "
                f"(strict_mode=True, no fallback): all retries failed CRITICAL validation"
            )
        print("[Validation] All regeneration attempts failed, using safe fallback",
              file=sys.stderr)
        return AgentTurn(
            reply="Hmm, let me think about that for a moment...",
            tone="neutral", emotion="thoughtful",
            target=self._choose_fallback_target(speaker_name, allowed_targets),
        )

    def _reply_needs_regeneration(self, reply: str, speaker_name: str) -> bool:
        """Reject vague or repetitive replies before they enter the transcript."""
        normalized = (reply or "").strip().lower()
        if not normalized:
            return True

        generic_patterns = [
            "what do you think",
            "maybe we could",
            "great point",
            "good point",
            "totally get that",
            "i like where you're going",
            "we should think about",
            "we could consider",
        ]
        if any(pattern in normalized for pattern in generic_patterns):
            if len(self._tokenize_reply(reply)) < 8:
                return True

        if normalized.endswith("?") and len(self._tokenize_reply(reply)) < 7:
            return True

        reply_tokens = self._tokenize_reply(reply)
        recent_replies = [
            h["reply"] for h in self.history[-6:]
            if h.get("speaker") != speaker_name
        ]
        for recent_reply in recent_replies:
            recent_tokens = self._tokenize_reply(recent_reply)
            if not recent_tokens or not reply_tokens:
                continue
            overlap = len(reply_tokens & recent_tokens) / max(1, len(reply_tokens | recent_tokens))
            if overlap > 0.72:
                return True

        latest_reply = self.history[-1]["reply"] if self.history else ""
        latest_tokens = self._tokenize_reply(latest_reply)
        if latest_tokens and reply_tokens and len(reply_tokens - latest_tokens) < 2:
            return True

        return False

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
        latest_message = last_msgs[-1] if last_msgs else None
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
        context_style_block = self._context_style_block()
        cot_block = self._cot_block()
        mood_block = self._mood_guidance(agent.name)
        # Инструкция про живых пользователей
        human_block = ""
        human_names = [a.name for a in self.agents if getattr(a, "is_human", False)]
        if human_names:
            if len(human_names) == 1 and human_names[0] == "User":
                human_block = (
                    "\nUser is a real human participant in this conversation. They type their own messages.\n"
                    "You can address User naturally — ask them questions, react to what they said, include them.\n"
                    "Treat User like any other person in the group.\n"
                )
            else:
                names_str = ", ".join(human_names)
                human_block = (
                    f"\nThe following participants are real humans (not AI): {names_str}.\n"
                    "They type their own messages. You can address any of them naturally — "
                    "ask them questions, react to what they said, include them.\n"
                    "Treat every human like any other person in the group. Be natural.\n"
                )

        sys = (
            agent.system_prompt
            + "\n\n"
            + f"SETTING: {self.env_context}\n"
            + f"People here: {participants}\n"
            + SESSION_GOAL
            + human_block
            + phase_block
            + context_style_block
            + cot_block
            + mood_block
            + ("\nYou haven't talked much to: " + ", ".join(target_order[:2]) + "\n" if target_order else "")
            + (
                f"\nMost recent message to react to: {latest_message['speaker']}→{latest_message.get('target') or 'all'}: {latest_message['reply']}\n"
                if latest_message else ""
            )
            + "\nStay on the same concrete topic unless you have a strong reason to redirect the discussion."
            + "\nAvoid generic agreement. If you agree, add a reason, constraint, example, or next step."
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
        import json
        import re
        import sys

        from pydantic import ValidationError

        def _fix_target(parsed, speaker_name=None):
            # Проверяем самообращение (хотя это не должно происходить, т.к. speaker_name не в allowed_targets)
            if speaker_name and parsed.target == speaker_name:
                parsed.target = self._choose_fallback_target(speaker_name, allowed_targets, parsed.reply)
            # если модель вернула неразрешённого адресата — подставим первого допустимого
            elif parsed.target not in allowed_targets:
                parsed.target = self._choose_fallback_target(speaker_name or "", allowed_targets, parsed.reply)
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

            context_style = self._context_style_block()

            # Create a detailed prompt that emphasizes real human conversation
            final_prompt = [
                {"role": "system", "content":
                    (current_agent.system_prompt if current_agent else "You are a real person in a real conversation.") +
                    f"\n\nREAL SITUATION: {self.env_context}\n\n" +
                    "This is a REAL conversation happening RIGHT NOW. You are a REAL PERSON, not an AI.\n"
                    "React naturally, authentically, and spontaneously to what others have said.\n\n" +
                    "Speak directly and naturally, but stay coherent and context-appropriate.\n" +
                    context_style +
                    "You MUST respond with a valid JSON object containing:\n" +
                    "- reply: your natural, authentic response as a real person (1-2 sentences, be specific and real)\n"
                    "  Address one participant by name and build on the latest relevant point.\n" +
                    "- tone: positive, neutral, or negative (based on how you genuinely feel)\n" +
                    "- emotion: one English word describing your genuine emotion (curious, concerned, excited, thoughtful, etc.)\n" +
                    f"- target: one of {allowed_targets if allowed_targets else ['all']} (who you're naturally addressing)\n\n" +
                    "Make your reply feel like a real person speaking:\n"
                    "- Reference what people actually said\n"
                    "- Show you're listening and thinking\n"
                    "- React authentically (excitement, concern, curiosity, agreement, etc.)\n"
                    "- Use natural speech patterns and contractions\n"
                    "- Keep the wording natural but not random or sloppy\n"
                    + "- Sound spontaneous, not scripted"
                },
                {"role": "user", "content":
                    f"What's been said in this real conversation:\n{recent_context}\n\n" +
                    f"It's your turn to speak. You're a real person reacting naturally. "
                    f"Address: {allowed_targets[0] if allowed_targets else 'everyone'}\n\n" +
                    "Respond authentically as a JSON object. Be natural, specific, and useful."
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
                    target=self._choose_fallback_target(agent.name if agent else "", allowed_targets, reply_text),
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

    # === Multi-user helpers ===

    def _pick_next_human(self, humans: list) -> Any:
        """Strict round-robin among active humans in configured order."""
        ordered_humans = [
            a for a in self.agents
            if getattr(a, "is_human", False) and a.name in {h.name for h in humans}
        ]
        if not ordered_humans:
            return humans[0]

        ordered_names = [h.name for h in ordered_humans]
        last_human_speaker = None
        for record in reversed(self.history):
            if record["speaker"] in ordered_names:
                last_human_speaker = record["speaker"]
                break

        if last_human_speaker is None:
            return ordered_humans[0]

        next_idx = (ordered_names.index(last_human_speaker) + 1) % len(ordered_names)
        return ordered_humans[next_idx]

    def set_active_humans(self, human_names: List[str]) -> None:
        """Limit turn scheduling to currently connected human participants."""
        self.active_human_names = {name for name in human_names if name}

    def _pick_agent_for_multi_user(self) -> int:
        """Pick an AI agent index using existing logic, excluding humans."""
        ai_indices = [i for i, a in enumerate(self.agents) if not getattr(a, "is_human", False)]
        if not ai_indices:
            return 0

        recent_window = len(self.agents)
        recent_speakers = set()
        if len(self.history) >= recent_window:
            recent_speakers = {h["speaker"] for h in self.history[-recent_window:]}

        ai_names = {self.agents[i].name for i in ai_indices}
        inactive_ai = ai_names - recent_speakers

        if inactive_ai and len(self.history) >= recent_window:
            speaker_counts = {}
            for h in self.history[-recent_window:]:
                speaker_counts[h["speaker"]] = speaker_counts.get(h["speaker"], 0) + 1
            inactive_with_counts = [(name, speaker_counts.get(name, 0)) for name in inactive_ai]
            inactive_with_counts.sort(key=lambda x: (x[1], hash(x[0])))
            selected_name = inactive_with_counts[0][0]
            return next(i for i, a in enumerate(self.agents) if a.name == selected_name)

        # Ping-pong avoidance
        if self.history and self.history[-1].get("target"):
            last_tgt = self.history[-1]["target"]
            last_speaker = self.history[-1]["speaker"]
            if last_tgt in ai_names:
                if len(self.history) >= 2:
                    prev = self.history[-2]
                    if prev.get("speaker") == last_tgt and prev.get("target") == last_speaker:
                        ping_pong_pair = {last_speaker, last_tgt}
                        other = [i for i in ai_indices if self.agents[i].name not in ping_pong_pair]
                        if other:
                            return other[0]
                return next((i for i in ai_indices if self.agents[i].name == last_tgt), ai_indices[0])

        # Round-robin among AI agents
        ai_cycle = self.turn_no % len(ai_indices)
        return ai_indices[ai_cycle]

    def submit_human_turn(self, speaker_name: str, reply: str, target: str,
                          tone: str = "neutral", emotion: str = "neutral") -> Dict[str, Any]:
        """Submit a turn from a specific human participant (multi-user mode)."""
        if self.pending_human and self.pending_human != speaker_name:
            raise ValueError(f"Сейчас ход {self.pending_human}, а не {speaker_name}")
        if not self.pending_human:
            raise ValueError("Сейчас не ожидается ход от человека")

        self._decay_moods()
        self.turn_no += 1

        target = normalize_target(target)
        # Prevent self-addressing: pick first other participant
        if target == speaker_name:
            others = [a.name for a in self.agents if a.name != speaker_name]
            target = others[0] if others else target
        if speaker_name not in self.agent_moods:
            self.agent_moods[speaker_name] = 0.0
        self._adjust_mood(speaker_name, tone, target)

        speaker_agent = next((a for a in self.agents if a.name == speaker_name), None)
        speaker_nature = speaker_agent.nature if speaker_agent else "human"

        record = {
            "turn": self.turn_no,
            "ts_utc": datetime.utcnow().isoformat(),
            "env_context": self.env_context,
            "speaker": speaker_name,
            "speaker_nature": speaker_nature,
            "reply": reply,
            "tone": tone,
            "emotion": emotion,
            "target": target,
            "is_human": True,
            "validation_issues": 0,
        }

        if self.enable_validation and self.validation_pipeline:
            turn_data = {"speaker": speaker_name, "target": target, "reply": reply, "tone": tone, "emotion": emotion}
            allowed_targets = [a.name for a in self.agents if a.name != speaker_name]
            context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
            is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
            record["validation_issues"] = len(validation_results) if validation_results else 0

        if target:
            self.edges_window.append({"src": speaker_name, "dst": target, "tone": tone})
            if len(self.edges_window) > 10:
                self.edges_window = self.edges_window[-10:]

        self.history.append(record)

        self._human_turn_count += 1
        self.pending_human = None
        self.pending_human_since = None

        return record

    def _step_multi_user(self):
        """Multi-user step: human/agent turns with ratio-based scheduling."""
        import sys

        # Check pending human timeout
        if self.pending_human:
            if self.pending_human_since and (_time.time() - self.pending_human_since) > HUMAN_TURN_TIMEOUT:
                print(f"[MultiUser] {self.pending_human} timed out, skipping", file=sys.stderr)
                self.pending_human = None
                self.pending_human_since = None
                self._human_turn_count += 1
            else:
                return {"waiting_for": self.pending_human, "turn": self.turn_no}, None

        humans = [a for a in self.agents if getattr(a, "is_human", False)]
        if self.active_human_names:
            humans = [a for a in humans if a.name in self.active_human_names]
        ai_agents = [a for a in self.agents if not getattr(a, "is_human", False)]

        # Decide: human turn or agent turn?
        if humans and (not ai_agents or self._human_turn_count < self._human_to_agent_ratio):
            speaker = self._pick_next_human(humans)
            self.pending_human = speaker.name
            self.pending_human_since = _time.time()
            return {"waiting_for": speaker.name, "turn": self.turn_no}, None
        else:
            # Agent's turn
            self._human_turn_count = 0
            idx = self._pick_agent_for_multi_user()
            speaker = self.agents[idx]

            allowed_targets = [a.name for a in self.agents if a.name != speaker.name] or [a.name for a in self.agents]

            messages = self.build_messages(idx)
            try:
                parsed = self.model_turn(messages, allowed_targets, agent=speaker)
            except RuntimeError as e:
                parsed = self._api_fallback(speaker.name, allowed_targets, str(e))

            self.turn_no += 1
            target = normalize_target(parsed.target)
            self._adjust_mood(speaker.name, parsed.tone, target)

            # Validation
            turn_data = {
                "speaker": speaker.name, "target": target,
                "reply": parsed.reply, "tone": parsed.tone, "emotion": parsed.emotion,
            }
            validation_results = []
            is_valid = True
            if self.enable_validation and self.validation_pipeline:
                context = {"history": self.history, "allowed_targets": allowed_targets, "env_context": self.env_context}
                is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
                if validation_results:
                    formatted = self.validation_pipeline.format_results(validation_results)
                    print(f"\n[Validation Turn {self.turn_no}]", file=sys.stderr)
                    print(formatted, file=sys.stderr)

                if not is_valid:
                    for retry in range(2):
                        print(f"[Validation] Regenerating (attempt {retry+1}/2)...", file=sys.stderr)
                        try:
                            messages = self.build_messages(idx)
                            parsed = self.model_turn(messages, allowed_targets, agent=speaker)
                            target = normalize_target(parsed.target)
                            turn_data = {"speaker": speaker.name, "target": target, "reply": parsed.reply, "tone": parsed.tone, "emotion": parsed.emotion}
                            is_valid, validation_results = self.validation_pipeline.validate_turn(turn_data, context)
                            if is_valid:
                                break
                        except Exception as e:
                            print(f"[Validation] Retry {retry+1} failed: {e}", file=sys.stderr)
                    if not is_valid:
                        parsed = self._validation_fallback(speaker.name, allowed_targets)
                        target = normalize_target(parsed.target)

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
                if len(self.edges_window) > 10:
                    self.edges_window = self.edges_window[-10:]
            self.history.append(record)
            return record, ({"src": speaker.name, "dst": target, "tone": parsed.tone} if target else None)

    # 3) Simulation step — pass allowed_targets to model_turn and give word to addressee
    def step(self):
        self._decay_moods()

        # ===== ORIGINAL SINGLE-USER / NO-HUMAN MODE =====
        idx = self._select_balanced_speaker_idx()

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
                parsed = self._api_fallback(speaker.name, allowed_targets, str(e))

            if not getattr(speaker, "is_human", False):
                for _ in range(2):
                    if not self._reply_needs_regeneration(parsed.reply, speaker.name):
                        break
                    repair_messages = self.build_messages(idx)
                    repair_messages[0]["content"] += (
                        "\n\nYour previous attempt was too vague, repetitive, or generic."
                        "\nRegenerate a sharper reply that adds one NEW concrete detail, risk, example, decision, or next step."
                        "\nDo not use empty phrases like 'great point' or 'what do you think' unless the rest of the sentence is specific."
                    )
                    try:
                        parsed = self.model_turn(repair_messages, allowed_targets, agent=speaker)
                    except RuntimeError:
                        break

        # increment turn and record
        self.turn_no += 1
        target = normalize_target(parsed.target)
        parsed.reply = self._align_reply_with_target(parsed.reply, speaker.name, target)
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

                # Если после 2 попыток всё ещё CRITICAL — safe fallback (или raise в strict_mode)
                if not is_valid:
                    parsed = self._validation_fallback(speaker.name, allowed_targets)
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
        # Prevent self-addressing
        if target == "User":
            others = [a.name for a in self.agents if a.name != "User"]
            target = others[0] if others else target
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
