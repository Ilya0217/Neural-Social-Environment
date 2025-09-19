from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from pydantic import ValidationError

from .config import MODEL, TEMPERATURE, MAX_TOKENS, TURNS_BETWEEN_PLOTS, VIZ_EDGE_WINDOW
from .prompts import AgentTurn, STRUCTURE_INSTRUCTION, SESSION_GOAL
from .utils import normalize_target
from .human_io import HumanIO

@dataclass
class DialogueManager:
    client: OpenAI
    agents: List[Any]  # list[Agent]
    env_context: str   # Выбранное "Окружение: Контекст"
    human_io: Optional[HumanIO] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    turn_no: int = 0
    edges_window: List[Dict[str, Any]] = field(default_factory=list)  # для визуализаций последнего окна

    # 1) Индекс говорящего по кругу (фолбэк)
    def next_speaker_idx(self) -> int:
        # turn_no увеличивается в КОНЦЕ step(), поэтому здесь используем текущее значение
        return (self.turn_no) % len(self.agents)

    # 2) Сборка сообщений (если уже внесён текст про участников — оставь как есть)
    def build_messages(self, agent_idx: int) -> List[Dict[str, str]]:
        agent = self.agents[agent_idx]
        last_msgs = self.history[-8:]
        content_summary = "\n".join(
            f"{h['speaker']}→{h.get('target') or 'all'}: {h['reply']}" for h in last_msgs
        )
        participants = ", ".join(a.name for a in self.agents)
        sys = (
            f"Окружение и контекст: {self.env_context}\n\n"
            + agent.system_prompt
            + "\n\n"
            + SESSION_GOAL
            + "\n\nУчастники разговора: " + participants +
            "\nПравило адресации: обязательно выбери одного адресата (target) из списка участников, кроме себя. "
            "Никогда не используй all/null/всем. Пиши коротко — 1–2 предложения.\n\n"
            + "Контекст последних сообщений:\n"
            + (content_summary or "(пока пусто)")
            + "\n\n"
            + STRUCTURE_INSTRUCTION
            + "\n\n"
            + "If tools are available, respond ONLY by calling the function `submit_agent_turn` with proper JSON arguments."
        )
        return [
            {"role": "system", "content": sys},
            {"role": "user", "content": "Сформируй следующую реплику обсуждения."}
        ]

    def model_turn(self, messages, allowed_targets):
        import json, re, sys
        from pydantic import ValidationError
        
        def _fix_target(parsed):
            # если модель вернула неразрешённого адресата — подставим первого допустимого
            if parsed.target not in allowed_targets:
                parsed.target = allowed_targets[0] if allowed_targets else None
            return parsed

        # tools (function calling) с жестким перечислением адресатов
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "submit_agent_turn",
                    "description": "Строго структурированный ответ агента.",
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
        tool_choice = {"type": "function", "function": {"name": "submit_agent_turn"}}

        # A) chat.completions + tools
        try:
            resp = self.client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            choice = resp.choices[0]
            tool_calls = getattr(choice.message, "tool_calls", None) or []
            if tool_calls:
                args_str = tool_calls[0].function.arguments
                parsed = AgentTurn.model_validate_json(args_str)
                return _fix_target(parsed)   # ← ДОБАВЛЕНО
            # если почему-то tool не вызван — попробуем распарсить контент
            content = choice.message.content or ""
            m = re.search(r"\{.*\}", content, re.S)
            if m:
                parsed = AgentTurn.model_validate_json(m.group(0))
                return _fix_target(parsed)   # ← ДОБАВЛЕНО
        except Exception as e:
            print(f"[tools path error] {e}", file=sys.stderr)

            # B) Жёсткий текстовый запрос одного JSON-объекта
            tough_messages = messages + [
                {"role": "user", "content":
                    "Return ONLY one valid minified JSON object with keys: reply, tone, emotion, target. "
                    f"target must be one of: {allowed_targets}."
                }
            ]
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=tough_messages,
                )
                raw = resp.choices[0].message.content or ""
                m = re.search(r"\{.*\}", raw, re.S)
                raw_json = m.group(0) if m else None
                if raw_json:
                    parsed = AgentTurn.model_validate_json(raw_json)
                    return _fix_target(parsed)   # ← ДОБАВЛЕНО
                print(f"[no JSON in content] {raw[:300]}", file=sys.stderr)
            except Exception as e:
                print(f"[text path error] {e}", file=sys.stderr)

        # C) Фолбэк (на всякий случай выбираем первого допустимого)
        return AgentTurn(
            reply="(не удалось распарсить ответ модели)",
            tone="neutral",
            emotion="neutral",
            target=allowed_targets[0] if allowed_targets else None,
        )

    # 3) Шаг симуляции — ПЕРЕДАЁМ allowed_targets во model_turn и даём слово адресату
    def step(self):
        # выбор следующего говорящего (как у вас было)
        if self.history and self.history[-1].get("target"):
            tgt = self.history[-1]["target"]
            idx = next((i for i, a in enumerate(self.agents) if a.name == tgt), self.next_speaker_idx())
        else:
            idx = self.next_speaker_idx()

        speaker = self.agents[idx]

        # Кого можно адресовать (все, кроме себя)
        allowed_targets = [a.name for a in self.agents if a.name != speaker.name] or [a.name for a in self.agents]

        # НОВОЕ: если это «живой» агент — получаем ход через HumanIO
        if getattr(speaker, "is_human", False) and self.human_io:
            parsed: AgentTurn = self.human_io.get_user_turn(speaker.name, allowed_targets, self.history)
        else:
            messages = self.build_messages(idx)
            parsed: AgentTurn = self.model_turn(messages, allowed_targets)

        # увеличиваем номер хода и фиксируем
        self.turn_no += 1
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
        }

        if target:
            self.edges_window.append({"src": speaker.name, "dst": target, "tone": parsed.tone})
            if len(self.edges_window) > 10:  # (см. замечание про синхронизацию с конфигом)
                self.edges_window = self.edges_window[-10:]

        self.history.append(record)
        return record, ({"src": speaker.name, "dst": target, "tone": parsed.tone} if target else None)
    
    def should_plot(self) -> bool:
        # рисуем каждые TURNS_BETWEEN_PLOTS ходов
        return self.turn_no > 0 and (self.turn_no % TURNS_BETWEEN_PLOTS == 0)
