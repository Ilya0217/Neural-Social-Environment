from typing import List, Dict, Any, Optional
import re
from .prompts import AgentTurn

class HumanIO:
    """
    CLI-адаптер для «живого» агента. В web-версии можно заменить
    на UI-коллбеки — сигнатура get_user_turn та же.
    """
    def __init__(self, client=None, classify_with_model: bool = True):
        self.client = client
        self.classify_with_model = classify_with_model

    def _extract_tags(self, text: str) -> Dict[str, str]:
        # мини-парсер для /tone:positive /emotion:curious
        tags = {}
        m_tone = re.search(r"/tone:(positive|neutral|negative)", text, re.I)
        m_emo  = re.search(r"/emotion:([a-zA-Z\- ]{2,32})", text)
        if m_tone: tags["tone"] = m_tone.group(1).lower()
        if m_emo:  tags["emotion"] = m_emo.group(1).strip().lower()
        return tags

    def _classify(self, text: str) -> Dict[str, str]:
        # лёгкая классификация через модель (если доступна)
        if not (self.client and self.classify_with_model):
            return {"tone": "neutral", "emotion": "neutral"}
        prompt = [
            {"role": "system", "content": "Classify user utterance with JSON: {tone: positive|neutral|negative, emotion: one-word english}"},
            {"role": "user", "content": text.strip()},
        ]
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=20,
                messages=prompt,
            )
            raw = (resp.choices[0].message.content or "").strip()
            m = re.search(r"\{.*\}", raw, re.S)
            return eval(m.group(0)) if m else {"tone": "neutral", "emotion": "neutral"}
        except Exception:
            return {"tone": "neutral", "emotion": "neutral"}

    def get_user_turn(
        self,
        speaker_name: str,
        allowed_targets: List[str],
        history: List[Dict[str, Any]],
    ) -> AgentTurn:
        print(f"\n[{speaker_name}] К вам обратились. Ваша очередь отвечать.")
        # короткий контекст последних 6 сообщений
        for h in history[-6:]:
            tgt = h.get("target") or "всем"
            print(f"  {h['speaker']}→{tgt}: {h['reply']}")

        # 1) Текст
        user_text = input("\nВведите вашу реплику (можно добавить /tone:positive /emotion:curious):\n> ").strip()
        tags = self._extract_tags(user_text)
        # уберём подсказки из текста
        user_text = re.sub(r"/tone:[^\s]+", "", user_text, flags=re.I)
        user_text = re.sub(r"/emotion:[^\s]+", "", user_text, flags=re.I).strip()

        # 2) Адресат
        print("\nКому адресовать ответ?")
        for i, t in enumerate(allowed_targets,  start=1):
            print(f"  {i}) {t}")
        try:
            raw = input(f"Ваш выбор [1-{len(allowed_targets)}]: ").strip()
            tgt_idx = int(raw) - 1
        except Exception:
            tgt_idx = 0
        target = allowed_targets[tgt_idx] if 0 <= tgt_idx < len(allowed_targets) else allowed_targets[0]

        # 3) Тон/эмоция
        if "tone" not in tags or "emotion" not in tags:
            auto = self._classify(user_text)
            tone = tags.get("tone", auto["tone"])
            emotion = tags.get("emotion", auto["emotion"])
        else:
            tone = tags["tone"]
            emotion = tags["emotion"]

        return AgentTurn(reply=user_text, tone=tone, emotion=emotion, target=target)
