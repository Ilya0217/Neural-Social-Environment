import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class IOLogger:
    def __init__(self, jsonl_path: Path, md_path: Path, env_context: str = ""):
        self.jsonl_path = jsonl_path
        self.md_path = md_path
        if not self.md_path.exists():
            header = "# Диалог агентов\n\n"
            if env_context:
                header += f"**Окружение и контекст:** {env_context}\n\n---\n\n"
            self.md_path.write_text(header, encoding="utf-8")

    def write_jsonl(self, record: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_markdown(self, turn_no: int, speaker: str, target: str, tone: str, emotion: str, text: str) -> None:
        ts = datetime.utcnow().isoformat()
        block = (
            f"### Ход {turn_no} — {ts} UTC\n"
            f"**Говорит:** {speaker} → **Кому:** {target or 'всем'}\n\n"
            f"**Тон:** {tone}, **Эмоция:** {emotion}\n\n"
            f"{text}\n\n---\n"
        )
        with self.md_path.open("a", encoding="utf-8") as f:
            f.write(block)
