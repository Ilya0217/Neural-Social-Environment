import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class IOLogger:
    def __init__(self, jsonl_path: Path, md_path: Path, env_context: str = ""):
        self.jsonl_path = jsonl_path
        self.md_path = md_path
        if not self.md_path.exists():
            header = "# Agents dialogue\n\n"
            if env_context:
                header += f"**Environment and context:** {env_context}\n\n---\n\n"
            self.md_path.write_text(header, encoding="utf-8")

    def write_jsonl(self, record: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_metrics_block(self, title: str, md_text: str) -> None:
        block = f"\n\n## {title}\n\n{md_text}\n\n---\n"
        with self.md_path.open("a", encoding="utf-8") as f:
            f.write(block)

    def write_markdown(self, turn_no: int, speaker: str, target: str, tone: str, emotion: str, text: str) -> None:
        ts = datetime.utcnow().isoformat()
        block = (
            f"### Turn {turn_no} — {ts} UTC\n"
            f"**Speaker:** {speaker} → **To:** {target or 'all'}\n\n"
            f"**Tone:** {tone}, **Emotion:** {emotion}\n\n"
            f"{text}\n\n---\n"
        )
        with self.md_path.open("a", encoding="utf-8") as f:
            f.write(block)
