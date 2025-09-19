from dataclasses import dataclass
from typing import Dict
from .prompts import SYSTEM_PROMPTS

@dataclass
class Agent:
    name: str
    nature: str
    color: str
    is_human: bool = False

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS.get(self.nature, "You are a helpful AI assistant.")
