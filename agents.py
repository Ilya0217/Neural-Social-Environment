from dataclasses import dataclass
from .prompts import build_system_prompt

@dataclass
class Agent:
    name: str
    nature: str
    color: str
    is_human: bool = False
    persona: str = ""

    @property
    def system_prompt(self) -> str:
        return build_system_prompt(self.nature, self.persona)
