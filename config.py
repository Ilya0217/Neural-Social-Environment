from pathlib import Path
from dotenv import load_dotenv
import os
import tempfile

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

DATA_DIR = Path(__file__).resolve().parent

# Runtime directories (can be overridden via env vars to keep repo clean)
RUNTIME_BASE = Path(
    os.getenv("AGENT_RUNTIME_DIR")
    or Path(tempfile.gettempdir()) / "agent_dialogue_sim"
)
LOG_DIR = Path(os.getenv("AGENT_LOG_DIR") or (RUNTIME_BASE / "logs"))
OUT_DIR = Path(os.getenv("AGENT_OUTPUT_DIR") or (RUNTIME_BASE / "outputs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Can be overridden via .env
DEFAULT_ENV_CONTEXT_INDEX = int(os.getenv("ENV_CONTEXT_INDEX", "0"))  # 0..2
AGENT_PROFILES_PATH = os.getenv("AGENT_PROFILES_PATH", "")

TEMPERATURE = 0.85  # Higher temperature for more natural, varied, human-like responses
MAX_TOKENS = 600
TURNS_BETWEEN_PLOTS = 5

VIZ_EDGE_WINDOW = 12         # последние N ходов для рёбер
VIZ_TOP_EDGE_LABELS = 6      # подписывать не больше N самых сильных рёбер
VIZ_SEED = 42                # фиксируем раскладку, чтобы "картинка не прыгала"

# default agents
DEFAULT_AGENTS = [
    {
        "name": "Explorer",
        "nature": "explorer",
        "color": "#4F46E5",
    },
    {
        "name": "Critic",
        "nature": "critic",
        "color": "#DC2626",
    },
    {
        "name": "Facilitator",
        "nature": "facilitator",
        "color": "#059669",
    },
]

USER_AGENT = {
    "name": "User",
    "nature": "human",
    "color": "#405686",
    "is_human": True,
}

# === Dialogue flow configuration ===
DIALOGUE_PHASES = [
    {
        "name": "ideation",
        "length": 4,
        "instruction": "Stay open and playful. Capture as many concrete ideas as possible, even rough ones. Ask exploratory questions.",
    },
    {
        "name": "prioritization",
        "length": 4,
        "instruction": "Compare the collected ideas, highlight trade-offs, vote or challenge the most promising ones. Be candid but respectful.",
    },
    {
        "name": "planning",
        "length": 4,
        "instruction": "Agree on next steps, owners, and timelines. Keep it actionable and realistic. Close loops from previous phases.",
    },
]

MOOD_DECAY = 0.05
MOOD_DELTA = {
    "positive": 0.25,
    "neutral": -0.02,
    "negative": -0.3,
}
MOOD_GUIDANCE = [
    (-0.4, "You're sounding a bit tense — soften the tone, acknowledge others, and look for common ground."),
    (0.4, "You're in a great mood — channel that to encourage others and connect ideas."),
]