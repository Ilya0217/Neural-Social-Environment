from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DATA_DIR = Path(__file__).resolve().parent
LOG_DIR = DATA_DIR / "logs"
OUT_DIR = DATA_DIR / "outputs"
LOG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Can be overridden via .env
DEFAULT_ENV_CONTEXT_INDEX = int(os.getenv("ENV_CONTEXT_INDEX", "0"))  # 0..2

TEMPERATURE = 0.7
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
        "name": "Mediator",
        "nature": "mediator",
        "color": "#059669",
    },
]

USER_AGENT = {
    "name": "User",
    "nature": "human",
    "color": "#405686",
    "is_human": True,
}