from typing import List
from .prompts import ENV_CONTEXTS as _ENV_CONTEXTS

# Backward-compatible re-export that delegates to prompts.py

def list_env_contexts() -> List[str]:
    return _ENV_CONTEXTS

def get_env_context_by_index(i: int) -> str:
    if 0 <= i < len(_ENV_CONTEXTS):
        return _ENV_CONTEXTS[i]
    return _ENV_CONTEXTS[0]  # default: first
