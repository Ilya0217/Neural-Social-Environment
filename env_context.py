from typing import List

# Three simple versions (format strictly "Environment: Context")
ENV_CONTEXTS: List[str] = [
    "University: Planning an AI coursework project",
    "Startup: Prioritizing MVP features and roadmap",
    "Research seminar: Reviewing a paper and planning experiments",
]

def list_env_contexts() -> List[str]:
    return ENV_CONTEXTS

def get_env_context_by_index(i: int) -> str:
    if 0 <= i < len(ENV_CONTEXTS):
        return ENV_CONTEXTS[i]
    return ENV_CONTEXTS[0]  # default: first
