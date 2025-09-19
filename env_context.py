from typing import List

# ТРИ ПРОСТЫЕ ВЕРСИИ (формат строго "Окружение: Контекст")
ENV_CONTEXTS: List[str] = [
    "Университет: Обсуждение плана курсовой работы по ИИ",
    "Стартап: Планирование MVP и приоритетов фич",
    "Научный семинар: Разбор статьи и план экспериментов",
]

def list_env_contexts() -> List[str]:
    return ENV_CONTEXTS

def get_env_context_by_index(i: int) -> str:
    if 0 <= i < len(ENV_CONTEXTS):
        return ENV_CONTEXTS[i]
    return ENV_CONTEXTS[0]  # дефолт: первый
