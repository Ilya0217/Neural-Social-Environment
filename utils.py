from typing import Optional

def normalize_target(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    t = target.strip()
    return t if t else None
