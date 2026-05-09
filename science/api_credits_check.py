"""
Pre-flight проверка баланса OpenRouter API ДО старта дорогого эксперимента.

Мотивация: 2026-04-26 прогон H6 был полностью испорчен из-за исчерпания кредитов
в середине работы. Эта проверка предупреждает за минуту до старта, а не
обнаруживает проблему через 7.5 минут после её возникновения.

Использование:
    from agent_dialogue_sim.science.api_credits_check import check_credits, ensure_sufficient

    info = check_credits()
    print(info["remaining_usd"])
    ensure_sufficient(min_usd=0.50)  # raises CreditsExhaustedError if low

Эндпоинт: GET https://openrouter.ai/api/v1/credits
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)


class CreditsCheckError(Exception):
    """Не удалось получить информацию о балансе (сеть, неверный ключ, и т.п.)."""


class CreditsExhaustedError(Exception):
    """Баланс ниже минимального порога — эксперимент откладывается."""


@dataclass
class CreditsInfo:
    total_credits: float       # Сколько было пополнено всего, $
    total_usage: float          # Сколько потрачено, $
    remaining_usd: float        # Остаток
    is_openrouter: bool         # True если base_url выглядит как OpenRouter
    raw_response: dict          # Сырой JSON ответа

    def is_sufficient(self, min_usd: float) -> bool:
        return self.remaining_usd >= min_usd


def _is_openrouter(base_url: str) -> bool:
    if not base_url:
        return False
    host = urlparse(base_url).hostname or ""
    return "openrouter" in host.lower()


def check_credits(api_key: Optional[str] = None,
                  base_url: Optional[str] = None,
                  timeout: float = 10.0) -> CreditsInfo:
    """
    Запрашивает баланс через OpenRouter `/api/v1/credits`.
    Если base_url не указан или указывает не на OpenRouter, возвращает
    CreditsInfo с remaining_usd=inf и is_openrouter=False (не блокирует прогон).

    Не использует requests/httpx чтобы избежать лишней зависимости — стандартная
    библиотека urllib справляется.
    """
    # Берём из config.py если не передано явно
    if api_key is None or base_url is None:
        try:
            from ..config import OPENAI_API_KEY, OPENAI_BASE_URL
            api_key = api_key or OPENAI_API_KEY
            base_url = base_url or OPENAI_BASE_URL
        except ImportError:
            pass

    if not _is_openrouter(base_url or ""):
        # Для неоткорректированных провайдеров пропускаем проверку
        logger.debug("base_url is not OpenRouter (%s), skipping credits check", base_url)
        return CreditsInfo(
            total_credits=float("inf"),
            total_usage=0.0,
            remaining_usd=float("inf"),
            is_openrouter=False,
            raw_response={},
        )

    if not api_key:
        raise CreditsCheckError("OPENAI_API_KEY is empty; cannot check OpenRouter credits")

    url = base_url.rstrip("/") + "/credits"
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")
        except Exception:
            pass
        raise CreditsCheckError(f"HTTP {e.code} from {url}: {body[:200]}") from e
    except URLError as e:
        raise CreditsCheckError(f"Network error contacting {url}: {e}") from e
    except (ValueError, json.JSONDecodeError) as e:
        raise CreditsCheckError(f"Invalid JSON from {url}: {e}") from e

    data = raw.get("data", {}) if isinstance(raw, dict) else {}
    total = float(data.get("total_credits", 0))
    usage = float(data.get("total_usage", 0))
    remaining = total - usage
    return CreditsInfo(
        total_credits=total,
        total_usage=usage,
        remaining_usd=remaining,
        is_openrouter=True,
        raw_response=raw,
    )


def ensure_sufficient(min_usd: float = 0.50,
                       api_key: Optional[str] = None,
                       base_url: Optional[str] = None) -> CreditsInfo:
    """
    Проверка-блокатор для запуска эксперимента. Поднимает CreditsExhaustedError
    если баланса недостаточно. Возвращает CreditsInfo при успехе.
    """
    info = check_credits(api_key=api_key, base_url=base_url)
    if info.is_openrouter and not info.is_sufficient(min_usd):
        raise CreditsExhaustedError(
            f"OpenRouter balance ${info.remaining_usd:.4f} is below minimum "
            f"${min_usd:.2f}. Top up at https://openrouter.ai/settings/credits "
            f"before running large experiments."
        )
    return info


def estimate_experiment_cost(n_dialogues: int, n_turns: int,
                              avg_input_tokens_per_call: int = 1500,
                              avg_output_tokens_per_call: int = 80,
                              input_price_per_million: float = 0.15,
                              output_price_per_million: float = 0.60) -> float:
    """
    Грубая оценка стоимости эксперимента в $.
    Default цены — gpt-4o-mini через OpenRouter (на момент 2026-04-26).
    Реальная стоимость может быть выше из-за ретраев валидации.
    """
    n_calls = n_dialogues * n_turns
    input_cost = (n_calls * avg_input_tokens_per_call / 1_000_000) * input_price_per_million
    output_cost = (n_calls * avg_output_tokens_per_call / 1_000_000) * output_price_per_million
    return input_cost + output_cost


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Check OpenRouter credit balance.")
    parser.add_argument("--min-usd", type=float, default=0.0,
                        help="Exit non-zero if balance below this. 0 = report only.")
    args = parser.parse_args()

    try:
        info = check_credits()
    except CreditsCheckError as e:
        print(f"ERROR: {e}")
        return 2

    if not info.is_openrouter:
        print("base_url is not OpenRouter — skipping check")
        return 0

    print(f"Total credits:  ${info.total_credits:.4f}")
    print(f"Total usage:    ${info.total_usage:.4f}")
    print(f"Remaining:      ${info.remaining_usd:.4f}")

    if args.min_usd > 0 and info.remaining_usd < args.min_usd:
        print(f"FAIL: balance below required ${args.min_usd:.2f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
