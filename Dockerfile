# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    AGENT_RUNTIME_DIR=/data \
    AGENT_LOG_DIR=/data/logs \
    AGENT_OUTPUT_DIR=/data/outputs

# matplotlib/scipy/numpy нужны небольшие системные библиотеки
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6 \
        libpng16-16 \
        libopenblas0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала только requirements — кэшируем слой со слоем зависимостей
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir "gunicorn==22.0.0"

# Копируем пакет как agent_dialogue_sim, чтобы работали относительные импорты
# (webapp/app.py: from ..config import ...)
COPY __init__.py        /app/agent_dialogue_sim/__init__.py
COPY config.py          /app/agent_dialogue_sim/config.py
COPY utils.py           /app/agent_dialogue_sim/utils.py
COPY main.py            /app/agent_dialogue_sim/main.py
COPY analytics          /app/agent_dialogue_sim/analytics
COPY core               /app/agent_dialogue_sim/core
COPY science            /app/agent_dialogue_sim/science
COPY webapp             /app/agent_dialogue_sim/webapp
COPY experiments        /app/agent_dialogue_sim/experiments

# Каталоги под рантайм-артефакты (логи, графики, отчёты)
RUN mkdir -p /data/logs /data/outputs \
 && useradd --create-home --shell /usr/sbin/nologin app \
 && chown -R app:app /app /data

USER app

EXPOSE 5000

# Healthcheck по /api/health (см. CD smoke-test)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5000/api/health || exit 1

# ВАЖНО: ровно 1 воркер. Состояние сессии (STATE в webapp/app.py) живёт в памяти
# процесса и НЕ разделяется между воркерами. С несколькими воркерами запросы
# раскидываются round-robin, и /api/step попадает на воркер со старой сессией —
# в UI всплывает старый диалог. Concurrency обеспечиваем потоками, не воркерами.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "agent_dialogue_sim.webapp.app:app"]
