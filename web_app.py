from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, render_template, request, send_from_directory, send_file
from openai import OpenAI

from .config import (
    LOG_DIR,
    OUT_DIR,
    DEFAULT_AGENTS,
    OPENAI_API_KEY,
    DEFAULT_ENV_CONTEXT_INDEX,
    VIZ_EDGE_WINDOW,
    VIZ_SEED,
)
from .visualize import draw_interactions_pro
from .agents import Agent
from .dialogue_manager import DialogueManager
from .io_logger import IOLogger
from .env_context import list_env_contexts, get_env_context_by_index
from .analytics import compute_metrics, hypotheses_from_metrics, render_markdown_report, save_report_md


app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)


class AppState:
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.dm: Optional[DialogueManager] = None
        self.logger: Optional[IOLogger] = None
        self.agents_meta: Dict[str, Dict[str, str]] = {}
        self.env_context: str = ""
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.last_hyps: List[str] = []

    def reset(self, env_index: int):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")

        self.client = OpenAI()

        options = list_env_contexts()
        idx = env_index if 0 <= env_index < len(options) else DEFAULT_ENV_CONTEXT_INDEX
        self.env_context = get_env_context_by_index(idx)

        # cleanup outputs and logs for a fresh session (specific patterns)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # outputs patterns
        for pattern in [
            "graph_turn_*.png",
            "metrics_turn_*.md",
            "trend_delay_*.png",
            "trend_entropy_*.png",
            "trend_targeting_*.png",
            "trend_tone_*.png",
        ]:
            for p in OUT_DIR.glob(pattern):
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
        # logs patterns
        for p in LOG_DIR.glob("dialog.*"):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

        agents = [Agent(**a) for a in DEFAULT_AGENTS]
        # В web-версии опускаем живого пользователя, чтобы не блокировать UI
        self.dm = DialogueManager(client=self.client, agents=agents, env_context=self.env_context, human_io=None)

        self.logger = IOLogger(
            jsonl_path=LOG_DIR / "dialog.jsonl",
            md_path=LOG_DIR / "dialog.md",
            env_context=self.env_context,
        )
        # ensure fresh empty logs after IOLogger init
        try:
            (LOG_DIR / "dialog.jsonl").write_text("", encoding="utf-8")
        except Exception:
            pass
        try:
            (LOG_DIR / "dialog.md").write_text("", encoding="utf-8")
        except Exception:
            pass
        self.agents_meta = {a.name: {"color": a.color} for a in agents}
        self.last_metrics = None
        self.last_hyps = []


STATE = AppState()


@app.route("/")
def index():
    envs = list_env_contexts()
    return render_template("index.html", envs=envs)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    env_index = int(data.get("env_index", DEFAULT_ENV_CONTEXT_INDEX))
    STATE.reset(env_index)
    return jsonify({
        "ok": True,
        "env": STATE.env_context,
        "turn": 0,
        "history": [],
        "metrics_md": None,
    })


@app.route("/api/step", methods=["POST"])
def api_step():
    if not STATE.dm or not STATE.logger:
        return jsonify({"ok": False, "error": "Session not initialized. Call /api/start"}), 400

    record, _ = STATE.dm.step()

    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"],
        speaker=record["speaker"],
        target=record.get("target"),
        tone=record["tone"],
        emotion=record["emotion"],
        text=record["reply"],
    )

    image_url = None
    metrics_md = None
    hyps: List[str] = []

    if STATE.dm.should_plot():
        out_path = OUT_DIR / f"graph_turn_{record['turn']:04d}.png"
        draw_interactions_pro(
            agents_meta=STATE.agents_meta,
            history=list(STATE.dm.history),
            out_path=out_path,
            title=f"Взаимодействия и тональности (до хода {record['turn']})",
            window=VIZ_EDGE_WINDOW,
            seed=VIZ_SEED,
            top_edge_labels=0,
            per_message_edges=True,
        )
        image_url = f"/outputs/{out_path.name}"

        agents_order = list(STATE.agents_meta.keys())
        metrics = compute_metrics(STATE.dm.history, agents_order, window_size=VIZ_EDGE_WINDOW)
        hyps = hypotheses_from_metrics(metrics, user_name=None)
        metrics_md = render_markdown_report(metrics, turn=record["turn"], title="Dialogue — metrics")
        metrics_path = OUT_DIR / f"metrics_turn_{record['turn']:04d}.md"
        save_report_md(metrics_md, metrics_path)
        STATE.last_metrics = metrics
        STATE.last_hyps = hyps

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-20:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    if not STATE.dm:
        return jsonify({"ok": True, "turn": 0, "history": [], "image_url": None, "hypotheses": [], "metrics_md": None})

    # Найти последний сгенерированный граф
    latest_img = None
    if OUT_DIR.exists():
        cand = sorted(OUT_DIR.glob("graph_turn_*.png"))
        latest_img = cand[-1].name if cand else None

    metrics_path = None
    candidates = sorted(OUT_DIR.glob("metrics_turn_*.md"))
    if candidates:
        metrics_path = candidates[-1]
    metrics_md = metrics_path.read_text(encoding="utf-8") if metrics_path else None

    return jsonify({
        "ok": True,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-50:],
        "image_url": (f"/outputs/{latest_img}" if latest_img else None),
        "hypotheses": STATE.last_hyps,
        "metrics_md": metrics_md,
        "env": STATE.env_context,
    })


@app.route("/outputs/<path:filename>")
def serve_outputs(filename: str):
    return send_from_directory(OUT_DIR, filename)


@app.route("/download/log")
def download_log():
    path = LOG_DIR / "dialog.jsonl"
    if not path.exists():
        return jsonify({"ok": False, "error": "No log yet"}), 404
    return send_file(path, as_attachment=True, download_name="dialog.jsonl", mimetype="application/json")


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Для локального запуска: python -m agent_dialogue_sim.web_app
    app.run(host="0.0.0.0", port=5000, debug=True)


