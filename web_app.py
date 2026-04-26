from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
from openai import OpenAI

from .advanced_analytics import compute_advanced_analysis, render_advanced_report
from .agents import Agent
from .analytics import (
    compute_metrics,
    get_hypothesis_validation_report,
    get_scientific_hypotheses_full,
    hypotheses_from_metrics,
    render_markdown_report,
    save_report_md,
    validate_hypotheses,
)
from .config import (
    DEFAULT_ENV_CONTEXT_INDEX,
    LOG_DIR,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OUT_DIR,
    VIZ_EDGE_WINDOW,
    VIZ_SEED,
)
from .dialogue_manager import DialogueManager
from .env_context import get_env_context_by_index, list_env_contexts
from .io_logger import IOLogger
from .observer_agent import ObserverAgent, get_hypothesis_registry
from .profiles import initialize_agents, initialize_agents_from_config
from .scientific_analytics import (
    generate_scientific_hypotheses,
    render_scientific_report,
)
from .visualize import draw_interactions_pro

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
        self.agents_meta: Dict[str, Dict[str, Any]] = {}
        self.env_context: str = ""
        self.last_metrics: Optional[Dict[str, Any]] = None
        self.last_hyps: List[str] = []
        self.last_scientific_hyps: List[Dict[str, Any]] = []
        self.last_scientific_report: str = ""
        self.last_validation_report: str = ""
        self.previous_metrics: Optional[Dict[str, Any]] = None
        self.observer: Optional[ObserverAgent] = None
        self.last_observer_report: str = ""
        self.last_observer_payload: Optional[Dict[str, Any]] = None
        self.last_advanced_report: str = ""
        self.last_advanced_data: Optional[Dict[str, Any]] = None
        self.user_participating: bool = False
        self._step_lock = threading.Lock()

    def reset(self, env_index: int, agents_config: Optional[List[Dict[str, Any]]] = None,
              join_as_participant: bool = False, human_participants: Optional[List[Dict[str, Any]]] = None):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is not set")

        client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            client_kwargs["base_url"] = OPENAI_BASE_URL
        self.client = OpenAI(**client_kwargs)

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

        # Initialize agents from config or use defaults
        # Pass client so LLM can generate Big Five + persona
        if agents_config:
            agents = initialize_agents_from_config(agents_config, client=self.client)
        else:
            agents = initialize_agents(interactive=False)

        self.user_participating = join_as_participant
        if join_as_participant:
            # Original single-user mode
            from .config import USER_AGENT
            user_agent = Agent(
                name=USER_AGENT["name"],
                nature=USER_AGENT["nature"],
                color=USER_AGENT["color"],
                is_human=True,
            )
            agents.insert(0, user_agent)
        # В web-версии human_io=None — пользователь пишет через /api/user_message
        self.dm = DialogueManager(client=self.client, agents=agents, env_context=self.env_context, human_io=None)
        # Initialize observer agent for methodological triangulation
        self.observer = ObserverAgent(client=self.client)
        self.last_observer_report = ""
        self.last_observer_payload = None
        self.last_advanced_report = ""
        self.last_advanced_data = None

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
        self.agents_meta = {
            a.name: {
                "color": a.color,
                "nature": a.nature,
                "big_five": a.big_five.to_dict() if a.big_five else None,
                "is_observer": False,
                "is_human": getattr(a, "is_human", False),
            }
            for a in agents
        }
        # Add observer as +1 agent (non-participating)
        self.agents_meta["Observer"] = {
            "color": "#8E8E93",
            "nature": "observer",
            "big_five": None,
            "is_observer": True,
        }
        self.last_metrics = None
        self.last_hyps = []

    def _build_scientific_summary(self) -> Optional[Dict[str, Any]]:
        if not self.last_metrics or "scientific" not in self.last_metrics:
            return None
        sci = self.last_metrics["scientific"]
        if not hasattr(sci, "group_stage"):
            return None
        return {
            "group_stage": sci.group_stage,
            "network_density": sci.network_density,
            "dominant_emotion": sci.dominant_emotion,
        }

    def _serialize_observer_report(self, report) -> Dict[str, Any]:
        return {
            "report": self.observer.render_report(report) if self.observer else "",
            "convergence_score": report.convergence_score,
            "agreements": report.agreements,
            "divergences": report.divergences,
            "novel_insights": report.novel_insights,
            "hypothesis_summary": report.hypothesis_summary,
            "hypothesis_checks": report.hypothesis_checks,
            "triangulation_summary": self.observer.get_triangulation_summary() if self.observer else {},
        }

    def _merge_hypothesis_registry(self) -> List[Dict[str, Any]]:
        registry = get_hypothesis_registry()
        latest_checks = {
            (item.get("theory"), item.get("hypothesis")): item
            for item in ((self.last_observer_payload or {}).get("hypothesis_checks", []) or [])
        }
        merged: List[Dict[str, Any]] = []
        for item in registry:
            match = latest_checks.get((item["theory"], item["hypothesis"]))
            if match:
                merged.append(
                    {
                        "theory": item["theory"],
                        "citation": item["citation"],
                        "hypothesis": item["hypothesis"],
                        "status": match.get("status", "pending"),
                        "evidence": match.get("evidence", ""),
                    }
                )
            else:
                merged.append(item)
        return merged

    def _build_advanced_summary(self) -> Optional[Dict[str, Any]]:
        return self.last_advanced_data

    def compute_metrics_if_needed(self, record: Dict[str, Any]):
        """Compute graphs, metrics, hypotheses if should_plot(). Returns (image_url, metrics_md, hyps)."""
        image_url = None
        metrics_md = None
        hyps: List[str] = []

        if self.dm and self.dm.should_plot():
            out_path = OUT_DIR / f"graph_turn_{record['turn']:04d}.png"
            draw_interactions_pro(
                agents_meta=self.agents_meta,
                history=list(self.dm.history),
                out_path=out_path,
                title=f"Interactions and tones (before move {record['turn']})",
                window=VIZ_EDGE_WINDOW,
                seed=VIZ_SEED,
                top_edge_labels=0,
                per_message_edges=True,
            )
            image_url = f"/outputs/{out_path.name}"

            agents_order = [a.name for a in self.dm.agents]
            metrics = compute_metrics(self.dm.history, agents_order, window_size=VIZ_EDGE_WINDOW)
            hyps = hypotheses_from_metrics(metrics, user_name=None)
            metrics_md = render_markdown_report(metrics, turn=record["turn"], title="Dialogue — metrics")
            metrics_path = OUT_DIR / f"metrics_turn_{record['turn']:04d}.md"
            save_report_md(metrics_md, metrics_path)

            # Generate scientific report
            scientific = metrics.get("scientific")
            if scientific:
                sci_hyps = generate_scientific_hypotheses(scientific, agents_order)
                sci_report = render_scientific_report(scientific, sci_hyps, record["turn"])
                sci_report_path = OUT_DIR / f"scientific_turn_{record['turn']:04d}.md"
                save_report_md(sci_report, sci_report_path)
                self.last_scientific_hyps = sci_hyps
                self.last_scientific_report = sci_report

                # Validate hypotheses
                try:
                    validation_report = get_hypothesis_validation_report(
                        sci_hyps,
                        metrics,
                        self.previous_metrics,
                        record["turn"],
                        self.dm.history
                    )
                    validation_path = OUT_DIR / f"validation_turn_{record['turn']:04d}.md"
                    save_report_md(validation_report, validation_path)
                    self.last_validation_report = validation_report
                except Exception as e:
                    print(f"[Warning] Hypothesis validation failed: {e}", file=sys.stderr)
                    self.last_validation_report = ""

            try:
                advanced_result = compute_advanced_analysis(self.dm.history, agents_order)
                self.last_advanced_report = render_advanced_report(advanced_result, agents_order)
                self.last_advanced_data = {
                    "contagion_rate": advanced_result.emotional_contagion.contagion_rate,
                    "most_contagious": advanced_result.emotional_contagion.most_contagious_agent,
                    "most_susceptible": advanced_result.emotional_contagion.most_susceptible_agent,
                    "group_lsm": advanced_result.language_style_matching.group_lsm,
                    "discourse_coherence": advanced_result.discourse_coherence.overall_coherence,
                    "thread_continuity": advanced_result.discourse_coherence.thread_continuity,
                    "topic_drift_points": advanced_result.discourse_coherence.topic_drift_points,
                }
            except Exception as e:
                print(f"[Warning] Advanced analysis failed: {e}", file=sys.stderr)
                self.last_advanced_report = ""
                self.last_advanced_data = None

            if self.observer and len(self.dm.history) >= 3:
                try:
                    report = self.observer.observe(
                        dialogue_history=self.dm.history,
                        turn_no=record["turn"],
                        metrics_summary=metrics,
                        scientific_summary=self._build_scientific_summary(),
                        advanced_summary=self._build_advanced_summary(),
                    )
                    self.last_observer_payload = self._serialize_observer_report(report)
                    self.last_observer_report = self.last_observer_payload["report"]
                    observer_path = OUT_DIR / f"observer_turn_{record['turn']:04d}.md"
                    save_report_md(self.last_observer_report, observer_path)
                except Exception as e:
                    print(f"[Warning] Observer checkpoint failed: {e}", file=sys.stderr)

            # Store previous metrics for next validation
            self.previous_metrics = self.last_metrics
            self.last_metrics = metrics
            self.last_hyps = hyps

        return image_url, metrics_md, hyps


STATE = AppState()


@app.route("/")
def index():
    envs = list_env_contexts()
    return render_template("index.html", envs=envs)


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    env_index = int(data.get("env_index", DEFAULT_ENV_CONTEXT_INDEX))
    agents_config = data.get("agents")  # Optional: list of agent configurations
    join_as_participant = bool(data.get("join_as_participant", False))

    try:
        STATE.reset(env_index, agents_config, join_as_participant=join_as_participant)
    except RuntimeError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({
        "ok": True,
        "env": STATE.env_context,
        "turn": 0,
        "history": [],
        "metrics_md": None,
        "agents": list(STATE.agents_meta.keys()),
        "agents_data": [
            {"name": name, **meta}
            for name, meta in STATE.agents_meta.items()
        ],
        "hypothesis_registry": STATE._merge_hypothesis_registry(),
        "user_participating": STATE.user_participating,
    })


@app.route("/api/step", methods=["POST"])
def api_step():
    if not STATE.dm or not STATE.logger:
        return jsonify({"ok": False, "error": "Session not initialized. Call /api/start"}), 400

    record, _ = STATE.dm.step()

    # Multi-user: step() may return a "waiting" signal
    if "waiting_for" in record:
        return jsonify({
            "ok": True,
            "waiting_for": record["waiting_for"],
            "turn": STATE.dm.turn_no,
            "history": STATE.dm.history[-20:],
        })

    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"],
        speaker=record["speaker"],
        target=record.get("target"),
        tone=record["tone"],
        emotion=record["emotion"],
        text=record["reply"],
    )

    image_url, metrics_md, hyps = STATE.compute_metrics_if_needed(record)

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-20:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
        "scientific_hypotheses": STATE.last_scientific_hyps,
        "scientific_report": STATE.last_scientific_report if STATE.dm.should_plot() else None,
        "validation_report": STATE.last_validation_report if STATE.dm.should_plot() else None,
        "has_validation_report": bool(STATE.last_validation_report) or bool(list(OUT_DIR.glob("validation_turn_*.md"))),
        "observer_report": (STATE.last_observer_payload or {}).get("report"),
        "observer_hypothesis_checks": (STATE.last_observer_payload or {}).get("hypothesis_checks", []),
        "observer_hypothesis_summary": (STATE.last_observer_payload or {}).get("hypothesis_summary", ""),
        "hypothesis_registry": STATE._merge_hypothesis_registry(),
        "triangulation_summary": (STATE.last_observer_payload or {}).get("triangulation_summary", {}),
        "addressed_user": record.get("target") == "User",
    })


@app.route("/api/user_message", methods=["POST"])
def api_user_message():
    """Accept a message from the human participant."""
    if not STATE.dm or not STATE.logger:
        return jsonify({"ok": False, "error": "Session not initialized"}), 400

    if not STATE.user_participating:
        return jsonify({"ok": False, "error": "User is not a participant"}), 400

    data = request.get_json(silent=True) or {}
    reply = data.get("reply", "").strip()
    target = data.get("target", "")
    tone = data.get("tone", "auto")

    if not reply:
        return jsonify({"ok": False, "error": "Empty message"}), 400

    # Auto-classify tone — simple fallback
    if tone == "auto":
        tone = "neutral"

    emotion = "neutral"

    record = STATE.dm.inject_user_turn(reply=reply, target=target, tone=tone, emotion=emotion)

    STATE.logger.write_jsonl(record)
    STATE.logger.write_markdown(
        turn_no=record["turn"],
        speaker=record["speaker"],
        target=record.get("target"),
        tone=record["tone"],
        emotion=record["emotion"],
        text=record["reply"],
    )

    image_url, metrics_md, hyps = STATE.compute_metrics_if_needed(record)

    return jsonify({
        "ok": True,
        "record": record,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-20:],
        "image_url": image_url,
        "hypotheses": hyps or STATE.last_hyps,
        "metrics_md": metrics_md,
        "scientific_hypotheses": STATE.last_scientific_hyps,
        "scientific_report": STATE.last_scientific_report if STATE.dm.should_plot() else None,
        "validation_report": STATE.last_validation_report if STATE.dm.should_plot() else None,
        "has_validation_report": bool(STATE.last_validation_report),
        "observer_report": (STATE.last_observer_payload or {}).get("report"),
        "observer_hypothesis_checks": (STATE.last_observer_payload or {}).get("hypothesis_checks", []),
        "observer_hypothesis_summary": (STATE.last_observer_payload or {}).get("hypothesis_summary", ""),
        "hypothesis_registry": STATE._merge_hypothesis_registry(),
        "triangulation_summary": (STATE.last_observer_payload or {}).get("triangulation_summary", {}),
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

    # Проверяем наличие файлов валидации
    validation_files = sorted(OUT_DIR.glob("validation_turn_*.md")) if OUT_DIR.exists() else []
    has_validation = bool(STATE.last_validation_report) or bool(validation_files)

    return jsonify({
        "ok": True,
        "turn": STATE.dm.turn_no,
        "history": STATE.dm.history[-50:],
        "image_url": (f"/outputs/{latest_img}" if latest_img else None),
        "hypotheses": STATE.last_hyps,
        "metrics_md": metrics_md,
        "env": STATE.env_context,
        "scientific_hypotheses": STATE.last_scientific_hyps,
        "scientific_report": STATE.last_scientific_report,
        "validation_report": STATE.last_validation_report,
        "has_validation_report": has_validation,
        "observer_report": (STATE.last_observer_payload or {}).get("report"),
        "observer_hypothesis_checks": (STATE.last_observer_payload or {}).get("hypothesis_checks", []),
        "observer_hypothesis_summary": (STATE.last_observer_payload or {}).get("hypothesis_summary", ""),
        "hypothesis_registry": STATE._merge_hypothesis_registry(),
        "triangulation_summary": (STATE.last_observer_payload or {}).get("triangulation_summary", {}),
    })


@app.route("/outputs/<path:filename>")
def serve_outputs(filename: str):
    return send_from_directory(OUT_DIR, filename)


@app.route("/api/scientific", methods=["GET"])
def api_scientific():
    """Get the latest scientific analysis report."""
    if not STATE.dm or not STATE.last_scientific_report:
        return jsonify({
            "ok": True,
            "report": None,
            "hypotheses": [],
            "message": "No scientific analysis available yet. Run more dialogue turns."
        })

    return jsonify({
        "ok": True,
        "report": STATE.last_scientific_report,
        "hypotheses": STATE.last_scientific_hyps,
    })


@app.route("/api/observer", methods=["POST"])
def api_observer():
    """Trigger observer agent analysis (methodological triangulation)."""
    if not STATE.dm or not STATE.observer:
        return jsonify({"ok": False, "error": "Session not started"}), 400

    history = STATE.dm.history
    if len(history) < 3:
        return jsonify({"ok": False, "error": "Need at least 3 turns for observation"}), 400

    report = STATE.observer.observe(
        dialogue_history=history,
        turn_no=STATE.dm.turn_no,
        metrics_summary=STATE.last_metrics,
        scientific_summary=STATE._build_scientific_summary(),
        advanced_summary=STATE._build_advanced_summary(),
    )
    STATE.last_observer_payload = STATE._serialize_observer_report(report)
    STATE.last_observer_report = STATE.last_observer_payload["report"]

    # Save report to file
    report_path = OUT_DIR / f"observer_turn_{STATE.dm.turn_no:04d}.md"
    report_path.write_text(STATE.last_observer_report, encoding="utf-8")

    return jsonify({
        "ok": True,
        **STATE.last_observer_payload,
        "hypothesis_registry": STATE._merge_hypothesis_registry(),
    })


@app.route("/api/observer/history", methods=["GET"])
def api_observer_history():
    """Get all observer reports and triangulation summary."""
    if not STATE.observer:
        return jsonify({"ok": True, "reports": [], "summary": {}})
    return jsonify({
        "ok": True,
        "reports": [STATE.observer.render_report(r) for r in STATE.observer.history],
        "summary": STATE.observer.get_triangulation_summary(),
    })


@app.route("/api/advanced", methods=["GET"])
def api_advanced():
    """Get advanced analytics: emotional contagion, LSM, discourse coherence."""
    if not STATE.dm or len(STATE.dm.history) < 3:
        return jsonify({"ok": True, "report": None, "message": "Need at least 3 turns."})

    if STATE.last_advanced_report and STATE.last_advanced_data:
        return jsonify({
            "ok": True,
            "report": STATE.last_advanced_report,
            "data": STATE.last_advanced_data,
        })

    agents_list = [a.name for a in STATE.dm.agents]
    result = compute_advanced_analysis(STATE.dm.history, agents_list)
    report_md = render_advanced_report(result, agents_list)

    return jsonify({
        "ok": True,
        "report": report_md,
        "data": {
            "contagion_rate": result.emotional_contagion.contagion_rate,
            "most_contagious": result.emotional_contagion.most_contagious_agent,
            "most_susceptible": result.emotional_contagion.most_susceptible_agent,
            "group_lsm": result.language_style_matching.group_lsm,
            "discourse_coherence": result.discourse_coherence.overall_coherence,
            "thread_continuity": result.discourse_coherence.thread_continuity,
            "topic_drift_points": result.discourse_coherence.topic_drift_points,
        },
    })


@app.route("/download/log")
def download_log():
    path = LOG_DIR / "dialog.jsonl"
    if not path.exists():
        return jsonify({"ok": False, "error": "No log yet"}), 404
    return send_file(path, as_attachment=True, download_name="dialog.jsonl", mimetype="application/json")


@app.route("/download/log_md")
def download_log_md():
    path = LOG_DIR / "dialog.md"
    if not path.exists():
        return jsonify({"ok": False, "error": "No log yet"}), 404
    return send_file(path, as_attachment=True, download_name="dialog.md", mimetype="text/markdown")


@app.route("/download/validation")
def download_validation():
    """Download latest hypothesis validation report with proofs and evidence."""
    # Сначала проверяем, есть ли отчёт в памяти (самый свежий)
    if STATE.last_validation_report:
        # Сохраняем текущий отчёт в файл для скачивания
        turn = STATE.dm.turn_no if STATE.dm else 0
        temp_path = OUT_DIR / f"validation_turn_{turn:04d}_download.md"

        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(STATE.last_validation_report)

            return send_file(
                str(temp_path),
                as_attachment=True,
                download_name=f"hypothesis_validation_turn_{turn:04d}.md",
                mimetype="text/markdown"
            )
        except Exception as e:
            print(f"[Error] Failed to create validation download file: {e}", file=sys.stderr)
            # Fallback к поиску сохранённого файла

    # Если нет в памяти, ищем последний сохранённый файл
    validation_files = sorted(OUT_DIR.glob("validation_turn_*.md"))
    if not validation_files:
        return jsonify({
            "ok": False,
            "error": "No validation report available yet. Run more dialogue turns (validation runs every 5 turns)."
        }), 404

    path = validation_files[-1]  # Последний файл
    turn = path.stem.split("_")[-1]  # Извлекаем номер хода
    return send_file(
        str(path),
        as_attachment=True,
        download_name=f"hypothesis_validation_turn_{turn}.md",
        mimetype="text/markdown"
    )


# ==================== MULTI-USER EXPERIMENT ENDPOINTS ====================

@app.route("/experiment")
def experiment_page():
    return "<h1>Multi-user experiment mode is disabled</h1>", 410


@app.route("/experiment/links")
def experiment_links_page():
    return "<h1>Multi-user experiment mode is disabled</h1>", 410


@app.route("/api/experiment/start", methods=["POST"])
def api_experiment_start():
    return jsonify({"ok": False, "error": "Multi-user experiment mode is disabled"}), 410


@app.route("/api/experiment/join", methods=["POST"])
def api_experiment_join():
    return jsonify({"ok": False, "error": "Multi-user experiment mode is disabled"}), 410


@app.route("/api/experiment/step", methods=["POST"])
def api_experiment_step():
    return jsonify({"ok": False, "error": "Multi-user experiment mode is disabled"}), 410


@app.route("/api/experiment/human_message", methods=["POST"])
def api_experiment_human_message():
    return jsonify({"ok": False, "error": "Multi-user experiment mode is disabled"}), 410


@app.route("/api/experiment/status", methods=["GET"])
def api_experiment_status():
    return jsonify({"ok": False, "error": "Multi-user experiment mode is disabled"}), 410


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Для локального запуска: python -m agent_dialogue_sim.web_app
    app.run(host="0.0.0.0", port=8080, debug=True)
