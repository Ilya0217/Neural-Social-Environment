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

from ..analytics.advanced import compute_advanced_analysis, render_advanced_report
from ..analytics.basic import (
    compute_metrics,
    get_hypothesis_validation_report,
    get_scientific_hypotheses_full,
    hypotheses_from_metrics,
    render_markdown_report,
    save_report_md,
    validate_hypotheses,
)
from ..analytics.scientific import (
    generate_scientific_hypotheses,
    render_scientific_report,
)
from ..analytics.visualize import draw_interactions_pro
from ..config import (
    DEFAULT_ENV_CONTEXT_INDEX,
    LOG_DIR,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OUT_DIR,
    VIZ_EDGE_WINDOW,
    VIZ_SEED,
)
from ..core.agents import Agent
from ..core.dialogue_manager import DialogueManager
from ..core.env_context import get_env_context_by_index, list_env_contexts
from ..core.io_logger import IOLogger
from ..core.observer_agent import ObserverAgent, get_hypothesis_registry
from ..core.profiles import initialize_agents, initialize_agents_from_config

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
              join_as_participant: bool = False, human_participants: Optional[List[Dict[str, Any]]] = None,
              language: str = "ru",
              provider: Optional[str] = None, api_key: Optional[str] = None,
              model_override: Optional[str] = None):
        # Provider configuration: chatgpt (OpenAI) или deepseek (OpenAI-compatible).
        provider = (provider or "").strip().lower() or None
        provider_specs = {
            "chatgpt": {"base_url": None, "model": "gpt-4o-mini"},
            "openai":  {"base_url": None, "model": "gpt-4o-mini"},
            "deepseek": {"base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
        }
        spec = provider_specs.get(provider) if provider else None

        effective_key = (api_key or "").strip() or OPENAI_API_KEY
        if not effective_key:
            raise RuntimeError("API key is not set — выберите провайдера и введите токен на первом шаге.")

        if spec is not None:
            effective_base_url = spec["base_url"]
            effective_model = model_override or spec["model"]
        else:
            effective_base_url = OPENAI_BASE_URL
            effective_model = model_override or None

        # Применяем модель глобально (config.MODEL и уже импортированные ссылки на MODEL).
        if effective_model:
            from .. import config as _cfg
            from ..core import dialogue_manager as _dm_mod
            _cfg.MODEL = effective_model
            _dm_mod.MODEL = effective_model

        client_kwargs: Dict[str, Any] = {"api_key": effective_key}
        if effective_base_url:
            client_kwargs["base_url"] = effective_base_url
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
            from ..config import USER_AGENT
            user_agent = Agent(
                name=USER_AGENT["name"],
                nature=USER_AGENT["nature"],
                color=USER_AGENT["color"],
                is_human=True,
            )
            agents.insert(0, user_agent)
        # В web-версии human_io=None — пользователь пишет через /api/user_message
        self.dm = DialogueManager(
            client=self.client, agents=agents,
            env_context=self.env_context, human_io=None,
            language=(language if language in ("ru", "en") else "ru"),
        )
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

    def _build_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Компактный JSON для UI: только числа, никакого markdown.
        Включает доп. поля для проверки гипотез H1-H5: OCEAN-черты агентов,
        счётчик validation issues, флаг CoT, n_agents."""
        if not metrics:
            return {}
        s_all = (metrics.get("summary") or {}).get("all") or {}
        s_win = (metrics.get("summary") or {}).get("window") or {}
        per_agent = metrics.get("per_agent") or {}
        tone_fracs = s_win.get("tone_fracs") or {}
        talk_share = s_win.get("talk_share") or {}
        top_speaker = None
        top_share = 0.0
        for name, share in talk_share.items():
            if share > top_share:
                top_share = share
                top_speaker = name

        # Для H1 — OCEAN из агентов
        agents_ocean: Dict[str, Dict[str, float]] = {}
        for name, meta in self.agents_meta.items():
            bf = meta.get("big_five") or {}
            if bf:
                agents_ocean[name] = {k: float(v) for k, v in bf.items()}

        # Для H3 — суммарные validation issues по истории
        total_validation_issues = 0
        if self.dm:
            for rec in self.dm.history:
                total_validation_issues += int(rec.get("validation_issues") or 0)

        # Для H4 — флаг CoT
        cot_enabled = bool(self.dm.enable_cot) if self.dm else False
        # Для H5 — общее число агентов (без User/Observer)
        n_dialogue_agents = sum(
            1 for a in (self.dm.agents if self.dm else [])
            if not getattr(a, "is_human", False)
        )

        per_agent_dict = {
            name: {
                "msgs": int(info.get("msgs", 0)),
                "avg_words": float(info.get("avg_words", 0.0)),
                "avg_tone": float(info.get("avg_tone", 0.0)),
                "last_emotion": info.get("last_emotion", "neutral"),
                "targets_diversity": int(info.get("targets_diversity", 0)),
                "big_five": agents_ocean.get(name),
            }
            for name, info in per_agent.items()
        }

        return {
            "window_size": metrics.get("window_size"),
            "messages_window": s_win.get("messages", 0),
            "messages_total": s_all.get("messages", 0),
            "avg_tone": float(s_win.get("avg_tone", 0.0)),
            "targeting_rate": float(s_win.get("targeting_rate", 0.0)),
            "positive_frac": float(tone_fracs.get("pos", 0.0)),
            "negative_frac": float(tone_fracs.get("neg", 0.0)),
            "neutral_frac": float(tone_fracs.get("neu", 0.0)),
            "emotion_entropy": float(s_win.get("emotion_entropy", 0.0)),
            "emotion_diversity": int(s_win.get("emotion_diversity", 0)),
            "avg_reply_words": float(s_win.get("avg_reply_words", 0.0)),
            "reciprocity": float(s_win.get("reciprocity", 0.0)),
            "question_rate": float(s_win.get("question_rate", 0.0)),
            "actionability_rate": float(s_win.get("actionability_rate", 0.0)),
            "reference_rate": float(s_win.get("reference_rate", 0.0)),
            "addressing_delay": float(s_win.get("avg_addressing_delay", 0.0)),
            "top_speaker": top_speaker,
            "top_speaker_share": float(top_share),
            "talk_share": {k: float(v) for k, v in talk_share.items()},
            "per_agent": per_agent_dict,
            "agents_ocean": agents_ocean,
            "total_validation_issues": total_validation_issues,
            "cot_enabled": cot_enabled,
            "n_dialogue_agents": n_dialogue_agents,
        }

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
                title=f"Сеть взаимодействий — ход {record['turn']}",
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
    language = str(data.get("language") or "ru").lower()
    if language not in ("ru", "en"):
        language = "ru"
    provider = data.get("provider")
    api_key = data.get("api_key")
    model_override = data.get("model")

    try:
        STATE.reset(
            env_index, agents_config,
            join_as_participant=join_as_participant,
            language=language,
            provider=provider,
            api_key=api_key,
            model_override=model_override,
        )
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
        "metrics_summary": STATE._build_metrics_summary(STATE.last_metrics) if STATE.last_metrics else None,
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
        "metrics_summary": STATE._build_metrics_summary(STATE.last_metrics) if STATE.last_metrics else None,
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
        "metrics_summary": STATE._build_metrics_summary(STATE.last_metrics) if STATE.last_metrics else None,
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


# ==================== НАУЧНЫЙ ОТЧЁТ ПО ГИПОТЕЗАМ (для научрука) ====================

def _build_verdicts_for_report(m: Dict[str, Any]) -> List[Dict[str, str]]:
    """Проверка пяти научных гипотез H1–H5 на основе текущих метрик.

    H1. Big Five extraversion → длина реплики (Soto & John 2017; Park et al. 2015)
    H2. Воспроизведение стадий группового развития (Wheelan 2016; Bonebright 2010)
    H3. Эффективность многоуровневой валидации (Beauchamp & Childress 2019)
    H4. Эффективность Chain-of-Thought (Wei et al. 2022)
    H5. Соответствие turn-taking человеческим корпусам (Levinson & Torreira 2015; Dunbar et al. 2015)
    """
    if not m or not m.get("messages_window"):
        return []

    per_agent = m.get("per_agent") or {}
    n_agents = max(1, int(m.get("n_dialogue_agents") or len(per_agent) or 1))
    rec = m.get("reciprocity", 0.0)
    act = m.get("actionability_rate", 0.0)
    neg = m.get("negative_frac", 0.0)
    pos = m.get("positive_frac", 0.0)
    qrate = m.get("question_rate", 0.0)
    top_share = m.get("top_speaker_share", 0.0)
    top_speaker = m.get("top_speaker", "—")
    avg_words = m.get("avg_reply_words", 0.0)
    targeting = m.get("targeting_rate", 0.0)
    val_issues = int(m.get("total_validation_issues") or 0)
    total_msgs = int(m.get("messages_total") or 0)
    cot_on = bool(m.get("cot_enabled", False))

    out: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # H1. Влияние экстраверсии на длину реплики
    # Тест: Спирмен-ранг корреляция между Extraversion агента и его avg_words.
    # ------------------------------------------------------------------
    pairs = []
    for name, info in per_agent.items():
        bf = (info or {}).get("big_five") or {}
        ext = bf.get("extraversion")
        words = (info or {}).get("avg_words")
        msgs = (info or {}).get("msgs", 0)
        if ext is not None and words is not None and msgs > 0:
            pairs.append((float(ext), float(words), name))
    if len(pairs) >= 3:
        # Спирмен через ранги
        ext_ranks = {p[2]: r for r, p in enumerate(sorted(pairs, key=lambda x: x[0]))}
        wrd_ranks = {p[2]: r for r, p in enumerate(sorted(pairs, key=lambda x: x[1]))}
        n = len(pairs)
        d2 = sum((ext_ranks[name] - wrd_ranks[name])**2 for _, _, name in pairs)
        rho = 1.0 - (6.0 * d2) / (n * (n*n - 1))
        ranking_str = ", ".join(
            f"{name}: E={e:.2f}/words={w:.1f}" for e, w, name in sorted(pairs, key=lambda x: -x[0])
        )
        if rho >= 0.5:
            status = "confirmed"
            claim = "Экстраверсия каузально связана с длиной реплики (положительная корреляция)."
        elif rho > 0:
            status = "partial"
            claim = "Положительный, но слабый ранговый тренд: данных мало для надёжного вывода."
        else:
            status = "refuted"
            claim = "Связи не наблюдается или обратная (требует репликации с N>30)."
        evidence = f"Spearman ρ = {rho:+.3f} (n={n} агентов). Ранжирование: {ranking_str}."
    elif len(pairs) == 2:
        a, b = pairs
        same_dir = (a[0] > b[0]) == (a[1] > b[1])
        rho = +1.0 if same_dir else -1.0
        status = "partial" if same_dir else "refuted"
        claim = ("Только 2 точки — направление совпадает с прогнозом."
                 if same_dir else "Только 2 точки — направление противоречит прогнозу.")
        evidence = f"n=2 (недостаточно для статистики). {a[2]}: E={a[0]:.2f}/words={a[1]:.1f}; {b[2]}: E={b[0]:.2f}/words={b[1]:.1f}."
    else:
        status = "pending"
        claim = "Недостаточно говорящих агентов с OCEAN-профилями для проверки H1."
        evidence = f"Найдено {len(pairs)} активных агентов с extraversion."
    out.append(dict(
        theory="H1. Экстраверсия → длина реплики",
        framework="Big Five (BFI-2); LIWC-like behavioral signals",
        ref="Soto & John (2017) — BFI-2; Park et al. (2015) — корпус 65 тыс. пользователей",
        status=status, claim=claim, evidence=evidence,
    ))

    # ------------------------------------------------------------------
    # H2. Воспроизведение стадий группового развития (Wheelan)
    # Forming → Storming → Norming → Performing.
    # ------------------------------------------------------------------
    # Эвристическая классификация по (act, rec, neg, qrate):
    if act >= 0.3 and rec >= 0.5:
        stage = "Performing"; status = "confirmed"
    elif neg >= 0.3:
        stage = "Storming"; status = "confirmed"
    elif qrate >= 0.3 and act < 0.2 and rec < 0.4:
        stage = "Forming"; status = "confirmed"
    elif rec >= 0.4 and act < 0.3 and neg < 0.2:
        stage = "Norming"; status = "confirmed"
    else:
        stage = "переходное состояние"; status = "partial"
    claim = f"Текущая стадия группового развития: {stage}."
    evidence = (
        f"actionability = {round(act*100)}%, reciprocity = {round(rec*100)}%, "
        f"negative = {round(neg*100)}%, questions = {round(qrate*100)}%."
    )
    out.append(dict(
        theory="H2. Воспроизведение стадий группового развития",
        framework="Интегрированная модель развития малой группы",
        ref="Wheelan (2016); Bonebright (2010)",
        status=status, claim=claim, evidence=evidence,
    ))

    # ------------------------------------------------------------------
    # H3. Эффективность многоуровневой валидации
    # Тест: доля «непрошедших» (заменённых fallback'ом) сообщений должна быть низкой.
    # ------------------------------------------------------------------
    if total_msgs == 0:
        status = "pending"; claim = "Нет реплик для оценки."; evidence = "—"
    else:
        rate = val_issues / total_msgs
        if rate < 0.15:
            status = "confirmed"
            claim = "Многоуровневая валидация эффективно фильтрует акты речи."
            evidence = (f"{val_issues}/{total_msgs} реплик помечены валидаторами "
                        f"({rate*100:.1f}%, ниже порога 15%).")
        elif rate < 0.35:
            status = "partial"
            claim = "Валидация работает, но процент срабатываний высок."
            evidence = (f"{val_issues}/{total_msgs} реплик помечены валидаторами "
                        f"({rate*100:.1f}%).")
        else:
            status = "refuted"
            claim = "Слишком много замечаний — валидаторы либо слишком строги, либо модель отвечает плохо."
            evidence = (f"{val_issues}/{total_msgs} реплик помечены валидаторами "
                        f"({rate*100:.1f}%, выше порога 35%).")
    out.append(dict(
        theory="H3. Эффективность многоуровневой валидации",
        framework="Биоэтические принципы как критерии приемлемости речевых актов",
        ref="Beauchamp & Childress (2019) — Principles of Biomedical Ethics, 8th ed.",
        status=status, claim=claim, evidence=evidence,
    ))

    # ------------------------------------------------------------------
    # H4. Эффективность Chain-of-Thought (CoT)
    # Если CoT включён, проверяем, что actionability_rate выше типичной baseline (~0.20).
    # Если выключен — pending (нет условия для тестирования в этой сессии).
    # ------------------------------------------------------------------
    if not cot_on:
        status = "pending"
        claim = "CoT-промптинг в данной сессии не включён — гипотеза не тестируется."
        evidence = ("Для проверки H4 запустите эксперимент через "
                    "experiments/run_experiments.py --config h6_cot.yaml")
    else:
        if act >= 0.40:
            status = "confirmed"
            claim = "CoT существенно повышает долю конкретных шагов."
            evidence = f"actionability = {round(act*100)}% (≥40% — заметно выше baseline ~20%)."
        elif act >= 0.25:
            status = "partial"
            claim = "CoT даёт умеренное улучшение."
            evidence = f"actionability = {round(act*100)}% (выше baseline, но < 40%)."
        else:
            status = "refuted"
            claim = "CoT не привёл к ожидаемому росту actionability."
            evidence = f"actionability = {round(act*100)}%."
    out.append(dict(
        theory="H4. Эффективность Chain-of-Thought-промптинга",
        framework="CoT улучшает многошаговые рассуждения LLM",
        ref="Wei et al. (2022) — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
        status=status, claim=claim, evidence=evidence,
    ))

    # ------------------------------------------------------------------
    # H5. Соответствие turn-taking человеческим корпусам
    # Levinson & Torreira: ~равномерное распределение, минимальные паузы между ходами.
    # Dunbar et al.: в малых группах каждый говорит ~ 1/N доли (≤ 1.5/N для «человекоподобия»).
    # ------------------------------------------------------------------
    expected_share = 1.0 / n_agents
    threshold = min(0.55, expected_share * 1.5 + 0.05)
    if top_share <= threshold and targeting >= 0.8:
        status = "confirmed"
        claim = "Turn-taking соответствует человеческой норме: равномерно и адресно."
        evidence = (f"top speaker {top_speaker} = {round(top_share*100)}% "
                    f"(порог ~ 1.5/N = {round(threshold*100)}%); targeting = {round(targeting*100)}%.")
    elif top_share > expected_share * 2:
        status = "refuted"
        claim = f"Сильное доминирование «{top_speaker}» отклоняет от человеческой структуры."
        evidence = (f"{round(top_share*100)}% реплик от одного агента при ожидаемых "
                    f"{round(expected_share*100)}% (1/N).")
    else:
        status = "partial"
        claim = "Turn-taking близок к норме, но с перекосом."
        evidence = (f"top speaker {top_speaker} = {round(top_share*100)}%; "
                    f"targeting = {round(targeting*100)}%; ожидание 1/N = {round(expected_share*100)}%.")
    out.append(dict(
        theory="H5. Соответствие turn-taking человеческим корпусам",
        framework="Структура распределения реплик в реальных малых группах",
        ref="Levinson & Torreira (2015); Dunbar et al. (2015) — Group Size & Conversational Structure",
        status=status, claim=claim, evidence=evidence,
    ))

    return out


def _render_scientific_report_md() -> str:
    """Собирает markdown-отчёт по текущей сессии."""
    from datetime import datetime, timezone

    md: List[str] = []
    md.append("# Научный отчёт по диалогу")
    md.append("")
    md.append(f"**Дата:** {datetime.now(timezone.utc).isoformat()}  ")
    md.append(f"**Окружение:** {STATE.env_context}  ")
    if STATE.dm:
        md.append(f"**Число ходов:** {STATE.dm.turn_no}  ")
        md.append(f"**Язык диалога:** {STATE.dm.language}  ")
        md.append(f"**Участники:** {', '.join(a.name for a in STATE.dm.agents)}")
    md.append("")

    if not STATE.last_metrics:
        md.append("> Отчёт пуст — метрики ещё не накоплены (минимум 5 ходов).")
        return "\n".join(md)

    ms = STATE._build_metrics_summary(STATE.last_metrics)

    # --- Сводная таблица метрик ---
    md.append("## 1. Ключевые метрики окна")
    md.append("")
    md.append("| Метрика | Значение | Источник |")
    md.append("|---|---:|---|")
    md.append(f"| Средний тон | {ms['avg_tone']:+.3f} | TONE_SCORE |")
    md.append(f"| Доля позитивных реплик | {ms['positive_frac']*100:.1f}% | TONE_SCORE |")
    md.append(f"| Доля негативных реплик | {ms['negative_frac']*100:.1f}% | TONE_SCORE |")
    md.append(f"| Адресных обращений | {ms['targeting_rate']*100:.1f}% | DialogueAct (ISO 24617-2) |")
    md.append(f"| Взаимность связей | {ms['reciprocity']*100:.1f}% | Borgatti SNA |")
    md.append(f"| Разнообразие эмоций (Shannon) | {ms['emotion_entropy']:.3f} bits | Cowen-Keltner |")
    md.append(f"| Уникальных эмоций | {ms['emotion_diversity']} | Cowen-Keltner |")
    md.append(f"| Средняя длина реплики | {ms['avg_reply_words']:.1f} слов | LIWC (LSM) |")
    md.append(f"| Доля вопросов | {ms['question_rate']*100:.1f}% | ISO 24617-2 |")
    md.append(f"| Конкретные шаги (actionability) | {ms['actionability_rate']*100:.1f}% | Wheelan (Performing) |")
    md.append(f"| Доминирование | {ms['top_speaker_share']*100:.1f}% ({ms['top_speaker']}) | Sacks et al. |")
    md.append("")

    # --- По-агентная таблица ---
    if ms.get("per_agent"):
        md.append("## 2. По агентам")
        md.append("")
        md.append("| Агент | Реплик | Ср. слов | Ср. тон | Последняя эмоция | Разнообразие адресатов |")
        md.append("|---|---:|---:|---:|---|---:|")
        for name, info in sorted(ms["per_agent"].items(), key=lambda x: -x[1]["msgs"]):
            md.append(f"| {name} | {info['msgs']} | {info['avg_words']:.1f} | "
                      f"{info['avg_tone']:+.2f} | {info['last_emotion']} | {info['targets_diversity']} |")
        md.append("")

    # --- Verdicts ---
    verdicts = _build_verdicts_for_report(ms)
    badge_map = {
        "confirmed": "✅ ПОДТВЕРЖДЕНА",
        "partial":   "⚠️ ЧАСТИЧНО",
        "refuted":   "❌ ОТКЛОНЕНА",
        "pending":   "⏸ НЕОПРЕДЕЛЕНО",
    }
    md.append("## 3. Подтверждение научных теорий и гипотез")
    md.append("")
    counts = {"confirmed": 0, "partial": 0, "refuted": 0, "pending": 0}
    for v in verdicts:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    md.append(
        f"**Сводка:** ✅ {counts['confirmed']} подтверждено · "
        f"⚠️ {counts['partial']} частично · "
        f"❌ {counts['refuted']} отклонено · "
        f"⏸ {counts['pending']} неопределено"
    )
    md.append("")
    md.append("| # | Теория | Фреймворк | Статус |")
    md.append("|---:|---|---|:---:|")
    for i, v in enumerate(verdicts, 1):
        md.append(f"| {i} | {v['theory']} | {v['framework']} | {badge_map[v['status']]} |")
    md.append("")
    md.append("### Подробно по каждой теории")
    md.append("")
    for i, v in enumerate(verdicts, 1):
        md.append(f"#### {i}. {v['theory']} — {v['framework']} · {badge_map[v['status']]}")
        md.append("")
        md.append(f"**Утверждение:** {v['claim']}")
        md.append("")
        md.append(f"**Доказательство:** {v['evidence']}")
        md.append("")

    md.append("---")
    md.append("")
    md.append("*Отчёт сгенерирован автоматически модулем `agent_dialogue_sim.webapp` "
              "на основе метрик последнего окна диалога.*")
    return "\n".join(md)


@app.route("/api/scientific_report", methods=["GET"])
def api_scientific_report_text():
    """Возвращает markdown текст отчёта в JSON (для предпросмотра в UI)."""
    return jsonify({"ok": True, "report_md": _render_scientific_report_md()})


@app.route("/download/scientific_report", methods=["GET"])
def download_scientific_report():
    """Отдаёт markdown отчёт как файл для скачивания."""
    md = _render_scientific_report_md()
    from io import BytesIO
    from flask import Response
    return Response(
        md,
        mimetype="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="scientific_report.md"'
        },
    )


# ==================== CAUSAL INTERVENTION ENDPOINTS ====================

import json as _json
import uuid as _uuid

from ..science.causal_intervention import (
    CausalConfig as _CausalConfig,
    CausalRunner as _CausalRunner,
    TraitIntervention as _TraitIntervention,
    DEFAULT_OUTCOME_METRICS as _DEFAULT_OUTCOME_METRICS,
)
from ..experiments.run_causal import build_scientific_report as _build_scientific_report

_RESULTS_BASE = Path(__file__).resolve().parent.parent / "experiments" / "results"
_CAUSAL_JOBS: Dict[str, Dict[str, Any]] = {}
_CAUSAL_JOBS_LOCK = threading.Lock()


def _list_causal_experiments() -> List[Dict[str, Any]]:
    """Сканирует experiments/results/ и возвращает список каузальных прогонов."""
    items: List[Dict[str, Any]] = []
    if not _RESULTS_BASE.exists():
        return items
    for d in sorted(_RESULTS_BASE.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # каузальный summary отличается от обычного — у него есть ate_by_metric
        if "ate_by_metric" not in summary:
            continue
        items.append({
            "dir": d.name,
            "experiment_id": summary.get("experiment_id"),
            "timestamp": summary.get("timestamp"),
            "n_pairs_total": summary.get("n_pairs_total"),
            "n_pairs_valid": summary.get("n_pairs_valid"),
            "treatment_descriptor": (summary.get("intervention_treatment") or {}).get("descriptor"),
            "control_descriptor": (summary.get("intervention_control") or {}).get("descriptor"),
            "duration_seconds": summary.get("duration_seconds"),
            "has_report": (d / "REPORT.md").exists(),
            "has_nauchnyi": (d / "НАУЧНЫЙ_ОТЧЁТ.md").exists(),
        })
    return items


@app.route("/causal")
def causal_page():
    return render_template("causal.html")


@app.route("/api/causal/list", methods=["GET"])
def api_causal_list():
    return jsonify({"ok": True, "experiments": _list_causal_experiments()})


@app.route("/api/causal/experiment/<exp_dir>", methods=["GET"])
def api_causal_experiment(exp_dir: str):
    d = _RESULTS_BASE / exp_dir
    if not d.exists() or not d.is_dir():
        return jsonify({"ok": False, "error": "experiment not found"}), 404
    summary_path = d / "summary.json"
    if not summary_path.exists():
        return jsonify({"ok": False, "error": "summary.json missing"}), 404
    summary = _json.loads(summary_path.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {"ok": True, "summary": summary}
    report_md = d / "REPORT.md"
    if report_md.exists():
        out["report_md"] = report_md.read_text(encoding="utf-8")
    nauchnyi = d / "НАУЧНЫЙ_ОТЧЁТ.md"
    if nauchnyi.exists():
        out["nauchnyi_md"] = nauchnyi.read_text(encoding="utf-8")
    # Список seed для пар (без полных историй)
    pairs_path = d / "pairs.jsonl"
    pair_seeds: List[Dict[str, Any]] = []
    if pairs_path.exists():
        for line in pairs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                p = _json.loads(line)
            except Exception:
                continue
            pair_seeds.append({
                "seed": p.get("seed"),
                "valid": p.get("valid"),
                "ite": p.get("ite", {}),
            })
    out["pairs"] = pair_seeds
    return jsonify(out)


@app.route("/api/causal/pair/<exp_dir>/<int:seed>", methods=["GET"])
def api_causal_pair(exp_dir: str, seed: int):
    d = _RESULTS_BASE / exp_dir
    pairs_path = d / "pairs.jsonl"
    if not pairs_path.exists():
        return jsonify({"ok": False, "error": "pairs.jsonl missing"}), 404
    for line in pairs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            p = _json.loads(line)
        except Exception:
            continue
        if p.get("seed") == seed:
            return jsonify({
                "ok": True,
                "seed": seed,
                "valid": p.get("valid"),
                "ite": p.get("ite", {}),
                "treatment_history": (p.get("treatment") or {}).get("history", []),
                "control_history": (p.get("control") or {}).get("history", []),
                "treatment_metrics": {k: v for k, v in (p.get("treatment") or {}).items() if k != "history"},
                "control_metrics": {k: v for k, v in (p.get("control") or {}).items() if k != "history"},
            })
    return jsonify({"ok": False, "error": f"seed {seed} not found"}), 404


def _run_causal_job(job_id: str, params: Dict[str, Any]) -> None:
    """Запускает CausalRunner в фоне; обновляет _CAUSAL_JOBS[job_id]."""
    try:
        treat = _TraitIntervention(
            name="treatment",
            target_agent=params["target_agent"],
            trait=params["trait"],
            value=float(params["treatment_value"]),
        )
        ctrl = _TraitIntervention(
            name="control",
            target_agent=params["target_agent"],
            trait=params["trait"],
            value=float(params["control_value"]),
        )
        cfg = _CausalConfig(
            experiment_id=params["experiment_id"],
            intervention_treatment=treat,
            intervention_control=ctrl,
            n_pairs=int(params["n_pairs"]),
            n_turns=int(params["n_turns"]),
            seed_base=int(params.get("seed_base", 9000)),
            env_context_index=int(params.get("env_context_index", 0)),
            parallelism=int(params.get("parallelism", 3)),
            outcome_metrics=list(_DEFAULT_OUTCOME_METRICS),
            alpha=float(params.get("alpha", 0.05)),
            d_threshold=float(params.get("d_threshold", 0.3)),
        )
        runner = _CausalRunner(cfg)
        with _CAUSAL_JOBS_LOCK:
            _CAUSAL_JOBS[job_id]["status"] = "running"
        result = runner.run(_RESULTS_BASE)
        report_md = _build_scientific_report(result)
        out_dir = Path(result.config["output_dir"])
        (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")
        with _CAUSAL_JOBS_LOCK:
            _CAUSAL_JOBS[job_id]["status"] = "done"
            _CAUSAL_JOBS[job_id]["result_dir"] = out_dir.name
            _CAUSAL_JOBS[job_id]["duration"] = result.duration_seconds
    except Exception as e:
        import traceback as _tb
        with _CAUSAL_JOBS_LOCK:
            _CAUSAL_JOBS[job_id]["status"] = "error"
            _CAUSAL_JOBS[job_id]["error"] = f"{type(e).__name__}: {e}"
            _CAUSAL_JOBS[job_id]["traceback"] = _tb.format_exc()


@app.route("/api/causal/run", methods=["POST"])
def api_causal_run():
    data = request.get_json(silent=True) or {}
    required = ["target_agent", "trait", "treatment_value", "control_value", "n_pairs", "n_turns"]
    for k in required:
        if k not in data:
            return jsonify({"ok": False, "error": f"missing {k}"}), 400
    if "experiment_id" not in data or not data["experiment_id"]:
        data["experiment_id"] = (
            f"WEB_{data['trait']}_{data['target_agent']}_"
            f"T{float(data['treatment_value']):.2f}_C{float(data['control_value']):.2f}"
        )
    # ограничители безопасности
    if int(data["n_pairs"]) < 1 or int(data["n_pairs"]) > 24:
        return jsonify({"ok": False, "error": "n_pairs must be in [1, 24]"}), 400
    if int(data["n_turns"]) < 2 or int(data["n_turns"]) > 20:
        return jsonify({"ok": False, "error": "n_turns must be in [2, 20]"}), 400

    job_id = _uuid.uuid4().hex[:12]
    with _CAUSAL_JOBS_LOCK:
        _CAUSAL_JOBS[job_id] = {
            "job_id": job_id,
            "params": data,
            "status": "queued",
            "started_at": time.time(),
        }
    t = threading.Thread(target=_run_causal_job, args=(job_id, data), daemon=True)
    t.start()
    return jsonify({"ok": True, "job_id": job_id})


@app.route("/api/causal/jobs", methods=["GET"])
def api_causal_jobs():
    with _CAUSAL_JOBS_LOCK:
        jobs = list(_CAUSAL_JOBS.values())
    return jsonify({"ok": True, "jobs": jobs})


@app.route("/api/causal/job/<job_id>", methods=["GET"])
def api_causal_job_status(job_id: str):
    with _CAUSAL_JOBS_LOCK:
        job = _CAUSAL_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    return jsonify({"ok": True, "job": job})


def create_app() -> Flask:
    return app


if __name__ == "__main__":
    # Для локального запуска: python -m agent_dialogue_sim.web_app
    app.run(host="0.0.0.0", port=8080, debug=True)
