"""
Batch experiment runner for the dialogue simulator.

Usage:
    python -m agent_dialogue_sim.experiments.run_experiments --turns 12 --runs 3 --env 0
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from ..config import (
    DEFAULT_AGENTS,
    DEFAULT_ENV_CONTEXT_INDEX,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    VIZ_EDGE_WINDOW,
)
from ..agents import Agent
from ..dialogue_manager import DialogueManager
from ..env_context import get_env_context_by_index
from ..analytics import compute_metrics, hypotheses_from_metrics, render_markdown_report


def run_session(turns: int, env_index: int, seed: int | None = None) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    if seed is not None:
        random.seed(seed)

    client = OpenAI(**({"base_url": OPENAI_BASE_URL} if OPENAI_BASE_URL else {}))
    env_context = get_env_context_by_index(env_index)
    agents = [Agent(**cfg) for cfg in DEFAULT_AGENTS]
    dm = DialogueManager(client=client, agents=agents, env_context=env_context, human_io=None)

    history: List[Dict[str, Any]] = []
    for _ in range(turns):
        record, _ = dm.step()
        history.append(record)

    agent_names = [a.name for a in agents]
    metrics = compute_metrics(history, agent_names, window_size=VIZ_EDGE_WINDOW)
    hyps = hypotheses_from_metrics(metrics)
    metrics_md = render_markdown_report(metrics, turn=dm.turn_no, title="Dialogue — metrics")

    return {
        "env_context": env_context,
        "turns": dm.turn_no,
        "history": history,
        "metrics": metrics,
        "hypotheses": hyps,
        "metrics_md": metrics_md,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated dialogue experiments.")
    parser.add_argument("--turns", type=int, default=12, help="Number of turns per session.")
    parser.add_argument("--runs", type=int, default=3, help="How many sessions to run.")
    parser.add_argument("--env", type=int, default=DEFAULT_ENV_CONTEXT_INDEX, help="Environment index (0..N).")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (per run + index).")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_results"), help="Directory to store outputs.")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    index_entries = []
    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx if args.seed is not None else None
        print(f"[experiment] Run {run_idx + 1}/{args.runs} (seed={run_seed})")
        session = run_session(args.turns, args.env, run_seed)

        fname = f"session_{run_idx:02d}.json"
        (output_dir / fname).write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        md_name = f"session_{run_idx:02d}.md"
        (output_dir / md_name).write_text(session["metrics_md"], encoding="utf-8")

        index_entries.append(
            {
                "file": fname,
                "env_context": session["env_context"],
                "turns": session["turns"],
                "hypotheses": session["hypotheses"],
            }
        )

    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps({"generated": datetime.utcnow().isoformat(), "runs": index_entries}, indent=2), encoding="utf-8")
    print(f"[experiment] Saved {args.runs} runs to {output_dir}")


if __name__ == "__main__":
    main()

