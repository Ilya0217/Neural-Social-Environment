# Agent Dialogue Simulator (Web + CLI)

An agent-to-agent dialogue simulator with analytics, visualizations, and a clean web UI. Agents talk in short, human-like English messages, addressing a single addressee each turn. Every 5 turns the app updates metrics, hypotheses, and an interaction graph.

## Features
- Multi-agent dialogue with strict addressee selection per turn
- English-only prompts and replies
- Metrics and hypotheses generated every 5 turns
- Beautiful per-message interaction graph saved to `outputs/`
- Web UI: transcript, graph, hypotheses, metrics table
- One-click download of dialogue log as JSONL
- Auto-clean of `outputs/` images/reports and `logs/dialog.*` on each new session

## Requirements
- Python 3.10+
- An OpenAI API key

Install dependencies:
```bash
pip install -r requirements.txt
```

Set environment variables (create a `.env` if convenient):
```bash
# Required
OPENAI_API_KEY=sk-...
# Optional
OPENAI_MODEL=gpt-4o-mini
ENV_CONTEXT_INDEX=0   # 0..2
```

## Project Structure
```text
agent_dialogue_sim/
  analytics.py        # metrics, hypotheses, markdown report
  dialogue_manager.py # dialogue turn logic, model/tool-calling
  visualize.py        # interaction graph rendering
  web_app.py          # Flask app (UI + APIs)
  main.py             # CLI simulation (interactive)
  prompts.py          # agent schemas and system prompts
  config.py           # settings (models, plotting cadence, agents)
  env_context.py      # 3 environment/context presets (English)
  io_logger.py        # JSONL + markdown logging
  static/             # CSS/JS assets
  templates/          # Jinja templates (index.html)
  outputs/            # generated graphs/metrics (auto-cleaned per session)
  logs/               # dialog.jsonl, dialog.md (auto-cleaned per session)
```

## Run the Web UI
Start the Flask app:
```bash
python -m agent_dialogue_sim.web_app
```
Open `http://localhost:5000`.

1) Select an environment
2) Click “Start” (this clears `outputs/` and `logs/dialog.*` for a fresh session)
3) Click “Next message” to advance turns
4) Every 5 turns the Metrics and Graph update; Hypotheses refresh as well
5) Click “Download JSONL” to download `logs/dialog.jsonl`

### Endpoints
- `GET /` — UI
- `POST /api/start` — start/reset a session; cleans outputs and logs
- `POST /api/step` — advance one turn; returns latest record, (optionally) graph URL, hypotheses, metrics markdown
- `GET /api/state` — current turn, recent history, last graph URL, hypotheses, metrics md
- `GET /outputs/<file>` — serve generated images/reports
- `GET /download/log` — download `logs/dialog.jsonl`

## Run the CLI (terminal)
```bash
python -m agent_dialogue_sim.main
```
- Choose an environment (0..2)
- Press Enter to generate the next message
- On every 5th turn, the app saves a graph and a metrics markdown report into `outputs/`

## Outputs
- Graphs: `outputs/graph_turn_XXXX.png`
- Metrics report: `outputs/metrics_turn_XXXX.md`
- Trends (if present): `outputs/trend_*.png`
- Logs: `logs/dialog.jsonl` (each line is a JSON record), `logs/dialog.md`

## Configuration
Edit `config.py`:
- `OPENAI_MODEL`, `TEMPERATURE`, `MAX_TOKENS`
- `TURNS_BETWEEN_PLOTS` (default 5)
- `VIZ_EDGE_WINDOW` (window of recent turns for edges)
- Default agents and colors

## Notes & Tips
- English-only: system prompts enforce English; start a new session to ensure the constraint applies from turn 1.
- Cleaning: starting a session deletes `graph_turn_*.png`, `metrics_turn_*.md`, `trend_*.png`, and `logs/dialog.*`, then recreates fresh empty `dialog.jsonl` and `dialog.md`.
- JSONL format: download via the UI button; each line is a single turn with fields like `turn`, `speaker`, `target`, `tone`, `emotion`, `reply`.

## License
MIT (or your preferred license).
