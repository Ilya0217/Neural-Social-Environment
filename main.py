from openai import OpenAI
from pathlib import Path
from .config import LOG_DIR, OUT_DIR, OPENAI_API_KEY, DEFAULT_ENV_CONTEXT_INDEX, USER_AGENT
from .config import VIZ_EDGE_WINDOW, VIZ_TOP_EDGE_LABELS, VIZ_SEED
from .visualize import draw_interactions_pro
from .agents import Agent
from .dialogue_manager import DialogueManager
from .io_logger import IOLogger
from .env_context import list_env_contexts, get_env_context_by_index
from .human_io import HumanIO
from .analytics import compute_metrics, hypotheses_from_metrics, render_markdown_report, save_report_md
from .profiles import initialize_agents

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()  # SDK возьмёт ключ из окружения. 

    # === Environment selection ===
    options = list_env_contexts()
    print("Choose environment and context (enter 1-3, Enter for default):")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    raw = input(f"Your choice [1-{len(options)}] (default {DEFAULT_ENV_CONTEXT_INDEX+1}): ").strip()
    try:
        idx = int(raw) - 1 if raw else DEFAULT_ENV_CONTEXT_INDEX
    except ValueError:
        idx = DEFAULT_ENV_CONTEXT_INDEX
    env_context = get_env_context_by_index(idx)
    print(f"\nSelected: {env_context}\n")
    
    agents = initialize_agents(interactive=True)
    agents.append(Agent(**USER_AGENT))

    # 2) инициализируем диалог-менеджер с HumanIO
    human_io = HumanIO(client=client, classify_with_model=True)
    dm = DialogueManager(client=client, agents=agents, env_context=env_context, human_io=human_io)

    logger = IOLogger(
        jsonl_path=LOG_DIR / "dialog.jsonl",
        md_path=LOG_DIR / "dialog.md",
        env_context=env_context,
    )

    agents_meta = {a.name: {"color": a.color} for a in agents}

    print("Agents dialogue simulation. Press Enter for next message. Ctrl+C to exit.\n")
    turn = 0
    try:
        while True:
            input("Press Enter for the next message...")
            record, edge = dm.step()
            turn = record["turn"]
            print(f"[{turn}] {record['speaker']} → {record.get('target') or 'all'} "
                  f"({record['tone']}, {record['emotion']}): {record['reply']}\n")

            logger.write_jsonl(record)
            logger.write_markdown(
                turn_no=turn,
                speaker=record["speaker"],
                target=record.get("target"),
                tone=record["tone"],
                emotion=record["emotion"],
                text=record["reply"]
            )

            if dm.should_plot():
                out_path = OUT_DIR / f"graph_turn_{turn:04d}.png"
                draw_interactions_pro(
                    agents_meta=agents_meta,
                    history=list(dm.history),
                    out_path=out_path,
                    title=f"Interactions and tones (up to turn {turn})",
                    window=VIZ_EDGE_WINDOW,
                    seed=VIZ_SEED,
                    top_edge_labels=0,
                    per_message_edges=True,
                )
                print(f"Saved visualization: {out_path}")

                # ===== Metrics and hypotheses =====
                agents_order = [a.name for a in agents]
                metrics = compute_metrics(dm.history, agents_order, window_size=VIZ_EDGE_WINDOW)
                hyps = hypotheses_from_metrics(metrics, user_name="User")
                report_md = render_markdown_report(metrics, turn=turn, title="Dialogue — metrics")
                # append hypotheses to the report
                if hyps:
                    report_md += "\n\n## 🧠 Hypotheses\n" + "\n".join([f"- {h}" for h in hyps]) + "\n"

                metrics_path = OUT_DIR / f"metrics_turn_{turn:04d}.md"
                save_report_md(report_md, metrics_path)
                logger.write_metrics_block(f"Metrics — turn {turn}", report_md)
                print(f"Saved metrics report: {metrics_path}")
    except KeyboardInterrupt:
        print("\nDone. Logs and images saved.")

if __name__ == "__main__":
    main()
