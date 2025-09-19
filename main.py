from openai import OpenAI
from pathlib import Path
from .config import LOG_DIR, OUT_DIR, DEFAULT_AGENTS, OPENAI_API_KEY, DEFAULT_ENV_CONTEXT_INDEX, USER_AGENT
from .config import VIZ_EDGE_WINDOW, VIZ_TOP_EDGE_LABELS, VIZ_SEED
from .visualize import draw_interactions_pro
from .agents import Agent
from .dialogue_manager import DialogueManager
from .io_logger import IOLogger
from .env_context import list_env_contexts, get_env_context_by_index
from .human_io import HumanIO

def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Не задан OPENAI_API_KEY")

    client = OpenAI()  # SDK возьмёт ключ из окружения. 

    # === ВЫБОР ОКРУЖЕНИЯ ===
    options = list_env_contexts()
    print("Выберите окружение и контекст (введите 1-3, Enter — по умолчанию):")
    for i, opt in enumerate(options, start=1):
        print(f"  {i}) {opt}")
    raw = input(f"Ваш выбор [1-{len(options)}] (по умолчанию {DEFAULT_ENV_CONTEXT_INDEX+1}): ").strip()
    try:
        idx = int(raw) - 1 if raw else DEFAULT_ENV_CONTEXT_INDEX
    except ValueError:
        idx = DEFAULT_ENV_CONTEXT_INDEX
    env_context = get_env_context_by_index(idx)
    print(f"\nВыбрано: {env_context}\n")
    
    agents = [Agent(**a) for a in DEFAULT_AGENTS]
    agents.append(Agent(**USER_AGENT))                         # ← НОВОЕ

    # 2) инициализируем диалог-менеджер с HumanIO
    human_io = HumanIO(client=client, classify_with_model=True)  # ← НОВОЕ
    dm = DialogueManager(client=client, agents=agents, env_context=env_context, human_io=human_io)

    logger = IOLogger(
        jsonl_path=LOG_DIR / "dialog.jsonl",
        md_path=LOG_DIR / "dialog.md",
        env_context=env_context,
    )

    agents_meta = {a.name: {"color": a.color} for a in agents}

    print("Симуляция диалога агентов. Нажимайте Enter для следующей реплики. Ctrl+C — выход.\n")
    turn = 0
    try:
        while True:
            input("Нажмите Enter для следующей реплики...")  # при ходе пользователя вы увидите его интерактивные вопросы внутри step()
            record, edge = dm.step()
            turn = record["turn"]
            print(f"[{turn}] {record['speaker']} → {record.get('target') or 'всем'} "
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
                    title=f"Взаимодействия и тональности (до хода {turn})",
                    window=VIZ_EDGE_WINDOW,
                    seed=VIZ_SEED,
                    top_edge_labels=0,        # ← необязательно, но уже не используется в новом режиме
                    per_message_edges=True,   # ← ВКЛЮЧИЛИ режим «каждая реплика = стрелка»
                )
                print(f"Сохранена визуализация: {out_path}")
    except KeyboardInterrupt:
        print("\nЗавершение. Логи и изображения сохранены.")

if __name__ == "__main__":
    main()
