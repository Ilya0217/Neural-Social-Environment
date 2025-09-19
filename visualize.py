# agent_dialogue_sim/visualize.py
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import gridspec

# === Палитры ===
TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# цвета для тонов (края шкалы + нейтральный центр)
C_POS = "#10b981"  # зелёный
C_NEU = "#9ca3af"  # серый
C_NEG = "#ef4444"  # красный

# цвета для эмоций (обводка узла)
EMOTION_COLOR = {
    "calm": "#60a5fa",       # голубой
    "curious": "#a78bfa",    # фиолетовый
    "confident": "#f59e0b",  # янтарный
    "skeptical": "#f97316",  # оранжевый
    "angry": "#ef4444",      # красный
    "joyful": "#10b981",     # зелёный
    "sad": "#3b82f6",        # синий
    "neutral": "#6b7280",    # тёмно‑серый
}

def _to_rgb(hex_color: str) -> Tuple[float, float, float]:
    return mcolors.to_rgb(hex_color)

def _blend(c1: str, c2: str, t: float) -> str:
    """Линейная интерполяция по t ∈ [0,1] между двумя цветами."""
    r1, g1, b1 = _to_rgb(c1)
    r2, g2, b2 = _to_rgb(c2)
    r = r1 + (r2 - r1) * t
    g = g1 + (g2 - g1) * t
    b = b1 + (b2 - b1) * t
    return mcolors.to_hex((r, g, b))

def tone_to_color(avg_score: float) -> str:
    """
    avg_score ∈ [-1, 1]; <0 — красный→серый, >=0 — серый→зелёный
    """
    if avg_score >= 0:
        return _blend(C_NEU, C_POS, avg_score)
    t = avg_score + 1.0  # [-1..0] -> [0..1]
    return _blend(C_NEG, C_NEU, t)

def compute_stats(agents_meta: Dict[str, Dict[str, str]], history: List[Dict[str, Any]], window: int):
    """
    Считаем метрики по истории.
    Возвращает:
    - nodes: dict с метриками по узлам
    - edges: dict с метриками по рёбрам за последние `window` ходов
    """
    agents = list(agents_meta.keys())
    # --- Node‑level ---
    talks = {a: 0 for a in agents}
    out_deg = {a: 0 for a in agents}
    in_deg = {a: 0 for a in agents}
    tone_sum = {a: 0.0 for a in agents}
    tone_cnt = {a: 0 for a in agents}
    last_emotion = {a: "neutral" for a in agents}

    for rec in history:
        sp = rec["speaker"]
        talks[sp] = talks.get(sp, 0) + 1
        last_emotion[sp] = (rec.get("emotion") or "neutral").lower()
        sc = TONE_SCORE.get((rec.get("tone") or "neutral").lower(), 0.0)
        tone_sum[sp] = tone_sum.get(sp, 0.0) + sc
        tone_cnt[sp] = tone_cnt.get(sp, 0) + 1
        tgt = rec.get("target")
        if tgt:
            out_deg[sp] = out_deg.get(sp, 0) + 1
            in_deg[tgt] = in_deg.get(tgt, 0) + 1

    node_stats = {}
    for a in agents:
        avg_tone = (tone_sum.get(a, 0.0) / tone_cnt[a]) if tone_cnt[a] else 0.0
        node_stats[a] = {
            "talks": talks.get(a, 0),
            "out": out_deg.get(a, 0),
            "in": in_deg.get(a, 0),
            "avg_tone": avg_tone,
            "emotion": last_emotion.get(a, "neutral"),
        }

    # --- Edge‑level (по окну последних window реплик) ---
    last = history[-window:] if window > 0 else history[:]
    edge_counts = {}
    edge_tone_sum = {}
    for rec in last:
        sp = rec["speaker"]
        tgt = rec.get("target")
        if not tgt:
            continue
        key = (sp, tgt)
        edge_counts[key] = edge_counts.get(key, 0) + 1
        sc = TONE_SCORE.get((rec.get("tone") or "neutral").lower(), 0.0)
        edge_tone_sum[key] = edge_tone_sum.get(key, 0.0) + sc

    edge_stats = {}
    for key, cnt in edge_counts.items():
        s = edge_tone_sum[key] / cnt if cnt else 0.0
        edge_stats[key] = {"count": cnt, "avg_score": s}

    return node_stats, edge_stats

def draw_interactions_pro(
    agents_meta: Dict[str, Dict[str, str]],
    history: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    window: int = 12,
    seed: int = 42,
    top_edge_labels: int = 6,
    per_message_edges: bool = True,
):
    """
    Крутая визуализация:
    - Узлы: размер ~ числу сообщений, цвет заполнения — из агента, обводка — по последней эмоции.
    - Рёбра: цвет = средний тон (-1..1), ширина ~ частоте за окно; кривизна для обратных рёбер.
    - Метрики и легенда в правой панели.
    """
    agents = list(agents_meta.keys())
    if not agents or not history:
        # ... (пустая заготовка, как было)
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "Недостаточно данных для графа", ha="center", va="center")
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    # --- Метрики по узлам (как раньше)
    node_stats, edge_stats_agg = compute_stats(agents_meta, history, window)

    # --- Граф узлов
    G = nx.DiGraph()
    for a in agents:
        G.add_node(a)
    pos = nx.spring_layout(G, seed=seed, k=1.2) if len(agents) > 2 else nx.circular_layout(G)

    # --- Фигура: граф + панель метрик
    fig = plt.figure(figsize=(12.5, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0], figure=fig, wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax.set_title(title, fontsize=12, pad=10)

    # --- Узлы (как было)
    base_node_size = 1200
    sizes, facecolors, edgecolors, linewidths = [], [], [], []
    for a in agents:
        talks = node_stats[a]["talks"]
        sizes.append(base_node_size + 80 * math.sqrt(max(talks, 1)))
        facecolors.append(agents_meta[a].get("color", "#888888"))
        emo = node_stats[a]["emotion"]
        edgecolors.append(EMOTION_COLOR.get(emo, "#6b7280"))
        linewidths.append(3.0)

    # glow
    nx.draw_networkx_nodes(G, pos, node_color=facecolors, node_size=[s*1.25 for s in sizes], alpha=0.08, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=facecolors, node_size=[s*1.10 for s in sizes], alpha=0.18, ax=ax)

    nx.draw_networkx_nodes(
        G, pos,
        node_color=facecolors,
        node_size=sizes,
        linewidths=linewidths,
        edgecolors=edgecolors,
        ax=ax
    )

    labels = {}
    for a in agents:
        ns = node_stats[a]
        labels[a] = f"{a}\n{ns['talks']} репл. · out:{ns['out']} in:{ns['in']}"
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    # === Рёбра ===
    if per_message_edges:
        # КАЖДАЯ реплика за последние `window` ходов — отдельная стрелка
        last = history[-window:] if window > 0 else history[:]
        # слегка разводим параллельные стрелки по дуге
        def arc_for(i: int) -> float:
            # 0.02..0.22 с периодом, чтобы дуги не слипались
            return 0.02 + (i % 5) * 0.05

        drawn = 0
        for i, rec in enumerate(last):
            src = rec["speaker"]
            dst = rec.get("target")
            if not dst:
                continue
            tone = (rec.get("tone") or "neutral").lower()
            col = tone_to_color(TONE_SCORE.get(tone, 0.0))
            rad = 0.02 + (i % 5) * 0.05  # как у вас

            arts = nx.draw_networkx_edges(
                G, pos,
                edgelist=[(src, dst)],
                arrows=True,
                arrowstyle="-|>",      # заметная «голова» стрелки
                arrowsize=26,          # увеличенный размер головы
                width=1.8,
                edge_color=col,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                alpha=0.95,
                min_source_margin=18,  # отступ от узла-источника
                min_target_margin=18,  # отступ от узла-цели
            )
            # ↑ функция возвращает список Patch-объектов
            if arts:
                for a in arts:
                    a.set_zorder(3)      # поверх узлов
                    a.set_clip_on(False) # не обрезать голову стрелки
            drawn += 1

        # подписи к топ-рёбрам здесь не нужны
    else:
        # СТАРЫЙ режим: агрегирование и подписи к самым «сильным» рёбрам
        reversed_pairs = set()
        for (u, v) in edge_stats_agg.keys():
            if (v, u) in edge_stats_agg:
                reversed_pairs.add((min(u, v), max(u, v)))

        drawn_edges = []
        for (u, v), est in sorted(edge_stats_agg.items(), key=lambda kv: kv[1]["count"], reverse=True):
            cnt = est["count"]
            score = est["avg_score"]
            col = tone_to_color(score)
            width = 1.2 + 0.9 * math.sqrt(cnt)
            rad = 0.18 if (min(u, v), max(u, v)) in reversed_pairs else 0.06

            e = nx.draw_networkx_edges(
                G, pos, edgelist=[(src, dst)],
                arrows=True,
                arrowstyle="-|>",          # заметная голова стрелки
                arrowsize=24,              # ↑ размер головы (px)
                width=1.8,                 # толщина линии
                edge_color=col,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                alpha=0.95,
                min_source_margin=18,      # ↑ больше отступ от узла-источника
                min_target_margin=18,      # ↑ больше отступ от узла-цели
                zorder=3,                  # рисуем поверх узлов/подложки
                clip_on=False,             # не клиповать голову стрелки
            )
            if e:
                drawn_edges.append(((u, v), cnt, score, col))

        # подписи к топ-рёбрам
        top = drawn_edges[:top_edge_labels]
        for (u, v), cnt, score, col in top:
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, f"{cnt}× · avg {score:+.2f}", fontsize=8, color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    ax.axis("off")

    # === Правая панель
    ax_stats.set_facecolor("#fcfcfd")

    # Топ говорящих (как было)
    top_speakers = sorted(agents, key=lambda a: node_stats[a]["talks"], reverse=True)
    y = list(reversed(top_speakers))
    x = [node_stats[a]["talks"] for a in y]
    bar_colors = [agents_meta[a].get("color", "#888") for a in y]
    ax_stats.barh(y, x, color=bar_colors, edgecolor="#374151")
    ax_stats.set_xlabel("сообщений")
    ax_stats.set_title("Кто больше говорит", fontsize=10)

    # Текст: уточняем легенду под новый режим
    if per_message_edges:
        ax_stats.text(0.0, -0.15, f"Стрелки: каждое сообщение за последние {window} ходов", transform=ax_stats.transAxes, fontsize=9)
    else:
        ax_stats.text(0.0, -0.15, f"Агрегация рёбер за последние {window} ходов", transform=ax_stats.transAxes, fontsize=9)

    ax_stats.grid(axis="x", alpha=0.25)
    ax_stats.set_axisbelow(True)

    legend_elems = [
        Line2D([0], [0], color=C_NEG, lw=3, label="негативная тональность"),
        Line2D([0], [0], color=C_NEU, lw=3, label="нейтральная"),
        Line2D([0], [0], color=C_POS, lw=3, label="позитивная"),
        Patch(facecolor="#ffffff", edgecolor="#111827", label="обводка узла = последняя эмоция"),
        Line2D([0], [0], color="#111111", lw=2, marker='>', markersize=8, label="стрелка = 1 реплика"),
    ]
    ax_stats.legend(handles=legend_elems, loc="lower right", frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
