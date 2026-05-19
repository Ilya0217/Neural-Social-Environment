# agent_dialogue_sim/analytics/visualize.py
#
# Профессиональная научная визуализация сети взаимодействий.
# 4-панельный публикационного качества дашборд:
#   (A) Главный граф    — force-directed сеть с глубиной, центральностями
#   (B) Centralities    — bar chart по агентам (in / out / betweenness)
#   (C) Sociomatrix     — heatmap взаимодействий с tone-coloring
#   (D) Turn timeline   — лента ходов с цветовой кодировкой тона
#
# Научные ссылки (рендерятся в подписи):
#   — Borgatti et al. (2009) Social Network Analysis
#   — Moreno (1934); Wasserman & Faust (1994) Sociomatrix
#   — Sacks, Schegloff & Jefferson (1974) Turn-taking organization
#   — Newman (2010) Networks: an introduction
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (thread-safe for Flask)
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colors as mcolors
from matplotlib import gridspec
from matplotlib import patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Patch, Rectangle

# === Палитра тонов ===
TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
C_POS = "#10b981"  # emerald 500
C_NEU = "#94a3b8"  # slate 400
C_NEG = "#ef4444"  # red 500
C_AXIS = "#334155"  # slate 700
C_GRID = "#e2e8f0"  # slate 200
C_PANEL_BG = "#fafbfc"
C_FIG_BG = "#f4f6fa"

# === Палитра последних эмоций (обводка узла) ===
EMOTION_COLOR = {
    "calm": "#60a5fa",
    "curious": "#a78bfa",
    "thoughtful": "#8b5cf6",
    "confident": "#f59e0b",
    "skeptical": "#f97316",
    "concerned": "#f97316",
    "angry": "#dc2626",
    "frustrated": "#dc2626",
    "joyful": "#10b981",
    "excited": "#10b981",
    "encouraging": "#22c55e",
    "supportive": "#22c55e",
    "sad": "#3b82f6",
    "pragmatic": "#475569",
    "measured": "#64748b",
    "focused": "#0ea5e9",
    "friendly": "#fb7185",
    "neutral": "#94a3b8",
}

# === Утилиты ===
def _to_rgb(c: str) -> Tuple[float, float, float]:
    return mcolors.to_rgb(c)


def _blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = _to_rgb(c1)
    r2, g2, b2 = _to_rgb(c2)
    return mcolors.to_hex((r1 + (r2 - r1) * t, g1 + (g2 - g1) * t, b1 + (b2 - b1) * t))


def tone_to_color(s: float) -> str:
    s = max(-1.0, min(1.0, float(s)))
    if s >= 0:
        return _blend(C_NEU, C_POS, s)
    return _blend(C_NEG, C_NEU, s + 1.0)


def _emotion_color(name: str) -> str:
    return EMOTION_COLOR.get((name or "neutral").lower(), "#64748b")


# === Сбор статистики ===
def compute_stats(agents_meta: Dict[str, Dict[str, str]],
                  history: List[Dict[str, Any]],
                  window: int):
    agents = list(agents_meta.keys())
    talks = {a: 0 for a in agents}
    out_deg = {a: 0 for a in agents}
    in_deg = {a: 0 for a in agents}
    tone_sum = {a: 0.0 for a in agents}
    tone_cnt = {a: 0 for a in agents}
    last_emotion = {a: "neutral" for a in agents}

    for rec in history:
        sp = rec["speaker"]
        if sp not in talks:
            talks[sp] = out_deg[sp] = in_deg[sp] = tone_cnt[sp] = 0
            tone_sum[sp] = 0.0
            last_emotion[sp] = "neutral"
        talks[sp] += 1
        last_emotion[sp] = (rec.get("emotion") or "neutral").lower()
        tone = TONE_SCORE.get((rec.get("tone") or "neutral").lower(), 0.0)
        tone_sum[sp] += tone
        tone_cnt[sp] += 1
        tgt = rec.get("target")
        if tgt:
            out_deg[sp] = out_deg.get(sp, 0) + 1
            in_deg[tgt] = in_deg.get(tgt, 0) + 1

    node_stats = {}
    for a in agents:
        avg_tone = tone_sum[a] / tone_cnt[a] if tone_cnt[a] else 0.0
        node_stats[a] = {
            "talks": talks[a], "out": out_deg[a], "in": in_deg[a],
            "avg_tone": avg_tone, "emotion": last_emotion[a],
        }

    last = history[-window:] if window > 0 else list(history)
    edge_counts: Dict[Tuple[str, str], int] = {}
    edge_tone: Dict[Tuple[str, str], float] = {}
    for rec in last:
        sp, tgt = rec["speaker"], rec.get("target")
        if not tgt:
            continue
        k = (sp, tgt)
        edge_counts[k] = edge_counts.get(k, 0) + 1
        edge_tone[k] = edge_tone.get(k, 0.0) + TONE_SCORE.get((rec.get("tone") or "neutral").lower(), 0.0)
    edge_stats = {
        k: {"count": c, "avg_score": edge_tone[k] / c}
        for k, c in edge_counts.items()
    }
    return node_stats, edge_stats


# === Главная сетевая панель ===
def _draw_main_network(ax, G, pos, agents, node_stats, edge_stats,
                       agents_meta, history, window, betweenness,
                       max_messages):
    """Главная панель: чистая направленная сеть.

    Принципы качественной визуализации:
      • круговая раскладка для N ≤ 10 → узлы равномерно по кругу, без наложений;
      • узлы заметно меньше доступного пространства → много «воздуха»;
      • одно ребро на пару (sender→target), агрегированное;
      • реципрокные пары разводятся симметричными дугами в противоположные стороны;
      • крупные явные стрелки (head_length/head_width в явных точках);
      • shrinkA/shrinkB точно подогнаны под радиус узла → стрелка касается границы;
      • подписи в белых пилюлях, радиально снаружи от каждого узла.
    """
    ax.set_facecolor("#ffffff")
    # Чуть больший диапазон — нужен запас на подписи снаружи круга.
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)
    ax.set_aspect("equal")
    ax.axis("off")

    # === Раскладка: круговая для малых графов (гарантированно без наложений) ===
    n_agents = len(agents)
    if n_agents <= 10:
        pos = nx.circular_layout(G, scale=1.25)
    # Для очень больших графов оставляем kamada_kawai (передан извне),
    # просто масштабируем положения наружу.
    else:
        pos = {a: (p[0] * 1.35, p[1] * 1.35) for a, p in pos.items()}

    # === Радиусы узлов: умеренные, чтобы не «съесть» пространство ===
    R_MIN, R_MAX = 0.11, 0.18
    def node_radius(talks):
        return R_MIN + (R_MAX - R_MIN) * (
            math.log1p(talks) / math.log1p(max(max_messages, 1))
        )

    radii = {a: node_radius(node_stats[a]["talks"]) for a in agents if a in pos}

    # === Сколько display-points занимает 1 axis-unit (для shrinkA/shrinkB) ===
    # Аппроксимация: получается из размера панели. Берём bbox оси в дюймах.
    bbox = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted())
    panel_width_in = max(bbox.width, 0.1)
    axis_span = 3.9  # xlim from -1.95 to 1.95
    points_per_axis_unit = (panel_width_in / axis_span) * 72.0

    def _shrink(node_a: str) -> float:
        r = radii.get(node_a, R_MIN)
        # +5 точек — небольшой зазор между стрелкой и краем узла.
        return r * points_per_axis_unit + 5.0

    # === Рёбра: агрегированные, цвет=тон, толщина=log(count) ===
    if edge_stats:
        max_count = max(st["count"] for st in edge_stats.values()) or 1
    else:
        max_count = 1

    def _edge_width(count: int) -> float:
        # от 2.2 (для 1 сообщения) до 5.5 (для max)
        return 2.2 + 3.3 * (math.log1p(count) / math.log1p(max_count))

    for (sp, tgt), st in edge_stats.items():
        if sp not in pos or tgt not in pos:
            continue
        count = st["count"]
        if count <= 0:
            continue

        # Реципрокные пары: оба ребра дугами в противоположные стороны.
        reciprocal = (tgt, sp) in edge_stats and edge_stats[(tgt, sp)]["count"] > 0
        rad = 0.20 if reciprocal else 0.0
        col = tone_to_color(st["avg_score"])
        width = _edge_width(count)

        # Большие явные наконечники: head_length / head_width в точках.
        arrow = FancyArrowPatch(
            pos[sp], pos[tgt],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=14,head_width=10",
            mutation_scale=1.0,
            linewidth=width,
            color=col,
            alpha=0.95,
            zorder=2,
            shrinkA=_shrink(sp),
            shrinkB=_shrink(tgt),
            capstyle="round",
            joinstyle="round",
        )
        ax.add_patch(arrow)

        # Бейдж с количеством сообщений (только если ≥2 — иначе шум).
        if count >= 2:
            mx, my = (pos[sp][0] + pos[tgt][0]) / 2, (pos[sp][1] + pos[tgt][1]) / 2
            if rad != 0:
                # Смещаем перпендикулярно по нормали к ребру для дуги.
                dx, dy = pos[tgt][0] - pos[sp][0], pos[tgt][1] - pos[sp][1]
                length = math.hypot(dx, dy) or 1.0
                nx_, ny_ = -dy / length, dx / length
                mx += nx_ * rad * 0.65
                my += ny_ * rad * 0.65
            ax.text(mx, my, str(count), fontsize=10.5, fontweight="bold",
                    color="#0f172a", ha="center", va="center", zorder=3.5,
                    family="DejaVu Sans",
                    bbox=dict(boxstyle="circle,pad=0.22",
                              facecolor="white", edgecolor=col,
                              linewidth=1.4, alpha=0.98))

    # === Узлы: цветная заливка, тёмная обводка, белая внешняя «галя» ===
    for a in agents:
        if a not in pos:
            continue
        x, y = pos[a]
        r = radii[a]
        col = agents_meta[a].get("color", "#6366f1")
        # Внешний белый ободок: визуально отделяет узел от рёбер при пересечении.
        ax.add_patch(Circle((x, y), r + 0.020, color="white",
                            zorder=4, ec="white", lw=0))
        # Основной круг с тёмной обводкой.
        ax.add_patch(Circle((x, y), r, color=col, zorder=5,
                            ec="#0f172a", lw=1.6))

    # === Подписи: снаружи каждого узла, в белой пилюле, никогда не накладываются ===
    for a in agents:
        if a not in pos:
            continue
        x, y = pos[a]
        r = radii[a]
        theta = math.atan2(y, x) if (x != 0 or y != 0) else math.pi / 2
        # Отступ подписи: радиус + заметный воздух, чтобы пилюля не касалась круга.
        offset = r + 0.18
        tx = x + offset * math.cos(theta)
        ty = y + offset * math.sin(theta)
        ha = "left" if math.cos(theta) > 0.15 else ("right" if math.cos(theta) < -0.15 else "center")
        va = "bottom" if math.sin(theta) > 0.15 else ("top" if math.sin(theta) < -0.15 else "center")
        ns = node_stats[a]
        label = f"{a}\n{ns['talks']} реп  ↗{ns['out']}  ↙{ns['in']}"
        ax.text(tx, ty, label, fontsize=13, fontweight="bold",
                color="#0f172a", ha=ha, va=va, zorder=10,
                family="DejaVu Sans", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.36",
                          facecolor="white", edgecolor="#cbd5e1",
                          linewidth=1.0, alpha=0.96))

    # === Блок сетевых метрик: верхний правый угол, не пересекается с подписями ===
    metrics_text = _format_network_metrics(G, node_stats, edge_stats)
    ax.text(0.985, 0.985, metrics_text, transform=ax.transAxes,
            fontsize=10, family="DejaVu Sans Mono",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.65",
                      facecolor="white", edgecolor="#cbd5e1",
                      linewidth=1.0, alpha=0.97),
            zorder=20)


def _format_network_metrics(G, node_stats, edge_stats) -> str:
    """Считает и форматирует сетевые метрики для info-box на главной панели."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    try:
        reciprocity = nx.reciprocity(G) or 0.0
    except Exception:
        reciprocity = 0.0
    try:
        clustering = nx.transitivity(G)
    except Exception:
        clustering = 0.0
    try:
        n_components = nx.number_weakly_connected_components(G)
    except Exception:
        n_components = 1
    total_msg = sum(s["talks"] for s in node_stats.values())
    return (
        "СЕТЕВЫЕ МЕТРИКИ\n"
        f"  узлов:        {n}\n"
        f"  рёбер:        {m}\n"
        f"  density:      {density:.3f}\n"
        f"  reciprocity:  {reciprocity:.3f}\n"
        f"  clustering:   {clustering:.3f}\n"
        f"  компонент:    {n_components}\n"
        f"  сообщений:    {total_msg}"
    )


# === Панель центральностей ===
def _draw_centrality_panel(ax, agents, node_stats, betweenness, agents_meta):
    ax.set_facecolor(C_PANEL_BG)
    ax.set_title("Центральности агентов",
                 fontsize=12, color=C_AXIS, pad=10, fontweight="600", loc="left")

    if not agents:
        ax.axis("off"); return

    # нормированные in/out
    max_in = max((node_stats[a]["in"] for a in agents), default=1) or 1
    max_out = max((node_stats[a]["out"] for a in agents), default=1) or 1
    max_b = max(betweenness.values()) if betweenness else 0.0

    sorted_agents = sorted(agents, key=lambda a: node_stats[a]["talks"], reverse=True)
    y = list(range(len(sorted_agents)))
    bar_h = 0.25

    in_norm = [node_stats[a]["in"] / max_in for a in sorted_agents]
    out_norm = [node_stats[a]["out"] / max_out for a in sorted_agents]
    bet_norm = [(betweenness.get(a, 0.0) / max_b) if max_b > 0 else 0.0 for a in sorted_agents]

    ax.barh([yi + bar_h for yi in y], in_norm, height=bar_h,
            color="#3b82f6", label="in-degree", edgecolor="#1e40af", linewidth=0.5)
    ax.barh(y, out_norm, height=bar_h,
            color="#f97316", label="out-degree", edgecolor="#9a3412", linewidth=0.5)
    ax.barh([yi - bar_h for yi in y], bet_norm, height=bar_h,
            color="#a855f7", label="betweenness", edgecolor="#6b21a8", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_agents, fontsize=9.5, color=C_AXIS)
    ax.set_ylim(-0.7, len(sorted_agents) - 0.3)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("норм. значение [0..1]", fontsize=9, color=C_AXIS)
    ax.tick_params(axis="x", labelsize=8.5, colors=C_AXIS)
    ax.tick_params(axis="y", colors=C_AXIS)
    ax.grid(axis="x", color=C_GRID, alpha=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")

    # Легенда — снизу под xlabel, дальше отнесена, чтобы не наезжать на бары.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=3, fontsize=9, frameon=False, handlelength=1.4,
              columnspacing=1.6, handletextpad=0.5)


# === Sociomatrix ===
def _draw_sociomatrix(ax, agents, edge_stats):
    ax.set_facecolor(C_PANEL_BG)
    ax.set_title("Социограмма взаимодействий",
                 fontsize=12, color=C_AXIS, pad=10, fontweight="600", loc="left")

    n = len(agents)
    if n == 0:
        ax.axis("off"); return
    # Матрица: rows=отправители, cols=получатели
    counts = [[edge_stats.get((s, r), {}).get("count", 0) for r in agents] for s in agents]
    tones = [[edge_stats.get((s, r), {}).get("avg_score", 0.0) for r in agents] for s in agents]

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)  # ось Y перевёрнута, как у матриц
    ax.set_aspect("equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Подписи получателей — снизу, под наклоном, чтобы не наезжали на заголовок
    ax.set_xticklabels(agents, fontsize=9, color=C_AXIS, rotation=30, ha="right")
    ax.set_yticklabels(agents, fontsize=9, color=C_AXIS)
    ax.set_xlabel("получатель", fontsize=9, color=C_AXIS, labelpad=6)
    ax.set_ylabel("отправитель", fontsize=9, color=C_AXIS, labelpad=6)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True,
                   length=0, colors=C_AXIS)

    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")

    max_cnt = max((max(row) if row else 0) for row in counts) or 1
    for i in range(n):
        for j in range(n):
            cnt = counts[i][j]
            tone = tones[i][j]
            if i == j:
                # диагональ — серая
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="#e2e8f0", edgecolor="white", lw=1))
                continue
            if cnt == 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="white", edgecolor="#e2e8f0", lw=0.5))
                continue
            base = tone_to_color(tone)
            # яркость по числу сообщений: alpha
            alpha = 0.30 + 0.70 * (cnt / max_cnt)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                   facecolor=base, edgecolor="white", lw=1.5, alpha=alpha))
            # надпись: count
            text_col = "white" if cnt / max_cnt > 0.5 else C_AXIS
            ax.text(j, i, str(cnt), ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_col, zorder=5)


# === Timeline ===
def _draw_turn_timeline(ax, agents, history):
    ax.set_facecolor(C_PANEL_BG)
    ax.set_title("Хронология ходов",
                 fontsize=12, color=C_AXIS, pad=10, fontweight="600", loc="left")

    if not history:
        ax.axis("off"); return

    n = len(agents)
    agent_idx = {a: i for i, a in enumerate(agents)}
    total_turns = len(history)

    ax.set_xlim(0.5, total_turns + 0.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_yticks(range(n))
    ax.set_yticklabels(agents, fontsize=8.5, color=C_AXIS)
    ax.set_xlabel("ход", fontsize=8.5, color=C_AXIS)
    ax.tick_params(axis="x", labelsize=8, colors=C_AXIS)
    ax.tick_params(axis="y", colors=C_AXIS)
    ax.grid(axis="x", color=C_GRID, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")

    # горизонтальные «дорожки» под каждого агента
    for i in range(n):
        ax.add_patch(Rectangle((0.5, i - 0.35), total_turns, 0.7,
                               facecolor="white", edgecolor="none", alpha=0.6, zorder=0.5))

    # каждый ход = маркер
    for rec in history:
        sp = rec.get("speaker")
        if sp not in agent_idx:
            continue
        t = rec.get("turn", 0)
        tone = (rec.get("tone") or "neutral").lower()
        col = tone_to_color(TONE_SCORE.get(tone, 0.0))
        ax.add_patch(Rectangle((t - 0.4, agent_idx[sp] - 0.3), 0.8, 0.6,
                               facecolor=col, edgecolor="#0f172a", lw=0.4,
                               alpha=0.92, zorder=3))


# === Sparkline тренда тона ===
def _draw_tone_trend(ax, history):
    ax.set_facecolor(C_PANEL_BG)
    ax.set_title("Тренд тона диалога",
                 fontsize=12, color=C_AXIS, pad=10, fontweight="600", loc="left")
    if not history:
        ax.axis("off"); return

    turns = [r.get("turn", i+1) for i, r in enumerate(history)]
    tones = [TONE_SCORE.get((r.get("tone") or "neutral").lower(), 0.0) for r in history]
    # Скользящее среднее по 3
    smoothed = []
    win = 3
    for i in range(len(tones)):
        lo = max(0, i - win // 2)
        hi = min(len(tones), i + win // 2 + 1)
        smoothed.append(sum(tones[lo:hi]) / max(1, hi - lo))

    ax.set_xlim(min(turns) - 0.5, max(turns) + 0.5)
    ax.set_ylim(-1.15, 1.15)
    ax.tick_params(labelsize=8, colors=C_AXIS)
    ax.set_xlabel("ход", fontsize=8.5, color=C_AXIS)
    ax.set_ylabel("тон", fontsize=8.5, color=C_AXIS)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cbd5e1")
    ax.grid(axis="y", color=C_GRID, alpha=0.5)
    ax.set_axisbelow(True)

    # горизонтальные зоны: позитив/нейтрал/негатив
    ax.axhspan(0.3, 1.15, color=C_POS, alpha=0.06, zorder=0)
    ax.axhspan(-0.3, 0.3, color=C_NEU, alpha=0.06, zorder=0)
    ax.axhspan(-1.15, -0.3, color=C_NEG, alpha=0.06, zorder=0)
    ax.axhline(0, color="#94a3b8", lw=0.7, ls="--", alpha=0.7)

    # сырые точки — небольшие, прозрачные
    for x, y in zip(turns, tones):
        ax.scatter([x], [y], color=tone_to_color(y), s=40,
                   edgecolors="#0f172a", linewidths=0.4, zorder=3, alpha=0.8)
    # сглаженная кривая
    ax.plot(turns, smoothed, color="#0f172a", lw=2.2, alpha=0.85, zorder=4)
    # заливка под кривой
    ax.fill_between(turns, smoothed, 0,
                    where=[s >= 0 for s in smoothed],
                    color=C_POS, alpha=0.15, interpolate=True, zorder=2)
    ax.fill_between(turns, smoothed, 0,
                    where=[s < 0 for s in smoothed],
                    color=C_NEG, alpha=0.15, interpolate=True, zorder=2)


# === Главный публичный API: обратносовместимая сигнатура ===
def draw_interactions_pro(
    agents_meta: Dict[str, Dict[str, str]],
    history: List[Dict[str, Any]],
    out_path: Path,
    title: str,
    window: int = 12,
    seed: int = 42,
    top_edge_labels: int = 6,        # параметр оставлен для обратной совместимости
    per_message_edges: bool = True,  # параметр оставлен для обратной совместимости
):
    """Публикационного качества 4-панельный дашборд:
       (A) сеть с глубиной (force-directed, центральности halo);
       (B) центральности (Freeman/Borgatti);
       (C) sociomatrix (Moreno);
       (D) хронология ходов (Sacks et al.).
    """
    agents = list(agents_meta.keys())
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not agents or not history:
        fig, ax = plt.subplots(figsize=(12, 7), facecolor=C_FIG_BG)
        ax.set_facecolor(C_PANEL_BG)
        ax.text(0.5, 0.5, "Недостаточно данных для построения сети",
                ha="center", va="center", fontsize=13, color=C_AXIS)
        ax.axis("off")
        plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        return

    # --- статистика
    node_stats, edge_stats = compute_stats(agents_meta, history, window)
    total_msg = sum(s["talks"] for s in node_stats.values())
    max_msg = max((s["talks"] for s in node_stats.values()), default=1)

    # --- граф
    G = nx.DiGraph()
    for a in agents:
        G.add_node(a)
    for (u, v), st in edge_stats.items():
        G.add_edge(u, v, weight=st["count"])
    # центральности
    try:
        betweenness = nx.betweenness_centrality(G, normalized=True) if len(agents) >= 3 else {a: 0.0 for a in agents}
    except Exception:
        betweenness = {a: 0.0 for a in agents}

    # позиции узлов — kamada-kawai даёт чистее, чем spring, для малых сетей
    if len(agents) <= 2:
        pos = nx.circular_layout(G, scale=1.0)
    else:
        try:
            pos = nx.kamada_kawai_layout(G, scale=1.0)
        except Exception:
            pos = nx.spring_layout(G, seed=seed, k=1.4, scale=1.0)

    # --- Figure: ландшафтный формат ≈5:3 (20×12), под широкий монитор.
    # На widescreen-экране изображение с сохранением пропорций заполняет
    # почти всю площадь панели — без растяжения, каждая подграфика чётко видна.
    fig = plt.figure(figsize=(20, 12), facecolor=C_FIG_BG, dpi=150)
    # 4-строчный grid: главная сеть (2 строки) + sociomatrix/trend (1 строка) + timeline (1 строка).
    # hspace/wspace и поля подобраны так, чтобы заголовки панелей, подписи осей
    # и легенды не накладывались друг на друга, но при этом не оставалось зияющих пустот.
    # Компоновка: главный граф доминирует слева (≈65% ширины),
    # справа компактным столбиком — центральности и социограмма.
    # Внизу — тонкая лента хронологии ходов.
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[1.85, 1.0],
        height_ratios=[3.4, 0.7],
        hspace=0.35, wspace=0.18,
        left=0.04, right=0.985, top=0.89, bottom=0.10,
        figure=fig,
    )
    # главный граф — большой блок слева
    ax_net = fig.add_subplot(gs[0, 0])
    # правый столбец: центральности сверху, sociomatrix снизу — через под-grid
    right_gs = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs[0, 1],
        height_ratios=[1.0, 1.0], hspace=0.45,
    )
    ax_cent = fig.add_subplot(right_gs[0, 0])
    ax_socio = fig.add_subplot(right_gs[1, 0])
    # хронология ходов — тонкая полоса внизу на всю ширину
    ax_time = fig.add_subplot(gs[1, :])

    # --- Заголовок и подзаголовок ---
    fig.text(0.04, 0.955, title, fontsize=28, fontweight="700",
             color="#0f172a", family="DejaVu Sans")
    subtitle = (
        f"N={len(agents)} агентов  ·  {total_msg} сообщений  ·  "
        f"окно анализа = {window} ходов"
    )
    fig.text(0.04, 0.925, subtitle, fontsize=13.5, color="#64748b",
             family="DejaVu Sans")

    # --- Панели ---
    _draw_main_network(ax_net, G, pos, agents, node_stats, edge_stats,
                       agents_meta, history, window, betweenness, max_msg)
    _draw_centrality_panel(ax_cent, agents, node_stats, betweenness, agents_meta)
    _draw_sociomatrix(ax_socio, agents, edge_stats)
    _draw_turn_timeline(ax_time, agents, history)

    # --- Глобальная легенда — компактно внизу ---
    legend_handles = [
        Line2D([0], [0], color=C_POS, lw=4, label="позитивный тон"),
        Line2D([0], [0], color=C_NEU, lw=4, label="нейтральный"),
        Line2D([0], [0], color=C_NEG, lw=4, label="негативный"),
        Patch(facecolor="white", edgecolor="#0f172a", lw=1.5,
              label="число в кружке = сообщений по ребру"),
        Patch(facecolor="#94a3b8", edgecolor="#0f172a", lw=1,
              label="размер узла ∝ числу реплик агента"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=5, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 0.015))

    fig.savefig(out_path, dpi=160, facecolor=fig.get_facecolor(),
                bbox_inches=None, pad_inches=0.1)
    plt.close(fig)
