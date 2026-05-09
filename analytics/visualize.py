# agent_dialogue_sim/visualize.py
from pathlib import Path
from typing import List, Dict, Any, Tuple
import math
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for thread safety
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import gridspec

# Palettes and color helpers
TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# tone colors (scale ends + neutral center)
C_POS = "#10b981"  # зелёный
C_NEU = "#9ca3af"  # серый
C_NEG = "#ef4444"  # красный

# emotion colors (node outline)
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
    """Linear interpolation for t ∈ [0,1] between two hex colors."""
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
    """Compute node-level and windowed edge metrics from history."""
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

    # --- Edge‑level (last `window` messages) ---
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
    """Visualization:
    - Nodes: size ~ #messages, fill from agent color, outline by last emotion.
    - Edges: color = avg tone (-1..1), width ~ frequency within window; curvature for reverse pairs.
    - Metrics and legend in the right panel.
    """
    agents = list(agents_meta.keys())
    if not agents or not history:
        # placeholder when insufficient data
        plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "Not enough data to draw graph", ha="center", va="center")
        plt.savefig(out_path, dpi=160)
        plt.close()
        return

    # --- Node metrics
    node_stats, edge_stats_agg = compute_stats(agents_meta, history, window)

    # --- Граф узлов
    G = nx.DiGraph()
    for a in agents:
        G.add_node(a)
    pos = nx.spring_layout(G, seed=seed, k=1.2) if len(agents) > 2 else nx.circular_layout(G)

    # --- Figure: graph + metrics panel
    fig = plt.figure(figsize=(12.5, 7.5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1.0], figure=fig, wspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    ax.set_title(title, fontsize=12, pad=10)

    # --- Nodes
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

    # === Edges ===
    if per_message_edges:
        # Each message in the last `window` turns is a separate arrow
        last = history[-window:] if window > 0 else history[:]
        # spread parallel arrows slightly via arc radius
        def arc_for(i: int) -> float:
            # 0.02..0.22 periodic to avoid overlapping arcs
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
                arrowstyle="-|>",
                arrowsize=26,
                width=1.8,
                edge_color=col,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                alpha=0.95,
                min_source_margin=18,
                min_target_margin=18,
            )
            # ↑ функция возвращает список Patch-объектов
            if arts:
                for a in arts:
                    a.set_zorder(3)      # поверх узлов
                    a.set_clip_on(False) # не обрезать голову стрелки
            drawn += 1

        # no top-edge labels in per-message mode
    else:
        # Aggregated mode: label strongest edges
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
                arrowstyle="-|>",
                arrowsize=24,
                width=1.8,
                edge_color=col,
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                alpha=0.95,
                min_source_margin=18,
                min_target_margin=18,
                zorder=3,
                clip_on=False,
            )
            if e:
                drawn_edges.append(((u, v), cnt, score, col))

        # labels for strongest edges
        top = drawn_edges[:top_edge_labels]
        for (u, v), cnt, score, col in top:
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            ax.text(x, y, f"{cnt}× · avg {score:+.2f}", fontsize=8, color=col,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=col, alpha=0.8))

    ax.axis("off")

    # === Right panel
    ax_stats.set_facecolor("#fcfcfd")

    # Top speakers
    top_speakers = sorted(agents, key=lambda a: node_stats[a]["talks"], reverse=True)
    y = list(reversed(top_speakers))
    x = [node_stats[a]["talks"] for a in y]
    bar_colors = [agents_meta[a].get("color", "#888") for a in y]
    ax_stats.barh(y, x, color=bar_colors, edgecolor="#374151")
    ax_stats.set_xlabel("messages")
    ax_stats.set_title("Who speaks more", fontsize=10)

    # Legend text for mode
    if per_message_edges:
        ax_stats.text(0.0, -0.15, f"Arrows: each message in the last {window} turns", transform=ax_stats.transAxes, fontsize=9)
    else:
        ax_stats.text(0.0, -0.15, f"Aggregated edges over the last {window} turns", transform=ax_stats.transAxes, fontsize=9)

    ax_stats.grid(axis="x", alpha=0.25)
    ax_stats.set_axisbelow(True)

    legend_elems = [
        Line2D([0], [0], color=C_NEG, lw=3, label="negative tone"),
        Line2D([0], [0], color=C_NEU, lw=3, label="neutral"),
        Line2D([0], [0], color=C_POS, lw=3, label="positive"),
        Patch(facecolor="#ffffff", edgecolor="#111827", label="node outline = last emotion"),
        Line2D([0], [0], color="#111111", lw=2, marker='>', markersize=8, label="arrow = 1 message"),
    ]
    ax_stats.legend(handles=legend_elems, loc="lower right", frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()
