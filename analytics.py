"""Analytics for agent dialogue.

Computes windowed and global metrics, renders markdown reports, and generates
lightweight heuristic hypotheses in English.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple
import math
from collections import Counter, defaultdict
from datetime import datetime

# tone palette
TONE_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def _window(history: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    return history[-n:] if n > 0 else history[:]

def _words(s: str) -> int:
    return len((s or "").strip().split())

def _unique_emotions(items: List[str]) -> int:
    return len({(e or "neutral").lower() for e in items})

def _entropy(items: List[str]) -> float:
    # Shannon entropy for emotion diversity
    if not items:
        return 0.0
    c = Counter([ (e or "neutral").lower() for e in items ])
    total = sum(c.values())
    ent = 0.0
    for v in c.values():
        p = v / total
        ent -= p * math.log2(p)
    return ent

def _addressing_delay(history: List[Dict[str, Any]]) -> float:
    """Average turns-to-first-response for addressed messages.

    If a message targets A, measure how many turns until A replies (first reply).
    """
    last_addr = defaultdict(list)  # agent -> turns when addressed
    delays = []
    for rec in history:
        t = rec["turn"]
        speaker = rec["speaker"]
        target = rec.get("target")
        # если спикер сейчас ответил, посчитаем задержку к ближайшей адресации ему
        if last_addr.get(speaker):
            # сколько ходов прошло с последней адресации этому спикеру
            last_t = last_addr[speaker].pop(0)
            delays.append(t - last_t)
        if target:
            last_addr[target].append(t)
    return sum(delays)/len(delays) if delays else 0.0

def compute_metrics(history: List[Dict[str, Any]], agents: List[str], window_size: int) -> Dict[str, Any]:
    win = _window(history, window_size)
    all_msgs = len(history)
    win_msgs = len(win)
    if all_msgs == 0:
        return {"summary": {}, "per_agent": {}, "edges": {}, "window_size": window_size}

    # aggregates over history
    def _aggregate(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        talk = Counter([r["speaker"] for r in sub])
        addr_ok = sum(1 for r in sub if r.get("target"))
        tone_vals = [TONE_SCORE.get((r.get("tone") or "neutral").lower(), 0.0) for r in sub]
        pos_frac = sum(1 for r in sub if (r.get("tone") or "neutral") == "positive") / max(1, len(sub))
        neg_frac = sum(1 for r in sub if (r.get("tone") or "neutral") == "negative") / max(1, len(sub))
        neu_frac = 1.0 - pos_frac - neg_frac
        emotions = [ (r.get("emotion") or "neutral") for r in sub ]
        avg_len = sum(_words(r.get("reply") or "") for r in sub) / max(1, len(sub))
        targets = [ (r["speaker"], r.get("target")) for r in sub if r.get("target") ]
        out_deg = Counter([ s for (s, _) in targets ])
        in_deg  = Counter([ t for (_, t) in targets ])
        reciprocity = _estimate_reciprocity(sub)
        return {
            "messages": len(sub),
            "talk_share": {a: talk.get(a, 0)/max(1, len(sub)) for a in agents},
            "targeting_rate": addr_ok / max(1, len(sub)),
            "avg_tone": sum(tone_vals)/max(1, len(tone_vals)),
            "tone_fracs": {"pos": pos_frac, "neu": neu_frac, "neg": neg_frac},
            "emotion_diversity": _unique_emotions(emotions),
            "emotion_entropy": _entropy(emotions),
            "avg_reply_words": avg_len,
            "out_degree": dict(out_deg),
            "in_degree": dict(in_deg),
            "reciprocity": reciprocity,
            "avg_addressing_delay": _addressing_delay(sub),
        }

    def _per_agent(sub: List[Dict[str, Any]]) -> Dict[str, Any]:
        res = {}
        for a in agents:
            msgs = [r for r in sub if r["speaker"] == a]
            res[a] = {
                "msgs": len(msgs),
                "avg_words": sum(_words(r.get("reply") or "") for r in msgs)/max(1, len(msgs)),
                "avg_tone": sum(TONE_SCORE.get((r.get("tone") or "neutral").lower(), 0.0) for r in msgs)/max(1, len(msgs)),
                "last_emotion": (msgs[-1].get("emotion") if msgs else "neutral"),
                "targets_diversity": len({r.get("target") for r in msgs if r.get("target")}),
            }
        return res

    summary_all = _aggregate(history)
    summary_win = _aggregate(win)
    per_agent_win = _per_agent(win)

    return {
        "window_size": window_size,
        "summary": {"all": summary_all, "window": summary_win},
        "per_agent": per_agent_win,
    }

def _estimate_reciprocity(sub: List[Dict[str, Any]]) -> float:
    """
    Доля «взаимных» обменов: (A->B) за окном сопровождалось хотя бы одним (B->A).
    """
    pairs = [(r["speaker"], r.get("target")) for r in sub if r.get("target")]
    if not pairs:
        return 0.0
    set_pairs = set(pairs)
    mutual = sum(1 for (u, v) in set_pairs if (v, u) in set_pairs)
    return mutual / max(1, len(set_pairs))

def hypotheses_from_metrics(metrics: Dict[str, Any], user_name: str | None = "User") -> List[str]:
    """Simple heuristic hypotheses generator (can be replaced with LLM)."""
    s = metrics["summary"]["window"]
    hyp = []
    # dominance
    talk = s["talk_share"]
    top_speaker, top_share = max(talk.items(), key=lambda kv: kv[1]) if talk else ("-", 0)
    if top_share > 0.45:
        hyp.append(f"Hypothesis: agent '{top_speaker}' dominates the discussion (~{int(top_share*100)}%). Check role balance.")
    # mood / tone
    if s["avg_tone"] < -0.25:
        hyp.append("Hypothesis: average tone is negative — risk of conflict; consider facilitation.")
    elif s["avg_tone"] > 0.35:
        hyp.append("Hypothesis: positive bias — good cooperation; you can accelerate decisions.")
    # addressing
    if s["targeting_rate"] < 0.7:
        hyp.append("Hypothesis: low addressing (<70%) — possible misunderstandings; enforce 'each message is addressed'.")
    # delays
    if s["avg_addressing_delay"] > 2.5:
        hyp.append("Hypothesis: high average response delay (>2.5 turns) — redistribute focus.")
    # emotion diversity
    if s["emotion_entropy"] > 1.3:
        hyp.append("Hypothesis: emotions are highly mixed — discussion may be fragmented.")
    # attention to user
    if user_name and s["in_degree"].get(user_name, 0) == 0:
        hyp.append("Hypothesis: the user hasn't been addressed — explicitly ask for their input.")

    return hyp or ["Hypotheses: no significant deviations — communication is stable."]

def render_markdown_report(metrics: Dict[str, Any], turn: int, title: str) -> str:
    s_all = metrics["summary"]["all"]
    s_win = metrics["summary"]["window"]
    pa = metrics["per_agent"]
    ts = datetime.utcnow().isoformat()

    def pct(x): return f"{x*100:.0f}%"
    def tone_emoji(x):
        return "🟢" if x > 0.2 else ("🟡" if abs(x) <= 0.2 else "🔴")

    # top block
    md = []
    md.append(f"# 📈 Dialogue Metrics — turn {turn} ({ts} UTC)")
    md.append("")
    md.append("**Window**: last {win} messages · **Total**: {all_} messages".format(win=s_win["messages"], all_=s_all["messages"]))
    md.append("")
    md.append("| Metric | Window | Total |")
    md.append("|---|---:|---:|")
    md.append(f"| Avg tone | {s_win['avg_tone']:+.2f} {tone_emoji(s_win['avg_tone'])} | {s_all['avg_tone']:+.2f} |")
    md.append(f"| Addressing, % | {pct(s_win['targeting_rate'])} | {pct(s_all['targeting_rate'])} |")
    md.append(f"| Emotion entropy | {s_win['emotion_entropy']:.2f} | {s_all['emotion_entropy']:.2f} |")
    md.append(f"| Avg reply words | {s_win['avg_reply_words']:.1f} | {s_all['avg_reply_words']:.1f} |")
    md.append(f"| Reciprocity | {s_win['reciprocity']:.2f} | {s_all['reciprocity']:.2f} |")
    md.append(f"| Response delay (turns) | {s_win['avg_addressing_delay']:.2f} | {s_all['avg_addressing_delay']:.2f} |")
    md.append("")
    # per-agent stats
    md.append("## Per agent (window)")
    md.append("| Agent | Msgs | Avg words | Avg tone | Last emotion | Target diversity |")
    md.append("|---|---:|---:|---:|---|---:|")
    for a, st in sorted(pa.items(), key=lambda kv: kv[1]["msgs"], reverse=True):
        md.append(f"| {a} | {st['msgs']} | {st['avg_words']:.1f} | {st['avg_tone']:+.2f} | {st['last_emotion']} | {st['targets_diversity']} |")
    md.append("")
    # speaking shares (as table)
    md.append("## Speaking shares (window)")
    tt = s_win["talk_share"]
    md.append("| Agent | Share |")
    md.append("|---|---:|")
    for a, v in sorted(tt.items(), key=lambda kv: kv[1], reverse=True):
        md.append(f"| {a} | {pct(v)} |")
    md.append("")
    return "\n".join(md)

def save_report_md(md_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_text, encoding="utf-8")
