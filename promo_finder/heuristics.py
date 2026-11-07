"""Rule-based promo finder. No API key needed.

Two jobs:
1. Offline fallback, so the tool still produces something with no model.
2. A baseline. If the LLM pipeline can't beat a keyword-and-punctuation scorer,
   the LLM isn't earning its cost.

It is deliberately simple and you can read the whole thing in five minutes.
"""
from __future__ import annotations

import re

from .parse import Beat, Episode

# Words that tend to show up around slaps, reveals, threats and betrayals.
HOT_WORDS = {
    3: ["slap", "strikes", "struck", "punch", "stab", "blood", "kill", "murder", "dead", "die", "hang",
        "execute", "execution", "poison", "arsenic", "witch", "burn", "fire", "explosion", "chains",
        "dungeon", "duke", "duchess", "pregnant", "divorce", "betray", "liar", "kneels", "propos",
        "marry me", "kiss", "sword", "knife", "dagger", "gallows", "iron", "arrest", "guards"],
    2: ["never", "how dare", "you knew", "disown", "banish", "humiliat", "scream", "gasp", "collapse",
        "grabs", "shoves", "throws", "tears", "trembl", "furious", "rage", "shatter", "crash",
        "truth", "secret", "who are you", "mine", "wife", "husband", "wedding", "ring"],
}
HOT_TAGS = ["outrag", "furious", "scream", "shout", "gasp", "shock", "devastat", "menac", "threat",
            "venom", "sadist", "gloat", "cruel", "desperate", "crying", "horrif", "enraged", "cold",
            "deadly", "lethal", "triumph", "mock", "defiant", "commanding"]
CLIFF_SFX = ["cliffhanger", "cut to black", "silence", "boom", "sting", "swell"]


def beat_heat(b: Beat) -> float:
    """How 'loud' a beat is, dramatically. Rough by design."""
    t = b.text.lower()
    score = 0.0
    for w, words in HOT_WORDS.items():
        score += w * sum(1 for k in words if k in t)
    if b.kind == "DIALOGUE":
        score += 0.5
        score += 1.0 * b.text.count("!")
        score += 0.5 * b.text.count("?")
        tags = " ".join(b.tags).lower()
        score += 1.0 * sum(1 for k in HOT_TAGS if k in tags)
    elif b.kind == "SFX":
        score += 1.5 if re.search(r"slap|crack|impact|thud|boom|scream|shatter|explo", t) else 0
    elif b.kind == "CAMERA":
        score *= 0.4   # camera notes repeat what action/dialogue already say
    return score


def _span(beats: list[Beat], i: int, end: float) -> list[Beat]:
    """Beats from index i that sit fully inside [beats[i].start, end]. Beats are time-sorted."""
    out = []
    for b in beats[i:]:
        if b.start > end:
            break
        if b.end <= end + 0.01:
            out.append(b)
    return out


def _window_score(ep: Episode, i: int, j: int, heat: dict[int, float]) -> dict:
    beats = ep.beats
    start, end = beats[i].start, beats[j].end
    span = _span(beats, i, end)
    hook = max([heat[b.id] for b in span if b.start - start <= 3.0] or [0])
    tail = [b for b in span if end - b.end <= 5.0]
    cliff = max([heat[b.id] for b in tail] or [0])
    last_line = next((b for b in reversed(span) if b.kind == "DIALOGUE"), None)
    if last_line and (last_line.text.endswith("?") or last_line.text.endswith("...")):
        cliff += 2
    if any(k in b.text.lower() for b in tail if b.kind == "SFX" for k in CLIFF_SFX):
        cliff += 2
    density = sum(heat[b.id] for b in span) / max(end - start, 1)
    return {"hook": hook, "cliff": cliff, "density": density,
            "score": 0.45 * hook + 0.35 * cliff + 2.0 * density}


def find_promos(ep: Episode, min_len=20.0, max_len=60.0, per_episode=2) -> list[dict]:
    """Slide a window over the episode and keep the best non-overlapping ones."""
    beats = ep.beats
    heat = {b.id: beat_heat(b) for b in beats}
    starts = [k for k, b in enumerate(beats) if b.kind in ("DIALOGUE", "ACTION", "SFX")]
    ends = [k for k, b in enumerate(beats) if b.kind in ("DIALOGUE", "SFX", "ACTION")]
    cands = []
    for i in starts:
        for j in ends:
            if j <= i:
                continue
            d = beats[j].end - beats[i].start
            if d < min_len:
                continue
            if d > max_len:
                break
            s = _window_score(ep, i, j, heat)
            cands.append((s["score"], i, j, s))
    cands.sort(reverse=True)
    picked = []
    for score, i, j, s in cands:
        a, b = beats[i].start, beats[j].end
        if any(not (b <= p["start"] or a >= p["end"]) for p in picked):
            continue
        picked.append({"start": a, "end": b, "start_beat": i, "end_beat": j, **s})
        if len(picked) >= per_episode:
            break
    out = []
    for p in picked:
        i, j = p["start_beat"], p["end_beat"]
        span = _span(beats, i, p["end"])
        hook_b = max((b for b in span if b.start - p["start"] <= 3.0), key=lambda b: heat[b.id], default=beats[i])
        lines = [b for b in span if b.kind == "DIALOGUE"]
        cliff_b = lines[-1] if lines else beats[j]
        out.append({
            "source": "heuristic",
            "title": f"Ep {ep.number} high-intensity stretch",
            "segments": [{"episode": ep.number, "start_beat": i, "end_beat": j}],
            "hook_beat": hook_b.id, "cliffhanger_beat": cliff_b.id,
            "hook": hook_b.text, "cliffhanger": cliff_b.text,
            "why": f"keyword/punctuation heat: hook {p['hook']:.1f}, ending {p['cliff']:.1f}, density {p['density']:.2f}/s",
            "context_needed": "unknown (rules can't judge this)",
            "scout_scores": {"hook": min(10, round(p["hook"])), "cliffhanger": min(10, round(p["cliff"])),
                             "standalone": None},
        })
    return out
