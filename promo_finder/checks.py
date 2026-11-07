"""Hard rules a promo must pass, checked in code (not by asking a model).

resolve()  -> beat ids to real timestamps and durations
problems() -> list of human-readable rule violations
autofix()  -> safe, mechanical fixes (start on the hook, cut on the cliffhanger)
render()   -> the clip as a viewer would experience it, timestamps from 00:00
"""
from __future__ import annotations

import copy

from .parse import Beat, Episode, fmt

MIN_LEN, MAX_LEN = 20.0, 60.0
HOOK_WINDOW = 3.0       # hook must start within this many seconds of the clip start
CLIFF_WINDOW = 4.0      # cliffhanger beat must end within this many seconds of the cut
TOL = 0.5               # timing tolerance in seconds


def _int(x):
    try:
        return int(str(x).lstrip("Bb"))
    except (TypeError, ValueError):
        return None


def segment_beats(ep: Episode, start: float, end: float) -> list[Beat]:
    """Beats the viewer sees inside [start, end]: anything that starts in the window."""
    return [b for b in ep.beats if start - 0.01 <= b.start < end]


def resolve(promo: dict, eps: dict[int, Episode]) -> dict:
    """Fill in start/end seconds for every segment. Leaves ids untouched."""
    p = promo
    total = 0.0
    for s in p["segments"]:
        ep = eps.get(_int(s.get("episode")))
        i, j = _int(s.get("start_beat")), _int(s.get("end_beat"))
        s["start_beat"], s["end_beat"] = i, j
        if ep is None or i is None or j is None or not (0 <= i < len(ep.beats)) or not (0 <= j < len(ep.beats)):
            s["start"] = s["end"] = s["duration"] = None
            continue
        s["start"], s["end"] = ep.beats[i].start, ep.beats[j].end
        s["duration"] = round(s["end"] - s["start"], 2)
        total += max(s["duration"], 0)
    p["duration"] = round(total, 2)
    return p


def problems(p: dict, eps: dict[int, Episode], same_episode_order: bool = True) -> list[str]:
    errs = []
    segs = p.get("segments") or []
    if not segs:
        return ["no segments"]
    for k, s in enumerate(segs, 1):
        ep = eps.get(_int(s.get("episode")))
        if ep is None:
            errs.append(f"segment {k}: episode {s.get('episode')} not in this show")
            continue
        if s.get("start") is None:
            errs.append(f"segment {k}: beat ids {s.get('start_beat')}..{s.get('end_beat')} don't exist "
                        f"(episode {ep.number} has B0..B{len(ep.beats) - 1})")
            continue
        if s["end_beat"] < s["start_beat"] or s["duration"] <= 0:
            errs.append(f"segment {k}: end_beat B{s['end_beat']} is before start_beat B{s['start_beat']}")
        elif s["duration"] < 1.5:
            errs.append(f"segment {k} is only {s['duration']}s, too short to read on screen")
    if errs:
        return errs
    if same_episode_order:
        for a, b in zip(segs, segs[1:]):
            if a["episode"] == b["episode"] and b["start"] < a["end"] - 0.01:
                errs.append(f"segments overlap or are out of order in episode {a['episode']} "
                            f"({fmt(a['start'])}-{fmt(a['end'])} then {fmt(b['start'])}-{fmt(b['end'])})")
    d = p["duration"]
    if d < MIN_LEN - TOL:
        errs.append(f"total length {d:.1f}s is under {MIN_LEN:.0f}s")
    if d > MAX_LEN + TOL:
        errs.append(f"total length {d:.1f}s is over {MAX_LEN:.0f}s")

    first, last = segs[0], segs[-1]
    hb = _int(p.get("hook_beat"))
    if hb is not None:
        ep = eps[first["episode"]]
        if not (first["start_beat"] <= hb <= first["end_beat"]) or hb >= len(ep.beats):
            errs.append(f"hook_beat B{hb} is not inside the first segment (B{first['start_beat']}..B{first['end_beat']})")
        elif ep.beats[hb].start - first["start"] > HOOK_WINDOW + 0.01:
            errs.append(f"hook B{hb} starts {ep.beats[hb].start - first['start']:.1f}s into the clip; "
                        f"must be within {HOOK_WINDOW:.0f}s")
    cb = _int(p.get("cliffhanger_beat"))
    if cb is not None:
        ep = eps[last["episode"]]
        if not (last["start_beat"] <= cb <= last["end_beat"]) or cb >= len(ep.beats):
            errs.append(f"cliffhanger_beat B{cb} is not inside the last segment (B{last['start_beat']}..B{last['end_beat']})")
        elif last["end"] - ep.beats[cb].end > CLIFF_WINDOW + 0.01:
            errs.append(f"clip runs {last['end'] - ep.beats[cb].end:.1f}s past the cliffhanger B{cb}; cut closer to it")
    return errs


def autofix(p: dict, eps: dict[int, Episode]) -> tuple[dict, list[str]]:
    """Mechanical edits an editor would make anyway. Returns (new promo, notes)."""
    q = copy.deepcopy(p)
    notes = []
    segs = q.get("segments") or []
    if not segs or any(s.get("start") is None for s in segs):
        return q, notes
    first, last = segs[0], segs[-1]
    hb, cb = _int(q.get("hook_beat")), _int(q.get("cliffhanger_beat"))
    # Hook lands late -> start the clip on the hook.
    if hb is not None and first["start_beat"] <= hb <= first["end_beat"]:
        ep = eps[first["episode"]]
        if ep.beats[hb].start - first["start"] > HOOK_WINDOW:
            notes.append(f"moved start from B{first['start_beat']} to hook B{hb}")
            first["start_beat"] = hb
    # Clip drags past the cliffhanger -> cut on it.
    if cb is not None and last["start_beat"] <= cb <= last["end_beat"]:
        ep = eps[last["episode"]]
        if last["end"] - ep.beats[cb].end > CLIFF_WINDOW:
            notes.append(f"moved end from B{last['end_beat']} to cliffhanger B{cb}")
            last["end_beat"] = cb
    resolve(q, eps)
    return q, notes


def snap_to_shots(p: dict, eps: dict[int, Episode], max_shift: float = 1.0) -> None:
    """Editors cut on shot changes, not mid-shot. Widen each segment by at most
    max_shift seconds to the camera shot that contains its edge, as long as the
    clip stays under 60s and the hook stays inside the first 3 seconds.
    Run this last: it edits start/end directly and resolve() would undo it."""
    for k, s in enumerate(p["segments"]):
        ep = eps[s["episode"]]
        shots = [b for b in ep.beats if b.kind == "CAMERA"]
        new_start = next((b.start for b in shots if 0 < s["start"] - b.start <= max_shift and b.end > s["start"]),
                         s["start"])
        new_end = next((b.end for b in shots if 0 < b.end - s["end"] <= max_shift and b.start < s["end"]), s["end"])
        if k == 0 and p.get("hook_beat") is not None:
            try:
                if ep.beats[int(p["hook_beat"])].start - new_start > HOOK_WINDOW:
                    new_start = s["start"]
            except (IndexError, ValueError, TypeError):
                pass
        grow = (s["start"] - new_start) + (new_end - s["end"])
        if grow > 0 and p["duration"] + grow <= MAX_LEN:
            s["start"], s["end"] = new_start, new_end
            s["duration"] = round(s["end"] - s["start"], 2)
            p["duration"] = round(p["duration"] + grow, 2)


def _cliffiness(b: Beat) -> int:
    """How much a beat feels like a place to cut: an open question, a trailing-off line, a sting."""
    t = b.text.strip()
    if b.kind == "SFX":
        return 3 if any(k in t.lower() for k in ("cliffhanger", "cut to black", "silence", "sting", "swell")) else 0
    if b.kind == "CAMERA":
        return 2 if "cut to black" in t.lower() else 0
    return 3 if t.endswith("?") or t.endswith("?!") else 2 if t.endswith(("...", "…", "—", "-")) else 1 if t.endswith("!") else 0


def fit_length(p: dict, eps: dict[int, Episode]) -> tuple[dict, str] | None:
    """Last-resort fix when the model's pick is too short or too long and its own repair
    failed (in testing this dropped ep 41's cathedral explosion at 15 s). Keep the start
    (the hook) and move the cut to the nearest beat that leaves something hanging."""
    q = copy.deepcopy(p)
    segs = q.get("segments") or []
    if not segs or any(s.get("start") is None for s in segs):
        return None
    last = segs[-1]
    ep = eps[last["episode"]]
    other = q["duration"] - last["duration"]
    lo, hi = MIN_LEN - other, MAX_LEN - other          # allowed length for the last segment
    if lo > hi or hi <= 1.5:
        return None
    old = q["duration"]
    options = [b for b in ep.beats if b.kind != "CAMERA" or "cut to black" in b.text.lower()]
    options = [b for b in options if b.id > last["start_beat"] and lo <= b.end - last["start"] <= hi]
    if options:
        if q["duration"] < MIN_LEN:   # extend: earliest good cut point; prefer a real cliff within ~45 s
            pool = [b for b in options if _cliffiness(b) >= 2] or options
            pick = min(pool, key=lambda b: (b.end - last["start"] > 45, -_cliffiness(b) // 2, b.end))
        else:                         # trim: latest good cut point under the limit
            pick = max(options, key=lambda b: (_cliffiness(b) >= 2, b.end))
        last["end_beat"] = pick.id
        q["cliffhanger_beat"] = pick.id
        q["cliffhanger"] = pick.text
        resolve(q, eps)
        how = "extended" if q["duration"] > old else "trimmed"
        return q, f"{how} by code from {old:.1f}s to {q['duration']:.1f}s; now cuts on B{pick.id}"
    # Nothing after it to extend into: the moment is the episode's ending (episodes end on their
    # biggest beat). Open earlier instead, on the strongest nearby line, so the climax lands last.
    if q["duration"] < MIN_LEN and len(segs) == 1:
        from .heuristics import beat_heat
        first = segs[0]
        earlier = [b for b in ep.beats if b.id < first["start_beat"] and b.kind in ("DIALOGUE", "ACTION", "SFX")
                   and MIN_LEN <= first["end"] - b.start <= min(45.0, MAX_LEN)]
        if not earlier:
            return None
        pick = max(earlier, key=lambda b: (beat_heat(b), b.start))
        first["start_beat"] = pick.id
        q["hook_beat"] = pick.id
        q["hook"] = pick.text
        resolve(q, eps)
        return q, f"extended by code from {old:.1f}s to {q['duration']:.1f}s; now opens on B{pick.id}"
    return None


def infer_hook_and_cliff(p: dict, eps: dict[int, Episode]) -> None:
    """For montage promos (built from moments) pick hook/cliff beats in code."""
    first, last = p["segments"][0], p["segments"][-1]
    ep = eps[first["episode"]]
    early = [b for b in segment_beats(ep, first["start"], first["end"])
             if b.start - first["start"] <= HOOK_WINDOW and b.kind != "CAMERA"]
    p.setdefault("hook_beat", (early[0].id if early else first["start_beat"]))
    ep = eps[last["episode"]]
    tail = [b for b in segment_beats(ep, last["start"], last["end"]) if b.kind in ("DIALOGUE", "ACTION")]
    p.setdefault("cliffhanger_beat", (tail[-1].id if tail else last["end_beat"]))


def render(p: dict, eps: dict[int, Episode], with_source=False, compact=False) -> str:
    """The ad as the viewer experiences it: one timeline from 00:00, cuts marked.
    compact=True trims camera notes (about a third of the tokens) for the judge."""
    out, offset = [], 0.0
    for k, s in enumerate(p["segments"]):
        ep = eps[s["episode"]]
        if k:
            out.append("--- CUT ---")
        for b in segment_beats(ep, s["start"], s["end"]):
            rs = b.start - s["start"] + offset
            re_ = min(b.end, s["end"]) - s["start"] + offset
            line = b.line()
            body = line.split("] ", 1)[1]
            if compact and b.kind == "CAMERA":
                words = b.text.split()
                body = "CAMERA: " + " ".join(words[:14]) + ("..." if len(words) > 14 else "")
            src = f"  <ep{ep.number} B{b.id} {fmt(b.start)}>" if with_source else ""
            out.append(f"[{fmt(rs)}-{fmt(re_)}] {body}{src}")
        offset += s["duration"]
    return "\n".join(out)


def same_scene(a: dict, b: dict, eps: dict[int, Episode], edge: float = 30.0) -> bool:
    """Two single-episode promos from back-to-back episodes, one near the end of
    episode N and one near the start of N+1, are usually one scene split by the
    episode break (found in testing: ep 15's standoff and ep 16's fight)."""
    ea = {s["episode"] for s in a["segments"]}
    eb = {s["episode"] for s in b["segments"]}
    if len(ea) != 1 or len(eb) != 1:
        return False
    (na,), (nb,) = ea, eb
    if abs(na - nb) != 1:
        return False
    first, second = (a, b) if na < nb else (b, a)
    n1 = min(na, nb)
    tail_gap = eps[n1].duration - max(s["end"] for s in first["segments"])
    head_gap = min(s["start"] for s in second["segments"])
    return tail_gap <= edge and head_gap <= edge


def iou(a: dict, b: dict) -> float:
    """Time overlap between two promos (only counts segments in the same episode)."""
    inter = 0.0
    for s in a["segments"]:
        for t in b["segments"]:
            if s["episode"] == t["episode"]:
                inter += max(0.0, min(s["end"], t["end"]) - max(s["start"], t["start"]))
    union = a["duration"] + b["duration"] - inter
    return inter / union if union > 0 else 0.0
