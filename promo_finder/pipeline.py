"""The promo finder pipeline.

  parse (code) -> show bible (LLM, 1 call) -> scout each episode (LLM, 1 call/ep)
  -> rule checks + autofix (code) -> repair what's still broken (LLM, only if needed)
  -> cross-episode trailers from the best moments (LLM, 1 call)
  -> blind judge: a "new viewer" model sees only the clip (LLM, 1 call/promo)
  -> rank (code, weights below) -> write outputs
"""
from __future__ import annotations

import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

from . import checks, heuristics, prompts
from .llm import LLM, Cascade, LLMError
from .parse import Episode, fmt, load_show


@dataclass
class Config:
    # Default = free Groq tier: Qwen scouts, a different model family (gpt-oss) judges.
    # Gemini works too (--provider gemini) but its free tier stops at ~20 requests/day/model.
    provider: str = "groq"
    model: str | None = None              # None = provider default (groq: qwen/qwen3.8-27b)
    base_url: str | None = None
    fallbacks: list[str] | None = None    # e.g. ["groq:openai/gpt-oss-20b@low"]; used when a quota runs out
    judge_provider: str | None = "groq"   # "same" or None = reuse the scout model
    judge_model: str | None = "openai/gpt-oss-120b"
    judge_base_url: str | None = None
    mode: str = "llm"                     # llm | heuristic
    temperature: float = 0.4
    reasoning: str | None = None          # reasoning_effort for thinking models (low/medium/high)
    judge_reasoning: str | None = "low"
    workers: int = 4
    judge_workers: int = 2                # judges often sit on a tighter free tier
    rpm: float = 0
    judge_rpm: float = 0
    tpm: float = 0                        # 0 = provider default (Groq: 7000)
    judge_tpm: float = 0
    max_candidates: int = 3               # per episode
    montage_n: int = 4
    judge: bool = True
    judge_per_episode: int = 3            # judge only the best N scout picks per episode (0 = all)
    early_episode_bonus: bool = True
    # The keyword baseline tied the LLM per episode in a blind test (20 episodes), so its best
    # window per episode also goes to the judge and competes in the ranking. Costs 1 judge call/ep.
    rules_proposer: bool = True
    cache_dir: str = ".cache/llm"
    episodes: list[int] | None = None     # run on a subset, e.g. [1,2,3]
    weights: dict = field(default_factory=lambda: {
        "judge_hook": 0.30, "judge_cliff": 0.30, "judge_standalone": 0.15, "judge_overall": 0.25,
        "judge_share": 0.8, "scout_share": 0.2,
        "verdict_post": 0.3, "verdict_skip": -0.5, "sweet_spot": 0.2,
        "early_ep_1_3": 0.3, "early_ep_4_10": 0.15})


Progress = Callable[[str, int, int], None]


def _noop(*_):
    pass


# ------------------------------------------------------------------ helpers
def beats_listing(ep: Episode) -> str:
    return "\n".join(f"B{b.id} {b.line()}" for b in ep.beats)


def bible_text(bible: dict) -> str:
    if not bible:
        return "(none)"
    lines = [f"Premise: {bible.get('premise', '')}"]
    for c in bible.get("characters", [])[:14]:
        labels = ", ".join(c.get("labels", [])[:6])
        lines.append(f"- {c.get('name')} [{labels}]: {c.get('role', '')}. {c.get('note', '')}")
    if bible.get("big_moments"):
        lines.append("Turning points: " + " | ".join(bible["big_moments"][:8]))
    return "\n".join(lines)


def _num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _clean_candidates(raw: dict, ep: Episode, limit: int) -> list[dict]:
    """Normalise whatever shape the model sent. Models drift from the schema: gpt-oss-20b
    once sent "scores": 5 instead of an object, which crashed a 50-episode run."""
    cands = raw.get("candidates") if isinstance(raw, dict) else None
    if isinstance(cands, dict):
        cands = [cands]
    out = []
    for c in (cands if isinstance(cands, list) else [])[:limit]:
        if not isinstance(c, dict):
            continue
        segs_raw = c.get("segments")
        if isinstance(segs_raw, dict):
            segs_raw = [segs_raw]
        if not isinstance(segs_raw, list):   # some models put start/end on the candidate itself
            segs_raw = [{"start_beat": c.get("start_beat"), "end_beat": c.get("end_beat")}]
        segs = [{"episode": ep.number, "start_beat": s.get("start_beat"), "end_beat": s.get("end_beat")}
                for s in segs_raw if isinstance(s, dict)]
        sc = c.get("scores")
        if not isinstance(sc, dict):
            sc = {k: sc for k in ("hook", "cliffhanger", "standalone")} if isinstance(sc, (int, float)) else {}
        out.append({
            "source": "llm-scout", "title": str(c.get("title") or ""), "segments": segs,
            "hook_beat": c.get("hook_beat"), "cliffhanger_beat": c.get("cliffhanger_beat"),
            "hook": str(c.get("hook") or ""), "cliffhanger": str(c.get("cliffhanger") or ""),
            "why": str(c.get("why") or ""), "context_needed": str(c.get("context_needed") or ""),
            "scout_scores": {k: _num(sc.get(k)) for k in ("hook", "cliffhanger", "standalone")},
        })
    return out


def _check_all(cands: list[dict], eps: dict, same_order=True) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    ok, bad = [], []
    for c in cands:
        checks.resolve(c, eps)
        errs = checks.problems(c, eps, same_order)
        if errs:
            fixed, notes = checks.autofix(c, eps)
            errs2 = checks.problems(fixed, eps, same_order)
            if not errs2:
                fixed["fixes"] = notes
                ok.append(fixed)
                continue
            c["_problems"] = errs2
            bad.append((c, errs2))
        else:
            c["fixes"] = []
            ok.append(c)
    return ok, bad


# -------------------------------------------------------------------- stages
def build_bible(llm: LLM, title: str, eps: list[Episode]) -> dict:
    speakers = Counter(b.speaker for e in eps for b in e.beats if b.speaker)
    msg = prompts.BIBLE_USER.format(
        title=title,
        loglines="\n".join(f"Ep {e.number}: {e.logline}" for e in eps),
        speakers=", ".join(f"{s} ({n})" for s, n in speakers.most_common(45)))
    try:
        return llm.chat_json([{"role": "system", "content": prompts.BIBLE_SYSTEM},
                              {"role": "user", "content": msg}], stage="bible", temperature=0.2)
    except LLMError as e:
        return {"premise": "", "characters": [], "big_moments": [], "_error": str(e)}


def scout_episode(llm: LLM, title: str, bible: str, ep: Episode, prev: list[Episode],
                  eps: dict, cfg: Config, raw_dir: Path) -> dict:
    """One episode, isolated: any failure is logged for that episode and the run goes on."""
    try:
        return _scout_episode(llm, title, bible, ep, prev, eps, cfg, raw_dir)
    except Exception as e:  # noqa: BLE001 - deliberate: one episode must not kill a show
        err = f"{type(e).__name__}: {e}"
        _dump(raw_dir / f"scout_ep{ep.number:03d}_error.json", {"episode": ep.number, "error": err})
        return {"episode": ep.number, "ok": [], "rejected": [], "moments": [], "error": err}


def _scout_episode(llm: LLM, title: str, bible: str, ep: Episode, prev: list[Episode],
                   eps: dict, cfg: Config, raw_dir: Path) -> dict:
    user = prompts.SCOUT_USER.format(
        title=title, bible=bible,
        recent=" ".join(f"Ep {e.number}: {e.logline}" for e in prev[-2:]) or "(this is the first episode)",
        number=ep.number, duration=round(ep.duration, 1), logline=ep.logline, beats=beats_listing(ep))
    messages = [{"role": "system", "content": prompts.SCOUT_SYSTEM}, {"role": "user", "content": user}]
    record = {"episode": ep.number, "attempts": []}
    try:
        raw = llm.chat_json(messages, stage=f"scout:ep{ep.number}")
    except LLMError as e:
        record["error"] = str(e)
        _dump(raw_dir / f"scout_ep{ep.number:03d}.json", record)
        return {"episode": ep.number, "ok": [], "rejected": [], "moments": [], "error": str(e)}
    record["attempts"].append(raw)
    cands = _clean_candidates(raw, ep, cfg.max_candidates)
    ok, bad = _check_all(cands, eps)

    if bad:  # one repair round, with the exact rule failures and only the beats that matter
        problems = "\n".join(f"- \"{c['title']}\" ({c.get('duration', 0):.1f}s): " + "; ".join(errs) for c, errs in bad)
        ids = [int(s[k]) for c, _ in bad for s in c["segments"] for k in ("start_beat", "end_beat")
               if isinstance(s.get(k), int) and 0 <= s[k] < len(ep.beats)]
        lo, hi = (max(0, min(ids) - 12), min(len(ep.beats) - 1, max(ids) + 12)) if ids else (0, len(ep.beats) - 1)
        shown = [{k: c.get(k) for k in ("title", "segments", "hook_beat", "hook", "cliffhanger_beat", "cliffhanger")}
                 for c, _ in bad]
        for c in shown:
            c["segments"] = [{"start_beat": s.get("start_beat"), "end_beat": s.get("end_beat")} for s in c["segments"]]
        rep_user = prompts.REPAIR_USER.format(
            title=title, number=ep.number, candidates=json.dumps(shown, ensure_ascii=False, indent=1),
            problems=problems, beats="\n".join(f"B{b.id} {b.line()}" for b in ep.beats[lo:hi + 1]))
        rep_msgs = [{"role": "system", "content": prompts.SCOUT_SYSTEM}, {"role": "user", "content": rep_user}]
        try:
            raw2 = llm.chat_json(rep_msgs, stage=f"repair:ep{ep.number}")
            record["attempts"].append(raw2)
            known = {o["title"] for o in ok}       # the model resends good ones too; skip those
            retry = [c for c in _clean_candidates(raw2, ep, cfg.max_candidates) if c["title"] not in known]
            ok2, bad2 = _check_all(retry, eps)
            for c in ok2:
                c["fixes"] = c.get("fixes", []) + ["repaired by model after rule check"]
            ok += ok2
            if ok2 or bad2:
                bad = bad2
            if raw2.get("moments"):
                raw = {**raw, "moments": raw2["moments"]}
        except LLMError as e:
            record["repair_error"] = str(e)

    # Last resort for picks that are still too short/long: code moves the cut (see checks.fit_length).
    still_bad = []
    for c, errs in bad:
        if any(("under" in e or "over" in e) and "s is" in e for e in errs):
            fitted = checks.fit_length(c, eps)
            if fitted and not checks.problems(fitted[0], eps):
                f, note = fitted
                f["fixes"] = (c.get("fixes") or []) + [note]
                ok.append(f)
                continue
        still_bad.append((c, errs))
    bad = still_bad

    moments = []
    for m in (raw.get("moments") or [])[:6]:
        if not isinstance(m, dict):
            continue
        mm = {"source": "moment", "segments": [{"episode": ep.number, "start_beat": m.get("start_beat"),
                                                "end_beat": m.get("end_beat")}],
              "type": m.get("type", "other"), "what": m.get("what", "")}
        checks.resolve(mm, eps)
        s = mm["segments"][0]
        if s["start"] is not None and 1.5 <= (s["duration"] or 0) <= 15:
            moments.append(mm)
    for k, c in enumerate(ok):
        c["id"] = f"E{ep.number:02d}-{chr(65 + k)}"
        c["scout_model"] = llm.last_name
    record["accepted"] = [c["id"] for c in ok]
    record["rejected"] = [{"title": c["title"], "problems": errs} for c, errs in bad]
    record["model"] = llm.last_name
    _dump(raw_dir / f"scout_ep{ep.number:03d}.json", record)
    return {"episode": ep.number, "ok": ok, "rejected": [c for c, _ in bad], "moments": moments}


def build_montages(llm: LLM, title: str, bible: str, moments: list[dict], eps: dict,
                   cfg: Config, raw_dir: Path) -> tuple[list[dict], list[dict]]:
    if len(moments) < 3 or cfg.montage_n <= 0:
        return [], []
    by_id = {}
    lines = []
    for k, m in enumerate(moments, 1):
        mid = f"M{k}"
        by_id[mid] = m
        s = m["segments"][0]
        ep = eps[s["episode"]]
        first_line = next((b.text for b in checks.segment_beats(ep, s["start"], s["end"]) if b.kind == "DIALOGUE"), "")
        lines.append(f'{mid} ep{s["episode"]} {s["duration"]:.1f}s {m["type"]}: {m["what"]}'
                     + (f' | "{first_line[:90]}"' if first_line else ""))
    user = prompts.MONTAGE_USER.format(title=title, bible=bible, moments="\n".join(lines), n=cfg.montage_n)
    messages = [{"role": "system", "content": prompts.MONTAGE_SYSTEM}, {"role": "user", "content": user}]

    def to_promos(raw):
        out = []
        for k, p in enumerate((raw.get("promos") or [])[: cfg.montage_n + 2]):
            ids = [str(i).strip().upper() for i in (p.get("moment_ids") or [])]
            ids = [i if i.startswith("M") else f"M{i}" for i in ids]
            missing = [i for i in ids if i not in by_id]
            segs = [dict(by_id[i]["segments"][0]) for i in ids if i in by_id]
            out.append({"source": "llm-montage", "title": p.get("title", ""), "segments": segs,
                        "moment_ids": ids, "hook": p.get("hook", ""), "cliffhanger": p.get("cliffhanger", ""),
                        "why": p.get("why", ""), "angle": p.get("angle", ""),
                        "context_needed": "", "scout_scores": {}, "_missing": missing})
        return out

    record = {"moments": lines, "attempts": []}
    try:
        raw = llm.chat_json(messages, stage="montage", temperature=0.6)
    except LLMError as e:
        record["error"] = str(e)
        _dump(raw_dir / "montage.json", record)
        return [], []
    record["attempts"].append(raw)
    cands = to_promos(raw)
    ok, bad = [], []
    for c in cands:
        checks.resolve(c, eps)
        errs = [f"unknown moment ids {c['_missing']}"] if c["_missing"] else []
        errs += checks.problems(c, eps, same_episode_order=False) if c["segments"] else ["no valid moments"]
        errs += _montage_shape(c)
        c["_problems"] = errs
        (bad.append((c, errs)) if errs else ok.append(c))
    if bad:
        problems = "\n".join(f"- \"{c['title']}\" ({sum(s['duration'] or 0 for s in c['segments']):.1f}s): "
                             + "; ".join(e) for c, e in bad)
        try:
            raw2 = llm.chat_json(messages + [
                {"role": "assistant", "content": json.dumps(raw, ensure_ascii=False)},
                {"role": "user", "content": "These promos failed the checks:\n" + problems +
                 "\nFix them by adding, removing or swapping moments. Return the full JSON again."}],
                stage="montage:repair", temperature=0.3)
            record["attempts"].append(raw2)
            bad = []
            known = {c["title"] for c in ok}
            for c in to_promos(raw2):
                checks.resolve(c, eps)
                errs = [f"unknown moment ids {c['_missing']}"] if c["_missing"] else []
                errs += checks.problems(c, eps, same_episode_order=False) if c["segments"] else ["no valid moments"]
                errs += _montage_shape(c)
                c["_problems"] = errs
                if errs:
                    bad.append((c, errs))
                elif c["title"] not in known:
                    c["fixes"] = ["repaired by model after rule check"]
                    ok.append(c)
        except LLMError as e:
            record["repair_error"] = str(e)
    for k, c in enumerate(ok[: cfg.montage_n], 1):
        checks.infer_hook_and_cliff(c, eps)
        c.setdefault("fixes", [])
        c["id"] = f"X-{k}"
        c["scout_model"] = llm.last_name
    record["accepted"] = [c["id"] for c in ok[: cfg.montage_n]]
    record["rejected"] = [{"title": c["title"], "problems": e} for c, e in bad]
    _dump(raw_dir / "montage.json", record)
    return ok[: cfg.montage_n], [c for c, _ in bad]


def _montage_shape(c: dict) -> list[str]:
    """Trailer rules the prompt asks for but models don't always follow."""
    errs = []
    if len(c["segments"]) > 5:
        errs.append(f"uses {len(c['segments'])} moments; max is 5")
    if len({s["episode"] for s in c["segments"]}) < 2:
        errs.append("all moments are from one episode; a trailer must mix at least 2")
    return errs


def judge_promo(llm: LLM, promo: dict, eps: dict) -> dict:
    transcript = checks.render(promo, eps, compact=True)
    user = prompts.JUDGE_USER.format(duration=promo["duration"], transcript=transcript)
    try:
        j = llm.chat_json([{"role": "system", "content": prompts.JUDGE_SYSTEM},
                           {"role": "user", "content": user}], stage=f"judge:{promo['id']}", temperature=0)
    except LLMError as e:
        return {"error": str(e)}
    if not isinstance(j, dict):
        return {"error": f"judge returned {type(j).__name__}"}
    sc = j.get("scores") if isinstance(j.get("scores"), dict) else {}
    return {"first_3_seconds": j.get("first_3_seconds", ""),
            "scores": {k: _num(sc.get(k)) for k in ("hook", "cliffhanger", "standalone", "overall")},
            "confusing": j.get("confusing") or [], "verdict": str(j.get("verdict", "")).lower().strip(),
            "reason": j.get("reason", "")}


# --------------------------------------------------------------------- rank
def score_promo(p: dict, cfg: Config) -> float:
    w = cfg.weights
    j = (p.get("judge") or {}).get("scores") or {}
    s = p.get("scout_scores") or {}
    s_vals = [v for v in s.values() if v is not None]
    scout = statistics.mean(s_vals) if s_vals else None
    if all(j.get(k) is not None for k in ("hook", "cliffhanger", "standalone", "overall")):
        judge = (w["judge_hook"] * j["hook"] + w["judge_cliff"] * j["cliffhanger"]
                 + w["judge_standalone"] * j["standalone"] + w["judge_overall"] * j["overall"])
        base = w["judge_share"] * judge + w["scout_share"] * (scout if scout is not None else judge)
    else:
        # The scout grades its own homework and runs high, so unjudged picks sit lower.
        base = (scout - 1.0) if scout is not None else 0.0
        if "not judged" not in p.setdefault("flags", []):
            p["flags"].append("not judged")
    verdict = (p.get("judge") or {}).get("verdict", "")
    base += w["verdict_post"] if verdict == "post" else w["verdict_skip"] if verdict == "skip" else 0
    if 25 <= p["duration"] <= 45:
        base += w["sweet_spot"]
    if cfg.early_episode_bonus:
        first_ep = min(s["episode"] for s in p["segments"])
        base += w["early_ep_1_3"] if first_ep <= 3 else w["early_ep_4_10"] if first_ep <= 10 else 0
    return round(base, 3)


def why_post(p: dict, rank: int) -> str:
    j = p.get("judge") or {}
    sc = j.get("scores") or {}
    eps_used = sorted({s["episode"] for s in p["segments"]})
    bits = []
    n = lambda k: "?" if sc.get(k) is None else f"{sc[k]:.0f}"
    if sc:
        bits.append(f"blind viewer gave it {n('overall')}/10 (hook {n('hook')}, "
                    f"cliffhanger {n('cliffhanger')}, clarity {n('standalone')})")
    if eps_used[0] <= 3:
        bits.append("comes from the first episodes, so a new viewer who taps lands right where it continues")
    if len(eps_used) > 1:
        bits.append(f"trailer cut across episodes {', '.join(map(str, eps_used))}")
    if j.get("confusing"):
        bits.append("watch out: " + "; ".join(map(str, j["confusing"][:2])))
    return "; ".join(bits)


def rank(promos: list[dict], cfg: Config, eps: dict | None = None) -> list[dict]:
    for p in promos:
        p["final_score"] = score_promo(p, cfg)
        # "Why it works" = the blind judge's one line. The scout's own note is kept separately:
        # it is written before repair/fit_length, so it can describe a cut that no longer exists.
        reason = (p.get("judge") or {}).get("reason")
        if reason:
            p.setdefault("scout_note", p.get("why", ""))
            p["why"] = str(reason)
    # Judged promos always rank above unjudged ones: the scout's self-scores run high
    # (8-10 on nearly everything in testing) and can't be compared with the judge's.
    promos.sort(key=lambda p: ("not judged" not in (p.get("flags") or []), p["final_score"]), reverse=True)
    kept = []
    for p in promos:
        dup = next((k for k in kept if checks.iou(p, k) >= 0.5), None)
        if dup is None and eps:
            dup = next((k for k in kept if checks.same_scene(p, k, eps)), None)
            if dup is not None:
                p["duplicate_reason"] = "same scene, continued across the episode break"
        if dup:
            p["duplicate_of"] = dup["id"]
            p.setdefault("duplicate_reason", "overlaps the same footage")
        else:
            kept.append(p)
    for r, p in enumerate(kept, 1):
        p["rank"] = r
        p["why_post"] = why_post(p, r)
    return kept + [p for p in promos if p.get("duplicate_of")]


# ---------------------------------------------------------------------- run
def run(path: str | Path, out_dir: str | Path, cfg: Config, progress: Progress = _noop) -> dict:
    title, episodes = load_show(path)
    return run_episodes(title, episodes, out_dir, cfg, progress)


def run_episodes(title: str, episodes: list[Episode], out_dir: str | Path, cfg: Config,
                 progress: Progress = _noop) -> dict:
    from .report import write_outputs   # local import to keep module graph simple

    if cfg.episodes:
        episodes = [e for e in episodes if e.number in set(cfg.episodes)]
    if not episodes:
        raise ValueError("No episodes parsed. Is this the right folder/zip?")
    eps = {e.number: e for e in episodes}
    out = Path(out_dir)
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "llm_calls.jsonl").unlink(missing_ok=True)

    # Baseline (always; it's free and gives us something to compare with)
    baseline = []
    for e in episodes:
        for k, c in enumerate(heuristics.find_promos(e)):
            checks.resolve(c, eps)
            checks.snap_to_shots(c, eps)
            c["id"] = f"H{e.number:02d}-{chr(65 + k)}"
            c["fixes"] = []
            baseline.append(c)

    meta = {"show": title, "episodes": len(episodes), "minutes": round(sum(e.duration for e in episodes) / 60, 1),
            "mode": cfg.mode, "config": {k: v for k, v in asdict(cfg).items() if k != "weights"},
            "weights": cfg.weights}

    if cfg.mode == "heuristic":
        for p in baseline:
            p["final_score"] = round(heuristics_score(p), 3)
        baseline.sort(key=lambda p: p["final_score"], reverse=True)
        for r, p in enumerate(baseline, 1):
            p["rank"] = r
            p["why_post"] = p["why"]
        result = {"title": title, "meta": meta, "promos": baseline, "rejected": [], "baseline": baseline,
                  "bible": {}}
        write_outputs(result, out, eps)
        return result

    log = raw_dir / "llm_calls.jsonl"
    llm = LLM(provider=cfg.provider, model=cfg.model, base_url=cfg.base_url, temperature=cfg.temperature,
              reasoning_effort=cfg.reasoning, rpm=cfg.rpm, tpm=cfg.tpm, cache_dir=Path(cfg.cache_dir), log_path=log)
    backups = []
    for f in cfg.fallbacks or []:
        try:
            backups.append(Cascade.parse(f, temperature=cfg.temperature, cache_dir=Path(cfg.cache_dir), log_path=log))
        except LLMError as e:        # e.g. no key for that provider: skip it, don't fail the run
            print(f"  (skipping fallback {f}: {e})")
    if backups:
        llm = Cascade([llm] + backups)
    judge_llm = llm
    if cfg.judge_provider not in (None, "same"):
        judge_llm = LLM(provider=cfg.judge_provider, model=cfg.judge_model,
                        base_url=cfg.judge_base_url, temperature=0, reasoning_effort=cfg.judge_reasoning,
                        rpm=cfg.judge_rpm or cfg.rpm, tpm=cfg.judge_tpm, cache_dir=Path(cfg.cache_dir), log_path=log)
    meta["model"], meta["judge_model"] = llm.name, judge_llm.name

    progress("Reading the whole show (bible)", 0, 1)
    bible = build_bible(llm, title, episodes)
    _dump(raw_dir / "bible.json", bible)
    btxt = bible_text(bible)

    # Scout and judge run as a pipeline: an episode's picks go to the judge as soon as
    # that episode is scouted. On free tiers each model has its own per-minute budget,
    # so overlapping the two roughly halves wall-clock time.
    judge_ex = ThreadPoolExecutor(max_workers=max(1, cfg.judge_workers))
    judge_futs: dict = {}

    def send_to_judge(cands: list[dict], top_n: int | None):
        if not cfg.judge:
            return
        cands = sorted(cands, reverse=True,
                       key=lambda c: statistics.mean([v for v in (c.get("scout_scores") or {}).values()
                                                      if v is not None] or [0]))
        for p in (cands[:top_n] if top_n else cands):
            judge_futs[judge_ex.submit(judge_promo, judge_llm, p, eps)] = p

    progress("Scouting episodes (judging alongside)", 0, len(episodes))
    results, done = [], 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as ex:
        futs = {ex.submit(scout_episode, llm, title, btxt, e, episodes[:i], eps, cfg, raw_dir): e
                for i, e in enumerate(episodes)}
        for f in as_completed(futs):
            r = f.result()
            for c in r["ok"]:
                checks.snap_to_shots(c, eps)      # final edit points before anyone judges the cut
            send_to_judge(r["ok"], cfg.judge_per_episode)
            results.append(r)
            done += 1
            progress("Scouting episodes (judging alongside)", done, len(episodes))
    results.sort(key=lambda r: r["episode"])
    singles = [c for r in results for c in r["ok"]]
    rejected = [c for r in results for c in r["rejected"]]
    moments = [m for r in results for m in r["moments"]]
    scout_errors = {r["episode"]: r["error"] for r in results if r.get("error")}

    progress("Cutting cross-episode trailers", 0, 1)
    try:
        montages, m_rejected = build_montages(llm, title, btxt, moments, eps, cfg, raw_dir)
    except Exception as e:  # noqa: BLE001 - trailers are a bonus; never lose the singles over them
        _dump(raw_dir / "montage_error.json", {"error": f"{type(e).__name__}: {e}"})
        montages, m_rejected = [], []
    rejected += m_rejected
    for c in montages:
        checks.snap_to_shots(c, eps)
    send_to_judge(montages, None)

    rules_picks = []
    if cfg.rules_proposer:
        best = {}
        for b in baseline:                       # baseline list is best-first within each episode
            best.setdefault(b["segments"][0]["episode"], b)
        for b in best.values():
            if b["segments"][0]["episode"] in scout_errors:
                continue
            q = json.loads(json.dumps(b))         # independent copy; baseline file stays as it was
            q.update({"source": "rules", "scout_scores": {}, "scout_model": "keyword rules"})
            rules_picks.append(q)
        send_to_judge(rules_picks, None)

    total, done = len(judge_futs), 0
    progress("Blind judging", 0, total)
    for f in as_completed(judge_futs):
        judge_futs[f]["judge"] = f.result()
        done += 1
        progress("Blind judging", done, total)
    judge_ex.shutdown(wait=True)
    judge_errors = sorted(p["id"] for p in judge_futs.values() if (p.get("judge") or {}).get("error"))

    promos = rank(singles + montages + rules_picks, cfg, eps)
    usage = {"main": llm.usage}
    if judge_llm is not llm:
        usage["judge"] = judge_llm.usage
    meta.update({"usage": usage, "scout_errors": scout_errors, "judge_errors": judge_errors, "moments": len(moments),
                 "candidates_accepted": len(singles) + len(montages), "candidates_rejected": len(rejected)})
    result = {"title": title, "meta": meta, "bible": bible, "promos": promos, "rejected": rejected,
              "baseline": baseline, "moments": moments}
    write_outputs(result, out, eps)
    progress("Done", 1, 1)
    return result


def heuristics_score(p: dict) -> float:
    s = p.get("scout_scores") or {}
    return (s.get("hook") or 0) * 0.5 + (s.get("cliffhanger") or 0) * 0.5


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
