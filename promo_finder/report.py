"""Write a run's results: JSON (for the app), CSV (for a spreadsheet), Markdown (for a human)."""
from __future__ import annotations

import csv
import json

from pathlib import Path

from . import checks
from .parse import Episode, fmt


def seg_str(p: dict) -> str:
    return " + ".join(f"Ep{s['episode']} {fmt(s['start'])}-{fmt(s['end'])}" for s in p["segments"])


def _sc(p, who, k):
    d = (p.get("judge") or {}).get("scores") if who == "judge" else p.get("scout_scores")
    v = (d or {}).get(k)
    return "" if v is None else f"{v:.0f}"


def _beat_text(eps, ep_num, bid):
    try:
        b = eps[ep_num].beats[int(bid)]
        return b.text
    except (KeyError, IndexError, TypeError, ValueError):
        return ""


def write_outputs(result: dict, out: Path, eps: dict[int, Episode]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    promos = result["promos"]
    for p in promos + result.get("baseline", []):
        p["transcript"] = checks.render(p, eps, with_source=True)
        p["segments_str"] = seg_str(p)
        # Exact script text at the hook / cliffhanger beats, so nobody has to trust the model's quote
        first, last = p["segments"][0], p["segments"][-1]
        p["hook_line"] = _beat_text(eps, first["episode"], p.get("hook_beat"))
        p["cliffhanger_line"] = _beat_text(eps, last["episode"], p.get("cliffhanger_beat"))

    (out / "promos.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(result["meta"], ensure_ascii=False, indent=2, default=str),
                                       encoding="utf-8")
    if result.get("rejected"):
        (out / "rejected.json").write_text(json.dumps(
            [{"title": c.get("title"), "source": c.get("source"), "segments": c.get("segments"),
              "problems": c.get("_problems") or c.get("_missing")} for c in result["rejected"]],
            ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _write_csv(out / "promos_ranked.csv", promos)
    if result.get("baseline") and result["meta"]["mode"] != "heuristic":
        _write_csv(out / "baseline_heuristic.csv", result["baseline"])
    _write_md(out / "report.md", result)


def _write_csv(path: Path, promos: list[dict]) -> None:
    cols = ["rank", "id", "type", "proposed_by", "title", "segments", "duration_s", "hook", "hook_line", "cliffhanger",
            "cliffhanger_line", "why", "why_post_first", "scout_note", "judge_hook", "judge_cliff", "judge_clarity",
            "judge_overall", "verdict", "viewer_reason", "viewer_confused_by", "scout_hook", "scout_cliff",
            "scout_clarity", "final_score", "auto_fixes", "duplicate_of"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in promos:
            j = p.get("judge") or {}
            w.writerow([p.get("rank", ""), p["id"],
                        "cross-episode" if len({s["episode"] for s in p["segments"]}) > 1
                        else ("multi-cut" if len(p["segments"]) > 1 else "single"),
                        p.get("scout_model") or p.get("source", ""),
                        p.get("title", ""), p["segments_str"], p["duration"], p.get("hook", ""), p.get("hook_line", ""),
                        p.get("cliffhanger", ""), p.get("cliffhanger_line", ""), p.get("why", ""), p.get("why_post", ""),
                        p.get("scout_note", ""), _sc(p, "judge", "hook"), _sc(p, "judge", "cliffhanger"), _sc(p, "judge", "standalone"),
                        _sc(p, "judge", "overall"), j.get("verdict", ""), j.get("reason", ""),
                        " | ".join(map(str, j.get("confusing") or [])),
                        _sc(p, "scout", "hook"), _sc(p, "scout", "cliffhanger"), _sc(p, "scout", "standalone"),
                        p.get("final_score", ""), "; ".join(p.get("fixes") or []), p.get("duplicate_of", "")])


def _write_md(path: Path, r: dict) -> None:
    m = r["meta"]
    promos = [p for p in r["promos"] if not p.get("duplicate_of")]
    L = [f"# Promo picks: {r['title']}", ""]
    L.append(f"{m['episodes']} episodes ({m['minutes']} min of show). Mode: **{m['mode']}**"
             + (f", scout model `{m.get('model')}`, judge model `{m.get('judge_model')}`." if m["mode"] == "llm" else "."))
    log = path.parent / "raw" / "llm_calls.jsonl"
    if log.exists():   # totals from the call log: cached entries keep the usage of their original call
        calls = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        by_model: dict[str, list[int]] = {}
        for c in calls:
            u = c.get("usage") or {}
            t = by_model.setdefault(c["model"], [0, 0, 0])
            t[0] += 1
            t[1] += int(u.get("prompt_tokens") or 0)
            t[2] += int(u.get("completion_tokens") or 0)
        cached = sum(1 for c in calls if c.get("from_cache"))
        L.append(f"Model calls: {len(calls)} ({cached} served from cache in this run). Tokens when first run: "
                 + "; ".join(f"`{k}` {v[0]} calls, {v[1]:,} in / {v[2]:,} out" for k, v in by_model.items()) + ".")
    if m.get("candidates_accepted") is not None:
        L.append(f"Candidates that passed the rule checks: {m['candidates_accepted']}; "
                 f"thrown out: {m['candidates_rejected']}.")
    if m.get("scout_errors"):
        L.append(f"**Episodes that failed:** {sorted(m['scout_errors'])}")
    L += ["", "## Post these first", "",
          "| # | Promo | Segments | Len | Hook (first 3s) | Viewer score | Verdict |",
          "|---|---|---|---|---|---|---|"]
    for p in promos[:10]:
        j = p.get("judge") or {}
        hook = (p.get("hook_line") or p.get("hook") or "").replace("|", "/")[:70]
        L.append(f"| {p['rank']} | {p['id']} {p.get('title', '')} | {p['segments_str']} | {p['duration']:.0f}s | "
                 f"{hook} | {_sc(p, 'judge', 'overall') or '-'}/10 | {j.get('verdict', '-')} |")
    L += ["", "## Every promo, ranked", ""]
    for p in promos:
        j = p.get("judge") or {}
        L += [f"### {p['rank']}. {p['id']} {p.get('title', '')}  (score {p.get('final_score')})", ""]
        L.append("| Segment | Episode | Start | End | Length |")
        L.append("|---|---|---|---|---|")
        for k, s in enumerate(p["segments"], 1):
            L.append(f"| {k} | {s['episode']} | {fmt(s['start'])} | {fmt(s['end'])} | {s['duration']:.1f}s |")
        L += ["", f"- **Total:** {p['duration']:.1f}s",
              f"- **Hook:** {p.get('hook', '')}" + (f"  (script: \"{p['hook_line']}\")" if p.get("hook_line") else ""),
              f"- **Cliffhanger:** {p.get('cliffhanger', '')}"
              + (f"  (script: \"{p['cliffhanger_line']}\")" if p.get("cliffhanger_line") else ""),
              f"- **Why it works (blind viewer):** {p.get('why') or '-'}"]
        if p.get("scout_note"):
            L.append(f"- **Scout's note (written before any re-cut):** {p['scout_note']}")
        if p.get("why_post"):
            L.append(f"- **Why this rank:** {p['why_post']}")
        if j.get("first_3_seconds"):
            L.append(f"- **Blind viewer saw in 0-3s:** {j['first_3_seconds']}")
        if j.get("confusing"):
            L.append(f"- **Blind viewer was confused by:** {'; '.join(map(str, j['confusing']))}")
        if p.get("context_needed") and p["context_needed"].lower() not in ("none", ""):
            L.append(f"- **Scout's context warning:** {p['context_needed']}")
        if p.get("fixes"):
            L.append(f"- **Auto-fixes:** {'; '.join(p['fixes'])}")
        L += ["", "<details><summary>Clip transcript (as the viewer sees it)</summary>", "", "```",
              p["transcript"], "```", "</details>", ""]
    dups = [p for p in r["promos"] if p.get("duplicate_of")]
    if dups:
        L += ["## Near-duplicates folded away", ""]
        L += [f"- {p['id']} ({p['segments_str']}): {p.get('duplicate_reason', 'overlaps')} as {p['duplicate_of']}"
              for p in dups]
        L.append("")
    if r.get("rejected"):
        L += ["## Thrown out by the rule checks", ""]
        for c in r["rejected"]:
            L.append(f"- {c.get('title', '?')} ({c.get('source')}): {'; '.join(c.get('_problems') or [])}")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
