"""Build the 2-page solution PDF from a finished run.

Every number in the doc is pulled from the run's output files, so the doc and the
raw outputs can't disagree. The page is HTML + CSS (solution_template.html,
solution.css), printed to A4 by headless Chrome or Edge.

    python docs/build_doc.py outputs/<show>/<run> [--preview DIR]

Name and links go in docs/doc_meta.json:
    {"name": "...", "repo": "https://github.com/...", "outputs": "https://drive...", "video": "optional"}
"""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from string import Template

import pymupdf

HERE = Path(__file__).parent

# Paid list prices (USD per 1M tokens, in/out), from third-party trackers, Nov 2026. Approximate.
PRICES = {"groq:qwen/qwen3.8-27b": (0.80, 4.00), "groq:openai/gpt-oss-120b": (0.15, 0.60),
          "groq:openai/gpt-oss-20b": (0.075, 0.30), "gemini:gemini-3.5-flash-lite": (0.10, 0.40)}
FREE_TPM = 8000          # Groq free tier, tokens per minute per model

# My own read of the top 10 against the script (rank -> verdict). Anything not listed: post as is.
MY_READ = {5: "recut", 9: "recut", 10: "repeat"}
EXAMPLE_ID = "E23-A"     # the promo drawn in "anatomy of one promo" (falls back to the best multi-cut pick)
ALTAR_ID = "E02-A"       # the altar scene in the judge-persona dev trials


def fmt_t(sec: float) -> str:
    m, s = divmod(sec, 60)
    return f"{int(m):02d}:{s:04.1f}"


def short(s: str, n: int) -> str:
    """Trim to n characters at a word boundary."""
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[: n - 1]
    cut = cut[: cut.rfind(" ")] if " " in cut[n // 2:] else cut
    return cut.rstrip(" ,;:.—-") + "…"


def esc(s) -> str:
    return html.escape(str(s))


def beat_said(p: dict, which: str) -> str:
    """Hook/cliffhanger as a reader would want it: the quoted line if it's dialogue,
    otherwise the plain description of what's on screen."""
    seg = p["segments"][0] if which == "hook" else p["segments"][-1]
    bid = p.get("hook_beat" if which == "hook" else "cliffhanger_beat")
    tag = f"<ep{seg['episode']} B{bid} "
    for line in (p.get("transcript") or "").splitlines():
        if tag in line:
            body = line.split("] ", 1)[1].split("  <ep", 1)[0]
            who, _, text = body.partition(": ")
            if who.split(" (")[0] in ("CAMERA", "ACTION", "SFX"):
                break
            return f"{who.split(' (')[0].title()}: “{text}”"
    desc = p.get("hook" if which == "hook" else "cliffhanger") or ""
    return re.sub(r"^\s*(CAMERA|ACTION|SFX)\s*:\s*", "", desc)


def src_of(p: dict) -> tuple[str, str]:
    """(css class, label) for who proposed a promo."""
    if p.get("source") == "rules":
        return "rules", "keyword rules"
    model = (p.get("scout_model") or "").split("/")[-1].split(":")[-1]
    model = {"qwen3.8-27b": "Qwen 3.8"}.get(model, model)
    return "llm", ("LLM · " + model) if model else "LLM"


# ------------------------------------------------------------------ figures
def anatomy(p: dict) -> str:
    """One promo drawn as a timeline (segments, the cut between them, hook and cliffhanger zones) + its fields."""
    segs = p["segments"]
    bar, labs = [], []
    for i, s in enumerate(segs):
        d = max(s["duration"], 0.1)
        zones = ""
        if i == 0:
            zones += f"<i class='zone hook' style='width:{min(3, d) / d * 100:.1f}%'></i>"
        if i == len(segs) - 1:
            zones += f"<i class='zone cliff' style='width:{min(4, d) / d * 100:.1f}%'></i>"
        bar.append(f"<div class='tseg' style='flex:{d:.2f}'>{zones}</div>")
        labs.append(f"<div style='flex:{d:.2f}'><b>Segment {i + 1}</b> · Ep{s['episode']} "
                    f"{fmt_t(s['start'])}–{fmt_t(s['end'])} · {d:.1f} s</div>")
        if i < len(segs) - 1:
            nxt = segs[i + 1]
            gap = (f"skips {nxt['start'] - s['end']:.0f} s" if nxt["episode"] == s["episode"]
                   else f"→ ep {nxt['episode']}")
            bar.append(f"<div class='tgap'>✂ {gap}</div>")
            labs.append("<div class='g'></div>")
    j = p.get("judge") or {}
    sc = j.get("scores") or {}
    n = lambda k: "–" if sc.get(k) is None else f"{sc[k]:.0f}"
    fixes = "; ".join(p.get("fixes") or []) or "passed the checks first time"
    cls, who = src_of(p)
    return f"""
<div class="anat">
  <div class="anat-head"><b>Promo #{p['rank']}: “{esc(p.get('title', ''))}”</b>
    <span class="tag"><i class="dot {cls}"></i>proposed by {esc(who)} · {len(segs)} segment{'s' if len(segs) > 1 else ''} · {p['duration']:.1f} s total</span></div>
  <div class="tl-marks"><span>▼ hook zone (0–3 s)</span><span>cut on the cliffhanger (last 4 s) ▼</span></div>
  <div class="tl-bar">{''.join(bar)}</div>
  <div class="tl-labs">{''.join(labs)}</div>
  <div class="fields">
    <div><span class="lab">Hook (first 3 s)</span>{esc(short(beat_said(p, 'hook'), 80))}</div>
    <div><span class="lab">Cliffhanger (the cut)</span>{esc(short(beat_said(p, 'cliff'), 95))}</div>
    <div><span class="lab">Why it works (blind viewer, its words)</span><span class="why">“{esc(short(p.get('why', ''), 160))}”</span></div>
    <div><span class="lab">Judge scores</span>hook {n('hook')} · cliffhanger {n('cliffhanger')} · clarity {n('standalone')} ·
      <b>overall {n('overall')}/10</b> → “{esc(j.get('verdict', '–'))}”</div>
    <div><span class="lab">Rule checks</span>{esc(short(fixes, 70))}</div>
    <div><span class="lab">Why two segments?</span>The model cut out the slow middle and jumped straight to the threat at the door.</div>
  </div>
</div>"""


def top_rows(promos: list[dict]) -> str:
    rows = []
    for p in promos[:5]:
        sc = (p.get("judge") or {}).get("scores") or {}
        ov = sc.get("overall")
        cls, who = src_of(p)
        segs = "<br>".join(f"Ep{s['episode']} {fmt_t(s['start'])}–{fmt_t(s['end'])}" for s in p["segments"])
        rows.append(
            f"<tr><td class='rk'>{p['rank']}</td>"
            f"<td class='ttl'><b>{esc(short(p.get('title', ''), 28))}</b><span class='tag'><i class='dot {cls}'></i>{esc(who)}</span></td>"
            f"<td class='seg'>{segs}</td><td class='num'>{p['duration']:.0f} s</td>"
            f"<td>{esc(short(beat_said(p, 'hook'), 72))}</td>"
            f"<td>{esc(short(beat_said(p, 'cliff'), 66))}</td>"
            f"<td class='why'>“{esc(short(p.get('why', ''), 160))}”</td>"
            f"<td class='score'><b>{'' if ov is None else int(ov)}</b>/10"
            f"<div class='meter'><i style='width:{(ov or 0) * 10:.0f}%'></i></div></td></tr>")
    return "\n".join(rows)


def wtl_bar(w: int, t: int, l: int) -> str:
    segs = [("llm", w), ("tie", t), ("rules", l)]
    return "<div class='segbar'>" + "".join(
        f"<div class='{c}' style='flex:{v}'>{v}</div>" for c, v in segs if v) + "</div>"


def strip(promos: list[dict]) -> str:
    icon = {"post": "✓", "recut": "✂", "repeat": "≈"}
    cells = []
    for p in promos[:10]:
        v = MY_READ.get(p["rank"], "post")
        src = "R" if p.get("source") == "rules" else "L"
        cells.append(f"<div><div class='cell {v}'><span class='r'>#{p['rank']}</span><span class='ic'>{icon[v]}</span></div>"
                     f"<div class='src'>{src}</div></div>")
    return "<div class='strip'>" + "".join(cells) + "</div>"


def judge_chart(v1: list[float], v2: list[float], a1, a2) -> str:
    """Range of scores the first and rewritten judge gave on the same dev episodes, with the altar scene marked."""
    x = lambda s: 62 + s * 12.5           # 0..10 -> 62..187
    def row(y, label, vals, alt, fill):
        lo, hi = min(vals), max(vals)
        r = (f"<text x='0' y='{y + 3}' font-size='7.3' fill='#46484f'>{label}</text>"
             f"<rect x='{x(lo) - 3}' y='{y - 4}' width='{x(hi) - x(lo) + 6}' height='8' rx='4' fill='{fill}'/>")
        if alt is not None:
            r += (f"<circle cx='{x(alt)}' cy='{y}' r='4.4' fill='#141518' stroke='#f5f5f2' stroke-width='2'/>"
                  f"<text x='{x(alt)}' y='{y - 8}' font-size='6.4' text-anchor='middle' fill='#141518'>altar {alt:.0f}</text>")
        return r
    ticks = "".join(f"<line x1='{x(t)}' x2='{x(t)}' y1='54' y2='57' stroke='#c3c2b7'/>"
                    f"<text x='{x(t)}' y='65' font-size='6.3' text-anchor='middle' fill='#7a7c83'>{t}</text>" for t in (0, 5, 10))
    return (f"<svg viewBox='0 0 196 68' width='196' height='68' role='img' "
            f"aria-label='First judge scored {min(v1):.0f}-{max(v1):.0f}, rewritten judge {min(v2):.0f}-{max(v2):.0f}'>"
            f"<line x1='{x(0)}' x2='{x(10)}' y1='54' y2='54' stroke='#c3c2b7'/>{ticks}"
            + row(22, "First judge", v1, a1, "#c3c2b7") + row(43, "Rewritten", v2, a2, "#1f3a5f") + "</svg>")


# ------------------------------------------------------------------ PDF
def find_browser() -> str:
    cands = [os.getenv("CHROME_PATH")] + [shutil.which(n) for n in
             ("chrome", "google-chrome", "chromium", "chromium-browser", "msedge")]
    cands += [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    for c in cands:
        if c and Path(c).exists():
            return c
    raise SystemExit("Need Chrome or Edge to print the PDF (or set CHROME_PATH).")


def print_pdf(page: Path, pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--print-to-pdf-no-header", f"--user-data-dir={profile}", f"--print-to-pdf={pdf}",
                        page.resolve().as_uri()], check=True, capture_output=True, timeout=180)


# ------------------------------------------------------------------ main
def main(run_dir: str, preview: str | None = None):
    run = Path(run_dir)
    res = json.loads((run / "promos.json").read_text(encoding="utf-8"))
    meta = res["meta"]
    promos = [p for p in res["promos"] if not p.get("duplicate_of")]
    judged = [p for p in promos if (p.get("judge") or {}).get("scores")]
    trailers = [p for p in promos if len({s["episode"] for s in p["segments"]}) > 1]
    llm_kept = [p for p in promos if p.get("source") == "llm-scout"]
    rules_kept = [p for p in promos if p.get("source") == "rules"]
    autofixed = [p for p in res["promos"] if any("moved" in f or "by code" in f for f in p.get("fixes") or [])]
    repaired = [p for p in res["promos"] if any("repaired" in f for f in p.get("fixes") or [])]

    calls = [json.loads(l) for l in (run / "raw" / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    by_model: dict[str, list[int]] = {}
    stages: dict[str, int] = {}
    stage_tokens: dict[str, int] = {}
    scouted_by: dict[int, str] = {}
    for c in calls:
        u = c.get("usage") or {}
        pt, ct = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
        t = by_model.setdefault(c["model"], [0, 0])
        t[0] += pt
        t[1] += ct
        st = c["stage"].split(":")[0]
        stages[st] = stages.get(st, 0) + 1
        stage_tokens[st] = stage_tokens.get(st, 0) + pt + ct
        if c["stage"].startswith("scout:ep"):
            scouted_by[int(c["stage"][8:])] = c["model"]
    cost = sum(PRICES.get(m, (0, 0))[0] * v[0] / 1e6 + PRICES.get(m, (0, 0))[1] * v[1] / 1e6
               for m, v in by_model.items())
    tokens_total = sum(v[0] + v[1] for v in by_model.values())
    scout_side = sum(v for k, v in stage_tokens.items() if k != "judge")

    ev = {}
    if (run / "eval_vs_baseline.json").exists():
        ev = json.loads((run / "eval_vs_baseline.json").read_text(encoding="utf-8"))["summary"]

    # judge-persona dev trials (same episodes, first vs rewritten judge prompt)
    def trial(name):
        f = run.parent.parent / "_dev_trials" / name / "promos.json"
        if not f.exists():
            return None, None
        ps = json.loads(f.read_text(encoding="utf-8"))["promos"]
        sc = {p["id"]: p["judge"]["scores"]["overall"] for p in ps if (p.get("judge") or {}).get("scores")}
        return list(sc.values()), sc.get(ALTAR_ID)
    v1, a1 = trial("ep1-3_gemini-scout_judge-v1-generic-persona")
    v2, a2 = trial("ep1-3_gemini-scout_judge-v2-genre-persona")
    v1, v2 = v1 or [1, 2], v2 or [4, 8]
    a1, a2 = a1 if a1 is not None else 2, a2 if a2 is not None else 8
    rng = lambda v: f"{min(v):.0f}–{max(v):.0f}"

    example = next((p for p in promos if p["id"] == EXAMPLE_ID), None) or \
        next((p for p in promos if len(p["segments"]) > 1), promos[0])

    docmeta = {"name": "[[YOUR NAME]]", "repo": "[[REPO LINK]]", "outputs": "[[RAW OUTPUTS LINK]]", "video": ""}
    if (HERE / "doc_meta.json").exists():
        docmeta.update(json.loads((HERE / "doc_meta.json").read_text(encoding="utf-8")))

    def chip(label, url, main=False):
        if url.startswith("http"):
            return f"<a class='{'main' if main else ''}' href='{esc(url)}'>{label} ↗</a>"
        return f"<span class='todo'>{label}: {esc(url)}</span>"
    chips = [chip("Prototype (GitHub)", docmeta["repo"], True), chip("Raw outputs", docmeta["outputs"])]
    if docmeta.get("video"):
        chips.append(chip("3-min walkthrough", docmeta["video"]))
    name = docmeta["name"]
    name_html = esc(name) if not name.startswith("[[") else f"<span class='todo'>{esc(name)}</span>"

    vals = {
        "name": name_html, "link_chips": "".join(chips), "show": esc(res.get("title", meta.get("show", ""))),
        "episodes": meta["episodes"], "minutes": f"{meta['minutes']:.0f}",
        "n_promos": len(promos), "n_llm": len(llm_kept), "n_rules": len(rules_kept), "n_trailers": len(trailers),
        "dur_min": f"{min(p['duration'] for p in promos):.0f}", "dur_max": f"{max(p['duration'] for p in promos):.0f}",
        "top_score": f"{max(p['judge']['scores']['overall'] for p in judged):.0f}",
        "n_judged": len(judged), "n_post": sum(1 for p in judged if p["judge"].get("verdict") == "post"),
        "n_repaired": len(repaired), "n_autofixed": len(autofixed), "n_rejected": meta.get("candidates_rejected", 0),
        "n_scene_folds": sum(1 for p in res["promos"] if "scene" in (p.get("duplicate_reason") or "")),
        "top10_rules": sum(1 for p in promos[:10] if p.get("source") == "rules"),
        "n_ok": sum(1 for p in promos[:10] if MY_READ.get(p["rank"], "post") == "post"),
        "anatomy": anatomy(example), "top_rows": top_rows(promos), "strip": strip(promos),
        "ev_n": ev.get("n", "–"),
        "ev_llm": f"{(ev.get('llm_mean') or {}).get('overall', 0):.2f}".rstrip("0").rstrip("."),
        "ev_base": f"{(ev.get('baseline_mean') or {}).get('overall', 0):.2f}".rstrip("0").rstrip("."),
        "wtl_bar": wtl_bar(ev.get("llm_wins", 0), ev.get("ties", 0), ev.get("baseline_wins", 0)),
        "n_calls": len(calls), "scout_calls": stages.get("scout", 0), "repair_calls": stages.get("repair", 0),
        "judge_calls": stages.get("judge", 0),
        "other_calls": len(calls) - sum(stages.get(k, 0) for k in ("scout", "repair", "judge")),
        "tokens_k": f"{tokens_total / 1000:.0f}", "cost": f"{cost:.2f}",
        "free_min": f"{scout_side / FREE_TPM:.0f}", "scout_k": f"{scout_side / 1000:.0f}",
        "n_qwen": sum(1 for m in scouted_by.values() if "qwen" in m),
        "n_oss": sum(1 for m in scouted_by.values() if "gpt-oss-20b" in m),
        "v1_n": len(v1), "v1_range": rng(v1), "v2_range": rng(v2), "v1_altar": f"{a1:.0f}", "v2_altar": f"{a2:.0f}",
        "judge_chart": judge_chart(v1, v2, a1, a2),
    }
    page = HERE / "solution.html"
    page.write_text(Template((HERE / "solution_template.html").read_text(encoding="utf-8")).substitute(vals),
                    encoding="utf-8")
    pdf = HERE / "solution.pdf"
    print_pdf(page, pdf)

    doc = pymupdf.open(pdf)
    n = doc.page_count
    print(f"wrote {pdf} ({n} page{'s' if n > 1 else ''})" + ("  <-- OVER THE 2-PAGE LIMIT" if n > 2 else ""))
    if preview:
        Path(preview).mkdir(parents=True, exist_ok=True)
        for i, pg in enumerate(doc):
            pg.get_pixmap(dpi=110).save(str(Path(preview) / f"page{i + 1}.png"))
    print(json.dumps({k: v for k, v in vals.items() if not isinstance(v, str) or "<" not in v}, indent=1))


if __name__ == "__main__":
    args = sys.argv[1:]
    prev = None
    if "--preview" in args:
        k = args.index("--preview")
        prev = args[k + 1]
        del args[k:k + 2]
    main(args[0], prev)
