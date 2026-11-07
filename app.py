"""Promo Finder: review app for editors.

Run:  streamlit run app.py
Load a show (zip of episode .txt files), or open a finished run, then go
through the ranked promos and mark each one approve / reject.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from promo_finder.llm import PROVIDERS, load_dotenv
from promo_finder.parse import fmt, load_show_from_bytes
from promo_finder.pipeline import Config, run_episodes

load_dotenv()
st.set_page_config(page_title="Promo Finder", layout="wide")

# Streamlit Cloud: keys can live in st.secrets
try:
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = st.secrets[k]
except Exception:
    pass

st.title("Promo Finder")
st.caption("Reads microdrama screenplays and proposes 20-60s promo clips: hook in the first 3 seconds, "
           "cut on a cliffhanger, understandable to a stranger. An editor approves the final picks.")

OUT_ROOT = Path("outputs")

with st.sidebar:
    src = st.radio("Start from", ["Open a finished run", "Run on a new show"], index=0)
    result = None
    if src == "Open a finished run":
        runs = sorted(OUT_ROOT.glob("**/promos.json"), key=lambda p: p.stat().st_mtime, reverse=True) \
            if OUT_ROOT.exists() else []
        up = st.file_uploader("...or upload a promos.json", type=["json"])
        if up:
            result = json.loads(up.read().decode("utf-8"))
            st.session_state["run_dir"] = None
        elif runs:
            pick = st.selectbox("Run", runs, format_func=lambda p: str(p.parent.relative_to(OUT_ROOT)))
            result = json.loads(Path(pick).read_text(encoding="utf-8"))
            st.session_state["run_dir"] = str(Path(pick).parent)
        else:
            st.info("No runs found in outputs/. Switch to 'Run on a new show'.")
    else:
        upload = st.file_uploader("Episodes (.zip of .txt scripts, or one .txt)", type=["zip", "txt"])
        mode = st.selectbox("Mode", ["llm", "heuristic"], help="heuristic = keyword rules only, no API key")
        provider = st.selectbox("Provider", list(PROVIDERS), index=list(PROVIDERS).index("groq"))
        model = st.text_input("Model", value=PROVIDERS[provider]["model"] or "")
        base_url = st.text_input("Base URL (optional)", value="" if provider != "custom" else os.getenv("LLM_BASE_URL", ""))
        key_env = PROVIDERS[provider]["key_env"]
        if key_env and mode == "llm":
            key = st.text_input(f"{key_env}", type="password", value="",
                                help="Leave blank to use the server's .env / secrets. Used for this session only.")
            if key:
                os.environ[key_env] = key
        j_options = ["(same)"] + list(PROVIDERS)
        j_provider = st.selectbox("Judge provider", j_options, index=j_options.index("groq"),
                                  help="Using a different model to judge avoids a model grading its own picks")
        j_default = "openai/gpt-oss-120b" if j_provider == "groq" else (PROVIDERS.get(j_provider, {}).get("model") or "")
        j_model = st.text_input("Judge model", value="" if j_provider == "(same)" else j_default)
        if j_provider not in ("(same)",) and PROVIDERS[j_provider]["key_env"] and mode == "llm":
            jk = st.text_input(PROVIDERS[j_provider]["key_env"], type="password", value="")
            if jk:
                os.environ[PROVIDERS[j_provider]["key_env"]] = jk
        ep_range = st.text_input("Episodes (blank = all, e.g. 1-5)", value="")
        rpm = st.number_input("Max requests/minute (0 = no limit)", min_value=0, value=0)
        go = st.button("Find promos", type="primary", disabled=upload is None)
        if go and upload is not None:
            title, episodes = load_show_from_bytes(upload.getvalue(), upload.name)
            st.write(f"Parsed **{len(episodes)}** episodes of *{title}*")
            eps_sel = None
            if ep_range.strip():
                from promo_finder.__main__ import parse_eps
                eps_sel = parse_eps(ep_range)
            cfg = Config(provider=provider, model=model or None, base_url=base_url or None, mode=mode,
                         judge_provider="same" if j_provider == "(same)" else j_provider,
                         judge_model=j_model or None, episodes=eps_sel, rpm=rpm)
            out = Path(tempfile.mkdtemp(prefix="promo_run_"))
            bar = st.progress(0.0, text="Starting")

            def progress(stage, done, total):
                bar.progress(min(done / max(total, 1), 1.0), text=f"{stage} {done}/{total}" if total > 1 else stage)

            try:
                result = run_episodes(title, episodes, out, cfg, progress)
                st.session_state["result"] = result
                st.session_state["run_dir"] = str(out)
            except Exception as e:  # show the real error to the editor
                st.error(f"Run failed: {e}")
        result = result or st.session_state.get("result")

if not result:
    st.stop()

meta = result["meta"]
promos = [p for p in result["promos"] if not p.get("duplicate_of")]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Episodes", meta["episodes"])
c2.metric("Show length", f"{meta['minutes']} min")
c3.metric("Promos found", len(promos))
chain = meta.get("model", "llm").split(" -> ")
c4.metric("Scout", meta["mode"] if meta["mode"] == "heuristic" else chain[0].split(":")[-1]
          + (" + fallback" if len(chain) > 1 else ""))
if len(chain) > 1:
    used = sorted({p.get("scout_model") for p in result["promos"] if p.get("scout_model")})
    st.caption("Scout models actually used: " + ", ".join(u.split(":")[-1] for u in used))
if meta.get("judge_model") and meta.get("judge_model") != meta.get("model"):
    st.caption(f"Blind judge: {meta['judge_model']}")
if meta.get("scout_errors"):
    st.warning(f"These episodes failed and have no picks: {sorted(meta['scout_errors'])}")

if result.get("bible", {}).get("premise"):
    with st.expander("What the model understood the show to be"):
        st.write(result["bible"]["premise"])
        for c in result["bible"].get("characters", []):
            st.write(f"- **{c.get('name')}** ({', '.join(c.get('labels', []))}): {c.get('role', '')}. {c.get('note', '')}")

# --------------------------------------------------------------- the table
rows = []
for p in promos:
    j = p.get("judge") or {}
    sc = j.get("scores") or {}
    rows.append({"rank": p.get("rank"), "id": p["id"], "title": p.get("title", ""), "segments": p["segments_str"],
                 "sec": round(p["duration"]), "viewer": sc.get("overall"), "hook": sc.get("hook"),
                 "cliff": sc.get("cliffhanger"), "clarity": sc.get("standalone"), "verdict": j.get("verdict", ""),
                 "score": p.get("final_score")})
df = pd.DataFrame(rows)
f1, f2 = st.columns([1, 3])
only_post = f1.checkbox("Only 'post' verdicts", value=False)
kinds = f2.multiselect("Type", ["single episode", "cross-episode"], default=["single episode", "cross-episode"])
show = df
if only_post:
    show = show[show["verdict"] == "post"]
is_x = show["segments"].str.contains(r"\+")
show = show[(is_x & ("cross-episode" in kinds)) | (~is_x & ("single episode" in kinds))]
st.dataframe(show, hide_index=True, width="stretch")

# ------------------------------------------------------------- review list
st.subheader("Review")
review = st.session_state.setdefault("review", {})
for p in promos:
    if p["id"] not in set(show["id"]):
        continue
    j = p.get("judge") or {}
    sc = j.get("scores") or {}
    head = (f"#{p.get('rank')}  {p['id']}  {p.get('title', '')}  ·  {p['segments_str']}  ·  {p['duration']:.0f}s"
            + (f"  ·  viewer {sc.get('overall'):.0f}/10" if sc.get("overall") is not None else ""))
    with st.expander(head, expanded=p.get("rank", 99) <= 3):
        a, b = st.columns([3, 2])
        with a:
            st.markdown(f"**Hook (0-3s):** {p.get('hook', '')}")
            if p.get("hook_line"):
                st.caption(f"Script line at the hook: \"{p['hook_line']}\"")
            st.markdown(f"**Cliffhanger:** {p.get('cliffhanger', '')}")
            if p.get("cliffhanger_line"):
                st.caption(f"Script line at the cut: \"{p['cliffhanger_line']}\"")
            st.markdown(f"**Why it works:** {p.get('why', '')}")
            if p.get("scout_note"):
                st.caption(f"Scout's note (written before any re-cut): {p['scout_note']}")
            if p.get("why_post"):
                st.markdown(f"**Why this rank:** {p['why_post']}")
            if p.get("fixes"):
                st.caption("Auto-fixes: " + "; ".join(p["fixes"]))
        with b:
            st.table(pd.DataFrame([{"episode": s["episode"], "start": fmt(s["start"]), "end": fmt(s["end"]),
                                    "sec": round(s["duration"], 1)} for s in p["segments"]]))
            if j.get("first_3_seconds"):
                st.caption(f"Blind viewer, first 3s: {j['first_3_seconds']}")
            if j.get("confusing"):
                st.caption("Confused by: " + "; ".join(map(str, j["confusing"])))
        st.code(p.get("transcript", ""), language=None)
        r1, r2 = st.columns([1, 3])
        verdict = r1.radio("Editor", ["-", "approve", "reject", "maybe"], horizontal=True, key=f"v_{p['id']}",
                           index=["-", "approve", "reject", "maybe"].index(review.get(p["id"], {}).get("verdict", "-")))
        note = r2.text_input("Note (what's wrong / what to change)", key=f"n_{p['id']}",
                             value=review.get(p["id"], {}).get("note", ""))
        review[p["id"]] = {"verdict": verdict, "note": note}

done = {k: v for k, v in review.items() if v["verdict"] != "-"}
if done:
    rev = pd.DataFrame([{"id": k, **v, "segments": next(p["segments_str"] for p in promos if p["id"] == k)}
                        for k, v in done.items()])
    st.download_button("Download editor review (CSV)", rev.to_csv(index=False).encode("utf-8"), "editor_review.csv")
    if st.session_state.get("run_dir"):
        if st.button("Save review next to the run"):
            Path(st.session_state["run_dir"], "editor_review.csv").write_text(rev.to_csv(index=False), encoding="utf-8")
            st.success("Saved editor_review.csv")
st.download_button("Download all promos (JSON)", json.dumps(result, ensure_ascii=False, default=str).encode("utf-8"),
                   "promos.json")
