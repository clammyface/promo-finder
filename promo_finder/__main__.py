"""CLI.  python -m promo_finder <folder|zip> [options]

Examples
  python -m promo_finder data/show.zip                     # Groq free tier: Qwen scouts, gpt-oss judges
  python -m promo_finder data/show.zip --provider gemini --model gemini-3.5-flash   # swap the scout
  python -m promo_finder data/show.zip --provider custom --base-url https://x/v1 --model m --judge-provider same
  python -m promo_finder data/show.zip --mode heuristic                   # no API key at all
  python -m promo_finder data/show.zip --episodes 1-5                     # quick test
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from .llm import PROVIDERS, load_dotenv
from .pipeline import Config, run


DEFAULT_FALLBACKS = ["groq:openai/gpt-oss-20b@low", "gemini:gemini-3.5-flash-lite"]


def parse_eps(s: str | None):
    if not s:
        return None
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    load_dotenv()
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find promo clips in microdrama screenplays.")
    ap.add_argument("input", help="folder of episode .txt files, a .zip of them, or one .txt")
    ap.add_argument("--out", help="output folder (default: outputs/<show>/<provider-model>)")
    ap.add_argument("--mode", choices=["llm", "heuristic"], default="llm")
    ap.add_argument("--provider", default="groq", choices=list(PROVIDERS))
    ap.add_argument("--model", help="model name for that provider (default: provider's default)")
    ap.add_argument("--base-url", help="override endpoint (any OpenAI-compatible /v1 URL)")
    ap.add_argument("--fallback", action="append", default=None, metavar="PROVIDER:MODEL[@EFFORT]",
                    help="model to switch to when the scout's daily quota runs out (repeatable, in order). "
                         "Default: groq:openai/gpt-oss-20b@low then gemini:gemini-3.5-flash-lite (skipped if no key)")
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--no-rules", action="store_true", help="don't let the keyword rules propose promos")
    ap.add_argument("--judge-provider", default="groq", choices=list(PROVIDERS) + ["same"],
                    help="'same' = the scout model judges its own picks (cheaper, less honest)")
    ap.add_argument("--judge-model", default="openai/gpt-oss-120b")
    ap.add_argument("--judge-base-url")
    ap.add_argument("--no-judge", action="store_true", help="skip the blind-viewer pass")
    ap.add_argument("--episodes", help="subset, e.g. 1-5 or 1,4,9")
    ap.add_argument("--reasoning", choices=["low", "medium", "high"], help="reasoning effort for the scout model")
    ap.add_argument("--judge-reasoning", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--judge-workers", type=int, default=2)
    ap.add_argument("--rpm", type=float, default=0, help="max requests/minute (free tiers: try 8-12)")
    ap.add_argument("--judge-rpm", type=float, default=0)
    ap.add_argument("--tpm", type=float, default=0, help="tokens/minute budget (0 = provider default)")
    ap.add_argument("--judge-tpm", type=float, default=0)
    ap.add_argument("--judge-per-episode", type=int, default=3, help="judge the best N scout picks per episode")
    ap.add_argument("--montages", type=int, default=4, help="how many cross-episode trailers to cut")
    ap.add_argument("--no-early-bonus", action="store_true", help="don't favour promos from early episodes")
    a = ap.parse_args(argv)

    cfg = Config(provider=a.provider, model=a.model, base_url=a.base_url, judge_provider=a.judge_provider,
                 judge_model=a.judge_model, judge_base_url=a.judge_base_url, mode=a.mode,
                 judge=not a.no_judge, episodes=parse_eps(a.episodes), workers=a.workers, rpm=a.rpm,
                 reasoning=a.reasoning, judge_reasoning=a.judge_reasoning, judge_workers=a.judge_workers,
                 fallbacks=[] if a.no_fallback else (a.fallback or DEFAULT_FALLBACKS),
                 rules_proposer=not a.no_rules, judge_rpm=a.judge_rpm, tpm=a.tpm, judge_tpm=a.judge_tpm, judge_per_episode=a.judge_per_episode,
                 montage_n=a.montages, early_episode_bonus=not a.no_early_bonus)
    if a.out:
        out = Path(a.out)
    else:
        show = re.sub(r"[^A-Za-z0-9]+", "_", Path(a.input).stem.split("-scripts")[0]).strip("_")[:40]
        tag = "heuristic" if a.mode == "heuristic" else f"{a.provider}_{a.model or PROVIDERS[a.provider]['model'] or 'custom'}"
        out = Path("outputs") / show / re.sub(r"[^A-Za-z0-9._-]+", "_", tag)

    t0 = time.time()
    last = {"stage": None}

    def progress(stage, done, total):
        if stage != last["stage"]:
            print(f"\n{stage}", end="", flush=True)
            last["stage"] = stage
        if total > 1:
            print(f"\r{stage}: {done}/{total}", end="", flush=True)

    res = run(a.input, out, cfg, progress)
    print(f"\n\nDone in {time.time() - t0:.0f}s -> {out}")
    top = [p for p in res["promos"] if not p.get("duplicate_of")][:5]
    for p in top:
        print(f"  #{p['rank']} {p['id']:7s} {p['duration']:5.1f}s  {p['segments_str']}")
    if res["meta"].get("usage"):
        u = res["meta"]["usage"]["main"]
        print(f"  calls: {u['calls']} live, {u['cached']} cached; tokens in/out: "
              f"{u['prompt_tokens']:,}/{u['completion_tokens']:,}")
    if res["meta"].get("scout_errors"):
        print(f"  WARNING: episodes failed: {sorted(res['meta']['scout_errors'])} (re-run; finished calls are cached)")


if __name__ == "__main__":
    main()
