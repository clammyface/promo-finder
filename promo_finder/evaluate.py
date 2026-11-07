"""Head-to-head: does the LLM pipeline beat the keyword baseline?

For a sample of episodes, take the pipeline's best single-episode promo and the
baseline's best, and let the SAME blind judge score both. The judge never sees
which is which (it only ever sees a transcript).

    python -m promo_finder.evaluate <run_dir> --input <show.zip> --sample 20
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from .llm import LLM, load_dotenv
from .parse import load_show
from .pipeline import judge_promo


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--input", required=True)
    ap.add_argument("--sample", type=int, default=20, help="number of episodes, spread evenly")
    a = ap.parse_args(argv)

    run = Path(a.run_dir)
    res = json.loads((run / "promos.json").read_text(encoding="utf-8"))
    meta = res["meta"]
    _, episodes = load_show(a.input)
    eps = {e.number: e for e in episodes}
    jprov, jmodel = meta["judge_model"].split(":", 1)
    judge = LLM(provider=jprov, model=jmodel, temperature=0,
                reasoning_effort=meta["config"].get("judge_reasoning"),
                cache_dir=Path(meta["config"]["cache_dir"]), log_path=run / "raw" / "eval_calls.jsonl")

    llm_best, base_best = {}, {}
    for p in res["promos"]:
        if (p.get("source") == "llm-scout" and (p.get("judge") or {}).get("scores")
                and len({s["episode"] for s in p["segments"]}) == 1):
            e = p["segments"][0]["episode"]
            if e not in llm_best or p["final_score"] > llm_best[e]["final_score"]:
                llm_best[e] = p
    for p in res["baseline"]:
        e = p["segments"][0]["episode"]
        base_best.setdefault(e, p)          # baseline list is already best-first per episode
    common = sorted(set(llm_best) & set(base_best))
    step = max(1, len(common) / a.sample)
    sample = sorted({common[int(i * step)] for i in range(min(a.sample, len(common)))})

    rows = []
    for e in sample:
        b = base_best[e]
        if not b.get("judge"):
            b["judge"] = judge_promo(judge, b, eps)
        l = llm_best[e]
        bs, ls = b["judge"].get("scores") or {}, l["judge"]["scores"]
        rows.append({"episode": e, "llm_id": l["id"], "llm": ls, "llm_verdict": l["judge"].get("verdict"),
                     "base_id": b["id"], "base": bs, "base_verdict": b["judge"].get("verdict"),
                     "base_reason": b["judge"].get("reason", ""), "llm_reason": l["judge"].get("reason", "")})
        print(f"ep{e:02d}  LLM {ls.get('overall')}  vs  baseline {bs.get('overall')}")

    def mean(side, k):
        v = [r[side].get(k) for r in rows if r[side].get(k) is not None]
        return round(statistics.mean(v), 2) if v else None

    summary = {"episodes": sample, "n": len(rows), "judge": meta["judge_model"],
               "llm_mean": {k: mean("llm", k) for k in ("hook", "cliffhanger", "standalone", "overall")},
               "baseline_mean": {k: mean("base", k) for k in ("hook", "cliffhanger", "standalone", "overall")},
               "llm_wins": sum(1 for r in rows if (r["llm"].get("overall") or 0) > (r["base"].get("overall") or 0)),
               "ties": sum(1 for r in rows if (r["llm"].get("overall") or 0) == (r["base"].get("overall") or 0)),
               "baseline_wins": sum(1 for r in rows if (r["llm"].get("overall") or 0) < (r["base"].get("overall") or 0)),
               "llm_post_rate": round(sum(r["llm_verdict"] == "post" for r in rows) / max(len(rows), 1), 2),
               "baseline_post_rate": round(sum(r["base_verdict"] == "post" for r in rows) / max(len(rows), 1), 2),
               "judge_usage": judge.usage}
    out = {"summary": summary, "rows": rows}
    (run / "eval_vs_baseline.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
