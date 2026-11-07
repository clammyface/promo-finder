"""One small client for any OpenAI-compatible chat endpoint.

Gemini, Groq, OpenRouter, OpenAI, Anthropic, Ollama and most in-house gateways
all speak /chat/completions, so swapping the model is a flag change:

    --provider gemini --model gemini-2.5-flash
    --provider groq   --model llama-3.3-70b-versatile
    --provider custom (reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL)

Every call is cached on disk (same prompt + model = free re-run) and logged
in full to the run's raw/ folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

# Defaults checked against each provider's /models list on 2026-11-07. Model
# line-ups change often (gemini-2.5-flash and Groq's llama-3.3-70b were already
# gone for new keys), so pass --model rather than trusting these forever.
PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINI_API_KEY", "model": "gemini-3.5-flash", "json_mode": True,
               "rpm": 10},     # free tier: ~10 requests/min and only ~20 requests/day per model
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_env": "GROQ_API_KEY", "model": "qwen/qwen3.8-27b", "json_mode": True,
             "tpm": 7600},   # free tier is 8000 tokens/min per model; stay just under it
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_env": "OPENROUTER_API_KEY", "model": "meta-llama/llama-3.3-70b-instruct:free",
                   "json_mode": False},
    "openai": {"base_url": "https://api.openai.com/v1",
               "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini", "json_mode": True},
    "anthropic": {"base_url": "https://api.anthropic.com/v1",
                  "key_env": "ANTHROPIC_API_KEY", "model": "claude-haiku-5-5", "json_mode": False},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "key_env": None, "model": "llama3.1:8b", "json_mode": True},
    "custom": {"base_url": None, "key_env": "LLM_API_KEY", "model": None, "json_mode": False},
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env reader so we don't need python-dotenv."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# One lock for log files across ALL clients. Per-client locks let the scout and the judge
# interleave writes into the same llm_calls.jsonl (it happened: one corrupted line in 174).
_FILE_LOCK = threading.Lock()


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    """Daily quota gone. Retrying is pointless; re-run later and the cache keeps finished calls."""


@dataclass
class LLM:
    provider: str = "gemini"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.3
    max_tokens: int = 16000      # thinking models spend part of this on reasoning
    reasoning_effort: str | None = None   # "low" | "medium" | "high"; None = provider default
    rpm: float = 0               # client-side request throttle, 0 = off
    tpm: float = 0               # client-side tokens/minute budget (Groq free tier: 8000), 0 = off
    cache_dir: Path | None = Path(".cache/llm")
    log_path: Path | None = None
    timeout: int = 180
    usage: dict = field(default_factory=lambda: {"calls": 0, "cached": 0, "prompt_tokens": 0,
                                                  "completion_tokens": 0, "seconds": 0.0})
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last_call: float = 0.0
    _bucket: float | None = field(default=None, repr=False)   # tokens available right now
    _bucket_t: float = field(default=0.0, repr=False)

    def __post_init__(self):
        if self.provider not in PROVIDERS:
            raise LLMError(f"Unknown provider '{self.provider}'. Pick one of {list(PROVIDERS)}")
        spec = PROVIDERS[self.provider]
        self.base_url = (self.base_url or os.getenv("LLM_BASE_URL") if self.provider == "custom"
                         else self.base_url or spec["base_url"])
        self.model = self.model or (os.getenv("LLM_MODEL") if self.provider == "custom" else spec["model"])
        if spec["key_env"]:
            self.api_key = self.api_key or os.getenv(spec["key_env"])
        self.json_mode = spec["json_mode"]
        self.tpm = self.tpm or spec.get("tpm", 0)
        self.rpm = self.rpm or spec.get("rpm", 0)
        if not self.base_url or not self.model:
            raise LLMError("custom provider needs LLM_BASE_URL and LLM_MODEL (env or flags)")
        if spec["key_env"] and not self.api_key:
            raise LLMError(f"No API key: set {spec['key_env']} in .env or the environment")
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def last_name(self) -> str:      # same interface as Cascade
        return self.name

    def _body_and_cache(self, messages, temperature, json_out):
        temp = self.temperature if temperature is None else temperature
        body = {"model": self.model, "messages": messages, "temperature": temp,
                "max_tokens": self.max_tokens}
        if json_out and self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        key = hashlib.sha256(json.dumps([self.base_url, body], sort_keys=True).encode()).hexdigest()[:24]
        return temp, body, (self.cache_dir / f"{key}.json" if self.cache_dir else None)

    def has_cached(self, messages: list[dict], temperature: float | None = None, json_out: bool = True) -> bool:
        return bool((f := self._body_and_cache(messages, temperature, json_out)[2]) and f.exists())

    # ------------------------------------------------------------------ core
    def chat(self, messages: list[dict], stage: str = "", temperature: float | None = None,
             json_out: bool = True) -> str:
        temp, body, cache_file = self._body_and_cache(messages, temperature, json_out)

        if cache_file and cache_file.exists():
            rec = json.loads(cache_file.read_text(encoding="utf-8"))
            with self._lock:
                self.usage["cached"] += 1
            self._log({**rec, "stage": stage, "from_cache": True})
            return rec["response"]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"

        url = self.base_url.rstrip("/") + "/chat/completions"
        # rough tokens in + typical out (measured on this data: ~4.2 chars/token, ~500 out)
        est = len(json.dumps(messages, ensure_ascii=False)) / 4.2 + 500
        delay, last_err = 5.0, None
        for attempt in range(12):                            # free tiers rate-limit a lot; be patient
            self._throttle(est)
            t0 = time.time()
            try:
                r = requests.post(url, headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = str(e)
                self._event(stage, f"network error, retry in {delay:.0f}s: {e}")
                time.sleep(delay); delay = min(delay * 2, 90)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                wait = _retry_after(r, cap=None) or delay
                if r.status_code == 429 and (wait > 600 or re.search(r"per ?day|PerDay|TPD|RPD", r.text)):
                    self._event(stage, f"DAILY QUOTA USED UP: {r.text[:160]}")
                    raise QuotaExhausted(f"{self.name}: daily free-tier quota used up "
                                         f"(provider says retry in ~{wait / 3600:.1f}h). {r.text[:200]}")
                wait = min(wait, 120)
                if self.tpm:   # a rejected request didn't spend tokens; give the estimate back
                    with self._lock:
                        self._bucket += est
                self._event(stage, f"HTTP {r.status_code}, retry {attempt + 1} in {wait:.0f}s: {r.text[:140]}")
                with self._lock:
                    self.usage["retries"] = self.usage.get("retries", 0) + 1
                time.sleep(wait + 0.5); delay = min(delay * 2, 90)
                continue
            if r.status_code == 400 and "response_format" in body and "generate JSON" in r.text:
                # Groq's strict JSON mode sometimes refuses; our own extractor is more forgiving.
                self._event(stage, "provider JSON mode refused; retrying once without it")
                body = {k: v for k, v in body.items() if k != "response_format"}
                continue
            if r.status_code >= 400:
                self._event(stage, f"HTTP {r.status_code} (not retried): {r.text[:200]}")
                raise LLMError(f"HTTP {r.status_code} from {self.name}: {r.text[:500]}")
            data = r.json()
            try:
                choice = data["choices"][0]
                text = (choice.get("message") or {}).get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                raise LLMError(f"Unexpected response shape: {str(data)[:500]}")
            if not text.strip():
                # Seen with Gemini 3.x when reasoning eats the whole token budget,
                # and on transient overload. Treat as retryable, don't cache it.
                last_err = f"empty reply (finish_reason={choice.get('finish_reason')})"
                msg = choice.get("message") or {}
                self._event(stage, f"{last_err}; other fields: {sorted(msg)}; usage={data.get('usage')}")
                if attempt >= 2:
                    raise LLMError(f"{self.name}: {last_err} three times")
                time.sleep(delay); delay = min(delay * 2, 90)
                continue
            u = data.get("usage") or {}
            if self.tpm:       # settle the estimate against the real count
                with self._lock:
                    self._bucket -= float(u.get("total_tokens") or est) - est
            rec = {"model": self.name, "stage": stage, "temperature": temp, "messages": messages,
                   "response": text, "usage": u, "latency_s": round(time.time() - t0, 2),
                   "finish_reason": data["choices"][0].get("finish_reason")}
            with self._lock:
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
                self.usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
                self.usage["seconds"] += rec["latency_s"]
            if cache_file:
                cache_file.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
            self._log({**rec, "from_cache": False})
            return text
        raise LLMError(f"Gave up after retries on {self.name}: {last_err}")

    def chat_json(self, messages: list[dict], stage: str = "", temperature: float | None = None,
                  retries: int = 1) -> dict:
        """Ask for JSON; if it doesn't parse, show the model its own error once."""
        text = self.chat(messages, stage=stage, temperature=temperature)
        for i in range(retries + 1):
            try:
                return extract_json(text)
            except ValueError as e:
                if i == retries:
                    raise LLMError(f"[{stage}] model did not return valid JSON: {e}; got: {text[:300]}")
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": f"That was not valid JSON ({e}). "
                                        "Reply again with only the JSON object, no prose, no code fences."}]
                text = self.chat(messages, stage=stage + ":json-fix", temperature=0)
        raise AssertionError("unreachable")

    # --------------------------------------------------------------- helpers
    def _throttle(self, est_tokens: float = 0):
        """Wait until this request fits the request and token budgets.

        Tokens use a bucket that refills continuously (tpm/60 per second), which is how
        Groq meters it (its reset-tokens header counts down in milliseconds). A fixed
        60-second window was my first version: it blocked a full minute after every big
        call and ran at about half speed."""
        if not self.rpm and not self.tpm:
            return
        while True:
            with self._lock:
                now = time.time()
                wait = 0.0
                if self.rpm:
                    wait = max(wait, self._last_call + 60.0 / self.rpm - now)
                if self.tpm:
                    if self._bucket is None:
                        self._bucket, self._bucket_t = float(self.tpm), now
                    rate = self.tpm / 60.0
                    self._bucket = min(float(self.tpm), self._bucket + (now - self._bucket_t) * rate)
                    self._bucket_t = now
                    need = min(est_tokens, float(self.tpm))   # oversize requests wait for a full bucket
                    if self._bucket < need:
                        wait = max(wait, (need - self._bucket) / rate)
                if wait <= 0:
                    self._last_call = now
                    if self.tpm:
                        self._bucket -= est_tokens
                    return
            time.sleep(min(wait, 5) + 0.05)

    def _event(self, stage: str, msg: str):
        """Retries, waits and errors: one line to stderr and to raw/llm_events.log."""
        line = f"{time.strftime('%H:%M:%S')} [{self.name} {stage}] {msg}"
        print("\n  " + line, file=sys.stderr, flush=True)
        if self.log_path:
            with _FILE_LOCK:
                with open(self.log_path.parent / "llm_events.log", "a", encoding="utf-8") as f:
                    f.write(line + "\n")

    def _log(self, rec: dict):
        if not self.log_path:
            return
        with _FILE_LOCK:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Cascade:
    """A primary model plus fallbacks. When a model's daily quota runs out, the rest of
    the run moves to the next one (and stays there, so a show isn't flip-flopping
    between models). A request any model in the chain already answered comes from that
    model's cache, so a re-run reproduces the original run exactly.

    Spec strings: "groq:openai/gpt-oss-20b@low" = provider:model[@reasoning_effort]."""

    def __init__(self, models: list[LLM]):
        self.models = models
        self.idx = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @staticmethod
    def parse(spec: str, **kw) -> LLM:
        prov, _, rest = spec.strip().partition(":")
        model, _, effort = rest.partition("@")
        return LLM(provider=prov, model=model or None, reasoning_effort=effort or None, **kw)

    @property
    def name(self) -> str:
        return " -> ".join(m.name for m in self.models)

    @property
    def last_name(self) -> str:
        return getattr(self._local, "last", self.models[self.idx].name)

    @property
    def usage(self) -> dict:
        agg = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "retries": 0}
        for m in self.models:
            for k in agg:
                agg[k] += m.usage.get(k, 0)
        agg["by_model"] = {m.name: dict(m.usage) for m in self.models}
        return agg

    def chat_json(self, messages, stage="", temperature=None, retries=1):
        # Any model in the chain already answered this exact request? Reuse it before going live.
        # (Checking only models we've moved past isn't enough: on a re-run the primary's daily quota
        # may have come back, and it would re-scout an episode a fallback already did.)
        for m in self.models:
            if m.has_cached(messages, temperature):
                self._local.last = m.name
                return m.chat_json(messages, stage=stage, temperature=temperature, retries=retries)
        while True:
            with self._lock:
                i = self.idx
            m = self.models[i]
            try:
                out = m.chat_json(messages, stage=stage, temperature=temperature, retries=retries)
                self._local.last = m.name
                return out
            except QuotaExhausted:
                with self._lock:
                    if self.idx == i and i + 1 < len(self.models):
                        self.idx = i + 1
                        m._event(stage, f"switching to fallback {self.models[i + 1].name} for the rest of the run")
                    elif self.idx == i:
                        raise


def _retry_after(r: requests.Response, cap: float | None = 120) -> float | None:
    """How long the provider asked us to wait. Handles the header and the wording
    Groq ('try again in 1m2.5s' / '577ms') and Gemini ('retry in 20h22m8.9s',
    'retryDelay': '14s') use."""
    secs = None
    h = r.headers.get("retry-after")
    if h:
        try:
            secs = float(h)
        except ValueError:
            pass
    if secs is None:
        m = re.search(r"(?:try again|retry) in (?:(\d+)h)?(?:(\d+)m(?!s))?([\d.]+)(ms|s)", r.text)
        if m:
            secs = (float(m.group(1) or 0) * 3600 + float(m.group(2) or 0) * 60
                    + float(m.group(3)) / (1000 if m.group(4) == "ms" else 1)) + 1
    if secs is None:
        m = re.search(r'"retryDelay":\s*"(\d+)s"', r.text)
        secs = float(m.group(1)) + 1 if m else None
    if secs is None:
        return None
    return min(secs, cap) if cap else secs


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a reply, tolerating ``` fences and chatter."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.S)
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            return {"items": obj}
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start == -1:
        raise ValueError("no '{' found")
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[start:i + 1])
                except json.JSONDecodeError as e:
                    raise ValueError(str(e))
    raise ValueError("unbalanced braces (reply probably cut off)")
