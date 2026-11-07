"""Turn Kuku-style microdrama screenplays into structured, timestamped beats.

This part is plain code on purpose. Timestamps are facts, so nothing here is
left to a model. Later stages refer to beats by id and code maps ids back to
real timestamps.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path

TS = r"(\d{1,2}:)?\d{1,2}:\d{1,2}(?:\.\d+)?"
LABEL_RE = re.compile(r"^\*\*(?P<label>[^*]+?):?\*\*\s*$")
TIMING_RE = re.compile(rf"Start:\s*\(?(?P<s>{TS})\)?\s*\|\s*End:\s*\(?(?P<e>{TS})\)?", re.I)
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
SCENE_FIELD_RE = re.compile(r"^SCENE\s+(HEADING|START|END)\s*:\s*(.+)$", re.I)
HEADER_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES)\s*:\s*(.*)$", re.I)
EP_NUM_RE = re.compile(r"(\d+)")

NON_DIALOGUE = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX",
                "MUSIC": "SFX", "SOUND": "SFX", "VISUAL": "ACTION", "TEXT": "ACTION"}
VOICE_MODIFIERS = {"O.S.", "V.O.", "OFF-SCREEN", "CONT'D", "O.C."}


def to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    return round(secs, 2)


def fmt(t: float) -> str:
    """Seconds -> mm:ss.ss, the same style the scripts use."""
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Beat:
    id: int
    kind: str              # DIALOGUE | ACTION | CAMERA | SFX
    start: float
    end: float
    text: str
    speaker: str | None = None
    tags: list[str] = field(default_factory=list)   # emotion / delivery notes
    voice: str | None = None                          # O.S. / V.O. if any
    scene: int = 1

    @property
    def dur(self) -> float:
        return round(self.end - self.start, 2)

    def line(self, rel_to: float | None = None) -> str:
        """Compact one-line rendering used in prompts and reports."""
        s, e = self.start, self.end
        if rel_to is not None:
            s, e = s - rel_to, e - rel_to
        ts = f"[{fmt(s)}-{fmt(e)}]"
        if self.kind == "DIALOGUE":
            who = self.speaker or "?"
            if self.voice:
                who += f" ({self.voice})"
            tag = f" ({', '.join(self.tags)})" if self.tags else ""
            return f"{ts} {who}{tag}: {self.text}"
        return f"{ts} {self.kind}: {self.text}"


@dataclass
class Episode:
    number: int
    source: str
    logline: str = ""
    continuity: str = ""
    scenes: list[dict] = field(default_factory=list)
    beats: list[Beat] = field(default_factory=list)

    @property
    def duration(self) -> float:
        ends = [b.end for b in self.beats] + [s.get("end", 0) for s in self.scenes]
        return max(ends) if ends else 0.0

    def beat(self, bid: int) -> Beat:
        return self.beats[bid]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = self.duration
        return d


def _split_label(label: str) -> tuple[str, str | None, str | None, list[str]]:
    """'MATRIARCH (O.S.) (Firm)' -> kind, speaker, voice, tags."""
    label = label.strip().rstrip(":").strip()
    upper = label.upper()
    if upper in NON_DIALOGUE:
        return NON_DIALOGUE[upper], None, None, []
    name = re.split(r"\s*\(", label, maxsplit=1)[0].strip()
    voice, tags = None, []
    for chunk in re.findall(r"\(([^)]*)\)", label):
        if chunk.strip().upper() in VOICE_MODIFIERS:
            voice = chunk.strip()
        else:
            tags += [t.strip() for t in chunk.split(",") if t.strip()]
    return "DIALOGUE", name.upper(), voice, tags


def parse_episode(text: str, source: str = "", fallback_number: int | None = None) -> Episode:
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    header: dict[str, str] = {}
    scenes: list[dict] = []
    raw_beats: list[dict] = []
    cur_scene = 1
    pending_label: str | None = None
    current: dict | None = None

    def flush():
        nonlocal current
        if current is not None:
            current["text"] = " ".join(current["text"]).strip().strip('"').strip()
            if current["text"]:
                raw_beats.append(current)
        current = None

    for raw in lines:
        line = raw.strip()
        if not line:
            if current is not None and current["text"]:
                flush()
            continue
        if line.startswith("--- SCRIPT"):
            flush()
            continue
        m = HEADER_RE.match(line)
        if m and not raw_beats and current is None:
            header[m.group(1).upper()] = m.group(2).strip()
            continue
        m = SCENE_RE.match(line)
        if m:
            flush()
            cur_scene = int(m.group(1))
            scenes.append({"number": cur_scene, "heading": "", "start": 0.0, "end": 0.0})
            continue
        m = SCENE_FIELD_RE.match(line)
        if m and scenes:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "heading":
                scenes[-1]["heading"] = val
            else:
                try:
                    scenes[-1][key] = to_seconds(val)
                except ValueError:
                    pass
            continue
        if line.lower().startswith("actions and dialogue"):
            continue
        m = LABEL_RE.match(line)
        if m:
            flush()
            pending_label = m.group("label")
            continue
        m = TIMING_RE.search(line)
        if m and pending_label is not None:
            kind, speaker, voice, tags = _split_label(pending_label)
            current = {"kind": kind, "speaker": speaker, "voice": voice, "tags": tags,
                       "start": to_seconds(m.group("s")), "end": to_seconds(m.group("e")),
                       "text": [], "scene": cur_scene}
            pending_label = None
            continue
        if current is not None:
            current["text"].append(line)
    flush()

    num = None
    if "EPISODE NUMBER" in header:
        mm = EP_NUM_RE.search(header["EPISODE NUMBER"])
        num = int(mm.group(1)) if mm else None
    if num is None:
        mm = re.search(r"episode[-_ ]?(\d+)", source, re.I)
        num = int(mm.group(1)) if mm else (fallback_number or 0)

    # Sort by time so beat ids follow what the viewer sees; stable sort keeps
    # script order for beats that start together (camera + line).
    raw_beats.sort(key=lambda b: b["start"])
    beats = [Beat(id=i, **b) for i, b in enumerate(raw_beats)]
    return Episode(number=num, source=source, logline=header.get("EPISODE LOGLINE", ""),
                   continuity=header.get("CONTINUITY NOTES", ""), scenes=scenes, beats=beats)


# ---------------------------------------------------------------- fallbacks
# A new show may not come in exactly this format. Two cheap fallbacks so the tool
# still runs: subtitle files (SRT/VTT), and any "timestamp(s) + text" line.
SRT_TIME = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})")
LOOSE_LINE = re.compile(rf"^\s*\[?\(?(?P<s>{TS})\)?\]?\s*(?:[-–]+>?|to)?\s*(?:\[?\(?(?P<e>{TS})\)?\]?)?\s*[\]:|-]*\s*(?P<rest>.+)$")


def _speaker_split(text: str) -> tuple[str | None, str]:
    m = re.match(r"^([A-Z][A-Z .'\-]{1,40})(?:\s*\(([^)]*)\))?\s*:\s*(.+)$", text)
    return (m.group(1).strip(), m.group(3).strip()) if m else (None, text.strip())


def parse_subtitles(text: str, source: str, number: int) -> Episode:
    beats, blocks = [], re.split(r"\n\s*\n", text.replace("\r\n", "\n"))
    for blk in blocks:
        m = SRT_TIME.search(blk)
        if not m:
            continue
        body = " ".join(l.strip() for l in blk[m.end():].strip().splitlines() if l.strip())
        body = re.sub(r"<[^>]+>", "", body)
        if not body:
            continue
        who, said = _speaker_split(body)
        beats.append(Beat(id=len(beats), kind="DIALOGUE", start=to_seconds(m.group(1).replace(",", ".")),
                          end=to_seconds(m.group(2).replace(",", ".")), text=said, speaker=who or "VOICE"))
    return Episode(number=number, source=source, beats=beats)


def parse_loose(text: str, source: str, number: int) -> Episode:
    """Last resort: any line that starts with a timestamp (or range) followed by text."""
    rows = []
    for line in text.splitlines():
        m = LOOSE_LINE.match(line)
        if not m or len(m.group("rest").strip()) < 2:
            continue
        rows.append((to_seconds(m.group("s")), to_seconds(m.group("e")) if m.group("e") else None,
                     m.group("rest").strip()))
    beats = []
    for i, (s, e, rest) in enumerate(rows):
        if e is None:   # no end time: run until the next line starts (max 6 s)
            nxt = rows[i + 1][0] if i + 1 < len(rows) else s + 3
            e = min(max(nxt, s + 0.5), s + 6)
        who, said = _speaker_split(rest)
        kind = "DIALOGUE" if who else ("ACTION" if not rest.startswith('"') else "DIALOGUE")
        beats.append(Beat(id=len(beats), kind=kind, start=s, end=e, text=said.strip('"'), speaker=who))
    beats.sort(key=lambda b: b.start)
    for i, b in enumerate(beats):
        b.id = i
    return Episode(number=number, source=source, beats=beats)


def parse_any(text: str, source: str, fallback_number: int) -> Episode:
    """Kuku screenplay format first; then subtitles; then loose timestamped lines."""
    ep = parse_episode(text, source, fallback_number)
    if ep.beats:
        return ep
    num = ep.number or fallback_number
    if SRT_TIME.search(text):
        sub = parse_subtitles(text, source, num)
        if sub.beats:
            return sub
    return parse_loose(text, source, num)


TEXT_EXT = (".txt", ".srt", ".vtt", ".md")


def load_show(path: str | Path) -> tuple[str, list[Episode]]:
    """Load every episode from a folder, a .zip, or a single .txt file."""
    path = Path(path)
    texts: list[tuple[str, str]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                if name.lower().endswith(TEXT_EXT) and not name.startswith("__MACOSX"):
                    texts.append((name, zf.read(name).decode("utf-8", errors="replace")))
        title = path.stem
    elif path.is_file():
        texts.append((path.name, path.read_text(encoding="utf-8", errors="replace")))
        title = path.parent.name
    else:
        files = sorted(f for f in path.rglob("*") if f.suffix.lower() in TEXT_EXT)
        texts = [(str(f.relative_to(path)), f.read_text(encoding="utf-8", errors="replace")) for f in files]
        title = path.name
    episodes = [parse_any(t, src, i + 1) for i, (src, t) in enumerate(texts)]
    episodes = [e for e in episodes if e.beats]
    episodes.sort(key=lambda e: e.number)
    # Zip names like "Show-scripts (2) (1)" -> use the folder inside if there is one.
    inner = {Path(src).parts[0] for src, _ in texts if len(Path(src).parts) > 1}
    if len(inner) == 1:
        title = inner.pop()
    title = re.sub(r"-scripts.*$", "", title).strip()
    return title, episodes


def load_show_from_bytes(data: bytes, name: str = "upload.zip") -> tuple[str, list[Episode]]:
    """Same as load_show, for Streamlit uploads (zip or single txt)."""
    if name.lower().endswith(".zip"):
        tmp = io.BytesIO(data)
        texts = []
        with zipfile.ZipFile(tmp) as zf:
            for n in sorted(zf.namelist()):
                if n.lower().endswith(TEXT_EXT) and not n.startswith("__MACOSX"):
                    texts.append((n, zf.read(n).decode("utf-8", errors="replace")))
        episodes = [parse_any(t, src, i + 1) for i, (src, t) in enumerate(texts)]
        inner = {Path(src).parts[0] for src, _ in texts if len(Path(src).parts) > 1}
        title = inner.pop() if len(inner) == 1 else Path(name).stem
    else:
        episodes = [parse_any(data.decode("utf-8", errors="replace"), name, 1)]
        title = Path(name).stem
    episodes = sorted([e for e in episodes if e.beats], key=lambda e: e.number)
    return re.sub(r"-scripts.*$", "", title).strip(), episodes
