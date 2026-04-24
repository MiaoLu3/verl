"""Local Flask app for browsing ALFWorld trajectory dumps.

Each JSONL file under TRAJ_ROOT/<run>/ contains a single JSON object
with keys: request_id, gamefile, won, final_reward, num_turns,
num_invalid_actions, turns (list), ...

Run:  python app.py [--port 5000]
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import (
    Flask,
    abort,
    render_template,
    request,
    url_for,
)
from markupsafe import Markup

TRAJ_ROOT = Path("/scratch/m000069-pm05/miaolu/verl/trajectories")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Index cache
# ---------------------------------------------------------------------------
# Per-run index: list of small dicts, one per trajectory file (no turns,
# no full_decoded_sequence). Built on first request for that run.
# Structure:
#   _INDEX[run_name] = {
#       "built_at": float,
#       "entries": [ {..lightweight fields..}, ... ],
#   }
_INDEX: Dict[str, Dict[str, Any]] = {}
_INDEX_LOCK = threading.Lock()

# Aggregate stats for the home page. Built lazily per run too.
_AGG: Dict[str, Dict[str, Any]] = {}

LIGHT_KEYS = (
    "request_id",
    "gamefile",
    "won",
    "final_reward",
    "num_turns",
    "num_invalid_actions",
    "prompt_length_final",
    "response_length_final",
    "response_mask_1_count",
    "response_mask_0_count",
    "global_step",
    "validate",
)


def _step_from_path(p: Path, run_dir: Path) -> tuple[Optional[int], bool]:
    """Parse ``step_<N>[_val]`` from the immediate parent of ``p``.

    Returns ``(step, validate)`` where step is None for legacy flat layouts.
    """
    try:
        rel = p.relative_to(run_dir).parts
    except ValueError:
        return None, False
    if len(rel) < 2:
        return None, False
    parent = rel[0]
    m = re.match(r"^step_(\d+)(_val)?$", parent)
    if not m:
        if parent == "step_unknown":
            return None, False
        if parent == "step_unknown_val":
            return None, True
        return None, False
    return int(m.group(1)), bool(m.group(2))


def _safe_run(run_name: str) -> Path:
    """Resolve a run name to a directory inside TRAJ_ROOT; guard traversal."""
    if not run_name or "/" in run_name or run_name.startswith("."):
        abort(400, "bad run name")
    run_dir = (TRAJ_ROOT / run_name).resolve()
    try:
        run_dir.relative_to(TRAJ_ROOT.resolve())
    except ValueError:
        abort(400, "path escapes traj root")
    if not run_dir.is_dir():
        abort(404, "run not found")
    return run_dir


def _light_read(p: Path, run_dir: Path) -> Optional[Dict[str, Any]]:
    """Read one trajectory file, return only the lightweight top-level fields.

    Uses json.load; the files are single-object JSON so this is fine.
    Returns None on read/parse failure.
    """
    try:
        with open(p, "r") as f:
            data = json.load(f)
    except Exception:
        return None
    entry = {k: data.get(k) for k in LIGHT_KEYS}
    # relpath (e.g. "step_46/abc.jsonl") is the URL-friendly handle;
    # also fill step/validate from the parent dir when the file itself
    # predates the global_step field.
    rel = str(p.relative_to(run_dir))
    entry["filename"] = rel
    if entry.get("global_step") is None or entry.get("validate") is None:
        step, validate = _step_from_path(p, run_dir)
        if entry.get("global_step") is None:
            entry["global_step"] = step
        if entry.get("validate") is None:
            entry["validate"] = validate
    return entry


def _build_index(run_name: str) -> Dict[str, Any]:
    """Build (or reuse) the index for a single run. Thread-safe."""
    with _INDEX_LOCK:
        if run_name in _INDEX:
            return _INDEX[run_name]

    run_dir = _safe_run(run_name)
    # Recursive: handles both flat legacy layout and new step_<N>/ layout.
    files = sorted(run_dir.rglob("*.jsonl"))
    entries: List[Dict[str, Any]] = []
    t0 = time.time()
    for p in files:
        e = _light_read(p, run_dir)
        if e is not None:
            entries.append(e)
    elapsed = time.time() - t0

    # aggregates
    n = len(entries)
    won_n = sum(1 for e in entries if e.get("won"))
    mean_turns = (
        sum((e.get("num_turns") or 0) for e in entries) / n if n else 0.0
    )
    mean_invalid = (
        sum((e.get("num_invalid_actions") or 0) for e in entries) / n
        if n
        else 0.0
    )
    agg = {
        "count": n,
        "won": won_n,
        "win_rate": (won_n / n) if n else 0.0,
        "mean_turns": mean_turns,
        "mean_invalid": mean_invalid,
        "build_secs": elapsed,
    }

    with _INDEX_LOCK:
        _INDEX[run_name] = {"built_at": time.time(), "entries": entries}
        _AGG[run_name] = agg
    return _INDEX[run_name]


def _get_agg(run_name: str) -> Optional[Dict[str, Any]]:
    return _AGG.get(run_name)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def home():
    if not TRAJ_ROOT.is_dir():
        abort(500, f"TRAJ_ROOT does not exist: {TRAJ_ROOT}")
    runs = sorted(
        [p.name for p in TRAJ_ROOT.iterdir() if p.is_dir()],
        reverse=True,
    )
    # Lightweight: for each run, file count via glob (cheap) + cached aggs.
    rows = []
    for r in runs:
        run_dir = TRAJ_ROOT / r
        try:
            file_count = sum(1 for _ in run_dir.rglob("*.jsonl"))
        except Exception:
            file_count = 0
        agg = _get_agg(r)
        rows.append(
            {
                "name": r,
                "file_count": file_count,
                "agg": agg,  # None if not yet indexed
            }
        )
    return render_template("home.html", rows=rows, traj_root=str(TRAJ_ROOT))


@app.route("/run/<run_name>/index")
def build_run_index(run_name: str):
    """Force-build index for a run (used by home 'Index' button)."""
    _build_index(run_name)
    return ("", 204)


@app.route("/run/<run_name>/")
def run_view(run_name: str):
    _safe_run(run_name)
    idx = _build_index(run_name)
    entries = idx["entries"]

    # Filters
    won_filter = request.args.get("won", "all")  # all | won | lost
    if won_filter == "won":
        entries = [e for e in entries if e.get("won")]
    elif won_filter == "lost":
        entries = [e for e in entries if not e.get("won")]

    # Sort
    sort_key = request.args.get("sort", "request_id")
    sort_dir = request.args.get("dir", "asc")
    allowed = {
        "request_id",
        "gamefile",
        "won",
        "final_reward",
        "num_turns",
        "num_invalid_actions",
    }
    if sort_key not in allowed:
        sort_key = "request_id"

    def _key(e):
        v = e.get(sort_key)
        if sort_key == "gamefile":
            v = os.path.basename(v or "")
        if v is None:
            # Sort None consistently last in asc
            return (1, "")
        return (0, v)

    entries = sorted(entries, key=_key, reverse=(sort_dir == "desc"))

    # Paginate
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    per_page = 50
    total = len(entries)
    total_pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    page_entries = entries[start : start + per_page]

    # Gamefile basenames for display
    for e in page_entries:
        e["_gamefile_base"] = os.path.basename(e.get("gamefile") or "")

    return render_template(
        "run.html",
        run_name=run_name,
        agg=_get_agg(run_name),
        entries=page_entries,
        page=page,
        total_pages=total_pages,
        total=total,
        won_filter=won_filter,
        sort_key=sort_key,
        sort_dir=sort_dir,
    )


# Raw response rendering ----------------------------------------------------
# We escape once (HTML-safe) then swap in the colored <think>/<action> blocks.
# _THINK_RE and _ACTION_RE run on already-escaped text, so we look for the
# escaped forms (&lt;think&gt;, etc).

_THINK_RE = re.compile(
    r"&lt;think&gt;(.*?)&lt;/think&gt;", flags=re.DOTALL
)
_ACTION_RE = re.compile(
    r"&lt;action&gt;(.*?)&lt;/action&gt;", flags=re.DOTALL
)
_OPEN_THINK_RE = re.compile(r"&lt;think&gt;")
_CLOSE_THINK_RE = re.compile(r"&lt;/think&gt;")


def render_raw_response(raw: str) -> Tuple[str, bool]:
    """Return (html, truncated_mid_think) for a raw_response string."""
    escaped = html.escape(raw or "")
    n_open = len(_OPEN_THINK_RE.findall(escaped))
    n_close = len(_CLOSE_THINK_RE.findall(escaped))
    truncated = n_open > n_close

    def _repl_think(m: re.Match) -> str:
        inner = m.group(1)
        return f'<span class="think">&lt;think&gt;{inner}&lt;/think&gt;</span>'

    def _repl_action(m: re.Match) -> str:
        inner = m.group(1)
        return f'<span class="action">&lt;action&gt;{inner}&lt;/action&gt;</span>'

    out = _THINK_RE.sub(_repl_think, escaped)
    out = _ACTION_RE.sub(_repl_action, out)

    # If truncated, wrap any dangling open <think>...EOF into a think span.
    if truncated:
        # Find last unclosed <think>: all <think> positions minus matched ones.
        last_open = None
        for m in _OPEN_THINK_RE.finditer(out):
            last_open = m
        if last_open is not None:
            # Only wrap if it's not already inside a rendered span
            # (simple heuristic: the matched &lt;think&gt; is still literal).
            pre = out[: last_open.start()]
            tail = out[last_open.start() :]
            tail = (
                '<span class="think think-truncated">' + tail + "</span>"
            )
            out = pre + tail

    return out, truncated


@app.route("/run/<run_name>/traj/<path:filename>")
def traj_view(run_name: str, filename: str):
    run_dir = _safe_run(run_name)
    if not filename.endswith(".jsonl") or ".." in filename.split("/"):
        abort(400, "bad filename")
    path = (run_dir / filename).resolve()
    try:
        path.relative_to(run_dir.resolve())
    except ValueError:
        abort(400, "path escape")
    if not path.is_file():
        abort(404, "trajectory not found")

    with open(path, "r") as f:
        data = json.load(f)

    show_full = request.args.get("full") == "1"
    full_seq = data.pop("full_decoded_sequence", None) if not show_full else (
        data.get("full_decoded_sequence")
    )
    full_seq_len = len(full_seq) if show_full and full_seq else None

    # Pre-render each turn's raw_response HTML (Markup so Jinja doesn't
    # escape it again).
    for t in data.get("turns", []):
        raw = t.get("raw_response", "") or ""
        body, trunc = render_raw_response(raw)
        t["_raw_html"] = Markup(body)
        t["_truncated_think"] = trunc

    # Decide prev/next for nav inside the currently-cached (filtered) list
    # skipped for simplicity -- user can go back.

    return render_template(
        "traj.html",
        run_name=run_name,
        filename=filename,
        data=data,
        show_full=show_full,
        full_seq=full_seq,
        full_seq_len=full_seq_len,
    )


# ---------------------------------------------------------------------------
# Jinja filters
# ---------------------------------------------------------------------------


@app.template_filter("basename")
def _basename_filter(p: str) -> str:
    return os.path.basename(p or "")


@app.template_filter("pct")
def _pct_filter(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%"


@app.template_filter("round2")
def _round2(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return str(v)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
