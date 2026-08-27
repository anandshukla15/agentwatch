"""
agentwatch.replay
------------------
Pretty-prints a trace file step by step so you can see exactly what an agent
did: which node ran, what went in, what came out, how long it took, and
whether it errored.
"""

from __future__ import annotations

import json
from pathlib import Path

from .tracer import load_trace


def _short(value, limit: int = 300) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def replay(path: str | Path, show_errors_only: bool = False) -> None:
    records = load_trace(path)
    print(f"\n=== Trace: {path} ({len(records)} events) ===\n")

    for i, r in enumerate(records):
        rtype = r.get("type")

        if rtype == "run_start":
            print(f"[start] run_id={r.get('run_id')} name={r.get('run_name')}")
            continue
        if rtype == "run_end":
            print(f"[end]   run_id={r.get('run_id')}")
            continue
        if rtype == "checkpoint":
            print(f"[checkpoint] {r.get('label')} -> {_short(r.get('data'))}")
            continue

        has_error = bool(r.get("error"))
        if show_errors_only and not has_error:
            continue

        marker = "FAILED" if has_error else "ok"
        print(f"#{i} [{marker}] {r.get('type')} node={r.get('node')} "
              f"({r.get('duration_s')}s)")
        print(f"   in : {_short(r.get('input'))}")
        print(f"   out: {_short(r.get('output'))}")
        if has_error:
            print(f"   error:\n{r.get('error')}")
        print()


if __name__ == "__main__":
    import sys
    replay(sys.argv[1])
