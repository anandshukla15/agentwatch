"""
agentwatch.replay
------------------
Pretty-prints a trace file step by step so you can see exactly what an agent
did: which node ran, what went in, what came out, how long it took, and
whether it errored.

Supports both: 
   agentwatch replay traces/calc-agent-123.jsonl 
and glob patterns: 
   agentwatch replay "traces/calc-agent-*.jsonl"
"""

from __future__ import annotations

import json
from pathlib import Path

from .tracer import load_trace


def _short(value, limit: int = 300) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[:limit] + "... [truncated]"


def _replay_file(path: str | Path, show_errors_only: bool = False) -> None:
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


def replay(path: str | Path, show_errors_only: bool = False) -> None:
    path=str(path)
    if any(char in path for char in "*?["):
        matches = sorted(Path(".").glob(path))
        if not matches:
            raise FileNotFoundError( f"No trace files matched pattern: {path}" )
        print(f"Found {len(matches)} trace file(s).\n")
        for trace_file in matches:
            _replay_file(trace_file, show_errors_only=show_errors_only)
    else:
        trace_file = Path(path)
        if not trace_file.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_file}")
        _replay_file(trace_file, show_errors_only=show_errors_only)    


if __name__ == "__main__":
    import sys
    replay(sys.argv[1])
