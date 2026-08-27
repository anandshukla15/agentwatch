"""
agentwatch.tracer
------------------
Framework-agnostic tracing for LLM agents (LangGraph, AutoGen, raw scripts, etc).

Every "event" (an LLM call, a tool call, or a custom checkpoint) gets appended
as one JSON line to a trace file. JSONL was chosen on purpose: it's greppable,
diffable in git, streams well for long-running agents, and needs no extra
tooling to inspect.

Basic usage
-----------
    from agentwatch.tracer import Tracer

    tracer = Tracer(run_name="fix-bug-123")

    with tracer.step("llm_call", node="planner") as step:
        response = call_gemini(prompt)
        step.log_input({"prompt": prompt})
        step.log_output({"response": response})

    with tracer.step("tool_call", node="run_tests") as step:
        result = run_tests()
        step.log_input({"cmd": "pytest"})
        step.log_output({"passed": result.passed, "failed": result.failed})

    tracer.close()

The trace file lives at ./traces/<run_name>-<timestamp>.jsonl by default.
"""

from __future__ import annotations

import json
import time
import uuid
import contextlib
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional


class Step:
    """Represents one event inside a trace. Populated via log_input/log_output."""

    def __init__(self, tracer: "Tracer", step_type: str, node: Optional[str] = None):
        self._tracer = tracer
        self.step_id = str(uuid.uuid4())[:8]
        self.step_type = step_type
        self.node = node
        self.input: Any = None
        self.output: Any = None
        self.error: Optional[str] = None
        self.started_at = time.time()
        self.ended_at: Optional[float] = None

    def log_input(self, data: Any) -> None:
        self.input = data

    def log_output(self, data: Any) -> None:
        self.output = data

    def to_record(self) -> dict:
        return {
            "step_id": self.step_id,
            "type": self.step_type,
            "node": self.node,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at,
            "duration_s": round((self.ended_at or time.time()) - self.started_at, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class Tracer:
    """Writes one JSONL file per run. Thread-unsafe by design (keep it simple);
    wrap in your own lock if you parallelize agent branches."""

    def __init__(self, run_name: str = "run", trace_dir: str = "traces"):
        self.run_id = f"{run_name}-{int(time.time())}"
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.trace_dir / f"{self.run_id}.jsonl"
        self._fh = open(self.path, "a", encoding="utf-8")
        self._write_meta_header(run_name)

    def _write_meta_header(self, run_name: str) -> None:
        header = {
            "type": "run_start",
            "run_id": self.run_id,
            "run_name": run_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._fh.write(json.dumps(header) + "\n")
        self._fh.flush()

    @contextlib.contextmanager
    def step(self, step_type: str, node: Optional[str] = None):
        """Context manager for one traced event. Automatically records
        exceptions instead of swallowing them, then re-raises."""
        s = Step(self, step_type, node)
        try:
            yield s
        except Exception:
            s.error = traceback.format_exc()
            raise
        finally:
            s.ended_at = time.time()
            self._fh.write(json.dumps(s.to_record(), default=str) + "\n")
            self._fh.flush()

    def checkpoint(self, label: str, data: Any = None) -> None:
        """Free-form marker, e.g. tracer.checkpoint('replanned', {'reason': ...})."""
        record = {
            "type": "checkpoint",
            "label": label,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        footer = {"type": "run_end", "run_id": self.run_id,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
        self._fh.write(json.dumps(footer) + "\n")
        self._fh.close()

    def __enter__(self) -> "Tracer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def load_trace(path: str | Path) -> list[dict]:
    """Read a trace file back into a list of records, in order."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
