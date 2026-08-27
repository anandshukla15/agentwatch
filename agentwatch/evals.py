"""
agentwatch.evals
------------------
A tiny, pytest-flavored framework for writing assertions against agent
traces, so agent regressions get caught in CI the same way code regressions
do.

Write eval functions in a file like tests/test_example_evals.py:

    from agentwatch.evals import eval_case, TraceAssert

    @eval_case(trace="traces/fix-bug-123-*.jsonl")
    def test_no_infinite_loop(trace: TraceAssert):
        trace.assert_max_steps(20)
        trace.assert_no_errors()

    @eval_case(trace="traces/fix-bug-123-*.jsonl")
    def test_tests_were_run(trace: TraceAssert):
        trace.assert_tool_called("run_tests")

Then run:

    agentwatch eval tests/

Exit code is non-zero if any eval fails, so it plugs straight into CI.
"""

from __future__ import annotations

import glob
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .tracer import load_trace


class EvalFailure(AssertionError):
    pass


@dataclass
class TraceAssert:
    """Assertion helpers bound to one loaded trace."""

    records: list[dict]
    path: str

    def _steps(self) -> list[dict]:
        return [r for r in self.records if r.get("type") not in
                ("run_start", "run_end", "checkpoint")]

    def assert_max_steps(self, n: int) -> None:
        count = len(self._steps())
        if count > n:
            raise EvalFailure(f"expected <= {n} steps, got {count}")

    def assert_no_errors(self) -> None:
        errors = [r for r in self._steps() if r.get("error")]
        if errors:
            raise EvalFailure(f"{len(errors)} step(s) errored: "
                               f"{[e.get('node') for e in errors]}")

    def assert_tool_called(self, node_name: str, at_least: int = 1) -> None:
        calls = [r for r in self._steps() if r.get("node") == node_name]
        if len(calls) < at_least:
            raise EvalFailure(
                f"expected node '{node_name}' called >= {at_least} times, "
                f"got {len(calls)}")

    def assert_completed(self) -> None:
        types = [r.get("type") for r in self.records]
        if "run_end" not in types:
            raise EvalFailure("trace has no run_end event (agent may have crashed)")

    def assert_duration_under(self, seconds: float) -> None:
        total = sum(r.get("duration_s", 0) for r in self._steps())
        if total > seconds:
            raise EvalFailure(f"total step duration {total}s exceeds {seconds}s")


@dataclass
class EvalCase:
    name: str
    fn: Callable
    trace_glob: str


_REGISTRY: list[EvalCase] = []


def eval_case(trace: str):
    """Decorator to register an eval function against a trace file glob."""

    def decorator(fn: Callable):
        _REGISTRY.append(EvalCase(name=fn.__name__, fn=fn, trace_glob=trace))
        return fn

    return decorator


def _load_eval_files(directory: str) -> None:
    for path in sorted(Path(directory).rglob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)  # registers eval_case-decorated fns


def run_evals(directory: str) -> bool:
    """Discovers eval files, runs them, prints a report. Returns True if all pass."""
    _REGISTRY.clear()
    _load_eval_files(directory)

    if not _REGISTRY:
        print(f"No eval cases found under {directory} (looking for test_*.py files "
              f"with @eval_case functions).")
        return True

    passed, failed = 0, 0
    for case in _REGISTRY:
        matches = glob.glob(case.trace_glob)
        if not matches:
            print(f"SKIP  {case.name}  (no trace matched '{case.trace_glob}')")
            continue
        trace_path = sorted(matches)[-1]  # most recent by name/timestamp
        records = load_trace(trace_path)
        ta = TraceAssert(records=records, path=trace_path)
        try:
            case.fn(ta)
        except EvalFailure as e:
            print(f"FAIL  {case.name}  ({trace_path})\n      -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {case.name}  ({trace_path})\n      -> {e!r}")
            failed += 1
        else:
            print(f"PASS  {case.name}  ({trace_path})")
            passed += 1

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0
