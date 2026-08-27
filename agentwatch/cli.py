"""
agentwatch CLI

    agentwatch replay traces/fix-bug-123-1234567.jsonl
    agentwatch replay traces/fix-bug-123-1234567.jsonl --errors-only
    agentwatch eval tests/
"""

from __future__ import annotations

import argparse
import sys

from .replay import replay
from .evals import run_evals


def main() -> None:
    parser = argparse.ArgumentParser(prog="agentwatch")
    sub = parser.add_subparsers(dest="command", required=True)

    p_replay = sub.add_parser("replay", help="Step through a trace file")
    p_replay.add_argument("trace_path")
    p_replay.add_argument("--errors-only", action="store_true")

    p_eval = sub.add_parser("eval", help="Run eval cases against traces")
    p_eval.add_argument("directory", nargs="?", default="tests")

    args = parser.parse_args()

    if args.command == "replay":
        replay(args.trace_path, show_errors_only=args.errors_only)
    elif args.command == "eval":
        ok = run_evals(args.directory)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
