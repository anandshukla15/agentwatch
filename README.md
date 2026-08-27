# agentwatch

Lightweight, framework-agnostic tracing + eval framework for LLM agents
(LangGraph, AutoGen, or your own agent loop). Catch agent regressions in CI
the same way you catch code regressions.

- **Trace** every LLM call and tool call to a simple JSONL file
- **Replay** a run step-by-step in the terminal to see exactly where it went wrong
- **Eval** traces with pytest-style assertions, and fail a PR in CI if the agent regresses

No servers, no accounts, no dashboards to host — just files and a CLI.

## Install

```bash
git clone <your-fork-url> agentwatch
cd agentwatch
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .
```

This gives you the `agentwatch` command plus the `agentwatch` Python package.

## Quickstart with the LangGraph + Gemini example

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
cd examples
python langgraph_gemini_example.py "What's 17 * 24, then add 5?"
```

This writes a trace to `traces/calc-agent-<timestamp>.jsonl`. Look at it:

```bash
agentwatch replay traces/calc-agent-<timestamp>.jsonl
```

Sample output:

```
=== Trace: traces/calc-agent-1735300000.jsonl (5 events) ===

[start] run_id=calc-agent-1735300000 name=calc-agent
[checkpoint] run_started -> {"question": "What's 17 * 24, then add 5?"}
#2 [ok] llm_call node=planner (0.8123s)
   in : {"prompt": "You are a planning step..."}
   out: {"plan": "17*24+5"}

#3 [ok] tool_call node=calculator (0.0001s)
   in : {"expression": "17*24+5"}
   out: {"result": "413"}

#4 [ok] llm_call node=responder (0.6210s)
   in : {"prompt": "The question was..."}
   out: {"answer": "17 times 24 plus 5 equals 413."}

[checkpoint] run_finished -> {"answer": "17 times 24 plus 5 equals 413."}
[end]   run_id=calc-agent-1735300000
```

## Writing evals

Evals live in `tests/test_*.py` and use `@eval_case` + `TraceAssert`:

```python
from agentwatch.evals import eval_case, TraceAssert

@eval_case(trace="traces/calc-agent-*.jsonl")
def test_no_infinite_loop(trace: TraceAssert):
    trace.assert_max_steps(5)

@eval_case(trace="traces/calc-agent-*.jsonl")
def test_used_calculator(trace: TraceAssert):
    trace.assert_tool_called("calculator")
```

Run them:

```bash
agentwatch eval tests/
```

Non-zero exit code on failure, so it plugs straight into CI — see
`.github/workflows/agent-evals.yml` for a working GitHub Action.

## Using it in your own agent

`agentwatch` doesn't care what framework you use. Wrap any call:

```python
from agentwatch.tracer import Tracer

with Tracer(run_name="my-agent") as tracer:
    with tracer.step("llm_call", node="planner") as step:
        step.log_input({"prompt": prompt})
        result = my_llm_call(prompt)
        step.log_output({"result": result})
```

Available `TraceAssert` methods: `assert_max_steps`, `assert_no_errors`,
`assert_tool_called`, `assert_completed`, `assert_duration_under`. Adding new
assertion helpers is a one-function change in `agentwatch/evals.py` — PRs
welcome.

## Project layout

```
agentwatch/
  agentwatch/
    tracer.py   # core: Tracer, Step, load_trace
    replay.py   # terminal replay viewer
    evals.py    # eval_case decorator, TraceAssert, run_evals
    cli.py      # `agentwatch replay` / `agentwatch eval`
  examples/
    langgraph_gemini_example.py
  tests/
    test_example_evals.py
  .github/workflows/agent-evals.yml
```

## Roadmap ideas (good first issues)

- [ ] AutoGen wrapper example (mirrors the LangGraph one)
- [ ] Cost/token tracking per step
- [ ] HTML trace viewer (single static file, no server)
- [ ] Diff two traces to spot behavioral drift between agent versions
- [ ] `assert_output_contains` / regex-based content assertions

## License

MIT — do whatever you want with it.
