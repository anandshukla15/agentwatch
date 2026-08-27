"""
Example evals for the calc-agent example.

Run the example agent first so a trace file exists:
    python examples/langgraph_gemini_example.py "what's 12 * 8?"

Then run these evals:
    agentwatch eval tests/
"""

from agentwatch.evals import eval_case, TraceAssert

TRACE_GLOB = "traces/calc-agent-*.jsonl"


@eval_case(trace=TRACE_GLOB)
def test_agent_completed(trace: TraceAssert):
    trace.assert_completed()


@eval_case(trace=TRACE_GLOB)
def test_no_errors(trace: TraceAssert):
    trace.assert_no_errors()


@eval_case(trace=TRACE_GLOB)
def test_used_calculator_tool(trace: TraceAssert):
    trace.assert_tool_called("calculator", at_least=1)


@eval_case(trace=TRACE_GLOB)
def test_reasonable_step_count(trace: TraceAssert):
    # planner -> calculator -> responder == 3 steps. Catches accidental loops.
    trace.assert_max_steps(5)


@eval_case(trace=TRACE_GLOB)
def test_fast_enough(trace: TraceAssert):
    trace.assert_duration_under(30.0)
