"""
Example: a tiny LangGraph agent (planner -> tool -> responder) powered by
Gemini, fully traced with agentwatch.

Setup:
    pip install -r ../requirements.txt
    export GOOGLE_API_KEY="your-gemini-api-key"

Run:
    python langgraph_gemini_example.py "What's 17 * 24, then add 5?"

This writes a trace to ./traces/<run>.jsonl. Try:
    agentwatch replay traces/<the-file-that-got-created>.jsonl
"""

from __future__ import annotations

import os
import sys
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import google.generativeai as genai

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agentwatch.tracer import Tracer

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
MODEL_NAME = "gemini-2.5-flash"


class AgentState(TypedDict):
    question: str
    plan: str
    tool_result: str
    answer: str
    messages: Annotated[list, operator.add]


def safe_calculator(expression: str) -> str:
    """A deliberately tiny, safe 'tool' the agent can call."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "error: expression contains disallowed characters"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"error: {e}"


def make_graph(tracer: Tracer):
    model = genai.GenerativeModel(MODEL_NAME)

    def plan_node(state: AgentState) -> AgentState:
        with tracer.step("llm_call", node="planner") as step:
            prompt = (
                "You are a planning step for a math assistant. Given the "
                "user's question, output ONLY a single arithmetic expression "
                "(digits and + - * / ( ) only) that would answer it. No words.\n\n"
                f"Question: {state['question']}"
            )
            step.log_input({"prompt": prompt})
            response = model.generate_content(prompt)
            plan = response.text.strip()
            step.log_output({"plan": plan})
        return {"plan": plan}

    def tool_node(state: AgentState) -> AgentState:
        with tracer.step("tool_call", node="calculator") as step:
            step.log_input({"expression": state["plan"]})
            result = safe_calculator(state["plan"])
            step.log_output({"result": result})
        return {"tool_result": result}

    def respond_node(state: AgentState) -> AgentState:
        with tracer.step("llm_call", node="responder") as step:
            prompt = (
                f"The question was: {state['question']}\n"
                f"The computed result was: {state['tool_result']}\n"
                "Write a one-sentence natural-language answer."
            )
            step.log_input({"prompt": prompt})
            response = model.generate_content(prompt)
            answer = response.text.strip()
            step.log_output({"answer": answer})
        return {"answer": answer}

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("tool", tool_node)
    graph.add_node("respond", respond_node)
    graph.set_entry_point("plan")
    graph.add_edge("plan", "tool")
    graph.add_edge("tool", "respond")
    graph.add_edge("respond", END)
    return graph.compile()


def run(question: str) -> str:
    with Tracer(run_name="calc-agent") as tracer:
        app = make_graph(tracer)
        tracer.checkpoint("run_started", {"question": question})
        final_state = app.invoke({
            "question": question, "plan": "", "tool_result": "",
            "answer": "", "messages": [],
        })
        tracer.checkpoint("run_finished", {"answer": final_state["answer"]})
        return final_state["answer"]


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What's 17 * 24, then add 5?"
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY first: export GOOGLE_API_KEY=your-key")
        sys.exit(1)
    print(run(q))
