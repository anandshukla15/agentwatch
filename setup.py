from setuptools import setup, find_packages

setup(
    name="agentwatch",
    version="0.1.0",
    description="Lightweight tracing and eval framework for LLM agents (LangGraph, AutoGen, etc.)",
    packages=find_packages(exclude=["tests", "examples"]),
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "agentwatch=agentwatch.cli:main",
        ],
    },
)
