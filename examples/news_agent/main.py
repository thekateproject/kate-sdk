"""
News Aggregator Agent — KATE SDK Example

A LangGraph pipeline that fetches, ranks, and summarizes today's top 5 news
stories using DuckDuckGo + an LLM (OpenAI or Anthropic).

Pipeline: fetch_news -> rank_stories -> summarize_stories -> format_output

This example demonstrates:
  - projectkate.init() to configure the SDK
  - @projectkate.trace() to capture each LLM call as a traced span
  - projectkate.run() context manager for run lifecycle + auto-eval

Requirements:
  pip install projectkate[langchain] langchain-openai langgraph duckduckgo-search

Environment variables:
  KATE_API_URL       KATE server URL (default: http://localhost:8000)
  KATE_API_KEY       Your KATE API key
  KATE_AGENT_ID      Your agent's UUID in KATE
  OPENAI_API_KEY     or ANTHROPIC_API_KEY — LLM provider key

Usage:
  python main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

import projectkate


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    raw_results: str
    ranked_stories: list[dict]
    summaries: list[dict]
    final_output: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, stripping markdown fences."""
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    payload = match.group(1).strip() if match else text.strip()
    return json.loads(payload)


def _get_llm():
    """Return an LLM instance — prefers OpenAI, falls back to Anthropic."""
    if os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=2048)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0, max_tokens=2048)
    else:
        print("Error: Set OPENAI_API_KEY or ANTHROPIC_API_KEY.", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Pipeline nodes — each LLM call is traced via @projectkate.trace
# ---------------------------------------------------------------------------

def fetch_news(state: AgentState) -> dict:
    """Search DuckDuckGo for today's breaking news."""
    from ddgs import DDGS

    today = datetime.now().strftime("%B %d, %Y")
    query = f"latest breaking news today {today}"

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=20))

    raw = json.dumps(results, indent=2) if results else "[]"
    return {"raw_results": raw}


@projectkate.trace("rank_stories")
def rank_stories(state: AgentState) -> dict:
    """Use an LLM to pick the top 5 most important, diverse stories."""
    llm = _get_llm()
    today = datetime.now().strftime("%B %d, %Y")

    prompt = f"""You are a news editor. Below are search results from {today}.

Pick the top 5 most important and diverse news stories. Return ONLY a JSON array
(no other text) where each element has these keys:
- "title": a clear, concise headline
- "url": the source URL
- "snippet": the original snippet from search results

Search results:
{state["raw_results"]}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        ranked = _parse_json(response.content)[:5]
    except (json.JSONDecodeError, TypeError):
        raw = json.loads(state["raw_results"])
        ranked = [
            {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in raw[:5]
        ]

    return {"ranked_stories": ranked}


@projectkate.trace("summarize_stories")
def summarize_stories(state: AgentState) -> dict:
    """Use an LLM to write a 2-3 sentence summary for each story."""
    llm = _get_llm()

    stories_json = json.dumps(state["ranked_stories"], indent=2)
    prompt = f"""Below are 5 news stories. For each one, write a summary as 3-4 bullet points
that capture the key facts. Return ONLY a JSON array (no other text) where each element has:
- "title": the headline (keep the original)
- "url": the source URL (keep the original)
- "summary": your 3-4 bullet point summary (use "* " prefix, separated by newlines)

Stories:
{stories_json}"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        summaries = _parse_json(response.content)[:5]
    except (json.JSONDecodeError, TypeError):
        summaries = [
            {"title": s.get("title", ""), "url": s.get("url", ""), "summary": s.get("snippet", "")}
            for s in state["ranked_stories"]
        ]

    return {"summaries": summaries}


def format_output(state: AgentState) -> dict:
    """Format summaries for terminal display."""
    today = datetime.now().strftime("%B %d, %Y")
    sep = "=" * 60

    lines = ["", sep, f"  TODAY'S TOP 5 NEWS STORIES -- {today}", sep, ""]

    for i, story in enumerate(state["summaries"], 1):
        lines.append(f"{i}. {story.get('title', 'Untitled')}")
        lines.append(f"   Source: {story.get('url', 'N/A')}")
        lines.append(f"   {story.get('summary', 'No summary available.')}")
        lines.append("")

    lines.extend([sep, "  Powered by LangGraph + KATE SDK", sep, ""])
    return {"final_output": "\n".join(lines)}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("fetch_news", fetch_news)
    graph.add_node("rank_stories", rank_stories)
    graph.add_node("summarize_stories", summarize_stories)
    graph.add_node("format_output", format_output)
    graph.set_entry_point("fetch_news")
    graph.add_edge("fetch_news", "rank_stories")
    graph.add_edge("rank_stories", "summarize_stories")
    graph.add_edge("summarize_stories", "format_output")
    graph.add_edge("format_output", END)
    return graph


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    # Initialize KATE SDK (reads config from env vars)
    projectkate.init()

    # Run the agent inside a KATE run context
    # On exit, the run is marked complete and auto-eval is triggered
    async with projectkate.run() as ctx:
        app = build_graph().compile()
        result = app.invoke({
            "raw_results": "",
            "ranked_stories": [],
            "summaries": [],
            "final_output": "",
        })
        # Record the final output for evaluation
        ctx.output(result["final_output"])
        print(result["final_output"])


if __name__ == "__main__":
    asyncio.run(main())
