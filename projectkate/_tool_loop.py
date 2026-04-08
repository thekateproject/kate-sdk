"""Agentic tool-use loop — auto-discover, call, and chain KATE marketplace tools."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LocalTool:
    """A locally-defined tool that runs in the buyer agent's process."""

    name: str
    description: str
    parameters: dict
    fn: Callable


@dataclass
class ToolCallInfo:
    """Metadata about a single tool invocation."""

    id: str
    name: str
    arguments: dict
    is_local: bool


@dataclass
class ToolLoopResult:
    """Result returned by tool_loop()."""

    content: str
    messages: list[dict]
    tool_calls_made: int
    rounds: int
    model: str


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------

class _ProviderAdapter(ABC):
    def __init__(self, client: Any) -> None:
        self.client = client

    @abstractmethod
    async def chat(
        self, messages: list[dict], tools: list[dict], model: str, **kwargs: Any,
    ) -> Any:
        ...

    @abstractmethod
    def extract_tool_calls(self, response: Any) -> list[ToolCallInfo]:
        ...

    @abstractmethod
    def extract_text(self, response: Any) -> str:
        ...

    @abstractmethod
    def format_tools(self, tools: list[dict]) -> list[dict]:
        ...

    @abstractmethod
    def build_assistant_message(self, response: Any) -> dict:
        ...

    @abstractmethod
    def build_tool_result_messages(
        self, tool_calls_with_results: list[tuple[ToolCallInfo, str]],
    ) -> list[dict]:
        ...


class _OpenAIAdapter(_ProviderAdapter):
    async def chat(
        self, messages: list[dict], tools: list[dict], model: str, **kwargs: Any,
    ) -> Any:
        call_kwargs: dict[str, Any] = {"model": model, "messages": messages, **kwargs}
        if tools:
            call_kwargs["tools"] = tools
        return await self.client.chat.completions.create(**call_kwargs)

    def extract_tool_calls(self, response: Any) -> list[ToolCallInfo]:
        message = response.choices[0].message
        if not message.tool_calls:
            return []
        calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Invalid JSON in tool call arguments for %s", tc.function.name)
                args = {}
            calls.append(
                ToolCallInfo(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    is_local=False,  # set later
                )
            )
        return calls

    def extract_text(self, response: Any) -> str:
        return response.choices[0].message.content or ""

    def format_tools(self, tools: list[dict]) -> list[dict]:
        return tools  # already OpenAI format

    def build_assistant_message(self, response: Any) -> dict:
        msg = response.choices[0].message
        d: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return d

    def build_tool_result_messages(
        self, tool_calls_with_results: list[tuple[ToolCallInfo, str]],
    ) -> list[dict]:
        return [
            {"role": "tool", "tool_call_id": tc.id, "content": result}
            for tc, result in tool_calls_with_results
        ]


class _AnthropicAdapter(_ProviderAdapter):
    async def chat(
        self, messages: list[dict], tools: list[dict], model: str, **kwargs: Any,
    ) -> Any:
        # Anthropic requires system as a separate kwarg
        filtered_messages = []
        system_parts: list[str] = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m.get("content", ""))
            else:
                filtered_messages.append(m)

        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": filtered_messages,
            **kwargs,
        }
        if system_parts:
            call_kwargs["system"] = "\n\n".join(system_parts)
        if tools:
            call_kwargs["tools"] = tools
        call_kwargs.setdefault("max_tokens", 4096)
        return await self.client.messages.create(**call_kwargs)

    def extract_tool_calls(self, response: Any) -> list[ToolCallInfo]:
        calls = []
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                calls.append(
                    ToolCallInfo(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if isinstance(block.input, dict) else {},
                        is_local=False,
                    )
                )
        return calls

    def extract_text(self, response: Any) -> str:
        parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts)

    def format_tools(self, tools: list[dict]) -> list[dict]:
        converted = []
        for t in tools:
            fn = t.get("function", {})
            converted.append({
                "name": fn.get("name", t.get("name", "")),
                "description": fn.get("description", t.get("description", "")),
                "input_schema": fn.get("parameters", t.get("input_schema", {})),
            })
        return converted

    def build_assistant_message(self, response: Any) -> dict:
        content_blocks = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                content_blocks.append({"type": "text", "text": block.text})
            elif getattr(block, "type", None) == "tool_use":
                content_blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input if isinstance(block.input, dict) else {},
                })
        return {"role": "assistant", "content": content_blocks}

    def build_tool_result_messages(
        self, tool_calls_with_results: list[tuple[ToolCallInfo, str]],
    ) -> list[dict]:
        # Anthropic batches all tool results into a single user message
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result,
                    }
                    for tc, result in tool_calls_with_results
                ],
            }
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_provider(client: Any) -> str:
    module = type(client).__module__ or ""
    if "openai" in module:
        return "openai"
    if "anthropic" in module:
        return "anthropic"
    raise TypeError(
        f"Unsupported LLM client: {type(client).__qualname__} "
        f"(module: {module}). Expected AsyncOpenAI or AsyncAnthropic."
    )


def _to_openai_tool(local_tool: LocalTool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": local_tool.name,
            "description": local_tool.description,
            "parameters": local_tool.parameters,
        },
    }


async def _run_local_tool(fn: Callable, args: dict) -> str:
    # Filter args to only keys the function actually accepts
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        filtered = args  # function accepts **kwargs, pass everything
    else:
        valid_keys = set(sig.parameters.keys())
        filtered = {k: v for k, v in args.items() if k in valid_keys}

    if asyncio.iscoroutinefunction(fn):
        result = await fn(**filtered)
    else:
        result = fn(**filtered)
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)
    return str(result)


def _result_to_string(result: Any) -> str:
    """Convert a ToolResult or dict to a string for the LLM."""
    if hasattr(result, "output"):
        # ToolResult dataclass
        if result.error:
            return f"Error: {result.error}"
        if isinstance(result.output, (dict, list)):
            return json.dumps(result.output, default=str)
        return str(result.output) if result.output is not None else ""
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)
    return str(result)


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------

async def tool_loop(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    local_tools: list[LocalTool] | None = None,
    max_rounds: int = 10,
    on_tool_call: Callable | None = None,
    **llm_kwargs: Any,
) -> ToolLoopResult:
    """Run an agentic tool-use loop: LLM ↔ tool calls until the LLM stops.

    Args:
        client: AsyncOpenAI or AsyncAnthropic instance.
        model: Model identifier (e.g. "gpt-4o", "claude-sonnet-4-20250514").
        messages: Initial conversation messages.
        tools: KATE tool definitions (OpenAI format). None = auto-fetch.
        local_tools: Optional locally-defined tools to merge with KATE tools.
        max_rounds: Maximum LLM round-trips before forced stop.
        on_tool_call: Optional callback(name, args, result_str) per tool call.
        **llm_kwargs: Extra kwargs passed to the LLM client.

    Returns:
        ToolLoopResult with the final text, full message history, and stats.
    """
    import projectkate  # deferred to avoid circular import

    # Detect provider and create adapter
    provider = _detect_provider(client)
    adapter: _ProviderAdapter
    if provider == "openai":
        adapter = _OpenAIAdapter(client)
    else:
        adapter = _AnthropicAdapter(client)

    # Build local tool registry
    local_registry: dict[str, Callable] = {}
    local_tool_defs: list[dict] = []
    if local_tools:
        for lt in local_tools:
            local_registry[lt.name] = lt.fn
            local_tool_defs.append(_to_openai_tool(lt))

    # Fetch KATE tools if not provided
    kate_tools: list[dict] = tools if tools is not None else await projectkate.get_tools(format="openai")

    # Check for name collisions between local and KATE tools
    kate_tool_names = set()
    for t in kate_tools:
        fn_def = t.get("function", {})
        name = fn_def.get("name", t.get("name", ""))
        if name:
            kate_tool_names.add(name)

    for local_name in local_registry:
        if local_name in kate_tool_names:
            raise ValueError(
                f"Local tool name '{local_name}' conflicts with a KATE tool. "
                f"Use a unique name for local tools."
            )

    # Merge tool definitions
    all_tools_openai = kate_tools + local_tool_defs

    # Format for the provider
    all_tools_formatted = adapter.format_tools(all_tools_openai)

    # Copy messages to avoid mutating caller's list
    history = [dict(m) for m in messages]

    total_tool_calls = 0
    rounds = 0
    response = None
    max_messages = max_rounds * 10  # cap total message history

    for _ in range(max_rounds):
        rounds += 1

        # Trim oldest non-system messages if history is getting too long
        if len(history) > max_messages:
            system_msgs = [m for m in history if m.get("role") == "system"]
            other_msgs = [m for m in history if m.get("role") != "system"]
            # Keep the most recent messages
            trimmed = other_msgs[-(max_messages - len(system_msgs)):]
            history = system_msgs + trimmed

        response = await adapter.chat(history, all_tools_formatted, model, **llm_kwargs)
        history.append(adapter.build_assistant_message(response))

        tool_calls = adapter.extract_tool_calls(response)
        if not tool_calls:
            break

        results: list[tuple[ToolCallInfo, str]] = []
        for tc in tool_calls:
            total_tool_calls += 1
            result_str: str

            if tc.name in local_registry:
                tc.is_local = True
                try:
                    result_str = await _run_local_tool(local_registry[tc.name], tc.arguments)
                except Exception as exc:
                    logger.error("Local tool '%s' failed", tc.name, exc_info=True)
                    result_str = f"Error: tool '{tc.name}' failed with {type(exc).__name__}"
            else:
                tc.is_local = False
                try:
                    tool_result = await projectkate.call_tool(tc.name, tc.arguments)
                    result_str = _result_to_string(tool_result)
                except Exception as exc:
                    logger.error("KATE tool '%s' failed", tc.name, exc_info=True)
                    result_str = f"Error: tool '{tc.name}' failed with {type(exc).__name__}"

            results.append((tc, result_str))

            if on_tool_call is not None:
                try:
                    on_tool_call(tc.name, tc.arguments, result_str)
                except Exception:
                    logger.debug("on_tool_call callback failed", exc_info=True)

        history.extend(adapter.build_tool_result_messages(results))

    content = adapter.extract_text(response) if response is not None else ""

    return ToolLoopResult(
        content=content,
        messages=history,
        tool_calls_made=total_tool_calls,
        rounds=rounds,
        model=model,
    )
