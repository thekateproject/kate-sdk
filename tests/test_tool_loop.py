"""Tests for the agentic tool-use loop (projectkate.tool_loop)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from projectkate._tool_loop import (
    LocalTool,
    ToolCallInfo,
    ToolLoopResult,
    _AnthropicAdapter,
    _OpenAIAdapter,
    _detect_provider,
    _run_local_tool,
    _to_openai_tool,
    tool_loop,
)
from projectkate.models import ToolResult


# ---------------------------------------------------------------------------
# Mock helpers — simulate OpenAI / Anthropic response objects
# ---------------------------------------------------------------------------

@dataclass
class _OAIFunction:
    name: str
    arguments: str


@dataclass
class _OAIToolCall:
    id: str
    function: _OAIFunction


@dataclass
class _OAIMessage:
    content: str | None
    tool_calls: list[_OAIToolCall] | None = None


@dataclass
class _OAIChoice:
    message: _OAIMessage


@dataclass
class _OAIResponse:
    choices: list[_OAIChoice]


@dataclass
class _AnthTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _AnthToolUseBlock:
    type: str = "tool_use"
    id: str = ""
    name: str = ""
    input: dict | None = None


@dataclass
class _AnthResponse:
    content: list[Any] = None

    def __post_init__(self):
        if self.content is None:
            self.content = []


def _make_openai_text_response(text: str) -> _OAIResponse:
    return _OAIResponse(choices=[_OAIChoice(message=_OAIMessage(content=text))])


def _make_openai_tool_response(tool_calls: list[tuple[str, str, dict]]) -> _OAIResponse:
    """tool_calls: list of (id, name, args_dict)"""
    tcs = [
        _OAIToolCall(id=tc_id, function=_OAIFunction(name=name, arguments=json.dumps(args)))
        for tc_id, name, args in tool_calls
    ]
    return _OAIResponse(choices=[_OAIChoice(message=_OAIMessage(content="", tool_calls=tcs))])


def _make_anthropic_text_response(text: str) -> _AnthResponse:
    return _AnthResponse(content=[_AnthTextBlock(text=text)])


def _make_anthropic_tool_response(tool_uses: list[tuple[str, str, dict]], text: str = "") -> _AnthResponse:
    blocks: list[Any] = []
    if text:
        blocks.append(_AnthTextBlock(text=text))
    for tu_id, name, inp in tool_uses:
        blocks.append(_AnthToolUseBlock(type="tool_use", id=tu_id, name=name, input=inp))
    return _AnthResponse(content=blocks)


class _FakeOpenAICompletions:
    create = AsyncMock()


class _FakeOpenAIChat:
    completions = _FakeOpenAICompletions()


class _FakeAsyncOpenAI:
    """Stub whose __module__ starts with 'openai' for provider detection."""
    chat = _FakeOpenAIChat()

_FakeAsyncOpenAI.__module__ = "openai._client"


class _FakeAnthropicMessages:
    create = AsyncMock()


class _FakeAsyncAnthropic:
    """Stub whose __module__ starts with 'anthropic' for provider detection."""
    messages = _FakeAnthropicMessages()

_FakeAsyncAnthropic.__module__ = "anthropic._client"


def _make_openai_client() -> _FakeAsyncOpenAI:
    client = _FakeAsyncOpenAI()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


def _make_anthropic_client() -> _FakeAsyncAnthropic:
    client = _FakeAsyncAnthropic()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()
    return client


SAMPLE_KATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_song",
            "description": "Analyze a song",
            "parameters": {
                "type": "object",
                "properties": {"song": {"type": "string"}},
                "required": ["song"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# 1. Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    def test_detect_openai(self):
        client = _make_openai_client()
        assert _detect_provider(client) == "openai"

    def test_detect_anthropic(self):
        client = _make_anthropic_client()
        assert _detect_provider(client) == "anthropic"

    def test_detect_unknown_raises(self):
        client = MagicMock()
        client.__class__ = type("SomeClient", (), {"__module__": "some.random.lib"})
        with pytest.raises(TypeError, match="Unsupported LLM client"):
            _detect_provider(client)


# ---------------------------------------------------------------------------
# 2. Tool format conversion
# ---------------------------------------------------------------------------

class TestToolFormatConversion:
    def test_openai_passthrough(self):
        adapter = _OpenAIAdapter(None)
        result = adapter.format_tools(SAMPLE_KATE_TOOLS)
        assert result == SAMPLE_KATE_TOOLS

    def test_anthropic_conversion(self):
        adapter = _AnthropicAdapter(None)
        result = adapter.format_tools(SAMPLE_KATE_TOOLS)
        assert len(result) == 1
        t = result[0]
        assert t["name"] == "analyze_song"
        assert t["description"] == "Analyze a song"
        assert "input_schema" in t
        assert t["input_schema"]["type"] == "object"
        assert "type" not in t  # no OpenAI "type": "function"
        assert "function" not in t


# ---------------------------------------------------------------------------
# 3. No tools available — single LLM call
# ---------------------------------------------------------------------------

class TestNoTools:
    @pytest.mark.asyncio
    async def test_no_tools_single_call(self):
        client = _make_openai_client()
        client.chat.completions.create.return_value = _make_openai_text_response("Hello!")

        with patch("projectkate._tool_loop.tool_loop.__module__", "projectkate._tool_loop"):
            with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=[]):
                result = await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                )

        assert result.content == "Hello!"
        assert result.tool_calls_made == 0
        assert result.rounds == 1


# ---------------------------------------------------------------------------
# 4. Single KATE tool call (OpenAI)
# ---------------------------------------------------------------------------

class TestSingleKateToolOpenAI:
    @pytest.mark.asyncio
    async def test_kate_tool_call_openai(self):
        client = _make_openai_client()
        # Round 1: LLM requests tool call
        # Round 2: LLM returns text
        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("call_1", "analyze_song", {"song": "Summer Vibes"})]),
            _make_openai_text_response("The song has high energy."),
        ]

        mock_tool_result = ToolResult(success=True, output={"energy": 0.9, "mood": "happy"})

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, return_value=mock_tool_result):
                result = await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Analyze Summer Vibes"}],
                )

        assert result.content == "The song has high energy."
        assert result.tool_calls_made == 1
        assert result.rounds == 2


# ---------------------------------------------------------------------------
# 5. Single KATE tool call (Anthropic)
# ---------------------------------------------------------------------------

class TestSingleKateToolAnthropic:
    @pytest.mark.asyncio
    async def test_kate_tool_call_anthropic(self):
        client = _make_anthropic_client()
        client.messages.create.side_effect = [
            _make_anthropic_tool_response([("tu_1", "analyze_song", {"song": "Summer Vibes"})]),
            _make_anthropic_text_response("The song has high energy."),
        ]

        mock_tool_result = ToolResult(success=True, output={"energy": 0.9})

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, return_value=mock_tool_result):
                result = await tool_loop(
                    client, model="claude-sonnet-4-20250514",
                    messages=[
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Analyze Summer Vibes"},
                    ],
                )

        assert result.content == "The song has high energy."
        assert result.tool_calls_made == 1
        assert result.rounds == 2
        # Verify system message was extracted
        call_kwargs = client.messages.create.call_args_list[0]
        assert "system" in call_kwargs.kwargs or call_kwargs[1].get("system")


# ---------------------------------------------------------------------------
# 6. Local tool call
# ---------------------------------------------------------------------------

class TestLocalTool:
    @pytest.mark.asyncio
    async def test_local_tool_invoked(self):
        client = _make_openai_client()

        async def search_playlist(query: str) -> dict:
            return {"results": [{"title": "Summer Vibes", "artist": "DJ Sun"}]}

        local = LocalTool(
            name="search_playlist",
            description="Search playlist",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
            fn=search_playlist,
        )

        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("call_1", "search_playlist", {"query": "summer"})]),
            _make_openai_text_response("Found Summer Vibes by DJ Sun."),
        ]

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=[]):
            result = await tool_loop(
                client, model="gpt-4o",
                messages=[{"role": "user", "content": "Find summer songs"}],
                local_tools=[local],
            )

        assert result.content == "Found Summer Vibes by DJ Sun."
        assert result.tool_calls_made == 1


# ---------------------------------------------------------------------------
# 7. Mixed local + KATE tools
# ---------------------------------------------------------------------------

class TestMixedTools:
    @pytest.mark.asyncio
    async def test_mixed_local_and_kate(self):
        client = _make_openai_client()

        def get_favorites() -> list:
            return ["Summer Vibes", "Ocean Drive"]

        local = LocalTool(
            name="get_favorites",
            description="Get favorite songs",
            parameters={"type": "object", "properties": {}},
            fn=get_favorites,
        )

        mock_tool_result = ToolResult(success=True, output={"energy": 0.85})

        # Round 1: call local tool
        # Round 2: call KATE tool
        # Round 3: final text
        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("c1", "get_favorites", {})]),
            _make_openai_tool_response([("c2", "analyze_song", {"song": "Summer Vibes"})]),
            _make_openai_text_response("Summer Vibes has 0.85 energy!"),
        ]

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, return_value=mock_tool_result):
                result = await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Analyze my fav"}],
                    local_tools=[local],
                )

        assert result.tool_calls_made == 2
        assert result.rounds == 3
        assert "0.85" in result.content


# ---------------------------------------------------------------------------
# 8. Async and sync local tools both work
# ---------------------------------------------------------------------------

class TestAsyncSyncLocalTools:
    @pytest.mark.asyncio
    async def test_sync_local_tool(self):
        def my_sync_fn(x: int) -> int:
            return x * 2

        result = await _run_local_tool(my_sync_fn, {"x": 5})
        assert result == "10"

    @pytest.mark.asyncio
    async def test_async_local_tool(self):
        async def my_async_fn(x: int) -> dict:
            return {"doubled": x * 2}

        result = await _run_local_tool(my_async_fn, {"x": 5})
        assert json.loads(result) == {"doubled": 10}


# ---------------------------------------------------------------------------
# 9. Local tool error handling
# ---------------------------------------------------------------------------

class TestLocalToolError:
    @pytest.mark.asyncio
    async def test_local_tool_error_fed_to_llm(self):
        client = _make_openai_client()

        def broken_tool() -> str:
            raise RuntimeError("disk full")

        local = LocalTool(
            name="broken", description="Broken tool",
            parameters={"type": "object", "properties": {}},
            fn=broken_tool,
        )

        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("c1", "broken", {})]),
            _make_openai_text_response("Sorry, the tool failed."),
        ]

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=[]):
            result = await tool_loop(
                client, model="gpt-4o",
                messages=[{"role": "user", "content": "Do it"}],
                local_tools=[local],
            )

        assert result.tool_calls_made == 1
        # Check that the error was passed as tool result
        tool_result_msg = result.messages[2]  # user -> assistant (tool_call) -> tool result
        assert "disk full" in tool_result_msg["content"]


# ---------------------------------------------------------------------------
# 10. KATE tool error handling
# ---------------------------------------------------------------------------

class TestKateToolError:
    @pytest.mark.asyncio
    async def test_kate_tool_error_fed_to_llm(self):
        client = _make_openai_client()
        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("c1", "analyze_song", {"song": "X"})]),
            _make_openai_text_response("Tool had an error, sorry."),
        ]

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, side_effect=RuntimeError("timeout")):
                result = await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Analyze X"}],
                )

        assert result.tool_calls_made == 1
        assert result.content == "Tool had an error, sorry."


# ---------------------------------------------------------------------------
# 11. Max rounds respected
# ---------------------------------------------------------------------------

class TestMaxRounds:
    @pytest.mark.asyncio
    async def test_max_rounds_exits(self):
        client = _make_openai_client()
        # LLM always returns tool calls — never stops
        client.chat.completions.create.return_value = _make_openai_tool_response(
            [("c1", "analyze_song", {"song": "X"})]
        )

        mock_tool_result = ToolResult(success=True, output="ok")

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, return_value=mock_tool_result):
                result = await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Go"}],
                    max_rounds=3,
                )

        assert result.rounds == 3
        assert result.tool_calls_made == 3


# ---------------------------------------------------------------------------
# 12. on_tool_call callback
# ---------------------------------------------------------------------------

class TestOnToolCallCallback:
    @pytest.mark.asyncio
    async def test_callback_fires(self):
        client = _make_openai_client()
        client.chat.completions.create.side_effect = [
            _make_openai_tool_response([("c1", "analyze_song", {"song": "Y"})]),
            _make_openai_text_response("Done."),
        ]

        mock_tool_result = ToolResult(success=True, output="great")
        callback_log: list[tuple] = []

        def on_call(name, args, result_str):
            callback_log.append((name, args, result_str))

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with patch("projectkate.call_tool", new_callable=AsyncMock, return_value=mock_tool_result):
                await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Go"}],
                    on_tool_call=on_call,
                )

        assert len(callback_log) == 1
        assert callback_log[0][0] == "analyze_song"
        assert callback_log[0][1] == {"song": "Y"}


# ---------------------------------------------------------------------------
# 13. Messages not mutated
# ---------------------------------------------------------------------------

class TestMessagesNotMutated:
    @pytest.mark.asyncio
    async def test_original_messages_unchanged(self):
        client = _make_openai_client()
        client.chat.completions.create.return_value = _make_openai_text_response("Hi")

        original = [{"role": "user", "content": "Hello"}]
        original_copy = [dict(m) for m in original]

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=[]):
            await tool_loop(client, model="gpt-4o", messages=original)

        assert original == original_copy


# ---------------------------------------------------------------------------
# 14. Auto-fetch KATE tools
# ---------------------------------------------------------------------------

class TestAutoFetchTools:
    @pytest.mark.asyncio
    async def test_get_tools_called_when_none(self):
        client = _make_openai_client()
        client.chat.completions.create.return_value = _make_openai_text_response("Hi")

        mock_get = AsyncMock(return_value=SAMPLE_KATE_TOOLS)
        with patch("projectkate.get_tools", mock_get):
            await tool_loop(client, model="gpt-4o", messages=[{"role": "user", "content": "Hi"}])

        mock_get.assert_called_once_with(format="openai")

    @pytest.mark.asyncio
    async def test_get_tools_not_called_when_provided(self):
        client = _make_openai_client()
        client.chat.completions.create.return_value = _make_openai_text_response("Hi")

        mock_get = AsyncMock(return_value=[])
        with patch("projectkate.get_tools", mock_get):
            await tool_loop(
                client, model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                tools=SAMPLE_KATE_TOOLS,
            )

        mock_get.assert_not_called()


# ---------------------------------------------------------------------------
# 15. LocalTool name collision
# ---------------------------------------------------------------------------

class TestNameCollision:
    @pytest.mark.asyncio
    async def test_collision_raises_value_error(self):
        client = _make_openai_client()

        local = LocalTool(
            name="analyze_song",  # same as KATE tool
            description="Duplicate",
            parameters={"type": "object", "properties": {}},
            fn=lambda: None,
        )

        with patch("projectkate.get_tools", new_callable=AsyncMock, return_value=SAMPLE_KATE_TOOLS):
            with pytest.raises(ValueError, match="conflicts with a KATE tool"):
                await tool_loop(
                    client, model="gpt-4o",
                    messages=[{"role": "user", "content": "Hi"}],
                    local_tools=[local],
                )
