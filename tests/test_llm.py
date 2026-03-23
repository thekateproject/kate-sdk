"""Tests for LLM client null safety."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from kate_sdk._llm import AnthropicLLMClient, OpenAILLMClient


@pytest.mark.asyncio
async def test_anthropic_empty_content_raises():
    """Anthropic returning empty content list should raise ValueError."""
    client = AnthropicLLMClient(api_key="sk-test", model="test-model")

    mock_message = MagicMock()
    mock_message.content = []  # No content blocks
    mock_message.model = "test-model"

    mock_anthropic_client = AsyncMock()
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_message)
    client._client = mock_anthropic_client

    with pytest.raises(ValueError, match="no text content"):
        await client.generate("Hello")


@pytest.mark.asyncio
async def test_anthropic_tool_use_only_raises():
    """Anthropic returning only tool_use blocks (no text) should raise ValueError."""
    client = AnthropicLLMClient(api_key="sk-test", model="test-model")

    # Tool use block has no 'text' attribute
    tool_block = MagicMock(spec=["type", "id", "name", "input"])
    tool_block.type = "tool_use"

    mock_message = MagicMock()
    mock_message.content = [tool_block]
    mock_message.model = "test-model"

    mock_anthropic_client = AsyncMock()
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_message)
    client._client = mock_anthropic_client

    with pytest.raises(ValueError, match="no text content"):
        await client.generate("Hello")


@pytest.mark.asyncio
async def test_openai_null_content_raises():
    """OpenAI returning null content (tool call only) should raise ValueError."""
    client = OpenAILLMClient(api_key="sk-test", model="test-model")

    mock_choice = MagicMock()
    mock_choice.message.content = None

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "test-model"

    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai_client

    with pytest.raises(ValueError, match="no text content"):
        await client.generate("Hello")


@pytest.mark.asyncio
async def test_openai_empty_choices_raises():
    """OpenAI returning empty choices list should raise ValueError."""
    client = OpenAILLMClient(api_key="sk-test", model="test-model")

    mock_response = MagicMock()
    mock_response.choices = []
    mock_response.model = "test-model"

    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai_client

    with pytest.raises(ValueError, match="no choices"):
        await client.generate("Hello")


@pytest.mark.asyncio
async def test_openai_null_usage_returns_zeros():
    """OpenAI returning null usage should return zero token counts, not crash."""
    client = OpenAILLMClient(api_key="sk-test", model="test-model")

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello back!"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "test-model"
    mock_response.usage = None

    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai_client

    result = await client.generate("Hello")
    assert result.text == "Hello back!"
    assert result.usage["input_tokens"] == 0
    assert result.usage["output_tokens"] == 0


@pytest.mark.asyncio
async def test_anthropic_happy_path():
    """Normal Anthropic response is correctly extracted."""
    client = AnthropicLLMClient(api_key="sk-test", model="test-model")

    text_block = MagicMock()
    text_block.text = "Generated text"

    mock_message = MagicMock()
    mock_message.content = [text_block]
    mock_message.model = "claude-test"
    mock_message.usage.input_tokens = 10
    mock_message.usage.output_tokens = 20

    mock_anthropic_client = AsyncMock()
    mock_anthropic_client.messages.create = AsyncMock(return_value=mock_message)
    client._client = mock_anthropic_client

    result = await client.generate("Hello")
    assert result.text == "Generated text"
    assert result.model == "claude-test"
    assert result.usage == {"input_tokens": 10, "output_tokens": 20}


@pytest.mark.asyncio
async def test_openai_happy_path():
    """Normal OpenAI response is correctly extracted."""
    client = OpenAILLMClient(api_key="sk-test", model="test-model")

    mock_choice = MagicMock()
    mock_choice.message.content = "Generated text"

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-test"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20

    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
    client._client = mock_openai_client

    result = await client.generate("Hello")
    assert result.text == "Generated text"
    assert result.model == "gpt-test"
    assert result.usage == {"input_tokens": 10, "output_tokens": 20}
