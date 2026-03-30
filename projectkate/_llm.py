"""Multi-provider LLM client abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from projectkate.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL


@dataclass
class LLMResponse:
    text: str
    model: str
    usage: dict  # {"input_tokens": int, "output_tokens": int}


class LLMClient(ABC):
    """Abstract LLM client. All LLM calls go through this interface."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a single user message and return the response."""


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        import anthropic

        if self._client is None:
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        message = await self._get_client().messages.create(**kwargs)
        text_blocks = [b for b in message.content if hasattr(b, "text")]
        if not text_blocks:
            raise ValueError("LLM returned no text content")
        return LLMResponse(
            text=text_blocks[0].text,
            model=message.model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )


class OpenAILLMClient(LLMClient):
    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        import openai

        if self._client is None:
            self._client = openai.AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.0,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not response.choices:
            raise ValueError("LLM returned no choices")
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            raise ValueError("LLM returned no text content (tool call only?)")
        usage = response.usage
        return LLMResponse(
            text=content,
            model=response.model,
            usage={
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
            },
        )
