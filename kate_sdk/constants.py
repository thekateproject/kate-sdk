"""Shared constants for the KATE SDK."""

from __future__ import annotations

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4o"

VALID_TRIGGERS = frozenset({"automatic", "manual", "test"})
DEFAULT_TRIGGER = "manual"

KNOWN_LLM_PROVIDERS = frozenset({"anthropic", "openai"})
