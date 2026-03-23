"""Framework registry and auto-detection for instrumentation."""

from __future__ import annotations

import importlib
import logging

logger = logging.getLogger(__name__)

INSTRUMENTORS = [
    {
        "name": "openai",
        "detector": "openai",
        "instrumentor": "openinference.instrumentation.openai",
        "class": "OpenAIInstrumentor",
    },
    {
        "name": "anthropic",
        "detector": "anthropic",
        "instrumentor": "openinference.instrumentation.anthropic",
        "class": "AnthropicInstrumentor",
    },
    {
        "name": "langchain",
        "detector": "langchain",
        "instrumentor": "openinference.instrumentation.langchain",
        "class": "LangChainInstrumentor",
    },
    {
        "name": "mistral",
        "detector": "mistralai",
        "instrumentor": "openinference.instrumentation.mistralai",
        "class": "MistralAIInstrumentor",
    },
    {
        "name": "vertexai",
        "detector": "vertexai",
        "instrumentor": "openinference.instrumentation.vertexai",
        "class": "VertexAIInstrumentor",
    },
    {
        "name": "google_genai",
        "detector": "google.generativeai",
        "instrumentor": "openinference.instrumentation.google_generativeai",
        "class": "GoogleGenerativeAIInstrumentor",
    },
    {
        "name": "crewai",
        "detector": "crewai",
        "instrumentor": "openinference.instrumentation.crewai",
        "class": "CrewAIInstrumentor",
    },
]


def detect_installed() -> list[dict]:
    """Return entries where both the framework and instrumentor are installed."""
    found = []
    for entry in INSTRUMENTORS:
        try:
            importlib.import_module(entry["detector"])
        except ImportError:
            continue
        try:
            importlib.import_module(entry["instrumentor"])
        except ImportError:
            logger.debug(
                "Framework '%s' detected but instrumentor '%s' not installed. "
                "Install with: pip install kate-sdk[%s]",
                entry["name"],
                entry["instrumentor"],
                entry["name"],
            )
            continue
        found.append(entry)
    return found


def load_instrumentor(entry: dict):
    """Import and return an instrumentor instance."""
    mod = importlib.import_module(entry["instrumentor"])
    cls = getattr(mod, entry["class"])
    return cls()
