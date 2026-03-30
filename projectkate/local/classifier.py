"""Node classification for eval pipeline spans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from projectkate.local.models import NodeClassification

if TYPE_CHECKING:
    from projectkate._llm import LLMClient


@dataclass
class NodeInfo:
    name: str
    prompt_template: str
    input_keys: list[str]
    output_keys: list[str]


@dataclass
class ClassificationResult:
    node_name: str
    classification: NodeClassification
    confidence: float
    reasoning: str


_CLASSIFICATION_PROMPT = (
    "You are classifying an AI agent node to determine what "
    "evaluation metrics to apply.\n\n"
    "## Classification Options\n\n"
    "1. **summarization** — Condenses longer text into a shorter "
    "summary. Examples: article summarizer, meeting notes generator, "
    "TL;DR node.\n"
    "2. **rag_qa** — Answers questions using retrieved "
    "context/documents. Examples: knowledge base Q&A, "
    "document-grounded chat, RAG pipeline answer node.\n"
    "3. **selection_ranking** — Selects, ranks, or filters items "
    "from a list. Examples: news article selector, recommendation "
    "ranker, top-K picker.\n"
    "4. **extraction** — Extracts structured data from unstructured "
    "text. Examples: entity extractor, field parser, keyword "
    "extractor.\n"
    "5. **transformation** — Pure data transformation with no LLM "
    "reasoning. Examples: format converter, data reshaper, "
    "template filler.\n\n"
    "## Node to Classify\n\n"
    "**Name:** {node_name}\n"
    "**Prompt Template:** {prompt_template}\n"
    "**Input Keys:** {input_keys}\n"
    "**Output Keys:** {output_keys}\n\n"
    "## Instructions\n\n"
    "Analyze the node's prompt template, inputs, and outputs. "
    "Choose the single best classification.\n\n"
    "Respond with JSON only:\n"
    '{{"classification": "<one of: summarization, rag_qa, '
    'selection_ranking, extraction, transformation>", '
    '"confidence": <0.0-1.0>, "reasoning": "<brief explanation>"}}'
)


async def classify_node(
    node: NodeInfo, llm_client: LLMClient
) -> ClassificationResult:
    """Classify a single node. Falls back to EXTRACTION on parse error."""
    if not node.prompt_template:
        return ClassificationResult(
            node_name=node.name,
            classification=NodeClassification.TRANSFORMATION,
            confidence=1.0,
            reasoning="No prompt template — pure transformation node.",
        )

    prompt = _CLASSIFICATION_PROMPT.format(
        node_name=node.name,
        prompt_template=node.prompt_template,
        input_keys=", ".join(node.input_keys),
        output_keys=", ".join(node.output_keys),
    )

    response = await llm_client.generate(prompt, temperature=0.0)

    try:
        text = response.text.strip()
        fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        data = json.loads(text)
        classification = NodeClassification(data["classification"])
        return ClassificationResult(
            node_name=node.name,
            classification=classification,
            confidence=float(data.get("confidence", 0.8)),
            reasoning=data.get("reasoning", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        raw = response.text[:200]
        return ClassificationResult(
            node_name=node.name,
            classification=NodeClassification.EXTRACTION,
            confidence=0.5,
            reasoning=(
                "Failed to parse LLM response, defaulting to "
                f"extraction. Raw: {raw}"
            ),
        )
