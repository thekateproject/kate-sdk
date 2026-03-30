"""Auto-generate GEval criteria from classified spans."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate.local.models import NodeClassification

if TYPE_CHECKING:
    from projectkate._llm import LLMClient

_CRITERIA_PROMPT = (
    "You are generating evaluation criteria for a GEval metric "
    "that will judge the quality of an AI agent node's output.\n\n"
    "## Context\n\n"
    "**Node Name:** {node_name}\n"
    "**Node Classification:** {classification}\n"
    "**Node Prompt:**\n{node_prompt}\n\n"
    "## Reference Example\n\n"
    "For a news selection node with the prompt "
    '"Select the top 5 most relevant articles", '
    "good criteria would be:\n"
    '"Evaluate whether the selected articles demonstrate diversity '
    "across topics and sources, prioritize newsworthiness and "
    "timeliness, and maintain relevance to the user's interests. "
    "The selection should avoid redundancy and cover the most "
    'significant developments."\n\n'
    "## Instructions\n\n"
    "Given the node's prompt above, generate a GEval evaluation "
    "criteria string that defines what 'good' output looks like "
    "for this specific node. The criteria should:\n"
    "- Be specific to what this node does\n"
    "- Define measurable quality dimensions\n"
    "- Be a single paragraph\n\n"
    "Return ONLY the criteria paragraph, no JSON, no quotes, "
    "no extra text."
)


async def generate_criteria(
    node_name: str,
    node_prompt: str,
    classification: NodeClassification,
    llm_client: LLMClient,
) -> str:
    """Generate GEval criteria string from a node's prompt template."""
    prompt = _CRITERIA_PROMPT.format(
        node_name=node_name,
        classification=classification.value,
        node_prompt=node_prompt,
    )

    response = await llm_client.generate(prompt, temperature=0.0)
    return response.text.strip()
