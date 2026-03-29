"""Response dataclasses for the Kate Management Client."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Agent:
    id: str
    name: str
    domain: str
    objective: str | None = None
    status: str | None = None
    owner_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None

    # Allow extra keys from API without breaking
    def __post_init__(self) -> None:
        pass


@dataclass
class EvalSummary:
    intelligence_summary: str
    updated_at: str
    overall_score: float | None = None
    recommendations: list[str] = field(default_factory=list)
    regression_detected: bool = False


@dataclass
class DiscoveryConfig:
    agent_id: str
    enabled: bool = False
    budget_tokens: int = 0
    auto_purchase: bool = False
    max_price_tokens: int = 0
    preferred_domains: list[str] = field(default_factory=list)


@dataclass
class DiscoveryAction:
    id: str
    agent_id: str
    action_type: str
    status: str
    summary: dict | None = None
    created_at: str | None = None


@dataclass
class Brief:
    agent_id: str
    version: str
    previous_version: str | None = None
    compiled_at: str | None = None
    brief: str = ""
    gap_summary: list[str] = field(default_factory=list)


@dataclass
class BriefVersion:
    current_version: str
    latest_version: str
    has_update: bool = False
    latest_compiled_at: str | None = None


@dataclass
class BriefDiff:
    from_version: str
    to_version: str
    summary: str = ""
    generated_at: str | None = None


@dataclass
class Artifact:
    id: str
    title: str
    domain: str
    status: str
    user_id: str | None = None
    agent_id: str | None = None
    description: str = ""
    hosting_type: str | None = None
    creation_type: str | None = None
    visibility: str | None = None
    price_tokens: int = 0
    per_query_tokens: int = 0
    version: str = "1.0.0"
    quality_score: float | None = None
    covers_generated: bool = False
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class ArtifactAnalytics:
    subscriber_count: int = 0
    total_queries: int = 0
    tokens_earned: int = 0


@dataclass
class WalletEntry:
    id: str
    user_id: str
    amount: int
    balance_after: int
    entry_type: str
    description: str | None = None
    created_at: str | None = None


@dataclass
class RunSummary:
    id: str
    agent_id: str
    status: str
    trigger: str | None = None
    overall_score: float | None = None
    created_at: str | None = None
    completed_at: str | None = None
