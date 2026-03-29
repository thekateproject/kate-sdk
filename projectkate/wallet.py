"""WalletClient — read-only token balance and ledger access."""

from __future__ import annotations

from typing import TYPE_CHECKING

from projectkate.models import WalletEntry

if TYPE_CHECKING:
    from projectkate.client import KateClient


class WalletClient:
    def __init__(self, client: KateClient) -> None:
        self._client = client

    async def balance(self) -> int:
        """Return the current token balance (most recent ledger entry's balance_after)."""
        entries = await self.ledger(limit=1)
        if not entries:
            return 0
        return entries[0].balance_after

    async def ledger(self, limit: int = 50, offset: int = 0) -> list[WalletEntry]:
        data = await self._client._request(
            "GET", "/auth/me/wallet", params={"limit": limit, "offset": offset}
        )
        items = data if isinstance(data, list) else []
        return [
            WalletEntry(
                id=str(e["id"]),
                user_id=str(e["user_id"]),
                amount=e["amount"],
                balance_after=e["balance_after"],
                entry_type=e.get("entry_type", ""),
                description=e.get("description"),
                created_at=e.get("created_at"),
            )
            for e in items
        ]
