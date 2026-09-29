"""Enterprise Feature Flagging and Canary Control Subsystem."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FeatureFlag(BaseModel):
    name: str
    description: str = ""
    enabled: bool = False
    rollout_percentage: int = 100  # 0 to 100
    allowed_users: List[str] = Field(default_factory=list)
    allowed_tenants: List[str] = Field(default_factory=list)


class FeatureFlagManager:
    """Manages progressive rollouts, feature gating, and instant kill switches."""

    def __init__(self) -> None:
        self._flags: Dict[str, FeatureFlag] = {}
        self._init_defaults()

    def _init_defaults(self) -> None:
        self.register(
            FeatureFlag(
                name="nextgen_agents",
                description="Autonomous multi-agent research and project mode",
                enabled=True,
                rollout_percentage=100,
            )
        )
        self.register(
            FeatureFlag(
                name="cross_media_rag",
                description="Cross-media unified textbook & video grounding",
                enabled=True,
                rollout_percentage=100,
            )
        )
        self.register(
            FeatureFlag(
                name="adaptive_compute",
                description="Active learning compute escalation for high-uncertainty frames",
                enabled=True,
                rollout_percentage=50,
            )
        )

    def register(self, flag: FeatureFlag) -> None:
        self._flags[flag.name] = flag

    def is_enabled(
        self,
        flag_name: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        if flag_name not in self._flags:
            return False

        flag = self._flags[flag_name]
        if not flag.enabled:
            return False

        # Tenant override
        if tenant_id and tenant_id in flag.allowed_tenants:
            return True

        # User override
        if user_id and user_id in flag.allowed_users:
            return True

        # Percentage rollout (deterministic per user)
        if flag.rollout_percentage >= 100:
            return True
        if flag.rollout_percentage <= 0:
            return False

        identifier = user_id or "anonymous"
        h = int(hashlib.md5(f"{flag_name}:{identifier}".encode("utf-8")).hexdigest()[:8], 16)
        return (h % 100) < flag.rollout_percentage
