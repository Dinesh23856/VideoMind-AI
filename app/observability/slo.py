"""SLO, Latency Percentiles, Error Budget, and Cost Tracking."""

from __future__ import annotations

import statistics
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SLOHealthReport(BaseModel):
    service_name: str
    target_availability_percent: float
    current_availability_percent: float
    total_requests: int
    error_count: int
    error_budget_remaining_percent: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    status: str  # HEALTHY, WARNING, BREACHED


class SLOMonitor:
    """Calculates rolling SLO compliance, latency distributions, and error budget consumption."""

    def __init__(
        self,
        service_name: str = "VideoMind-Core",
        target_availability: float = 99.5,
    ) -> None:
        self.service_name = service_name
        self.target_availability = target_availability
        self._latencies_ms: List[float] = []
        self._total_requests = 0
        self._error_requests = 0

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        self._total_requests += 1
        self._latencies_ms.append(latency_ms)
        if is_error:
            self._error_requests += 1
        # Keep window of recent requests
        if len(self._latencies_ms) > 10000:
            self._latencies_ms.pop(0)

    def get_health_report(self) -> SLOHealthReport:
        if not self._total_requests:
            return SLOHealthReport(
                service_name=self.service_name,
                target_availability_percent=self.target_availability,
                current_availability_percent=100.0,
                total_requests=0,
                error_count=0,
                error_budget_remaining_percent=100.0,
                p50_latency_ms=0.0,
                p95_latency_ms=0.0,
                p99_latency_ms=0.0,
                status="HEALTHY",
            )

        avail = ((self._total_requests - self._error_requests) / self._total_requests) * 100.0
        allowed_error_ratio = (100.0 - self.target_availability) / 100.0
        allowed_errors = max(1, int(self._total_requests * allowed_error_ratio))
        budget_remaining = max(0.0, ((allowed_errors - self._error_requests) / allowed_errors) * 100.0)

        # Percentiles
        sorted_lats = sorted(self._latencies_ms)
        n = len(sorted_lats)
        p50 = sorted_lats[int(n * 0.50)]
        p95 = sorted_lats[min(n - 1, int(n * 0.95))]
        p99 = sorted_lats[min(n - 1, int(n * 0.99))]

        if avail < self.target_availability:
            status = "BREACHED"
        elif budget_remaining < 20.0:
            status = "WARNING"
        else:
            status = "HEALTHY"

        return SLOHealthReport(
            service_name=self.service_name,
            target_availability_percent=self.target_availability,
            current_availability_percent=round(avail, 3),
            total_requests=self._total_requests,
            error_count=self._error_requests,
            error_budget_remaining_percent=round(budget_remaining, 2),
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            p99_latency_ms=round(p99, 2),
            status=status,
        )


class CostAllocationTracker:
    """Tracks and forecasts multi-tenant AI inference spend."""

    def __init__(self) -> None:
        self._tenant_spend: Dict[str, float] = {}
        self._feature_spend: Dict[str, float] = {}

    def record_spend(self, tenant_id: str, feature_name: str, cost_usd: float) -> None:
        self._tenant_spend[tenant_id] = self._tenant_spend.get(tenant_id, 0.0) + cost_usd
        self._feature_spend[feature_name] = self._feature_spend.get(feature_name, 0.0) + cost_usd

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_spend_usd": round(sum(self._tenant_spend.values()), 4),
            "by_tenant": {k: round(v, 4) for k, v in self._tenant_spend.items()},
            "by_feature": {k: round(v, 4) for k, v in self._feature_spend.items()},
        }
