"""Metrics collection, latency tracking, token accounting, and Prometheus telemetry."""

from __future__ import annotations

import time
from typing import Dict


class MetricsCollector:
    """Thread-safe telemetry accumulator for API and pipeline observability."""

    def __init__(self):
        self.request_count: int = 0
        self.error_count: int = 0
        self.jobs_started: int = 0
        self.jobs_completed: int = 0
        self.jobs_failed: int = 0
        self.total_tokens_consumed: int = 0
        self.total_cost_usd: float = 0.0
        self.start_time: float = time.time()
        self.stage_durations: Dict[str, float] = {}

    def record_request(self, is_error: bool = False):
        self.request_count += 1
        if is_error:
            self.error_count += 1

    def record_job_start(self):
        self.jobs_started += 1

    def record_job_completion(self, success: bool, cost_usd: float = 0.0):
        if success:
            self.jobs_completed += 1
        else:
            self.jobs_failed += 1
        self.total_cost_usd += cost_usd

    def record_tokens(self, count: int):
        self.total_tokens_consumed += count

    def record_stage_duration(self, stage: str, duration_seconds: float):
        self.stage_durations[stage] = self.stage_durations.get(stage, 0.0) + duration_seconds

    def export_prometheus_text(self) -> str:
        uptime = time.time() - self.start_time
        lines = [
            "# HELP videomind_uptime_seconds Total runtime of the service in seconds.",
            "# TYPE videomind_uptime_seconds gauge",
            f"videomind_uptime_seconds {uptime:.2f}",
            "# HELP videomind_http_requests_total Total HTTP requests received.",
            "# TYPE videomind_http_requests_total counter",
            f"videomind_http_requests_total {self.request_count}",
            "# HELP videomind_http_errors_total Total HTTP errors encountered.",
            "# TYPE videomind_http_errors_total counter",
            f"videomind_http_errors_total {self.error_count}",
            "# HELP videomind_jobs_started_total Total pipeline jobs submitted.",
            "# TYPE videomind_jobs_started_total counter",
            f"videomind_jobs_started_total {self.jobs_started}",
            "# HELP videomind_jobs_completed_total Total pipeline jobs completed successfully.",
            "# TYPE videomind_jobs_completed_total counter",
            f"videomind_jobs_completed_total {self.jobs_completed}",
            "# HELP videomind_jobs_failed_total Total pipeline jobs failed.",
            "# TYPE videomind_jobs_failed_total counter",
            f"videomind_jobs_failed_total {self.jobs_failed}",
            "# HELP videomind_tokens_consumed_total Total LLM tokens consumed.",
            "# TYPE videomind_tokens_consumed_total counter",
            f"videomind_tokens_consumed_total {self.total_tokens_consumed}",
            "# HELP videomind_estimated_cost_usd Estimated processing cost incurred in USD.",
            "# TYPE videomind_estimated_cost_usd gauge",
            f"videomind_estimated_cost_usd {self.total_cost_usd:.4f}",
        ]
        return "\n".join(lines) + "\n"


_global_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _global_metrics
