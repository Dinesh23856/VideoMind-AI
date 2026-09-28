"""Observability layer: metrics collection, request tracing, and health probes."""
from .metrics import MetricsCollector, get_metrics_collector

__all__ = ["get_metrics_collector", "MetricsCollector"]
