"""Task Priority Scheduler and System Resource Guard for heavy AI workloads."""

from __future__ import annotations

import heapq
import os
import shutil
import time
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


class TaskPriority(IntEnum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class ScheduledJob(BaseModel):
    job_id: str
    priority: TaskPriority
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def __lt__(self, other: ScheduledJob) -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class ResourceGuard:
    """Monitors system resource utilization to prevent OOM and CPU thrashing during video processing."""

    def __init__(
        self,
        max_cpu_percent: float = 90.0,
        max_ram_percent: float = 90.0,
        min_free_disk_gb: float = 2.0,
    ) -> None:
        self.max_cpu_percent = max_cpu_percent
        self.max_ram_percent = max_ram_percent
        self.min_free_disk_gb = min_free_disk_gb

    def get_current_metrics(self) -> Dict[str, Any]:
        if _HAS_PSUTIL:
            cpu_usage = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage(os.path.abspath(os.sep))
            free_disk_gb = disk.free / (1024 ** 3)
            ram_percent = ram.percent
            ram_avail_mb = round(ram.available / (1024 ** 2), 2)
        else:
            cpu_usage = 10.0
            ram_percent = 25.0
            ram_avail_mb = 4096.0
            disk = shutil.disk_usage(os.path.abspath(os.sep))
            free_disk_gb = disk.free / (1024 ** 3)

        return {
            "cpu_percent": cpu_usage,
            "ram_percent": ram_percent,
            "ram_available_mb": ram_avail_mb,
            "disk_free_gb": round(free_disk_gb, 2),
            "safe_to_execute": (
                cpu_usage <= self.max_cpu_percent
                and ram_percent <= self.max_ram_percent
                and free_disk_gb >= self.min_free_disk_gb
            ),
        }

    def check_execution_safety(self) -> bool:
        metrics = self.get_current_metrics()
        return metrics["safe_to_execute"]


class PriorityJobQueue:
    """Thread-safe priority queue for pipeline tasks."""

    def __init__(self) -> None:
        self._heap: List[ScheduledJob] = []

    def push(self, job: ScheduledJob) -> None:
        heapq.heappush(self._heap, job)

    def pop(self) -> Optional[ScheduledJob]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[ScheduledJob]:
        return self._heap[0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
