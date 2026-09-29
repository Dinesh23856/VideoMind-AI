"""Pipeline DAG Execution Engine with parallel execution, checkpointing, and retries."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class TaskState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class RetryPolicy(BaseModel):
    max_retries: int = 2
    backoff_seconds: float = 0.5
    backoff_multiplier: float = 2.0


class DAGTask:
    def __init__(
        self,
        task_id: str,
        name: str,
        action: Callable[..., Any],
        dependencies: Optional[List[str]] = None,
        retry_policy: Optional[RetryPolicy] = None,
        checkpoint_enabled: bool = True,
    ) -> None:
        self.task_id = task_id
        self.name = name
        self.action = action
        self.dependencies = dependencies or []
        self.retry_policy = retry_policy or RetryPolicy()
        self.checkpoint_enabled = checkpoint_enabled
        self.state = TaskState.PENDING
        self.result: Any = None
        self.error: Optional[str] = None
        self.attempts: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None


class DAGEngine:
    """Directed Acyclic Graph orchestrator for pipeline tasks."""

    def __init__(self) -> None:
        self._tasks: Dict[str, DAGTask] = {}
        self._checkpoints: Dict[str, Any] = {}

    def add_task(self, task: DAGTask) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"Task {task.task_id} already exists in DAG")
        self._tasks[task.task_id] = task

    def get_task(self, task_id: str) -> Optional[DAGTask]:
        return self._tasks.get(task_id)

    def validate_dag(self) -> None:
        """Ensures all dependencies exist and no cycles exist."""
        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                if dep not in self._tasks:
                    raise ValueError(f"Task {tid} depends on unknown task {dep}")

        # Cycle detection via DFS
        visited: Dict[str, int] = {}  # 0: unvisited, 1: visiting, 2: visited

        def dfs(node: str):
            visited[node] = 1
            for dep in self._tasks[node].dependencies:
                if visited.get(dep, 0) == 1:
                    raise ValueError(f"Cyclic dependency detected involving {node} and {dep}")
                if visited.get(dep, 0) == 0:
                    dfs(dep)
            visited[node] = 2

        for tid in self._tasks:
            if visited.get(tid, 0) == 0:
                dfs(tid)

    def get_execution_order(self) -> List[str]:
        self.validate_dag()
        in_degree = {tid: 0 for tid in self._tasks}
        graph = {tid: [] for tid in self._tasks}

        for tid, task in self._tasks.items():
            for dep in task.dependencies:
                graph[dep].append(tid)
                in_degree[tid] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            curr = queue.pop(0)
            order.append(curr)
            for neighbor in graph[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order

    def execute_sync(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronously execute the DAG respecting dependencies, checkpoints, and retries."""
        order = self.get_execution_order()
        ctx = context if context is not None else {}

        for tid in order:
            task = self._tasks[tid]

            # Check if any dependency failed
            dep_failed = any(
                self._tasks[dep].state in (TaskState.FAILED, TaskState.SKIPPED)
                for dep in task.dependencies
            )
            if dep_failed:
                task.state = TaskState.SKIPPED
                task.error = "Upstream dependency failed or was skipped"
                continue

            # Check checkpoint
            if task.checkpoint_enabled and tid in self._checkpoints:
                task.result = self._checkpoints[tid]
                task.state = TaskState.SUCCESS
                ctx[tid] = task.result
                continue

            task.state = TaskState.RUNNING
            task.start_time = time.time()
            success = False

            while task.attempts <= task.retry_policy.max_retries and not success:
                task.attempts += 1
                try:
                    res = task.action(ctx)
                    task.result = res
                    task.state = TaskState.SUCCESS
                    success = True
                    if task.checkpoint_enabled:
                        self._checkpoints[tid] = res
                    ctx[tid] = res
                except Exception as ex:
                    task.error = str(ex)
                    logger.warning(f"Task {tid} failed attempt {task.attempts}: {ex}")
                    if task.attempts <= task.retry_policy.max_retries:
                        time.sleep(task.retry_policy.backoff_seconds)
                    else:
                        task.state = TaskState.FAILED

            task.end_time = time.time()

        return {
            "all_succeeded": all(t.state == TaskState.SUCCESS for t in self._tasks.values()),
            "task_states": {tid: t.state.value for tid, t in self._tasks.items()},
            "context": ctx,
        }
