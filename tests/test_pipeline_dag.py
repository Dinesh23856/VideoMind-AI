"""Tests for Pipeline DAG Engine, Priority Queue, Resource Guard, Distributed Lock, Idempotency, Circuit Breaker, and DLQ."""

import time
import pytest
from app.pipeline.dag import DAGEngine, DAGTask, RetryPolicy, TaskState
from app.pipeline.orchestrator import PriorityJobQueue, ResourceGuard, ScheduledJob, TaskPriority
from app.core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    DeadLetterQueueManager,
    DistributedLock,
    GlobalLockManager,
    IdempotencyManager,
    LockAcquisitionError,
    LockManager,
)


def test_dag_engine_execution_and_dependencies():
    engine = DAGEngine()

    # Task A: produce raw audio
    def task_a(ctx):
        return {"audio_frames": 100}

    # Task B: produce visual frames (concurrent with A)
    def task_b(ctx):
        return {"video_fps": 30}

    # Task C: depends on A and B
    def task_c(ctx):
        return {
            "synced": True,
            "frames": ctx["task_a"]["audio_frames"],
            "fps": ctx["task_b"]["video_fps"],
        }

    engine.add_task(DAGTask("task_a", "Extract Audio", task_a))
    engine.add_task(DAGTask("task_b", "Sample Video", task_b))
    engine.add_task(DAGTask("task_c", "Fuse Multimodal", task_c, dependencies=["task_a", "task_b"]))

    order = engine.get_execution_order()
    assert order.index("task_a") < order.index("task_c")
    assert order.index("task_b") < order.index("task_c")

    res = engine.execute_sync()
    assert res["all_succeeded"] is True
    assert res["context"]["task_c"]["synced"] is True
    assert res["context"]["task_c"]["frames"] == 100


def test_dag_retries_and_checkpoint():
    engine = DAGEngine()
    attempts = 0

    def flaky_action(ctx):
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("Temporary network timeout")
        return "recovered_data"

    engine.add_task(
        DAGTask(
            "flaky_task",
            "Flaky Task",
            flaky_action,
            retry_policy=RetryPolicy(max_retries=2, backoff_seconds=0.01),
        )
    )

    res = engine.execute_sync()
    assert res["all_succeeded"] is True
    assert res["context"]["flaky_task"] == "recovered_data"
    assert attempts == 2


def test_priority_job_queue():
    queue = PriorityJobQueue()

    job_low = ScheduledJob(job_id="j_low", priority=TaskPriority.LOW)
    job_crit = ScheduledJob(job_id="j_crit", priority=TaskPriority.CRITICAL)
    job_norm = ScheduledJob(job_id="j_norm", priority=TaskPriority.NORMAL)

    queue.push(job_low)
    queue.push(job_crit)
    queue.push(job_norm)

    assert len(queue) == 3
    # Critical must pop first
    first = queue.pop()
    assert first.job_id == "j_crit"
    second = queue.pop()
    assert second.job_id == "j_norm"
    third = queue.pop()
    assert third.job_id == "j_low"


def test_resource_guard():
    guard = ResourceGuard(max_cpu_percent=100.0, max_ram_percent=100.0, min_free_disk_gb=0.1)
    metrics = guard.get_current_metrics()
    assert "cpu_percent" in metrics
    assert "ram_percent" in metrics
    assert "disk_free_gb" in metrics
    assert guard.check_execution_safety() is True


def test_distributed_lock():
    mgr = LockManager()
    l1 = DistributedLock(key="video_resource_42", ttl_seconds=1.0)
    l2 = DistributedLock(key="video_resource_42", ttl_seconds=1.0)

    assert l1.acquire(mgr) is True
    # Second acquisition fails
    assert l2.acquire(mgr) is False

    # Release l1
    assert l1.release(mgr) is True
    # Now l2 can acquire
    assert l2.acquire(mgr) is True
    assert l2.release(mgr) is True


def test_idempotency_manager():
    mgr = IdempotencyManager(default_ttl_seconds=60.0)

    # First request
    is_new, cached = mgr.start_or_get("req_key_123")
    assert is_new is True
    assert cached is None

    mgr.complete("req_key_123", {"result": "processed_successfully"})

    # Duplicate request with same key
    is_new2, cached2 = mgr.start_or_get("req_key_123")
    assert is_new2 is False
    assert cached2 == {"result": "processed_successfully"}


def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=0.1)

    def failing_api():
        raise ConnectionResetError("API down")

    with pytest.raises(ConnectionResetError):
        cb.call(failing_api)
    with pytest.raises(ConnectionResetError):
        cb.call(failing_api)

    # 3rd attempt should fail fast with CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(failing_api)

    # Wait for recovery
    time.sleep(0.15)

    def successful_api():
        return "api_up"

    # Should succeed and reset to CLOSED
    assert cb.call(successful_api) == "api_up"


def test_dead_letter_queue():
    dlq = DeadLetterQueueManager()

    msg = dlq.send_to_dlq(
        queue_name="video_transcoding",
        payload={"video_id": "v999", "format": "mp4"},
        error_reason="Corrupt video header",
        attempts=3,
    )
    assert msg.message_id.startswith("dlq_")
    assert len(dlq.list_messages()) == 1

    redriven_payload = None

    def handler(payload):
        nonlocal redriven_payload
        redriven_payload = payload

    success = dlq.redrive(msg.message_id, handler)
    assert success is True
    assert redriven_payload["video_id"] == "v999"
    assert len(dlq.list_messages()) == 0
