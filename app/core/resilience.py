"""Production Resilience Subsystem: Distributed Locks, Idempotency, Circuit Breakers, and DLQ."""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class LockAcquisitionError(Exception):
    pass


class DistributedLock:
    """Safe token-based distributed lock with automatic TTL expiration."""

    def __init__(self, key: str, ttl_seconds: float = 30.0) -> None:
        self.key = key
        self.ttl = ttl_seconds
        self.token: Optional[str] = None
        self._acquired_at: Optional[float] = None

    def acquire(self, lock_manager: LockManager) -> bool:
        return lock_manager.acquire(self)

    def release(self, lock_manager: LockManager) -> bool:
        return lock_manager.release(self)

    def __enter__(self):
        # Default global in-memory manager
        if not GlobalLockManager.acquire(self):
            raise LockAcquisitionError(f"Could not acquire lock for key: {self.key}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        GlobalLockManager.release(self)


class LockManager:
    """In-memory distributed lock manager (simulates Redis SETNX with TTL)."""

    def __init__(self) -> None:
        self._locks: Dict[str, tuple[str, float]] = {}  # key -> (token, expiry_timestamp)

    def acquire(self, lock: DistributedLock) -> bool:
        now = time.time()
        # Clean expired
        if lock.key in self._locks:
            tok, expiry = self._locks[lock.key]
            if now > expiry:
                del self._locks[lock.key]

        if lock.key not in self._locks:
            token = uuid.uuid4().hex
            self._locks[lock.key] = (token, now + lock.ttl)
            lock.token = token
            lock._acquired_at = now
            return True
        return False

    def release(self, lock: DistributedLock) -> bool:
        if not lock.token or lock.key not in self._locks:
            return False
        stored_token, _ = self._locks[lock.key]
        if stored_token == lock.token:
            del self._locks[lock.key]
            lock.token = None
            return True
        return False


GlobalLockManager = LockManager()


class IdempotencyRecord(BaseModel):
    idempotency_key: str
    status: str = "IN_PROGRESS"  # IN_PROGRESS, COMPLETED, FAILED
    result: Any = None
    created_at: float = Field(default_factory=time.time)
    expires_at: float


class IdempotencyManager:
    """Ensures once-and-only-once execution of critical side-effects."""

    def __init__(self, default_ttl_seconds: float = 300.0) -> None:
        self.default_ttl = default_ttl_seconds
        self._records: Dict[str, IdempotencyRecord] = {}

    def start_or_get(self, key: str) -> tuple[bool, Optional[Any]]:
        """Returns (is_new, cached_result). If not new and COMPLETED, cached_result is returned."""
        now = time.time()
        # Prune if expired
        if key in self._records and now > self._records[key].expires_at:
            del self._records[key]

        if key in self._records:
            rec = self._records[key]
            return False, rec.result

        # Register new
        self._records[key] = IdempotencyRecord(
            idempotency_key=key,
            status="IN_PROGRESS",
            expires_at=now + self.default_ttl,
        )
        return True, None

    def complete(self, key: str, result: Any) -> None:
        if key in self._records:
            self._records[key].status = "COMPLETED"
            self._records[key].result = result


class CircuitBreakerState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    """Fails fast when external model or network dependencies encounter repeated errors."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout_seconds: float = 1.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_seconds
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.success_count = 0

    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        now = time.time()

        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time) >= self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN. Fast failing request.")

        try:
            res = func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
            return res
        except Exception as ex:
            self.failure_count += 1
            self.last_failure_time = now
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
            raise ex


class DLQMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: f"dlq_{uuid.uuid4().hex[:10]}")
    queue_name: str
    payload: Dict[str, Any]
    error_reason: str
    attempts: int = 1
    timestamp: float = Field(default_factory=time.time)


class DeadLetterQueueManager:
    """Manages undeliverable or repeatedly failing pipeline jobs for inspection and re-drive."""

    def __init__(self) -> None:
        self._messages: Dict[str, DLQMessage] = {}

    def send_to_dlq(self, queue_name: str, payload: Dict[str, Any], error_reason: str, attempts: int = 1) -> DLQMessage:
        msg = DLQMessage(
            queue_name=queue_name,
            payload=payload,
            error_reason=error_reason,
            attempts=attempts,
        )
        self._messages[msg.message_id] = msg
        return msg

    def list_messages(self, queue_name: Optional[str] = None) -> List[DLQMessage]:
        if queue_name:
            return [m for m in self._messages.values() if m.queue_name == queue_name]
        return list(self._messages.values())

    def redrive(self, message_id: str, target_handler: Callable[[Dict[str, Any]], Any]) -> bool:
        if message_id not in self._messages:
            return False
        msg = self._messages[message_id]
        target_handler(msg.payload)
        del self._messages[message_id]
        return True

    def purge(self) -> int:
        count = len(self._messages)
        self._messages.clear()
        return count
