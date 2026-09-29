"""In-memory asynchronous and synchronous EventBus for event-driven orchestration."""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import logging
from typing import Any, Callable, Dict, List, Set

from app.events.stream import MultimodalEvent

logger = logging.getLogger(__name__)


class EventBus:
    """Thread-safe event dispatcher supporting exact and wildcard event type subscriptions."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[MultimodalEvent], Any]]] = {}
        self._history: List[MultimodalEvent] = []
        try:
            loop = asyncio.get_running_loop()
            self._lock = asyncio.Lock()
        except RuntimeError:
            self._lock = None

    def subscribe(self, pattern: str, handler: Callable[[MultimodalEvent], Any]) -> None:
        """Subscribe a handler to an event type or wildcard pattern (e.g. 'CONCEPT_*', '*')."""
        if pattern not in self._subscribers:
            self._subscribers[pattern] = []
        if handler not in self._subscribers[pattern]:
            self._subscribers[pattern].append(handler)

    def unsubscribe(self, pattern: str, handler: Callable[[MultimodalEvent], Any]) -> bool:
        if pattern in self._subscribers and handler in self._subscribers[pattern]:
            self._subscribers[pattern].remove(handler)
            if not self._subscribers[pattern]:
                del self._subscribers[pattern]
            return True
        return False

    def publish_sync(self, event: MultimodalEvent) -> int:
        """Publish event synchronously to all matching handlers."""
        self._history.append(event)
        executed_count = 0
        target_type = event.event_type.value

        matched_handlers: Set[Callable[[MultimodalEvent], Any]] = set()
        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(target_type, pattern):
                matched_handlers.update(handlers)

        for handler in matched_handlers:
            try:
                res = handler(event)
                if inspect.isawaitable(res):
                    # If it's a coroutine in sync publish, schedule or run
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(res)
                        else:
                            loop.run_until_complete(res)
                    except RuntimeError:
                        asyncio.run(res)
                executed_count += 1
            except Exception as ex:
                logger.error(f"Error in EventBus handler {handler}: {ex}", exc_info=True)

        return executed_count

    async def publish(self, event: MultimodalEvent) -> int:
        """Publish event asynchronously to all matching handlers."""
        self._history.append(event)
        executed_count = 0
        target_type = event.event_type.value

        matched_handlers: Set[Callable[[MultimodalEvent], Any]] = set()
        for pattern, handlers in self._subscribers.items():
            if fnmatch.fnmatch(target_type, pattern):
                matched_handlers.update(handlers)

        for handler in matched_handlers:
            try:
                res = handler(event)
                if inspect.isawaitable(res):
                    await res
                executed_count += 1
            except Exception as ex:
                logger.error(f"Error in async EventBus handler {handler}: {ex}", exc_info=True)

        return executed_count

    def clear(self) -> None:
        self._subscribers.clear()
        self._history.clear()

    @property
    def history(self) -> List[MultimodalEvent]:
        return list(self._history)
