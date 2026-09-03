"""轻量 SSE 事件中心。

每个频道(channel)持有一组订阅者的 asyncio.Queue；
publish 必须在事件循环线程内调用——worker 线程通过
``loop.call_soon_threadsafe(hub.publish, ...)`` 桥接过来。
"""
from __future__ import annotations

import asyncio
from collections import defaultdict

# 单个订阅者的待发事件上限：慢消费者丢最旧事件，避免内存无限增长。
_MAX_QUEUE = 256


class EventHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, channel: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._subscribers[channel].add(queue)
        return queue

    def unsubscribe(self, channel: str, queue: asyncio.Queue) -> None:
        subscribers = self._subscribers.get(channel)
        if subscribers is not None:
            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(channel, None)

    def subscriber_count(self, channel: str) -> int:
        return len(self._subscribers.get(channel, ()))

    def publish(self, channel: str, event: str, data: dict) -> None:
        for queue in list(self._subscribers.get(channel, ())):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait({"event": event, "data": data})
