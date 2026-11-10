"""Streaming Modal map transport for independent async search trajectories."""

import asyncio
import time
from copy import deepcopy
from uuid import uuid4

from retrieval_service.dataset import SEARCH_APP
from retrieval_service.timing import CONTEXT, finish_timing, payload_digest


class AsyncSearchClient:
    """Route one shared unordered map stream to per-trajectory futures.

    Use as an async context manager, or explicitly call aclose(). Modal owns
    submission batching/backpressure and server batch formation. No retries or
    local batch timer. A map-level failure fails all outstanding futures because
    an untagged transport exception cannot safely identify an individual input.
    """

    def __init__(self, endpoint=None, trace_sink=None):
        """Bind an optional fake endpoint; create the stream lazily on first search."""
        self._endpoint = endpoint
        self._trace_sink = trace_sink
        self._timings = {}
        self._heartbeat_task = None
        self._session_id = uuid4().hex
        self._queue = asyncio.Queue()
        self._pending = {}
        self._worker = None
        self._closed = False
        self._failure = None

    def open_session(self):
        """Create a fresh owned map stream for one rollout pool/event loop."""
        return type(self)(endpoint=self._endpoint, trace_sink=self._trace_sink)

    async def __aenter__(self):
        """Own the map stream for a collection of concurrent trajectory chains."""
        if self._trace_sink is not None:
            self._heartbeat_deadline = asyncio.get_running_loop().time() + 0.05
            self._heartbeat_task = asyncio.create_task(self._heartbeat())
        return self

    async def _heartbeat(self):
        """Measure live event-loop scheduling delays with no remote monitoring RPCs."""
        loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(max(0, self._heartbeat_deadline - loop.time()))
            now = loop.time()
            self._trace_sink.heartbeat(
                {
                    "session_id": self._session_id,
                    "utc_ns": time.time_ns(),
                    "lag_seconds": max(0, now - self._heartbeat_deadline),
                }
            )
            self._heartbeat_deadline = now + 0.05

    async def __aexit__(self, *exc):
        """Stop submission and release the stream on success or interruption."""
        await self.aclose()

    async def _inputs(self):
        """Yield ready requests continuously until the client is closed."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if self._trace_sink is not None:
                row = self._timings.get(item["request_id"])
                if row is None:
                    self._pending.pop(item["request_id"], None)
                    continue
                row["sdk_handoff_ns"] = time.perf_counter_ns()
                row["sdk_handoff_utc_ns"] = time.time_ns()
            yield item

    async def _consume(self):
        """Resolve tagged outputs immediately; never guess identity on stream errors."""
        try:
            async for output in self._endpoint.map.aio(
                self._inputs(), order_outputs=False, return_exceptions=False
            ):
                key = output["request_id"]
                if key not in self._pending:
                    raise ValueError("Map output has unknown or duplicate request ID")
                response = output["response"]
                if self._trace_sink is not None and key in self._timings:
                    row = self._timings[key]
                    row["result_observed_ns"] = time.perf_counter_ns()
                    row["result_observed_utc_ns"] = time.time_ns()
                    row["server"] = output.get("timing")
                    row["server_timing_missing"] = output.get("timing") is None
                future = self._pending.pop(key)
                if not future.done():
                    future.set_result(response)
            if self._pending:
                raise RuntimeError("Map stream ended with unresolved searches")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- fail all unresolved map futures
            self._failure = exc
        finally:
            error = self._failure or RuntimeError("Search map stream closed")
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(error)
            self._pending.clear()

    async def __call__(self, request):
        """Enqueue one tagged search and await only its own response."""
        created_ns = time.perf_counter_ns()
        if self._closed:
            raise RuntimeError("Search client is closed")
        if self._failure is not None:
            raise self._failure
        if self._worker is None:
            if self._endpoint is None:
                import modal

                self._endpoint = modal.Function.from_name(SEARCH_APP, "dynamic_search")
            self._worker = asyncio.create_task(self._consume())
        key = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[key] = future
        item = {"request_id": key, "request": deepcopy(request)}
        if self._trace_sink is not None:
            self._timings[key] = {
                "request_id": key,
                "session_id": self._session_id,
                "payload_sha256": payload_digest(request),
                "created_ns": created_ns,
                "created_utc_ns": time.time_ns(),
                **(CONTEXT.get() or {}),
                "enqueued_ns": time.perf_counter_ns(),
            }
            item["trace"] = True
        self._queue.put_nowait(item)
        try:
            return await future
        except BaseException as exc:
            if self._trace_sink is not None:
                self._timings[key]["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if self._trace_sink is not None:
                row = self._timings.pop(key)
                row["resumed_ns"] = time.perf_counter_ns()
                row["resumed_utc_ns"] = time.time_ns()
                self._trace_sink.record(finish_timing(row))

    async def aclose(self):
        """Cancel outstanding searches and close the owned map iterator; no retries."""
        if self._closed:
            return
        self._closed = True
        self._queue.put_nowait(None)
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)

    async def batch(self, requests):
        """Convenience for experiments: submit unique requests concurrently.

        Results retain caller order; transport failures occupy their own slots.
        This does not promise one physical batch or a particular execution order.
        """
        import json

        keys = [json.dumps(r, sort_keys=True) for r in requests]
        unique = dict(zip(keys, requests, strict=True))
        values = await asyncio.gather(
            *(self(r) for r in unique.values()), return_exceptions=True
        )
        by_key = dict(zip(unique, values, strict=True))
        return [
            deepcopy(by_key[k]) if not isinstance(by_key[k], Exception) else by_key[k]
            for k in keys
        ]
