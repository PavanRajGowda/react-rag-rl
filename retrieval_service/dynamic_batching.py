"""Pure configuration and dispatch for Modal's native search batching."""

import json
import time
from copy import deepcopy
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSettings:
    """Deployment-time settings; provisional defaults, not measured optima."""

    max_batch_size: int = 512
    wait_ms: int = 10

    def __post_init__(self):
        """Stay within the existing Elasticsearch RPC contract."""
        if type(self.max_batch_size) is not int or not 1 <= self.max_batch_size <= 512:
            raise ValueError("max_batch_size must be 1..512")
        if type(self.wait_ms) is not int or not 0 <= self.wait_ms <= 60000:
            raise ValueError("wait_ms must be 0..60000")

    @classmethod
    def from_env(cls, env):
        """Read client/deployer environment, without contacting Modal."""
        return cls(
            int(env.get("SEARCH_DYNAMIC_BATCH_SIZE", "512")),
            int(env.get("SEARCH_DYNAMIC_WAIT_MS", "10")),
        )


async def dispatch(requests, execute):
    """Deduplicate full payloads, await one existing msearch RPC, restore membership.

    execute accepts a list and returns an awaitable ordered response list. Individual
    domain/transport errors stay in their slots. A shared exception becomes a
    structured retryable response for each slot; this function never retries.
    Copies prevent one caller from mutating another's response. No persistent cache.
    """
    if not requests:
        return [], {"requests": 0, "unique_queries": 0, "seconds": 0.0}
    if len(requests) > 512:
        raise ValueError("Dynamic search batch exceeds 512 requests")
    keys = [json.dumps(r, sort_keys=True) for r in requests]
    unique = dict(zip(keys, requests, strict=True))
    started = time.perf_counter()
    try:
        values = await execute(list(unique.values()))
        if len(values) != len(unique):
            raise ValueError("Search RPC changed batch membership")
    except Exception as exc:  # noqa: BLE001 -- serialize shared RPC failures, no retries
        error = {
            "status": "error",
            "code": "search_unavailable",
            "message": f"{type(exc).__name__}: {exc}",
            "retryable": True,
        }
        values = [error for _ in unique]
    by_key = dict(zip(unique, values, strict=True))
    return [deepcopy(by_key[k]) for k in keys], {
        "requests": len(requests),
        "unique_queries": len(unique),
        "seconds": time.perf_counter() - started,
    }


async def dispatch_envelopes(requests, execute, *, entered=None, settings=None):
    """Correlate optional telemetry outside deduplicated queries and model-visible evidence.

    execute accepts the existing list, plus trace_id only for traced batches. The
    returned envelope remains backward compatible for plain and untraced callers.
    Native batching/platform wait occurs before entry and is not timed as execution.
    """
    from uuid import uuid4

    entered = entered or (time.perf_counter_ns(), time.time_ns())
    tagged = ["request_id" in item and "request" in item for item in requests]
    payloads = [
        item["request"] if tag else item
        for item, tag in zip(requests, tagged, strict=True)
    ]
    traced = any(
        tag and item.get("trace") for item, tag in zip(requests, tagged, strict=True)
    )
    batch_id = uuid4().hex if traced else None
    server, took = {}, {}
    stages = {}

    async def forward(unique):
        """Time the downstream RPC including its serialization/queueing/delivery."""
        started = time.perf_counter_ns()
        stages["downstream_started_utc_ns"] = time.time_ns()
        stages["batch_prepare_seconds"] = (started - entered[0]) / 1e9
        try:
            if not traced:
                return await execute(unique)
            result = await execute(unique, trace_id=batch_id)
            server.update(result["timing"])
            values = server.pop("query_took_ms", [])
            for payload, value in zip(unique, values, strict=True):
                took[json.dumps(payload, sort_keys=True)] = value
            return result["responses"]
        finally:
            stages["downstream_rpc_seconds"] = (time.perf_counter_ns() - started) / 1e9
            stages["rpc_finished_ns"] = time.perf_counter_ns()
            stages["downstream_finished_utc_ns"] = time.time_ns()

    responses, metrics = await dispatch(payloads, forward)
    outputs = [
        {"request_id": item["request_id"], "response": response} if tag else response
        for item, tag, response in zip(requests, tagged, responses, strict=True)
    ]
    for item, payload, output, tag in zip(
        requests, payloads, outputs, tagged, strict=True
    ):
        if tag and item.get("trace"):
            output["timing"] = {
                "schema": "search-orchestration-v1",
                "batch_id": batch_id,
                "batch_entered_utc_ns": entered[1],
                "requests": metrics["requests"],
                "unique_queries": metrics["unique_queries"],
                "batch_prepare_seconds": stages.get("batch_prepare_seconds"),
                "downstream_rpc_seconds": stages.get("downstream_rpc_seconds"),
                "downstream_started_utc_ns": stages.get("downstream_started_utc_ns"),
                "downstream_finished_utc_ns": stages.get("downstream_finished_utc_ns"),
                "elasticsearch": server
                | {"query_took_ms": took.get(json.dumps(payload, sort_keys=True))},
            }
    returning, utc = time.perf_counter_ns(), time.time_ns()
    for output in outputs:
        if "timing" in output:
            output["timing"].update(
                batch_returning_utc_ns=utc,
                batch_function_seconds=(returning - entered[0]) / 1e9,
                response_prepare_seconds=(
                    returning - stages.get("rpc_finished_ns", returning)
                )
                / 1e9,
                configured_batch_size=settings.max_batch_size if settings else None,
                configured_wait_ms=settings.wait_ms if settings else None,
            )
    return outputs, metrics
