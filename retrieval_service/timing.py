"""Opt-in search timings kept outside query payloads, evidence and reward data."""

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

CONTEXT = ContextVar("search_timing_context", default=None)


@contextmanager
def search_context(episode_id, turn):
    """Associate a search with its agent turn without changing the query contract."""
    token = CONTEXT.set({"episode_id": episode_id, "turn": turn})
    try:
        yield
    finally:
        CONTEXT.reset(token)


def payload_digest(payload):
    """Correlate exact query payloads without writing their text into timing records."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class SearchTrace:
    """Buffer bounded diagnostic-run measurements; export after the run, never per RPC.

    Timers use monotonic clocks for same-host elapsed times. UTC one-way estimates
    include clock skew and never claim to separate SDK processing from network time.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.requests, self.heartbeats = [], []

    def record(self, row):
        """Keep one terminal timing record, including failures and missing boundaries."""
        self.requests.append(row)

    def heartbeat(self, row):
        """Keep one observed scheduling-lag sample from the actual rollout loop."""
        self.heartbeats.append(row)

    def save(self):
        """Persist raw measurements for offline correlation without pooled-percentile subtraction."""
        from baseline_agent.persistence import write_json

        write_json(
            self.directory / "search-timings.json",
            {
                "schema": "search-orchestration-v1",
                "requests": self.requests,
                "heartbeats": self.heartbeats,
                "qualification": "SDK handoff is iterator consumption, not confirmed wire submission. One-way UTC differences include clock skew. Server timings exclude response serialization after function return. Readiness rows have no episode_id. Heartbeat captures local scheduling, not a causal allocation of each wait.",
            },
        )


def elapsed(row, start, end):
    """Return a same-process duration only when both recorded boundaries exist."""
    if start not in row or end not in row:
        return None
    return (row[end] - row[start]) / 1e9


def finish_timing(row):
    """Derive additive client segments and labeled residuals, preserving negative estimates."""
    for name, first, last in (
        ("client_prepare_seconds", "created_ns", "enqueued_ns"),
        ("client_queue_seconds", "enqueued_ns", "sdk_handoff_ns"),
        ("sdk_handoff_to_result_seconds", "sdk_handoff_ns", "result_observed_ns"),
        ("resume_delay_seconds", "result_observed_ns", "resumed_ns"),
        ("client_total_seconds", "created_ns", "resumed_ns"),
    ):
        row[name] = elapsed(row, first, last)
    server = row.get("server") or {}
    total = server.get("batch_function_seconds")
    observed = row.get("sdk_handoff_to_result_seconds")
    if total is not None and observed is not None:
        row["outside_server_function_seconds"] = observed - total
    if "sdk_handoff_utc_ns" in row and "batch_entered_utc_ns" in server:
        row["approx_submission_to_batch_entry_seconds"] = (
            server["batch_entered_utc_ns"] - row["sdk_handoff_utc_ns"]
        ) / 1e9
    if "result_observed_utc_ns" in row and "batch_returning_utc_ns" in server:
        row["approx_server_return_to_client_seconds"] = (
            row["result_observed_utc_ns"] - server["batch_returning_utc_ns"]
        ) / 1e9
    es = server.get("elasticsearch") or {}
    if (
        server.get("downstream_rpc_seconds") is not None
        and es.get("service_total_seconds") is not None
    ):
        row["downstream_outside_service_seconds"] = (
            server["downstream_rpc_seconds"] - es["service_total_seconds"]
        )
    return row


def validate_timing(row):
    """Fail diagnostic readiness before sampling if deployment lacks required stages."""
    server = row.get("server") or {}
    es = server.get("elasticsearch") or {}
    if (
        server.get("schema") != "search-orchestration-v1"
        or not server.get("batch_id")
        or es.get("batch_id") != server.get("batch_id")
        or any(
            es.get(k) is None
            for k in ("service_total_seconds", "msearch_seconds", "format_seconds")
        )
        or any(
            row.get(k) is None
            for k in (
                "client_total_seconds",
                "client_queue_seconds",
                "sdk_handoff_to_result_seconds",
            )
        )
    ):
        raise RuntimeError(
            "Search instrumentation readiness failed; deploy current search code before paid generation"
        )
