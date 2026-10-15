"""Modal wiring tested with local fakes; no hydration, remote calls or image build."""

import asyncio
from types import SimpleNamespace as NS

import pytest

from grpo_training.modal_app import acquire_writer, release_writer, search_transport


class Registry:
    def __init__(self):
        self.values = {}

    def put(self, key, value, *, skip_if_exists):
        assert skip_if_exists
        if key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def pop(self, key):
        return self.values.pop(key)


def test_writer_lock_is_exclusive_and_owner_checked():
    registry = Registry()
    token = acquire_writer(registry, "run")
    with pytest.raises(RuntimeError, match="writer lock"):
        acquire_writer(registry, "run")
    release_writer(registry, "run", "someone-else")
    assert registry.get("run") == token
    release_writer(registry, "run", token)
    assert registry.get("run") is None


def test_transport_returns_response_and_bounds_wait():
    seen = []

    async def get():
        return {"status": "ok"}

    async def spawn(payload):
        seen.append(payload)
        return NS(get=NS(aio=get))

    service = NS(search=NS(spawn=NS(aio=spawn)))
    assert search_transport(service, timeout=7)({"query": "x"}) == {"status": "ok"}
    assert seen == [{"query": "x"}]


def test_transport_timeout_cancels_call_without_killing_search_container():
    cancelled = []

    async def get():
        await asyncio.sleep(10)

    async def cancel(**kwargs):
        cancelled.append(kwargs)

    async def spawn(payload):
        return NS(get=NS(aio=get), cancel=NS(aio=cancel))

    service = NS(search=NS(spawn=NS(aio=spawn)))
    with pytest.raises(TimeoutError, match="exceeded"):
        search_transport(service, timeout=0.01)({"query": "x"})
    assert cancelled == [{"terminate_containers": False}]


def test_transport_deadline_also_covers_submission():
    async def spawn(payload):
        await asyncio.sleep(10)

    service = NS(search=NS(spawn=NS(aio=spawn)))
    with pytest.raises(TimeoutError, match="exceeded"):
        search_transport(service, timeout=0.01)({"query": "x"})
