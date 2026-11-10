"""One round scheduler: batched generation, batched search, repeat."""


def search_many(search, payloads, batch_search=None):
    """Run independent searches and return responses or exceptions in input order.

    Args:
        search: Synchronous single-query callable; may expose a batch method.
        payloads: Query dictionaries in the same order as their waiting tasks.
        batch_search: Optional explicit batch callable overriding search.batch.

    Returns:
        One value per payload. A per-query exception occupies its own slot; a
        shared batch transport/membership error occupies every slot. The caller
        delivers these exceptions to the waiting controller, which owns retries.

    Without a batch callable, queries run sequentially. This helper does not spawn
    threads, classify model mistakes or turn failures into empty search results.
    """
    batch_search = batch_search or getattr(search, "batch", None)
    if batch_search is not None:
        try:
            values = list(batch_search(payloads))
            if len(values) != len(payloads):
                raise ValueError("Search batch membership changed")
            return values
        except Exception as exc:  # noqa: BLE001 -- deliver transport failure to controllers
            return [exc] * len(payloads)
    values = []
    for payload in payloads:
        try:
            values.append(search(payload))
        except Exception as exc:  # noqa: BLE001 -- controllers own retry policy
            values.append(exc)
    return values


def run_tasks(
    tasks,
    backend,
    search,
    *,
    batch_search=None,
    on_event=lambda kind, ready, batch: None,
    generation_errors="raise",
):
    """Drive resumable trajectory tasks without threads or asyncio.

    Tasks yield ("generation", GenerationRequest) and receive (output, batch),
    or yield ("search", payload) and receive a response. Exceptions are thrown
    into the waiting task. Tasks return their caller-specific trajectory result.
    Yields (task, result) as tasks finish. Controllers own parsing, retries and
    reward metadata; this scheduler alone owns batching and response routing.
    """
    tasks = list(tasks)
    # Each task is a suspended trajectory generator. active maps it to exactly
    # one next operation; completed holds return values awaiting delivery.
    active, completed = {}, []

    def advance(task, value=None, error=None, initial=False):
        """Resume one trajectory generator with a value or thrown exception. Updates its
        next pending operation in active; terminal StopIteration values move into
        completed. Unexpected task exceptions propagate to the enclosing scheduler.
        """
        try:
            pending = (
                next(task)
                if initial
                else task.throw(error)
                if error
                else task.send(value)
            )
            if pending[0] not in ("generation", "search"):
                raise ValueError("Unknown trajectory operation")
            active[task] = pending
        except StopIteration as done:
            active.pop(task, None)
            completed.append((task, done.value))

    try:
        for task in tasks:
            advance(task, initial=True)
        while active or completed:
            while completed:
                yield completed.pop(0)
            # Collect the next generation from every ready trajectory into one
            # wave. Finished trajectories leave active and no longer occupy it.
            ready = [
                (task, op[1]) for task, op in active.items() if op[0] == "generation"
            ]
            if ready:
                requests = tuple(request for _, request in ready)
                on_event("generation_started", ready, None)
                try:
                    batch = backend.generate(requests)
                    # Routing by order is safe only when the backend preserves
                    # every request identity, including explicit Failure records.
                    if tuple(g.request for g in batch.generations) != requests:
                        raise ValueError(
                            "Generation backend changed rollout membership/order"
                        )
                except Exception as exc:
                    if generation_errors == "raise":
                        raise
                    for task, _ in ready:
                        advance(task, error=exc)
                else:
                    on_event("searches_started", ready, batch)
                    for (task, _), generation in zip(
                        ready, batch.generations, strict=True
                    ):
                        advance(task, (generation, batch))
                    on_event("generation_batch", ready, batch)
            # Retryable searches may yield another search before the next generation.
            while searching := [
                (task, op[1]) for task, op in active.items() if op[0] == "search"
            ]:
                values = search_many(search, [p for _, p in searching], batch_search)
                for (task, _), value in zip(searching, values, strict=True):
                    advance(task, error=value) if isinstance(
                        value, Exception
                    ) else advance(task, value)
    # Closing suspended tasks also runs their own finally blocks on a shared
    # backend error or when a caller stops consuming the result iterator.
    finally:
        for task in tasks:
            task.close()
