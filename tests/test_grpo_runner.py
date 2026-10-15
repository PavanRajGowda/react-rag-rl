"""CPU-only control-flow tests. No Torch, model, live transport or cloud calls."""

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from grpo_training.checkpoints import Progress, RunStore
from grpo_training.collection import collect_groups
from grpo_training.config import TrainingConfig
from grpo_training.data import PreparedSplit, Question, RewardLabel
from grpo_training.environment import RetryPolicy, SearchEpisode
from grpo_training.observability import NullObserver
from grpo_training.records import PolicyVersion
from grpo_training.rollouts import template_digest
from grpo_training.runner import run_identity, run_training
from grpo_training.trainer import CapturedUpdate
from grpo_training.vllm_backend import BackendGeneration, GenerationBatch
from tests.test_grpo_advantages import make_group
from tests.test_grpo_replay import ANSWER, SEARCH, ToyTokenizer


def inputs(config):
    train = PreparedSplit(
        "train",
        tuple(
            Question(
                f"finqa_train_{i}", "Revenue?", "train", i, config.dataset_revision
            )
            for i in range(3)
        ),
        tuple(
            RewardLabel(f"finqa_train_{i}", "42", "diagnostic-only") for i in range(3)
        ),
        (),
        3,
    )
    return train


class Store:
    def __init__(self, root, identity, trace):
        self.root, self.identity, self.trace = root, identity, trace
        self.saved, self.events = [], []

    def save(self, trainer, artifact, progress, pending=None):
        self.trace.append(f"save:{progress.step}")
        self.saved.append((progress, pending))
        return self.root / f"step-{progress.step}"

    def audit(self, event):
        self.events.append(event)


class Backend:
    def __init__(self, trace):
        self.trace = trace

    def sleep(self):
        self.trace.append("sleep")

    def wake_up(self):
        self.trace.append("wake")

    def synchronize(self, artifact):
        self.trace.append(f"sync:{artifact.policy.optimizer_step}")


def setup_run(tmp_path, *, config=None):
    config = config or TrainingConfig(
        num_generations=2, questions_per_update=1, num_iterations=2
    )
    trace = []
    train = inputs(config)
    trainer = NS(
        state=NS(global_step=0), model=NS(zero_grad=lambda **kw: trace.append("zero"))
    )
    bundle = NS(config=config, tokenizer=NS())
    store = Store(tmp_path, run_identity(config, train), trace)

    def export(bundle, root, step):
        trace.append(f"export:{step}")
        return NS(
            policy=PolicyVersion(
                config.model_revision, config.model_revision, "a" * 64, step, "b" * 64
            )
        )

    def collect(
        questions, config, policy, backend, tokenizer, search, round_index, **kwargs
    ):
        trace.append(f"collect:{round_index}")
        return tuple(
            make_group(config, number=q.source_index, policy=policy) for q in questions
        )

    def capture(trainer, prepared, artifact, bundle):
        trace.append("capture")
        vectors = tuple((-2.0,) * len(r["target_ids"]) for r in prepared.rows())
        return CapturedUpdate(
            prepared, vectors, tuple((1.0,) * len(v) for v in vectors)
        )

    def step(trainer, config):
        trainer.state.global_step += 1
        trace.append(f"step:{trainer.state.global_step}")
        return 0.5

    kwargs = {
        "export_fn": export,
        "collect_fn": collect,
        "capture_fn": capture,
        "step_fn": step,
        "backward_fn": lambda *a: trace.append("backward") or 0.25,
        "release_memory": lambda: trace.append("release"),
        "emit": lambda *a, **kw: None,
        "observer": NullObserver(),
    }
    return trainer, bundle, Backend(trace), train, store, kwargs, trace


def test_runner_phase_order_reuse_and_per_step_checkpoints(tmp_path):
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)
    result = run_training(trainer, bundle, backend, train, None, store, **kwargs)
    assert result["status"] == "completed"
    assert [p.step for p, _ in store.saved] == [0, 1, 2, 3]
    assert trace.count("capture") == 2 and trace.count("backward") == 3
    assert [x for x in trace if x.startswith("sync:")] == ["sync:0", "sync:2"]
    assert (
        trace.index("sleep")
        < trace.index("capture")
        < trace.index("backward")
        < trace.index("step:1")
    )
    assert trace.index("save:2") < trace.index("wake") < trace.index("sync:2")
    assert store.saved[-1][0] == Progress(3, 2, 2, 1)
    assert (
        store.saved[-1][1] is not None
    )  # Preserve partial rollout reuse at max_steps.


def test_resume_mid_reuse_does_not_repeat_completed_step(tmp_path):
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)
    original = kwargs["step_fn"]

    def crash(trainer, config):
        if trainer.state.global_step == 1:
            raise RuntimeError("crash before second optimizer step")
        return original(trainer, config)

    with pytest.raises(RuntimeError, match="crash"):
        run_training(
            trainer,
            bundle,
            backend,
            train,
            None,
            store,
            **(kwargs | {"step_fn": crash}),
        )
    progress, pending = store.saved[-1]
    assert progress.step == 1 and pending is not None
    trace.clear()
    run_training(
        trainer,
        bundle,
        backend,
        train,
        None,
        store,
        progress=progress,
        pending=pending,
        **kwargs,
    )
    assert [x for x in trace if x.startswith("step:")] == ["step:2", "step:3"]
    assert [x for x in trace if x.startswith("collect:")] == ["collect:1"]


def test_bad_probability_agreement_refuses_update(tmp_path):
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)
    original = kwargs["capture_fn"]

    def mismatch(*args):
        captured = original(*args)
        return replace(
            captured,
            old_logprobs=tuple((-20.0,) * len(v) for v in captured.old_logprobs),
        )

    with pytest.raises(RuntimeError, match="agreement"):
        run_training(
            trainer,
            bundle,
            backend,
            train,
            None,
            store,
            **(kwargs | {"capture_fn": mismatch}),
        )
    assert [p.step for p, _ in store.saved] == [0]
    assert "backward" not in trace and trainer.state.global_step == 0


def test_failed_optimizer_does_not_create_new_checkpoint(tmp_path):
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)

    def fail(*args):
        raise RuntimeError("optimizer failed")

    with pytest.raises(RuntimeError, match="optimizer failed"):
        run_training(
            trainer, bundle, backend, train, None, store, **(kwargs | {"step_fn": fail})
        )
    assert [p.step for p, _ in store.saved] == [0]
    assert store.events[-1]["kind"] == "run_failed"
    assert trace[-1] == "zero"


def test_complete_resume_performs_no_rollout_or_optimizer_step(tmp_path):
    trainer, bundle, backend, train, store, kwargs, trace = setup_run(tmp_path)
    trainer.state.global_step = bundle.config.max_steps
    run_training(
        trainer,
        bundle,
        backend,
        train,
        None,
        store,
        progress=Progress(step=3),
        **kwargs,
    )
    assert "backward" not in trace and not store.saved and trace[-1] == "sleep"


def test_run_identity_rejects_dev_and_changed_question(tmp_path):
    config = TrainingConfig(num_generations=2, questions_per_update=1)
    train = inputs(config)
    with pytest.raises(ValueError, match="train questions"):
        run_identity(config, replace(train, split="dev"))
    before = run_identity(config, train)
    changed = replace(
        train,
        questions=(replace(train.questions[0], text="changed"), *train.questions[1:]),
    )
    assert run_identity(config, changed) != before
    store = RunStore(tmp_path, "run-1", before)
    with pytest.raises(FileExistsError):
        RunStore(tmp_path, "run-1", before)
    with pytest.raises(ValueError, match="identity changed"):
        RunStore(tmp_path, "run-1", run_identity(config, changed), resume=True)
    assert RunStore(tmp_path, "run-1", before, resume=True).root == store.root
    with pytest.raises(ValueError, match="Run ID"):
        RunStore(tmp_path, "../escape", before)


class ScriptedBackend:
    def __init__(self, tokenizer):
        self.tokenizer, self.batch_sizes = tokenizer, []

    def generate(self, requests):
        self.batch_sizes.append(len(requests))
        items = []
        for request in requests:
            text = (
                SEARCH.replace("revenue", request.episode_id)
                if request.turn == 1
                else ANSWER
            )
            ids = tuple(self.tokenizer.encode(text, add_special_tokens=False))
            items.append(
                BackendGeneration(request, ids, (-2.0,) * len(ids), "action", 1)
            )
        return GenerationBatch(tuple(items), 0.01)


def test_collection_batches_turns_and_retries_whole_failed_question(monkeypatch):
    import grpo_training.collection as module

    config = TrainingConfig(num_generations=2, questions_per_update=2)
    train, tokenizer = inputs(config), ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    backend, events = ScriptedBackend(tokenizer), []
    monkeypatch.setattr(
        module, "SearchEpisode", lambda *a: SearchEpisode(*a, retry=RetryPolicy(1, 0))
    )

    def search(payload):
        if "finqa_train_0-attempt-0" in payload["query"]:
            raise TimeoutError("search unavailable")
        return {
            "status": "ok",
            "corpus_version": config.corpus_version,
            "search_schema_version": 4,
            "results": [
                {
                    "context_id": "evidence",
                    "company_name": "Acme",
                    "text": "42",
                    "report_year": 2015,
                    "page_number": 1,
                }
            ],
        }

    groups = collect_groups(
        train.questions[:2],
        config,
        policy,
        backend,
        tokenizer,
        search,
        0,
        audit=events.append,
    )
    assert all(
        g.scorable and all(e.status == "answered" for e in g.episodes) for g in groups
    )
    assert "attempt-1" in groups[0].episodes[0].group_id
    assert "attempt-0" in groups[1].episodes[0].group_id
    assert backend.batch_sizes == [4, 2, 2, 2]
    assert sum(e["kind"] == "rollout_attempt" for e in events) == 2
    assert "gold_answer" not in str(events)


def test_collection_stops_after_bounded_search_failures(monkeypatch):
    import grpo_training.collection as module
    from grpo_training.replay import GroupRequiresRetry

    config = TrainingConfig(num_generations=2, questions_per_update=1)
    train, tokenizer = inputs(config), ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    monkeypatch.setattr(
        module, "SearchEpisode", lambda *a: SearchEpisode(*a, retry=RetryPolicy(1, 0))
    )

    def fail(payload):
        raise TimeoutError("down")

    backend = ScriptedBackend(tokenizer)
    with pytest.raises(GroupRequiresRetry):
        collect_groups(train.questions[:1], config, policy, backend, tokenizer, fail, 0)
    assert backend.batch_sizes == [2, 2]


def test_collection_generation_exception_is_not_retried():
    config = TrainingConfig(num_generations=2, questions_per_update=1)
    train, tokenizer = inputs(config), ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    calls = []

    def fail(requests):
        calls.append(1)
        raise RuntimeError("engine unhealthy")

    with pytest.raises(RuntimeError, match="unhealthy"):
        collect_groups(
            train.questions[:1],
            config,
            policy,
            NS(generate=fail),
            tokenizer,
            lambda payload: None,
            0,
        )
    assert calls == [1]


def test_collection_context_limits_do_not_generate_tokens():
    config = TrainingConfig(
        num_generations=2,
        questions_per_update=1,
        max_context_tokens=11,
        max_new_tokens=10,
    )
    train, tokenizer = inputs(config), ToyTokenizer()
    policy = PolicyVersion(
        config.model_revision,
        config.model_revision,
        "base",
        0,
        template_digest(tokenizer),
    )
    backend = ScriptedBackend(tokenizer)
    groups = collect_groups(
        train.questions[:1], config, policy, backend, tokenizer, lambda payload: None, 0
    )
    assert not backend.batch_sizes and all(
        e.status == "context_limit" for e in groups[0].episodes
    )


def test_runner_imports_no_training_or_cloud_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import grpo_training.runner; assert not {'torch','trl','vllm','transformers','modal'} & sys.modules.keys()",
        ],
        check=True,
    )
