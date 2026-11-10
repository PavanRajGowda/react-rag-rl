"""CPU-only hosted contracts, including checkpoint resume and token masks."""

import json
from types import SimpleNamespace as NS

import pytest

from tests.agent_fixtures import ToyTokenizer
from tests.rollout_fixtures import inputs
from tests.test_grpo_advantages import make_group
from tinker_training.advantages import prepare_update
from tinker_training.backend import TinkerBackend
from tinker_training.config import TrainingConfig
from tinker_training.loss import loss_metrics, pack_rows
from tinker_training.runner import run


class Tensor:
    def __init__(self, data, **kwargs):
        """Store fake tensor data and accept SDK-compatible constructor metadata."""
        self.data = data

    def tolist(self):
        """Expose fake tensor values as a Python list for metric/packing tests."""
        return self.data


class ModelInput:
    @staticmethod
    def from_ints(ids):
        """Wrap integer input tokens in a minimal SDK-compatible ModelInput stand-in."""
        return ids


TYPES = NS(
    ModelInput=ModelInput,
    TensorData=Tensor,
    Datum=lambda **kw: NS(**kw),
    SamplingParams=lambda **kw: NS(**kw),
    AdamParams=lambda **kw: NS(**kw),
)


class Future:
    def __init__(self, value):
        """Store a deferred fake service result or exception."""
        self.value = value

    def result(self):
        """Return the fake service value or raise the configured error when awaited
        synchronously.
        """
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class Sampler:
    def __init__(self, owner):
        """Bind fake service state and tokenizer for deterministic hosted-sampling tests."""
        self.owner = owner

    async def sample_async(self, prompt, num_samples, sampling_params):
        """Mirror Tinker's awaited sampling interface without network IO."""
        return self.sample(prompt, num_samples, sampling_params).result()

    def sample(self, prompt, num_samples, sampling_params):
        """Return a future containing scripted token/log-probability output; no model or
        network is used.
        """
        self.owner.samples += 1
        text = "".join(chr(i - 1) for i in prompt)
        if "[d] T" in text:
            self.owner.answer_calls = getattr(self.owner, "answer_calls", 0) + 1
            answer = "42" if self.owner.answer_calls % 2 else "43"
            action = f"Done.</reasoning><answer>{answer}</answer>"
        else:
            action = "Find.</reasoning><search>revenue</search>"
        ids = [ord(c) + 1 for c in action]
        return Future(NS(sequences=[NS(tokens=ids, logprobs=[-1.0] * len(ids))]))


class Client:
    def __init__(self, owner, step=0):
        """Bind fake service state and initialize the toy training client."""
        self.owner, self.step = owner, step

    def get_tokenizer(self):
        """Return the toy tokenizer used by both sampling and replay fixtures."""
        return self.owner.tokenizer

    def get_info(self):
        """Return serializable fake hosted-model metadata."""
        return NS(model_dump=lambda **kw: {"model": "fake"})

    def save_state(self, name):
        """Create a fake remote checkpoint including optimizer/update state; may inject a
        save failure.
        """
        if self.owner.fail_save_at == self.step:
            self.owner.fail_save_at = None
            raise RuntimeError("interrupted checkpoint")
        path = f"tinker://fake/{name}"
        self.owner.checkpoints[path] = self.step
        return Future(NS(path=path))

    def save_weights_and_get_sampling_client(self):
        """Return a sampler reflecting the fake client's current policy snapshot."""
        return Sampler(self.owner)

    def forward_backward(self, data, loss_fn, loss_fn_config):
        """Record packed replay Datum inputs and return deterministic current log
        probabilities for local loss checks.
        """
        self.owner.data.append(data)
        assert loss_fn == "ppo"
        assert loss_fn_config["clip_low_threshold"] == 0.8
        return Future(
            NS(
                loss_fn_outputs=[
                    {"logprobs": d.loss_fn_inputs["logprobs"]} for d in data
                ],
                metrics={"loss:sum": 0.0},
            )
        )

    def optim_step(self, params):
        """Advance fake optimizer state and return synthetic metrics for checkpoint/reuse
        tests.
        """
        self.step += 1
        return Future(NS(metrics={}))


class Service:
    def __init__(self):
        """Initialize checkpoint storage and failure controls for the fake Tinker service."""
        self.tokenizer = ToyTokenizer()
        self.tokenizer.eos_token_id = 0
        self.samples, self.data, self.checkpoints, self.restored = 0, [], {}, []
        self.fail_save_at = None

    def create_lora_training_client(self, **kw):
        """Return a fresh fake client while recording requested adapter settings."""
        return Client(self)

    def create_training_client_from_state_with_optimizer(self, path):
        """Restore a fake client from the saved optimizer checkpoint for resume assertions."""
        self.restored.append(path)
        return Client(self, self.checkpoints[path])


def settings():
    """Return a minimal Tinker-compatible TrainingConfig with test-specific overrides."""
    return TrainingConfig(
        temperature=1,
        top_p=1,
        num_generations=2,
        questions_per_update=1,
        num_iterations=2,
        max_steps=2,
        context_mode="replace",
    )


def search(payload):
    """Return deterministic document evidence for the fake hosted training loop."""
    cfg = settings()
    return {
        "status": "ok",
        "corpus_version": cfg.corpus_version,
        "search_schema_version": 6,
        "results": [{"document_id": "d", "title": "T", "text": "Fact"}],
    }


def test_loss_alignment_and_global_denominator():
    """Verify loss alignment and global denominator."""
    cfg = settings()
    group = make_group(cfg)
    train = inputs(cfg)
    prepared = prepare_update((group,), (train.labels[0],))
    data = pack_rows(prepared, TYPES)
    for datum, row in zip(data, prepared.rows(), strict=True):
        full = row["input_ids"]
        prefix = len(full) - len(row["target_ids"]) - 1
        assert datum.model_input == list(full[:-1])
        assert datum.loss_fn_inputs["target_tokens"].data == list(full[1:])
        assert datum.loss_fn_inputs["advantages"].data[:prefix] == [0.0] * prefix
        assert datum.loss_fn_inputs["advantages"].data[prefix:] == [
            row["advantage"] / prepared.token_count
        ] * len(row["target_ids"])
    output = NS(
        loss_fn_outputs=[{"logprobs": d.loss_fn_inputs["logprobs"]} for d in data]
    )
    metrics = loss_metrics(prepared, output)
    assert metrics["current_over_old_ratio"]["mean"] == 1
    assert metrics["entropy_nats"]["mean"] is None


def test_training_two_updates_and_idempotent_resume(tmp_path):
    """Verify training two updates and idempotent resume."""
    cfg, service = settings(), Service()
    train = inputs(cfg)
    result = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": 1},
        validation_count=0,
    )
    assert result["step"] == 2
    assert len(service.data) == 2
    assert result["metrics"]["mean_reward"] == 0.5
    assert service.samples == 4
    samples = service.samples
    resumed = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": 1},
        validation_count=0,
    )
    assert resumed["step"] == 2 and service.samples == samples
    assert service.restored == [result["checkpoint"]]
    index = json.loads((tmp_path / "observability/index.json").read_text())
    assert index["committed_step"] == 2
    assert (tmp_path / "steps.csv").exists()
    assert result["metrics"]["gradient_norm"] is None
    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "observability/validation").exists()


def test_resume_after_optimizer_failure_regenerates_from_checkpoint(tmp_path):
    """Lost uncheckpointed updates regenerate their in-memory pool on resume."""
    cfg, service = settings(), Service()
    train = inputs(cfg)
    service.fail_save_at = 2
    with pytest.raises(RuntimeError, match="interrupted checkpoint"):
        run(
            cfg,
            train,
            train,
            tmp_path,
            service,
            search,
            types=TYPES,
            identity={"test": 1},
            validation_count=0,
        )
    committed = json.loads((tmp_path / "progress.json").read_text())
    assert committed["step"] == 0 and committed["pending"] is None
    samples = service.samples
    result = run(
        cfg,
        train,
        train,
        tmp_path,
        service,
        search,
        types=TYPES,
        identity={"test": 1},
        validation_count=0,
    )
    assert result["step"] == 2 and service.samples > samples
    assert service.restored == [committed["checkpoint"]]


def test_partial_group_keeps_original_sample_indices():
    """Verify partial group keeps original sample indices."""
    from tinker_training.replay import QuestionGroup

    cfg = TrainingConfig(num_generations=4)
    group = make_group(
        cfg, answers=("42", "9", "42", "9"), lengths=((1,), (1,), (1,), (1,))
    )
    partial = QuestionGroup((group.episodes[0], group.episodes[3]), allow_partial=True)
    prepared = prepare_update((partial,), (inputs(cfg).labels[0],))
    assert len(prepared.signals) == 2
    assert {r["sample_index"] for r in prepared.rows()} == {0, 3}


def test_trajectory_credit_is_broadcast_and_nonfinite_rejected():
    """One trajectory coefficient reaches all completion targets; NaN is rejected."""
    row = {
        "input_ids": (10, 11, 12, 13),
        "target_ids": (12, 13),
        "advantage": 2.0,
        "sampling_logprobs": (0.0, 0.0, -1.0, -2.0),
    }
    prepared = NS(rows=lambda: [row], token_count=2)
    datum = pack_rows(prepared, TYPES)[0]
    assert datum.loss_fn_inputs["advantages"].data == [0.0, 1.0, 1.0]
    row["advantage"] = float("nan")
    with pytest.raises(ValueError):
        pack_rows(prepared, TYPES)


def test_backend_submits_whole_batch_before_wait_and_isolates_failure():
    """Verify backend submits whole batch before wait and isolates failure."""
    from tinker_training.records import GenerationRequest, SamplingParameters

    cfg, tokenizer = settings(), ToyTokenizer()
    tokenizer.eos_token_id = 0
    submitted = []

    class CheckFuture(Future):
        def result(self):
            """Assert all sample submissions occurred before the scheduler waits for any
            result, then return the fixture response.
            """
            assert len(submitted) == 2
            return super().result()

    def sample(**kwargs):
        """Return a future containing scripted token/log-probability output; no model or
        network is used.
        """
        submitted.append(kwargs)
        if len(submitted) == 1:
            return CheckFuture(TimeoutError("temporary"))
        text = "Done.</reasoning><answer>42</answer>ignored"
        return CheckFuture(
            NS(
                sequences=[
                    NS(
                        tokens=tokenizer.encode(text, add_special_tokens=False),
                        logprobs=[-1.0] * len(text),
                    )
                ]
            )
        )

    backend = TinkerBackend(NS(sample=sample), tokenizer, cfg, types=TYPES)
    requests = [
        GenerationRequest(
            str(i),
            str(i),
            str(i),
            0,
            1,
            42 + i,
            backend.policy,
            SamplingParameters(1.0, 1.0),
            "prompt<reasoning>",
            (1, 2),
        )
        for i in range(2)
    ]
    batch = backend.generate(requests)
    assert all(
        p["sampling_params"].stop == ["</search>", "</answer>"] for p in submitted
    )
    assert isinstance(batch.generations[0].error, TimeoutError)
    assert tokenizer.decode(
        batch.generations[1].completion_ids, skip_special_tokens=True
    ).endswith("</answer>")
    assert batch.generations[1].request == requests[1]
    assert batch.cache_stats[0]["returned_sample_tokens"] > len(
        batch.generations[1].completion_ids
    )


def test_sdk_types_without_authentication():
    """Verify sdk types without authentication."""
    sdk = pytest.importorskip("tinker")
    row = {
        "input_ids": (10, 11, 12),
        "target_ids": (12,),
        "advantage": 2.0,
        "sampling_logprobs": (0.0, 0.0, -1.0),
    }
    datum = pack_rows(NS(rows=lambda: [row], token_count=1), sdk.types)[0]
    assert datum.loss_fn_inputs["advantages"].tolist() == [0.0, 2.0]
