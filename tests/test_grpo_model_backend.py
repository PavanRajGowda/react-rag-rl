"""CPU contract tests; fake engine/weights do not establish CUDA compatibility."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest

from grpo_training.config import TrainingConfig
from grpo_training.model import (
    TrainingModel,
    export_adapter,
    text_lora_targets,
    trainable_names,
)
from grpo_training.records import GenerationRequest, SamplingParameters
from grpo_training.vllm_backend import BackendFailure, VLLMBackend, create_vllm_backend

CONFIG = TrainingConfig()
TARGETS = (
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.3.self_attn.q_proj",
)


class Tokenizer:
    chat_template = "test-template"
    eos_token_id = 0

    def get_vocab(self):
        return {"eos": 0, "x": 1}

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens
        return "".join(chr(i) for i in ids if i)


class Model:
    def __init__(self):
        self.weights = b"fake weights: hash tests only"

    def named_parameters(self):
        return [
            (
                f"base_model.model.{TARGETS[0]}.lora_A.default.weight",
                NS(requires_grad=True),
            ),
            ("model.visual.weight", NS(requires_grad=False)),
        ]

    def save_pretrained(self, path, **kwargs):
        from pathlib import Path

        assert kwargs == {"safe_serialization": True, "selected_adapters": ["default"]}
        path = Path(path)
        (path / "adapter_model.safetensors").write_bytes(self.weights)
        (path / "adapter_config.json").write_text(
            json.dumps(
                {
                    "target_modules": list(TARGETS),
                    "peft_type": "LORA",
                    "bias": "none",
                    "r": CONFIG.lora_rank,
                    "lora_alpha": CONFIG.lora_alpha,
                    "lora_dropout": CONFIG.lora_dropout,
                }
            )
        )


class Engine:
    def __init__(self):
        self.llm_engine = self
        self.loaded = {}
        self.calls = []
        self.reset_count = 0
        self.ack = True
        self.reply = None

    def add_lora(self, request):
        if self.ack:
            self.loaded[request.lora_int_id] = request
        return self.ack

    def list_loras(self):
        return set(self.loaded)

    def remove_lora(self, index):
        del self.loaded[index]
        return True

    def reset_prefix_cache(self):
        self.reset_count += 1

    def sleep(self, **kwargs):
        assert kwargs == {"level": 1}

    def wake_up(self):
        pass

    def generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return self.reply(prompts, kwargs)


@pytest.fixture
def setup(tmp_path):
    bundle = TrainingModel(Model(), Tokenizer(), tmp_path, CONFIG, TARGETS)
    artifact = export_adapter(bundle, tmp_path / "transfers", 0)
    engine = Engine()
    backend = VLLMBackend(
        engine, bundle.tokenizer, CONFIG, [0], sampling_factory=NS, lora_factory=NS
    )
    backend.synchronize(artifact)
    return bundle, artifact, engine, backend


def request(artifact, index=0):
    return GenerationRequest(
        f"episode-{index}",
        "group",
        "question",
        index,
        1,
        42 + index,
        artifact.policy,
        SamplingParameters(),
        "<reasoning>",
        (101, 102 + index),
    )


def result(prompt, lora, text, *, reason="stop", stop="</answer>", ids=None):
    ids = list(map(ord, text)) if ids is None else ids
    output = NS(
        token_ids=ids,
        logprobs=[{i: NS(logprob=-0.5)} for i in ids],
        finish_reason=reason,
        stop_reason=stop,
    )
    return NS(
        prompt_token_ids=prompt["prompt_token_ids"],
        finished=True,
        outputs=[output],
        lora_request=lora,
    )


def normal_reply(prompts, kwargs):
    return [
        result(p, kwargs["lora_request"], "Done.</reasoning><answer>42</answer>")
        for p in prompts
    ]


def test_imports_do_not_load_gpu_libraries():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import grpo_training.model; import grpo_training.vllm_backend; assert not {'torch', 'vllm', 'peft', 'transformers'} & sys.modules.keys()",
        ],
        check=True,
    )


def test_only_explicit_text_projections_are_selected():
    excluded = [
        "model.visual.blocks.0.attn.q_proj",
        "lm_head",
        "model.language_model.embed_tokens",
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
    ]
    assert text_lora_targets([*excluded, *reversed(TARGETS)]) == TARGETS
    with pytest.raises(ValueError, match="both"):
        text_lora_targets(excluded)
    with pytest.raises(ValueError, match="Unexpected"):
        trainable_names([("lm_head.weight", NS(requires_grad=True))])


def test_export_is_content_addressed_and_idempotent(setup):
    bundle, artifact, _, _ = setup
    assert artifact.path.name == artifact.policy.id
    assert export_adapter(bundle, artifact.path.parent, 0) == artifact
    bundle.model.weights += b"update"
    changed = export_adapter(bundle, artifact.path.parent, 1)
    assert changed.policy.adapter_sha256 != artifact.policy.adapter_sha256
    assert artifact.verify() == artifact


def test_same_transfer_identity_rejects_changed_experiment(setup):
    bundle, artifact, _, _ = setup
    bundle.config = replace(bundle.config, learning_rate=2e-5)
    with pytest.raises(ValueError, match="different experiment"):
        export_adapter(bundle, artifact.path.parent, 0)


@pytest.mark.parametrize(
    "filename", ["adapter_model.safetensors", "adapter_config.json", "transfer.json"]
)
def test_artifact_tampering_fails(setup, filename):
    _, artifact, _, _ = setup
    path = artifact.path / filename
    if filename.endswith("json"):
        data = json.loads(path.read_text())
        data["policy" if filename == "transfer.json" else "bias"] = "changed"
        path.write_text(json.dumps(data))
    else:
        path.write_bytes(b"different")
    with pytest.raises(ValueError):
        artifact.verify()


def test_sync_retires_old_adapter_and_rejects_rollback(setup):
    bundle, original, engine, backend = setup
    backend.synchronize(original)
    assert engine.reset_count == 1
    bundle.model.weights += b"updated"
    updated = export_adapter(bundle, original.path.parent, 1)
    backend.synchronize(updated)
    assert set(engine.loaded) == {2}
    assert backend.policy == updated.policy
    with pytest.raises(ValueError, match="newer"):
        backend.synchronize(original)
    with pytest.raises(ValueError, match="stale"):
        backend.generate([request(original)])


def test_failed_weight_ack_blocks_generation(setup):
    bundle, original, engine, backend = setup
    bundle.model.weights += b"updated"
    updated = export_adapter(bundle, original.path.parent, 1)
    engine.ack = False
    with pytest.raises(RuntimeError, match="acknowledge"):
        backend.synchronize(updated)
    assert backend.policy is None
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.generate([request(updated)])


def test_sleep_and_missing_adapter_block_generation(setup):
    _, artifact, engine, backend = setup
    backend.sleep()
    with pytest.raises(RuntimeError, match="awake"):
        backend.generate([request(artifact)])
    with pytest.raises(RuntimeError, match="Wake"):
        backend.synchronize(artifact)
    backend.wake_up()
    engine.loaded.clear()
    with pytest.raises(RuntimeError, match="registered"):
        backend.generate([request(artifact)])


def test_batch_preserves_tokens_probabilities_and_sampling(setup):
    _, artifact, engine, backend = setup
    engine.reply = normal_reply
    requests = [request(artifact, i) for i in range(2)]
    batch = backend.generate(requests)
    assert len(batch.generations) == 2 and batch.seconds >= 0
    for item, expected in zip(batch.generations, requests, strict=True):
        assert item.request == expected
        assert item.finish_reason == "action" and item.engine_segments == 1
        assert item.sampling_logprobs == (-0.5,) * len(item.completion_ids)
    params = engine.calls[0][1]["sampling_params"][0]
    assert (params.logprobs, params.temperature, params.top_p, params.top_k) == (
        0,
        0.7,
        1.0,
        0,
    )
    assert params.include_stop_str_in_output and not params.ignore_eos


@pytest.mark.parametrize(
    "corruption", ["prompt", "adapter", "probability", "count", "empty"]
)
def test_corrupted_engine_output_is_rejected(setup, corruption):
    _, artifact, engine, backend = setup

    def reply(prompts, kwargs):
        output = normal_reply(prompts, kwargs)
        if corruption == "prompt":
            output[0].prompt_token_ids = [999]
        elif corruption == "adapter":
            output[0].lora_request = None
        elif corruption == "probability":
            output[0].outputs[0].logprobs[0] = {}
        elif corruption == "empty":
            output[0].outputs[0].token_ids = []
        else:
            return []
        return output

    engine.reply = reply
    if corruption in ("probability", "empty"):
        failure = backend.generate([request(artifact)]).generations[0]
        assert isinstance(failure, BackendFailure)
        assert isinstance(failure.error, ValueError)
    else:
        with pytest.raises(ValueError):
            backend.generate([request(artifact)])


def test_quoted_closing_tag_resumes_same_generation(setup):
    _, artifact, engine, backend = setup
    first, second = 'Use "</search>', '" format.</reasoning><answer>42</answer>'

    def reply(prompts, kwargs):
        text = first if len(engine.calls) == 1 else second
        return [
            result(
                prompts[0],
                kwargs["lora_request"],
                text,
                stop="</search>" if len(engine.calls) == 1 else "</answer>",
            )
        ]

    engine.reply = reply
    original = request(artifact)
    item = backend.generate([original]).generations[0]
    assert item.engine_segments == 2 and item.finish_reason == "action"
    assert item.completion_ids == tuple(map(ord, first + second))
    assert engine.calls[1][0][0]["prompt_token_ids"] == list(
        original.prompt_ids
    ) + list(map(ord, first))
    assert engine.calls[1][1]["sampling_params"][
        0
    ].max_tokens == CONFIG.max_new_tokens - len(first)


@pytest.mark.parametrize("ending", ["eos", "length"])
def test_eos_and_length_are_not_invented_actions(setup, ending):
    _, artifact, engine, backend = setup
    engine.reply = lambda prompts, kw: [
        result(
            prompts[0],
            kw["lora_request"],
            "",
            ids=[0] if ending == "eos" else [120],
            reason="stop" if ending == "eos" else "length",
            stop=0,
        )
    ]
    item = backend.generate([request(artifact)]).generations[0]
    assert item.finish_reason == ending
    assert item.completion_ids == ((0,) if ending == "eos" else (120,))


def test_duplicate_or_changed_sampling_requests_rejected(setup):
    _, artifact, _, backend = setup
    original = request(artifact)
    with pytest.raises(ValueError, match="Duplicate"):
        backend.generate([original, original])
    with pytest.raises(ValueError, match="sampling"):
        backend.generate(
            [replace(original, sampling=SamplingParameters(temperature=1.0))]
        )


@pytest.mark.parametrize("quantization", [None, "fp8"])
@pytest.mark.parametrize("prefix_caching", [False, True])
@pytest.mark.parametrize("execution_mode", ["eager", "graphs", "compiled"])
@pytest.mark.parametrize("max_num_seqs", [None, 80])
def test_factory_uses_reviewed_runtime_settings(
    setup, monkeypatch, quantization, prefix_caching, execution_mode, max_num_seqs
):
    import grpo_training.vllm_backend as module

    bundle, _, _, _ = setup
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return NS(get_tokenizer=lambda: bundle.tokenizer)

    monkeypatch.setattr(module, "check_runtime", lambda: None)
    monkeypatch.setitem(sys.modules, "vllm", NS(LLM=factory))
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    backend = create_vllm_backend(
        bundle,
        quantization=quantization,
        enable_prefix_caching=prefix_caching,
        execution_mode=execution_mode,
        max_num_seqs=max_num_seqs,
    )
    assert backend.policy is None
    assert captured["model"] == str(bundle.snapshot)
    assert captured["dtype"] == "bfloat16" and captured["quantization"] == quantization
    assert captured["logprobs_mode"] == "processed_logprobs"
    assert captured["enable_sleep_mode"]
    assert captured["enforce_eager"] is (execution_mode == "eager")
    if execution_mode == "eager":
        assert "compilation_config" not in captured
    else:
        assert captured["compilation_config"]["mode"] == (
            0 if execution_mode == "graphs" else 3
        )
        assert captured["compilation_config"]["cudagraph_mode"] == "FULL_DECODE_ONLY"
        assert captured["cudagraph_metrics"] is True
        assert captured["max_num_seqs"] == (max_num_seqs or 16)
        assert captured["compilation_config"]["max_cudagraph_capture_size"] == (
            max_num_seqs or 16
        )
    if max_num_seqs is not None:
        assert captured["max_num_seqs"] == max_num_seqs
    assert captured["enable_prefix_caching"] is prefix_caching
    assert captured["limit_mm_per_prompt"] == {"image": 0, "video": 0}


def test_generation_preserves_prefix_hit_counters(setup):
    _, artifact, engine, backend = setup

    def reply(prompts, kwargs):
        outputs = normal_reply(prompts, kwargs)
        outputs[0].num_cached_tokens = 1
        return outputs

    engine.reply = reply
    batch = backend.generate([request(artifact), request(artifact, 1)])
    assert len(batch.generations) == 2
    assert batch.cache_stats[0]["cached_tokens"] == 1
    assert batch.cache_stats[0]["prompt_tokens"] == 2
    assert batch.cache_stats[1]["cached_tokens"] is None


def test_factory_rejects_tokenizer_disagreement(setup, monkeypatch):
    import grpo_training.vllm_backend as module

    bundle, _, _, _ = setup
    monkeypatch.setattr(module, "check_runtime", lambda: None)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        NS(
            LLM=lambda **kw: NS(
                get_tokenizer=lambda: NS(get_vocab=dict, eos_token_id=0)
            )
        ),
    )
    with pytest.raises(RuntimeError, match="tokenizers"):
        create_vllm_backend(bundle)


def test_loader_pins_snapshot_and_freezes_base_model(tmp_path, monkeypatch):
    import grpo_training.model as module

    snapshot = tmp_path / "snapshots" / CONFIG.model_revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text('{"model_type":"qwen3_5"}')
    calls = {}
    tokenizer = NS(eos_token_id=0, pad_token_id=None, eos_token="eos")

    class TrainModel(Model):
        config = NS(use_cache=True)

        def __init__(self):
            super().__init__()
            self.peft_config = {"default": NS(target_modules={"down_proj", "q_proj"})}

        def named_parameters(self):
            return [
                (
                    f"base_model.model.{name}.lora_A.default.weight",
                    NS(requires_grad=True),
                )
                for name in TARGETS
            ]

        def requires_grad_(self, value):
            calls["base_grad"] = value

        def named_modules(self):
            return [(name, None) for name in TARGETS]

        def gradient_checkpointing_enable(self, **kwargs):
            calls["checkpointing"] = kwargs

        def enable_input_require_grads(self):
            calls["input_grads"] = True

        def train(self):
            calls["training"] = True

    model = TrainModel()

    def download(model_id, **kwargs):
        calls["download"] = (model_id, kwargs)
        return str(snapshot)

    def from_pretrained(path, **kwargs):
        calls["model_load"] = (path, kwargs)
        return model

    def get_peft(base, config):
        assert base is model
        calls["lora"] = config
        return model

    monkeypatch.setattr(module, "check_runtime", lambda: None)
    monkeypatch.setattr(module, "check_optimized_kernels", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        NS(
            cuda=NS(is_available=lambda: True, is_bf16_supported=lambda: True),
            bfloat16="bf16",
            manual_seed=lambda seed: calls.update(seed=seed),
        ),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", NS(snapshot_download=download))
    monkeypatch.setitem(sys.modules, "peft", NS(LoraConfig=NS, get_peft_model=get_peft))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(
            AutoTokenizer=NS(from_pretrained=lambda *a, **kw: tokenizer),
            Qwen3_5ForConditionalGeneration=NS(from_pretrained=from_pretrained),
        ),
    )
    loaded = module.load_training_model(CONFIG)
    assert loaded.snapshot == snapshot and loaded.targets == TARGETS
    assert calls["download"][1]["revision"] == CONFIG.model_revision
    assert calls["base_grad"] is False and calls["input_grads"]
    assert calls["lora"].target_modules == list(TARGETS)
    assert calls["lora"].r == CONFIG.lora_rank
    assert calls["model_load"][1]["dtype"] == "bf16"
    assert calls["model_load"][1]["local_files_only"]
    assert calls["checkpointing"] == {
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
    }
    assert calls["training"] and not model.config.use_cache
    assert tokenizer.pad_token == "eos"


def test_identical_group_prompts_use_separate_sampling_requests(setup):
    _, artifact, engine, backend = setup
    backend.config = replace(CONFIG, temperature=2.0)
    engine.reply = normal_reply
    requests = tuple(
        replace(
            request(artifact, i),
            prompt_ids=(101, 102),
            prompt_text="Same question and initial context<reasoning>",
            sampling=SamplingParameters(temperature=2.0),
        )
        for i in range(4)
    )
    backend.generate(requests)
    call = engine.calls[0]
    parameters = call[1]["sampling_params"]
    assert len(parameters) == 4
    assert len({id(p) for p in parameters}) == 4
    assert {p.seed for p in parameters} == {r.seed for r in requests}
    assert len({p.seed for p in parameters}) == 4
    assert all(p.n == 1 and p.temperature == 2.0 for p in parameters)
    assert all(p.top_p == 1.0 and p.top_k == 0 for p in parameters)
    assert len({tuple(p["prompt_token_ids"]) for p in call[0]}) == 1


@pytest.mark.parametrize("stop", [None, 999])
def test_unrecognized_stop_preserves_diagnostics_without_inventing_eos(setup, stop):
    _, artifact, engine, backend = setup
    engine.reply = lambda prompts, kw: [
        result(prompts[0], kw["lora_request"], "x", reason="stop", stop=stop)
    ]
    failure = backend.generate([request(artifact)]).generations[0]
    assert isinstance(failure, BackendFailure)
    assert "Unrecognized stop" in str(failure.error)
    details = json.loads(str(failure.error).split(": ", 1)[1])
    assert details["episode_id"] == "episode-0"
    assert details["turn"] == 1
    assert details["finish_reason"] == "stop"
    assert details["stop_reason"] == stop
    assert details["tail_token_ids"] == [120]
    assert details["expected_eos_token_ids"] == [0]
    assert details["generated_tokens"] == 1


def test_one_response_failure_keeps_other_batch_generation(setup):
    _, artifact, engine, backend = setup

    def reply(prompts, kw):
        results = normal_reply(prompts, kw)
        results[0].outputs[0].logprobs[0] = {}
        return results

    engine.reply = reply
    batch = backend.generate([request(artifact, 0), request(artifact, 1)])
    assert isinstance(batch.generations[0], BackendFailure)
    assert batch.generations[0].raw_token_count > 0
    assert batch.generations[1].finish_reason == "action"
    assert batch.generations[1].request == request(artifact, 1)


def test_required_optimized_kernels():
    from grpo_training.model import check_optimized_kernels

    check_optimized_kernels(NS(is_fast_path_available=True))
    with pytest.raises(RuntimeError, match="optimized"):
        check_optimized_kernels(NS(is_fast_path_available=False))


@pytest.mark.parametrize("kind", ["unfinished", "multiple", "logprob_shape"])
def test_malformed_single_response_does_not_discard_valid_sibling(setup, kind):
    _, artifact, engine, backend = setup

    def reply(prompts, kwargs):
        results = normal_reply(prompts, kwargs)
        if kind == "unfinished":
            results[0].finished = False
        elif kind == "multiple":
            results[0].outputs *= 2
        else:
            results[0].outputs[0].logprobs = [None] * len(
                results[0].outputs[0].token_ids
            )
        return results

    engine.reply = reply
    batch = backend.generate([request(artifact, 0), request(artifact, 1)])
    assert isinstance(batch.generations[0], BackendFailure)
    assert batch.generations[1].finish_reason == "action"
