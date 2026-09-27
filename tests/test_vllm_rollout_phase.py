# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

from trl.generation.vllm_generation import VLLMGeneration


@pytest.fixture
def generation(monkeypatch):
    engine = object.__new__(VLLMGeneration)
    engine.mode = "colocate"
    engine.enable_sleep_mode = True
    engine._rollout_depth = 0
    engine._weights_dirty = True
    engine._lora_config = None
    engine._lora_request = None
    engine._llm_weights_sleeping = True
    engine._kv_cache_sleeping = True
    engine.model = torch.nn.Linear(2, 2, bias=False)
    engine._dist = SimpleNamespace(is_fsdp=False, gather_params=lambda params: nullcontext())
    engine.accelerator = SimpleNamespace(is_main_process=True)
    engine.tensor_parallel_size = 1
    engine.temperature = 0.0
    engine.top_p = 1.0
    engine.top_k = -1
    engine.min_p = 0.0
    engine.repetition_penalty = 1.0
    engine.max_completion_length = 2
    engine.logprobs = None
    engine.structured_outputs_regex = None
    engine.generation_kwargs = {}
    engine.llm = Mock()
    engine.llm.generate.return_value = [
        SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[2], logprobs=None)])
    ]
    monkeypatch.setattr("trl.generation.vllm_generation.SamplingParams", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr("trl.generation.vllm_generation.empty_cache", lambda: None)
    return engine


def generate(engine):
    return engine.generate([[1]], images=None, num_generations=1)


def test_unchanged_policy_stays_resident_for_all_turns(generation):
    with generation.rollout_phase():
        for _ in range(4):
            assert generate(generation)[1] == [[2]]
            generation.llm.sleep.assert_not_called()
        assert generation.llm.reset_prefix_cache.call_count == 1
        assert generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights.call_count == 1
        assert generation.llm.wake_up.call_args_list == [call(tags=["weights"]), call(tags=["kv_cache"])]
    generation.llm.sleep.assert_called_once_with(level=2)
    assert generation._llm_weights_sleeping
    assert not generation._weights_dirty  # Sleep discards residency; it does not change the policy.


def test_standalone_generation_and_next_phase_restore_discarded_weights(generation):
    generate(generation)
    generate(generation)
    assert generation.llm.reset_prefix_cache.call_count == 2
    assert generation.llm.sleep.call_count == 2


def test_gpu_tool_handoff_releases_and_restores_memory(generation):
    with generation.rollout_phase():
        generate(generation)
        generation.sleep()
        generation.llm.sleep.assert_called_once_with(level=2)
        assert generation._llm_weights_sleeping and generation._kv_cache_sleeping
        generate(generation)
        generate(generation)
        assert generation.llm.reset_prefix_cache.call_count == 2
    assert generation.llm.sleep.call_count == 2


def test_real_policy_update_invalidates_kv_before_loading(generation):
    with generation.rollout_phase():
        generate(generation)
        generation.llm.reset_mock()
        with torch.no_grad():
            generation.model.weight.add_(1)
        generation.sync_weights()
        generate(generation)
        assert generation.llm.mock_calls[0] == call.reset_prefix_cache()
        assert generation.llm.reset_prefix_cache.call_count == 1
        generation.llm.wake_up.assert_not_called()


@pytest.mark.parametrize("failure", ["weights", "kv_cache", "load", "generate", "tool"])
def test_failure_releases_memory_and_next_phase_can_retry(generation, failure):
    def wake_up(tags):
        if tags == [failure]:
            raise RuntimeError("failed")

    generation.llm.wake_up.side_effect = wake_up
    loader = generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    if failure == "load":
        loader.side_effect = RuntimeError("failed")
    if failure == "generate":
        generation.llm.generate.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"), generation.rollout_phase():
        generate(generation)
        if failure == "tool":
            raise RuntimeError("failed")
    generation.llm.sleep.assert_called_once_with(level=2)
    assert generation._rollout_depth == 0
    generation.llm.wake_up.side_effect = None
    loader.side_effect = None
    generation.llm.generate.side_effect = None
    generate(generation)
    assert generation.llm.sleep.call_count == 2


def test_grpo_custom_rollout_owns_phase_and_cleans_up(generation, make_grpo_trainer):
    trainer = make_grpo_trainer(use_vllm=True)
    trainer.vllm_generation = generation

    def rollout(prompts, trainer):
        generate(generation)
        generate(generation)
        generation.llm.sleep.assert_not_called()
        raise RuntimeError("tool failed")

    trainer.rollout_func = rollout
    with pytest.raises(RuntimeError, match="tool failed"):
        trainer._generate(["prompt"])
    assert generation.llm.reset_prefix_cache.call_count == 1
    generation.llm.sleep.assert_called_once_with(level=2)


@pytest.mark.parametrize("mode, sleep", [("server", True), ("colocate", False)])
def test_phase_without_colocated_sleep_does_not_handoff(generation, mode, sleep):
    generation.mode = mode
    generation.enable_sleep_mode = sleep
    with generation.rollout_phase():
        pass
    generation.llm.sleep.assert_not_called()


def test_grpo_builtin_tool_loop_shares_initial_phase(generation, server_tool_trainer, monkeypatch):
    from collections import defaultdict

    trainer = server_tool_trainer
    trainer.vllm_generation = generation
    trainer.vllm_mode = "colocate"
    trainer.rollout_func = None
    trainer.tools = ["calculator"]
    trainer._metrics = {"train": defaultdict(list)}
    trainer._tokenizer.response_template = "test"
    trainer._tokenize_prompts = lambda prompts: ([[1], [1]], None, {})
    generation.max_completion_length = 32
    generation.llm.llm_engine.model_config.max_model_len = 128

    def decode(tokenizer, ids, prefix):
        if ids[0] in (10, 20):
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"type": "function", "function": {"name": "calculator", "arguments": {"result": 30}}}],
            }
        return {"role": "assistant", "content": str(ids[0])}

    def backend(prompts, **kwargs):
        generation.llm.sleep.assert_not_called()
        return [
            SimpleNamespace(
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[
                    SimpleNamespace(token_ids=[token], logprobs=[{token: SimpleNamespace(rank=1, logprob=-0.1)}])
                ],
            )
            for prompt, token in zip(
                prompts, [10, 20] if len(prompts[0]["prompt_token_ids"]) == 1 else [130, 130], strict=True
            )
        ]

    monkeypatch.setattr("trl.trainer.grpo_trainer.parse_response", decode)
    generation.llm.generate.side_effect = backend
    result = trainer._generate([[{"role": "user", "content": "calculate"}]] * 2)
    assert result[1] == [[10, 30, 130], [20, 30, 130]]
    assert generation.llm.reset_prefix_cache.call_count == 1
    assert generation.llm.generate.call_count == 2
    generation.llm.sleep.assert_called_once_with(level=2)


@pytest.fixture
def peft_generation(generation):
    from contextlib import contextmanager

    peft = pytest.importorskip("peft")
    from peft.tuners.tuners_utils import BaseTunerLayer

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)).to(torch.bfloat16)
    model = peft.get_peft_model(model, peft.LoraConfig(r=2, lora_alpha=2, target_modules=["0", "1"]))
    # Nonzero, non-dyadic adapter updates expose BF16 merge/unmerge rounding drift.
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(917)
        for param in model.parameters():
            param.normal_(mean=0.0, std=0.2)
    generation.model = model
    generation.adapter_layers = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    generation.training_state = {name: value.clone() for name, value in model.state_dict().items()}
    generation.gather_exits = []

    @contextmanager
    def gather(params):
        try:
            yield
        finally:
            assert not any(layer.merged for layer in generation.adapter_layers)
            generation.gather_exits.append(True)

    generation._dist.gather_params = gather
    return generation


def assert_adapter_restored(engine):
    assert not any(layer.merged for layer in engine.adapter_layers)
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, engine.training_state[name], rtol=0, atol=0)
    assert engine.gather_exits


@pytest.fixture
def native_lora(peft_generation, monkeypatch):
    engine = peft_generation
    engine._lora_config = engine.model.peft_config["default"]
    engine._lora_snapshot = None
    engine._lora_version = 0
    engine._base_weights_loaded = True
    monkeypatch.setattr(
        "trl.generation.vllm_generation.LoRARequest",
        lambda name, version, path: SimpleNamespace(lora_name=name, lora_int_id=version, lora_path=path),
        raising=False,
    )
    yield engine
    if engine._lora_snapshot is not None:
        engine._lora_snapshot.cleanup()


def test_native_lora_snapshot_is_immutable_and_versions_are_retired(native_lora):
    from pathlib import Path

    from safetensors.torch import load_file

    engine = native_lora
    with engine.rollout_phase():
        generate(engine)
        first = engine._lora_request
        before = load_file(str(Path(first.lora_path) / "adapter_model.safetensors"))
        with torch.no_grad():
            for name, param in engine.model.named_parameters():
                if "lora_B" in name:
                    param.add_(1)
        generate(engine)
        assert engine.llm.llm_engine.add_lora.call_count == 1
        assert engine.llm.generate.call_args.kwargs["lora_request"] is first
        after = load_file(str(Path(first.lora_path) / "adapter_model.safetensors"))
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        engine.sync_weights()
        second = engine._lora_request
        assert second.lora_int_id != first.lora_int_id
        assert second.lora_name != first.lora_name
        assert not Path(first.lora_path).exists()
        engine.llm.llm_engine.remove_lora.assert_called_once_with(first.lora_int_id)
    engine.llm.sleep.assert_called_once_with(level=1)
    for name, param in engine.model.state_dict().items():
        if "lora_" not in name:
            torch.testing.assert_close(param, engine.training_state[name], rtol=0, atol=0)


def test_native_lora_tool_handoff_restores_without_republication(native_lora):
    engine = native_lora
    with engine.rollout_phase():
        generate(engine)
        engine.sleep()
        generate(engine)
    engine.llm.llm_engine.add_lora.assert_called_once()
    assert engine.llm.wake_up.call_args_list == [call(tags=["weights"]), call(tags=["kv_cache"])] * 2
    assert engine.llm.sleep.call_args_list == [call(level=1), call(level=1)]


@pytest.mark.parametrize("failure", [False, True])
def test_native_lora_initial_base_copy_restores_inference_wrappers(native_lora, monkeypatch, failure):
    import sys

    engine = native_lora

    class NativeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_layer = torch.nn.Linear(4, 4)

    inference = torch.nn.Sequential(NativeLayer(), NativeLayer()).to(torch.bfloat16)
    wrappers = list(inference.children())

    def load(weights):
        for name, weight in weights:
            inference.get_parameter(name).data.copy_(weight)
            if failure:
                raise RuntimeError("base transfer failed")

    inference.load_weights = load
    monkeypatch.setitem(sys.modules, "vllm.lora.layers", SimpleNamespace(BaseLayerWithLoRA=NativeLayer))
    engine.llm.llm_engine.model_executor.driver_worker.model_runner.model = inference
    engine._base_weights_loaded = False
    if failure:
        with pytest.raises(RuntimeError, match="base transfer failed"):
            engine.sync_weights()
        assert not engine._base_weights_loaded
        assert engine._weights_dirty
        engine.llm.llm_engine.add_lora.assert_not_called()
    else:
        engine.sync_weights()
        for index, wrapper in enumerate(wrappers):
            torch.testing.assert_close(
                wrapper.base_layer.weight, engine.model.base_model.model[index].base_layer.weight
            )
        assert engine._base_weights_loaded
    assert list(inference.children()) == wrappers


@pytest.mark.parametrize("options", [{}, {"use_dora": True}, {"bias": "all"}, {"rank_pattern": {"0": 4}}])
def test_native_lora_selection_preserves_other_peft_paths(monkeypatch, options):
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"], **options)
    )
    monkeypatch.setattr(VLLMGeneration, "_init_vllm", lambda self: None)
    monkeypatch.setattr("trl.generation.vllm_generation.is_vllm_available", lambda: True)
    monkeypatch.setattr("trl.generation.vllm_generation.vllm_version", "0.22.0", raising=False)
    engine = VLLMGeneration(model, Accelerator(cpu=True), None)
    assert (engine._lora_config is not None) == (options == {})


@pytest.mark.parametrize("version", ["0.21.0", "0.22.0"])
def test_native_lora_initialization_limits_wrappers_to_targets(monkeypatch, version):
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(4, 4)), LoraConfig(r=2, target_modules=["0"]))
    model.name_or_path = "test-model"
    llm = Mock()
    monkeypatch.setattr("trl.generation.vllm_generation.LLM", llm, raising=False)
    monkeypatch.setattr("trl.generation.vllm_generation.is_vllm_available", lambda: True)
    monkeypatch.setattr("trl.generation.vllm_generation.vllm_version", version, raising=False)
    engine = VLLMGeneration(model, Accelerator(cpu=True), None, enable_sleep_mode=True)
    if version == "0.22.0":
        assert llm.call_args.kwargs["lora_target_modules"] == ["0"]
        assert llm.call_args.kwargs["max_loras"] == 1
        engine.llm.sleep.assert_called_once_with(level=1)
    else:
        assert "enable_lora" not in llm.call_args.kwargs
        engine.llm.sleep.assert_called_once_with(level=2)


@pytest.mark.parametrize("failure", ["save", "load", "retire"])
def test_native_lora_failure_does_not_publish_or_mutate_training(native_lora, monkeypatch, failure):
    from pathlib import Path

    engine = native_lora
    engine.sync_weights()
    original = engine._lora_request
    engine.llm.reset_mock()
    if failure == "save":
        monkeypatch.setattr("safetensors.torch.save_file", Mock(side_effect=RuntimeError("failed")))
    elif failure == "load":
        engine.llm.llm_engine.add_lora.side_effect = RuntimeError("failed")
    else:
        engine.llm.llm_engine.remove_lora.side_effect = [RuntimeError("failed"), True]
    with pytest.raises(RuntimeError, match="failed"), engine.rollout_phase():
        engine.sync_weights()
    assert engine._weights_dirty
    assert engine._lora_request is original
    engine.llm.generate.assert_not_called()
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, engine.training_state[name], rtol=0, atol=0)
    assert Path(original.lora_path).exists()
    if failure == "load":
        failed = engine.llm.llm_engine.add_lora.call_args.args[0]
        assert not Path(failed.lora_path).exists()
        engine.llm.llm_engine.add_lora.side_effect = None
        generate(engine)
        assert engine._lora_request.lora_int_id > failed.lora_int_id


def test_closing_real_peft_export_unmerges_inside_gather(peft_generation):
    engine = peft_generation
    params = engine._iter_named_params()
    next(params)
    assert all(layer.merged for layer in engine.adapter_layers)
    params.close()
    assert_adapter_restored(engine)


@pytest.mark.parametrize("sleep", [False, True])
@pytest.mark.parametrize("failure", ["merge", "before", "during", "after", "cleanup"])
def test_real_peft_failed_export_cannot_generate_until_complete_retry(peft_generation, monkeypatch, sleep, failure):
    engine = peft_generation
    engine.enable_sleep_mode = sleep
    engine._llm_weights_sleeping = sleep
    engine._kv_cache_sleeping = sleep
    expected = {name: value.clone() for name, value in engine._iter_named_params()}
    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    published = {}
    calls = 0

    def load(weights):
        nonlocal calls
        calls += 1
        if failure == "before":
            raise RuntimeError("export failed")
        for name, value in weights:
            published[name] = value.clone()
        if (failure == "during" and calls == 2) or (failure == "after" and len(published) == len(expected)):
            raise RuntimeError("export failed")

    loader.side_effect = load
    original_merge = engine.model.merge_adapter
    original_unmerge = engine.model.unmerge_adapter
    if failure == "merge":

        def merge():
            engine.adapter_layers[0].merge()
            raise RuntimeError("export failed")

        monkeypatch.setattr(engine.model, "merge_adapter", merge)
    elif failure == "cleanup":

        def unmerge():
            original_unmerge()
            raise RuntimeError("export failed")

        monkeypatch.setattr(engine.model, "unmerge_adapter", unmerge)

    for _ in range(2):
        calls = 0
        with pytest.raises(RuntimeError, match="export failed"):
            generate(engine)
        assert_adapter_restored(engine)
        assert engine._weights_dirty
        engine.llm.generate.assert_not_called()

    monkeypatch.setattr(engine.model, "merge_adapter", original_merge)
    monkeypatch.setattr(engine.model, "unmerge_adapter", original_unmerge)
    loader.side_effect = lambda weights: published.update((name, value.clone()) for name, value in weights)
    generate(engine)
    assert_adapter_restored(engine)
    assert not engine._weights_dirty
    engine.llm.generate.assert_called_once()
    assert published.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(published[name], expected[name], rtol=0, atol=0)


def test_server_failed_export_retries_before_generation(peft_generation):
    engine = peft_generation
    engine.mode = "server"
    engine._weight_metadata = None
    retained = []

    def transfer(metadata, params):
        retained.append(params)  # Retaining the iterator must not delay unmerge until garbage collection.
        next(params)
        raise RuntimeError("export failed")

    engine.vllm_client = SimpleNamespace(
        weight_update=nullcontext, update_named_params=transfer, generate=Mock(), reset_prefix_cache=Mock()
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="export failed"):
            generate(engine)
        assert_adapter_restored(engine)
        assert engine._weights_dirty
        engine.vllm_client.generate.assert_not_called()
    assert all(params.gi_frame is None for params in retained)


@pytest.mark.parametrize("failure", ["merge", "before", "during", "after"])
def test_async_merged_failure_restores_adapters_without_publishing(peft_generation, monkeypatch, failure):
    from accelerate import PartialState

    from trl.experimental.async_grpo import AsyncGRPOTrainer

    PartialState(cpu=True)
    engine = peft_generation
    trainer = object.__new__(AsyncGRPOTrainer)
    trainer.model = engine.model
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model, device=torch.device("cpu"), is_main_process=True, wait_for_everyone=Mock()
    )
    trainer.model_version = 7
    trainer.rollout_worker = Mock()
    trainer.weight_transfer = Mock()

    if failure == "merge":

        def merge():
            engine.adapter_layers[0].merge()
            raise RuntimeError("export failed")

        monkeypatch.setattr(engine.model, "merge_adapter", merge)

    def send(params):
        assert all(layer.merged for layer in engine.adapter_layers)
        if failure == "during":
            next(params)
        elif failure == "after":
            list(params)
        raise RuntimeError("export failed")

    trainer.weight_transfer.send_weights.side_effect = send
    with pytest.raises(RuntimeError, match="export failed"):
        trainer._sync_weight_merged(0.0)
    assert not any(layer.merged for layer in engine.adapter_layers)
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, engine.training_state[name], rtol=0, atol=0)
    trainer.weight_transfer.pause.assert_called_once()
    trainer.weight_transfer.resume.assert_not_called()
    trainer.rollout_worker.update_model_version.assert_not_called()
    assert trainer.model_version == 7


def test_online_dpo_failed_export_invalidates_step_and_cache(peft_generation):
    from trl.experimental.online_dpo import OnlineDPOTrainer

    engine = peft_generation
    trainer = object.__new__(OnlineDPOTrainer)
    trainer.model = engine.model
    trainer.accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None), is_main_process=True)
    trainer.is_fsdp_enabled = False
    trainer.vllm_mode = "colocate"
    trainer.llm = engine.llm
    trainer._last_loaded_step = 3
    loader = trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = RuntimeError("export failed")
    with pytest.raises(RuntimeError, match="export failed"):
        trainer._move_model_to_vllm()
    assert trainer._last_loaded_step == -1
    assert trainer.llm.mock_calls[0] == call.reset_prefix_cache()
    assert not any(layer.merged for layer in engine.adapter_layers)
    for name, value in engine.model.state_dict().items():
        torch.testing.assert_close(value, engine.training_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("use_dora", [False, True])
def test_bf16_repeated_exports_preserve_policy_and_publish_adapter_updates(generation, use_dora):
    import copy

    peft = pytest.importorskip("peft")
    from transformers import LlamaConfig, LlamaForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(917)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
            )
        ).to(torch.bfloat16)
        model = peft.get_peft_model(
            model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], use_dora=use_dora)
        ).eval()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.2)
    generation.model = model
    ids = torch.tensor([[1, 5, 9, 3]])
    previous = None
    with torch.no_grad():
        for update in range(2):
            if update:
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.add_(0.07)
            training_state = {name: value.clone() for name, value in model.state_dict().items()}
            logprobs = model(ids).logits.float().log_softmax(-1)
            # A separately merged copy is the reference for exported inference weights. Merged BF16 inference
            # and the unmerged adapter forward need not be bit-identical: their arithmetic differs.
            reference = copy.deepcopy(model).merge_and_unload()
            expected = reference.state_dict()
            for _ in range(8):
                exported = {name: value.clone() for name, value in generation._iter_named_params()}
                for name, value in expected.items():
                    torch.testing.assert_close(exported[name], value, rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, training_state[name], rtol=0, atol=0)
                torch.testing.assert_close(model(ids).logits.float().log_softmax(-1), logprobs, rtol=0, atol=0)
            if update:
                assert not torch.equal(exported["model.layers.0.self_attn.v_proj.weight"], previous)
            previous = exported["model.layers.0.self_attn.v_proj.weight"]


def test_bf16_export_preserves_merged_adapter_bias(generation):
    peft = pytest.importorskip("peft", minversion="0.21.0")
    model = peft.get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(4, 4)).to(torch.bfloat16),
        peft.LoraConfig(r=2, lora_alpha=4, target_modules=["0"], lora_bias=True),
    )
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(917)
        for param in model.parameters():
            param.normal_(std=0.2)
    generation.model = model
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    for _ in range(8):
        list(generation._iter_named_params())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
