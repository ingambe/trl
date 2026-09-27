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
    from accelerate import Accelerator

    monkeypatch.setattr(VLLMGeneration, "_init_vllm", lambda self: None)
    engine = VLLMGeneration(
        torch.nn.Linear(2, 2, bias=False),
        Accelerator(cpu=True),
        None,
        enable_sleep_mode=True,
        temperature=0.0,
        top_k=-1,
        max_completion_length=2,
        logprobs=None,
    )
    engine._llm_weights_sleeping = True
    engine._kv_cache_sleeping = True
    engine.llm = Mock()
    engine.llm.generate.return_value = [
        SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[2], logprobs=None)])
    ]
    monkeypatch.setattr("trl.generation.vllm_generation.SamplingParams", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr("trl.generation.vllm_generation.empty_cache", lambda: None)
    return engine


def generate(engine):
    return engine.generate([[1]], images=None, num_generations=1)


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


@pytest.mark.parametrize("fail_continuation", [False, True])
def test_grpo_builtin_tool_loop_shares_initial_phase(generation, server_tool_trainer, monkeypatch, fail_continuation):
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
        if fail_continuation and len(prompts[0]["prompt_token_ids"]) > 1:
            raise RuntimeError("continuation failed")
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
    with pytest.raises(RuntimeError, match="continuation failed") if fail_continuation else nullcontext():
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

    @contextmanager
    def gather(params):
        try:
            yield
        finally:
            assert not any(layer.merged for layer in generation.adapter_layers)

    generation._dist.gather_params = gather
    return generation


def assert_adapter_restored(engine):
    assert not any(layer.merged for layer in engine.adapter_layers)
    torch.testing.assert_close(engine.model.state_dict(), engine.training_state, rtol=0, atol=0)


def test_closing_real_peft_export_unmerges_inside_gather(peft_generation):
    engine = peft_generation
    with engine._export_named_params() as params:
        next(params)
        assert all(layer.merged for layer in engine.adapter_layers)
    assert params.gi_frame is None
    assert_adapter_restored(engine)


@pytest.mark.parametrize("sleep", [False, True])
@pytest.mark.parametrize("failure", ["merge", "before", "during", "after", "cleanup"])
def test_real_peft_failed_export_cannot_generate_until_complete_retry(peft_generation, monkeypatch, sleep, failure):
    engine = peft_generation
    engine.enable_sleep_mode = sleep
    engine._llm_weights_sleeping = sleep
    engine._kv_cache_sleeping = sleep
    with engine._export_named_params() as params:
        expected = {name: value.clone() for name, value in params}
    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    published = {}

    def load(weights):
        if failure == "before":
            raise RuntimeError("export failed")
        for name, value in weights:
            published[name] = value.clone()
            if failure == "during" and len(published) == 2:
                raise RuntimeError("export failed")
        if failure == "after":
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
    torch.testing.assert_close(published, expected, rtol=0, atol=0)


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


def test_async_partial_export_restores_adapters_without_publishing(peft_generation):
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    engine = peft_generation
    trainer = object.__new__(AsyncGRPOTrainer)
    trainer.model = engine.model
    trainer.accelerator = SimpleNamespace(
        unwrap_model=lambda model: model, device=torch.device("cpu"), is_main_process=True, wait_for_everyone=Mock()
    )
    trainer.model_version = 7
    trainer.rollout_worker = Mock()
    trainer.weight_transfer = Mock()

    def send(params):
        next(params)
        raise RuntimeError("export failed")

    trainer.weight_transfer.send_weights.side_effect = send
    with pytest.raises(RuntimeError, match="export failed"):
        trainer._sync_weight_merged(0.0)
    assert not any(layer.merged for layer in engine.adapter_layers)
    torch.testing.assert_close(engine.model.state_dict(), engine.training_state, rtol=0, atol=0)
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
    torch.testing.assert_close(engine.model.state_dict(), engine.training_state, rtol=0, atol=0)


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
                with generation._export_named_params() as params:
                    exported = {name: value.clone() for name, value in params}
                torch.testing.assert_close(exported, expected, rtol=0, atol=0)
                torch.testing.assert_close(model.state_dict(), training_state, rtol=0, atol=0)
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
        with generation._export_named_params() as params:
            list(params)
        torch.testing.assert_close(model.state_dict(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("sharded", [False, True])
def test_publication_keeps_merged_weights_live_until_loader_returns(peft_generation, sharded):
    engine = peft_generation
    engine._dist = SimpleNamespace(is_zero3=sharded, is_fsdp=False, gather_params=engine._dist.gather_params)
    with engine._export_named_params() as params:
        expected = {name: value.clone() for name, value in params}
    published = {}

    def load(weights):
        # Exercise a loader that retains references until it has consumed the iterable.
        pending = list(weights)
        assert all(layer.merged for layer in engine.adapter_layers)
        published.update((name, value.clone()) for name, value in pending)

    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = load
    engine.sync_weights()
    assert loader.call_count == (len(expected) if sharded else 1)
    torch.testing.assert_close(published, expected, rtol=0, atol=0)
    assert_adapter_restored(engine)
