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


def test_failed_merged_adapter_transfer_unmerges_before_handoff(generation, monkeypatch):
    generation.model.merge_adapter = Mock()
    generation.model.unmerge_adapter = Mock()
    generation.model.prefix = "lora_"
    monkeypatch.setattr("trl.generation.vllm_generation.is_peft_model", lambda model: True)
    loader = generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"), generation.rollout_phase():
        generate(generation)
    generation.model.merge_adapter.assert_called_once()
    generation.model.unmerge_adapter.assert_called_once()
    assert generation._weights_dirty
    generation.llm.sleep.assert_called_once_with(level=2)


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


def test_online_dpo_merged_adapter_transfer_has_matching_cleanup(generation, monkeypatch):
    from trl.experimental.online_dpo import OnlineDPOTrainer

    trainer = object.__new__(OnlineDPOTrainer)
    trainer.accelerator = SimpleNamespace(state=SimpleNamespace(deepspeed_plugin=None), is_main_process=True)
    trainer.is_fsdp_enabled = False
    trainer.model = generation.model
    trainer.model.prefix = "lora_"
    trainer.model.merge_adapter = Mock()
    trainer.model.unmerge_adapter = Mock()
    trainer.vllm_mode = "colocate"
    trainer.llm = generation.llm
    monkeypatch.setattr("trl.experimental.online_dpo.online_dpo_trainer.is_peft_model", lambda model: True)
    trainer.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights.side_effect = RuntimeError(
        "failed"
    )
    with pytest.raises(RuntimeError, match="failed"):
        trainer._move_model_to_vllm_inner()
    trainer.model.merge_adapter.assert_called_once()
    trainer.model.unmerge_adapter.assert_called_once()
