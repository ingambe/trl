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

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from trl.generation.vllm_generation import VLLMGeneration


def _make_server_generation(accelerator, *, max_completion_length):
    """Keep the real server batching code and replace only the client that sends generation requests."""
    generation = object.__new__(VLLMGeneration)
    generation.accelerator = accelerator
    generation.mode = "server"
    generation.temperature = 1.0
    generation.top_p = 1.0
    generation.top_k = -1
    generation.min_p = None
    generation.repetition_penalty = 1.0
    generation.max_completion_length = max_completion_length
    generation.logprobs = 0
    generation.structured_outputs_regex = None
    generation.generation_kwargs = {}

    def generate_from_submitted_histories(prompts, n, **sampling_kwargs):
        # Match vLLM's prompt-major ordering and n outputs per submitted history. Answers encode the tool result
        # the server actually saw: 30 -> 130, 35 -> 135. Never infer an answer from the intended recipient.
        tool_results = [prompt[-1] for prompt in prompts for _ in range(n)]
        return {
            "prompt_ids": prompts,
            "completion_ids": [[result + 100] for result in tool_results],
            "logprobs": [[[-result / 100]] for result in tool_results],
            "logprob_token_ids": [[[result + 100]] for result in tool_results],
        }

    generation.vllm_client = SimpleNamespace(generate=Mock(side_effect=generate_from_submitted_histories))
    return generation


@pytest.fixture
def server_tool_trainer(make_grpo_trainer):
    """Provide a two-sibling tool scenario; restore patched dependencies after the test."""
    trainer = make_grpo_trainer(use_vllm=True)
    trainer.model.config = SimpleNamespace(max_position_embeddings=128)
    trainer.vllm_mode = "server"
    trainer.num_generations = 2
    trainer.max_tool_calling_iterations = 1
    trainer.max_completion_length = 32
    trainer._is_vlm = False
    trainer.vllm_generation = _make_server_generation(
        trainer.accelerator, max_completion_length=trainer.max_completion_length
    )

    def calculator(result):
        return result

    trainer._sync_tool_dicts = [{"calculator": calculator} for _ in range(2)]
    trainer._async_tool_dicts = [{}, {}]

    def encode_tool_result(messages):
        # Synthetic tokenization: a result of "30" becomes token 30.
        return [int(messages[-1]["content"])]

    def decode_assistant_response(tokenizer, ids, prefix):
        return {"role": "assistant", "content": str(ids[0])}

    def gather_on_single_process(values):
        return values

    # Patch only token formatting and distributed transport. The tool loop, single-turn generation, and server
    # grouping run unchanged. These replacements are active during yield and automatically restored afterward.
    with (
        patch.object(trainer, "_get_tool_suffix_ids", side_effect=encode_tool_result),
        patch("trl.trainer.grpo_trainer.parse_response", side_effect=decode_assistant_response),
        patch("trl.generation.vllm_generation.gather_object", side_effect=gather_on_single_process),
        patch("trl.generation.vllm_generation.broadcast_object_list", return_value=None),
    ):
        yield trainer


def test_server_tool_continuations_keep_their_own_histories(server_tool_trainer):
    """Two siblings with different tool results must each continue from their own history."""
    trainer = server_tool_trainer
    # The same initial prompt produced two different assistant tool calls (tokens 10 and 20), returning 30 and 35.
    prompts = [[{"role": "user", "content": "Calculate 3 * 10 + 5."}] for _ in range(2)]
    completions = [
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"type": "function", "function": {"name": "calculator", "arguments": {"result": result}}}
                ],
            }
        ]
        for result in (30, 35)
    ]

    tool_mask, messages, completion_ids, logprobs, tool_count, failure_count, _ = trainer._tool_call_loop(
        prompts=prompts,
        prompt_ids=[[1], [1]],
        completion_ids=[[10], [20]],
        completions=completions,
        logprobs=[[-0.1], [-0.2]],
        images=None,
        multimodal_fields={},
    )

    assert tool_count == 2, "Both siblings must execute their calculator call before continuing."
    assert failure_count == 0, "Calculator calls must succeed for this ownership test."
    assert [history[1]["content"] for history in messages] == ["30", "35"], (
        "Each history must retain its own calculator result: 30 for the first sibling, 35 for the second."
    )
    assert completion_ids == [[10, 30, 130], [20, 35, 135]], (
        "Continuations must use their own tool history: the second sibling must receive token 135; "
        "token 130 means its answer was generated from the first sibling's history."
    )
    # Distinct mock logprobs let us verify probability ownership as well as token ownership.
    assert logprobs == [[-0.1, 0.0, -0.3], [-0.2, 0.0, -0.35]], (
        "Continuation logprobs must stay with their originating history: -0.3 for the first sibling, "
        "-0.35 for the second; tool-result positions must have zero placeholders."
    )
    assert tool_mask == [[1, 0, 1], [1, 0, 1]], "Only tool-result tokens should be excluded from the training loss."
    assert [history[-1]["content"] for history in messages] == ["130", "135"], (
        "Decoded assistant messages must preserve the same sibling ownership as the completion tokens."
    )
    request = trainer.vllm_generation.vllm_client.generate.call_args.kwargs
    assert request["prompts"] == [[1, 10, 30], [1, 20, 35]], (
        "The server must receive both distinct continuation histories in their original order."
    )
    assert request["n"] == 1, "Each continuation history must request exactly one completion."
