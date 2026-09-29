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


def test_ranks_with_uneven_tool_calls_share_each_round(two_rank_server_tool_trainers):
    """A rank without tool calls must still join the generation round, and each rank must get its own completions."""
    results_per_rank = [[None], [30, 35]]

    def tool_call_loop(rank, trainer):
        completions = []
        for result in results_per_rank[rank]:
            message = {"role": "assistant", "content": ""}
            if result is not None:
                function = {"name": "calculator", "arguments": {"result": result}}
                message["tool_calls"] = [{"type": "function", "function": function}]
            completions.append([message])
        return trainer._tool_call_loop(
            prompts=[[{"role": "user", "content": "Calculate."}] for _ in completions],
            prompt_ids=[[1] for _ in completions],
            completion_ids=[[10 + rank] for _ in completions],
            completions=completions,
            logprobs=None,
            images=None,
            multimodal_fields={},
        )

    (trainer_0, outputs_0), (_, outputs_1) = two_rank_server_tool_trainers([1, 2], tool_call_loop)

    assert outputs_0[2] == [[10]]
    assert outputs_1[2] == [[11, 30, 130], [11, 35, 135]]
    request = trainer_0.vllm_generation.vllm_client.generate.call_args.kwargs
    assert request["prompts"] == [[1, 11, 30], [1, 11, 35]]


def test_tool_loop_marks_rolled_back_tool_call_as_truncated(server_tool_trainer):
    """A tool call dropped because its result would exceed max_completion_length must be flagged as truncated."""
    trainer = server_tool_trainer
    trainer.max_completion_length = 3
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

    # The first tool result overflows the completion budget and is rolled back
    _, messages, completion_ids, _, _, _, _, tool_truncated = trainer._tool_call_loop(
        prompts=prompts,
        prompt_ids=[[1], [1]],
        completion_ids=[[10, 11, 2], [20]],
        completions=completions,
        logprobs=[[-0.1, -0.1, -0.1], [-0.2]],
        images=None,
        multimodal_fields={},
    )

    assert completion_ids == [[10, 11, 2], [20, 35, 135]]
    assert "tool_calls" in messages[0][-1], "The first sample must end on its unexecuted tool call."
    assert tool_truncated == [True, False], "Only the sample whose tool call was rolled back is truncated."


def test_server_generates_each_prompt_when_group_prompts_differ(server_generation):
    """Group members with different prompts must each be generated from their own prompt."""
    prompt_ids, completion_ids, _, _ = server_generation.generate([[1, 30], [1, 35], [2, 40], [2, 40]], None, 2)

    request = server_generation.vllm_client.generate.call_args.kwargs
    assert (request["prompts"], request["n"]) == ([[1, 30], [1, 35], [2, 40], [2, 40]], 1)
    assert prompt_ids == [[1, 30], [1, 35], [2, 40], [2, 40]]
    assert completion_ids == [[130], [135], [140], [140]]
