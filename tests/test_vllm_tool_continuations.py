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
