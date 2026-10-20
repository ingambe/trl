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

import pytest
import torch
import transformers
from datasets import Dataset
from packaging.version import Version
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from trl.chat_template_utils import get_training_chat_template, is_chat_template_prefix_preserving

from ..testing_utils import TrlTestCase, require_response_parsing


def multiply_tool(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The product of the two integers.
    """
    return a * b


class TestRolloutTokenization(TrlTestCase):
    @pytest.mark.xfail(
        condition=Version(transformers.__version__) < Version("5.0.0"),
        reason="Tool parsing is not supported in transformers versions below 5.0.0",
        strict=True,
    )
    @require_response_parsing
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
            pytest.param(
                "trl-internal-testing/tiny-GptOssForCausalLM",
                marks=pytest.mark.xfail(
                    Version(transformers.__version__) < Version("5.5.0"),
                    reason="Upstream bug in response parsing (see #5753; fixed in transformers#45166)",
                    strict=True,
                ),
            ),
        ],
    )
    def test_tool_rollout_trains_on_sampled_tokens(self, model_id):
        # The trained sequence must be exactly the sampled tokens plus the tool turn rendered by the chat template, with
        # only the tool turn masked out.
        prompt = [{"role": "user", "content": "What is 3 * 4?"}]
        tool_call = {"type": "function", "function": {"name": "multiply_tool", "arguments": {"a": 3, "b": 4}}}
        turns = [
            {"role": "assistant", "content": "", "tool_calls": [tool_call]},
            {"role": "tool", "name": "multiply_tool", "content": "12"},
            {"role": "assistant", "content": "The answer is 12."},
        ]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # The template GRPOTrainer renders tool turns with
        chat_template = (
            None if is_chat_template_prefix_preserving(tokenizer) else get_training_chat_template(tokenizer)
        )

        def render(messages, **kwargs):
            return tokenizer.apply_chat_template(
                messages, tools=[multiply_tool], chat_template=chat_template, return_dict=False, **kwargs
            )

        def turn(context_ids, messages):
            # The rendered turn that follows `context_ids`
            ids = render(messages)
            assert ids[: len(context_ids)] == context_ids
            return ids[len(context_ids) :]

        prompt_ids = render(prompt, add_generation_prompt=True)
        context_ids = render(prompt + turns[:2], add_generation_prompt=True)
        first_ids = turn(prompt_ids, prompt + turns[:1])
        second_ids = turn(context_ids, prompt + turns)
        # Make the model sample the rendered turns by biasing each of their tokens given the exact preceding context
        sequence_bias = {
            tuple(ctx + ids[: j + 1]): 100.0
            for ctx, ids in [(prompt_ids, first_ids), (context_ids, second_ids)]
            for j in range(len(ids))
        }
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                generation_kwargs={"sequence_bias": sequence_bias},
                report_to="none",
            ),
            train_dataset=Dataset.from_dict({"prompt": [prompt]}),
            tools=[multiply_tool],
        )
        # Generation stops at the end-of-turn token
        first_ids = first_ids[: next(j for j, t in enumerate(first_ids) if t in trainer.eos_token_ids) + 1]
        second_ids = second_ids[: next(j for j, t in enumerate(second_ids) if t in trainer.eos_token_ids) + 1]

        output = trainer._generate_and_score_completions([{"prompt": prompt}] * 2)

        for i in range(2):
            ids = torch.cat([output["prompt_ids"][i], output["completion_ids"][i]])
            mask = torch.cat([output["prompt_mask"][i], output["completion_mask"][i]]).bool()
            assert ids[mask].tolist() == context_ids + second_ids
            tool_mask = output["tool_mask"][i][output["completion_mask"][i].bool()].tolist()
            num_tool_tokens = len(context_ids) - len(prompt_ids) - len(first_ids)
            assert tool_mask == [1] * len(first_ids) + [0] * num_tool_tokens + [1] * len(second_ids)
