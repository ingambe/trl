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
from datasets import Dataset

from trl import GRPOConfig, GRPOTrainer, RLOOConfig, RLOOTrainer

from ..testing_utils import TrlTestCase


MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    return a * b


class TestRepeatedPrompts(TrlTestCase):
    # Each prompt is repeated `num_generations` times: it is tokenized once and every copy gets the same ids
    @pytest.mark.parametrize("trainer_name", ["grpo", "rloo"])
    def test_repeated_prompts_tokenized_once(self, trainer_name):
        prompts = [[{"role": "user", "content": "What is 2*3?"}], [{"role": "user", "content": "Hi"}]]
        dataset = Dataset.from_dict({"prompt": prompts})
        kwargs = dict(output_dir=self.tmp_dir, report_to="none", chat_template_kwargs={"enable_thinking": False})
        if trainer_name == "grpo":
            trainer = GRPOTrainer(
                model=MODEL_ID,
                reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
                args=GRPOConfig(**kwargs),
                train_dataset=dataset,
                tools=[multiply],
            )
        else:
            trainer = RLOOTrainer(
                model=MODEL_ID,
                reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
                args=RLOOConfig(**kwargs),
                train_dataset=dataset,
            )
        batch = [prompts[0]] * 3 + [prompts[1]] * 3

        prompt_ids, _, _ = trainer._tokenize_prompts(batch)

        assert prompt_ids == [trainer._tokenize_prompts([prompt])[0][0] for prompt in batch]
        assert prompt_ids[0] is prompt_ids[2] and prompt_ids[3] is prompt_ids[5]
