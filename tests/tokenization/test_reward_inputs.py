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
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer, RewardConfig, RewardTrainer

from ..testing_utils import TrlTestCase


class TestRewardModelInputs(TrlTestCase):
    @pytest.mark.parametrize(
        "prompt,completion",
        [
            ("The sky is", " blue."),
            ([{"role": "user", "content": "The sky is"}], [{"role": "assistant", "content": "blue."}]),
        ],
    )
    def test_grpo_scores_rollouts_like_reward_trainer(self, prompt, completion):
        # The reward model must see the ids RewardTrainer trained it on.
        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        reward_trainer = RewardTrainer(
            model=reward_model_id,
            args=RewardConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": [prompt], "chosen": [completion], "rejected": [completion]}),
        )
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model_id,
            args=GRPOConfig(
                output_dir=self.tmp_dir, report_to="none", per_device_train_batch_size=2, num_generations=2
            ),
            train_dataset=Dataset.from_dict({"prompt": [prompt]}),
        )
        seen = []
        trainer.reward_funcs[0].register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["input_ids"]), with_kwargs=True
        )
        trainer._calculate_rewards([{"prompt": prompt}], [prompt], [completion], [[]])
        assert seen[0][0].tolist() == reward_trainer.train_dataset[0]["chosen_ids"]

    def test_grpo_scores_rollouts_without_reward_eos_token(self):
        # Reward tokenizers without an EOS token (e.g. BERT-like) score the raw text.
        reward_model_id = "trl-internal-testing/tiny-Qwen2ForSequenceClassification-2.5"
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)
        reward_tokenizer.eos_token = None
        trainer = GRPOTrainer(
            model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            reward_funcs=reward_model_id,
            reward_processing_classes=reward_tokenizer,
            args=GRPOConfig(
                output_dir=self.tmp_dir, report_to="none", per_device_train_batch_size=2, num_generations=2
            ),
            train_dataset=Dataset.from_dict({"prompt": ["The sky is"]}),
        )
        seen = []
        trainer.reward_funcs[0].register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["input_ids"]), with_kwargs=True
        )
        trainer._calculate_rewards([{"prompt": "The sky is"}], ["The sky is"], [" blue."], [[]])
        assert seen[0][0].tolist() == reward_tokenizer("The sky is blue.")["input_ids"]
