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

from trl import SFTConfig, SFTTrainer

from ..testing_utils import TrlTestCase


class TestSFTTokenization(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        ],
    )
    def test_prompt_completion_matches_chat_template(self, model_id):
        # The tokenized example must be the chat template rendering, with the loss starting right after the
        # generation prompt.
        prompt = [{"role": "user", "content": "What color is the sky?"}]
        completion = [{"role": "assistant", "content": "It is blue."}]
        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": [prompt], "completion": [completion]}),
        )
        tokenizer = trainer.processing_class
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
        full_ids = tokenizer.apply_chat_template(prompt + completion, return_dict=False)
        example = trainer.train_dataset[0]
        assert example["input_ids"] == full_ids
        assert full_ids[: len(prompt_ids)] == prompt_ids
        assert example["labels"] == [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        ],
    )
    def test_standard_text_gets_one_bos_and_eos(self, model_id):
        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": ["The sky is"], "completion": [" blue."]}),
        )
        tokenizer = trainer.processing_class
        example = trainer.train_dataset[0]
        prompt_ids = tokenizer("The sky is")["input_ids"]
        expected = tokenizer("The sky is blue.")["input_ids"] + [tokenizer.eos_token_id]
        assert example["input_ids"] == expected
        assert example["labels"] == [-100] * len(prompt_ids) + expected[len(prompt_ids) :]
