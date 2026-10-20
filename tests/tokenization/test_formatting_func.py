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

from trl import SFTConfig, SFTTrainer

from ..testing_utils import TrlTestCase


class TestFormattingFunc(TrlTestCase):
    # The formatted text must be trained on, not the conversational column it was built from
    @pytest.mark.parametrize("dataset_text_field", ["text", "content"])
    def test_sft_conversational(self, dataset_text_field):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]

        def formatting_func(example):
            return (
                f"### Question: {example['messages'][0]['content']}\n### Answer: {example['messages'][1]['content']}"
            )

        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, dataset_text_field=dataset_text_field, report_to="none"),
            train_dataset=Dataset.from_dict({"messages": [messages]}),
            formatting_func=formatting_func,
        )
        text = formatting_func({"messages": messages}) + tokenizer.eos_token
        assert trainer.train_dataset[0]["input_ids"] == tokenizer(text)["input_ids"]

    # A formatted dataset is plain text, so assistant masks would be silently lost
    def test_sft_rejects_assistant_only_loss(self):
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        with pytest.raises(ValueError, match="formatting function was provided while `assistant_only_loss=True`"):
            SFTTrainer(
                model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
                args=SFTConfig(output_dir=self.tmp_dir, assistant_only_loss=True, report_to="none"),
                train_dataset=Dataset.from_dict({"messages": [messages]}),
                formatting_func=lambda example: example["messages"][0]["content"],
            )
