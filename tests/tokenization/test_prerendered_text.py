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

from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
from trl.experimental.orpo import ORPOConfig, ORPOTrainer

from ..testing_utils import TrlTestCase


MODEL_IDS = ["trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "trl-internal-testing/tiny-LlamaForCausalLM-3.1"]


class TestPrerenderedText(TrlTestCase):
    # Text already rendered with the chat template must tokenize exactly like the template
    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_sft(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        trainer = SFTTrainer(
            model=model_id,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"text": [text]}),
        )
        assert trainer.train_dataset[0]["input_ids"] == tokenizer.apply_chat_template(messages, return_dict=False)

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_dpo(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = [{"role": "user", "content": "Hi"}]
        chosen = [{"role": "assistant", "content": "Hello!"}]
        prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        chosen_text = tokenizer.apply_chat_template(prompt + chosen, tokenize=False)[len(prompt_text) :]
        trainer = DPOTrainer(
            model=model_id,
            args=DPOConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict(
                {"prompt": [prompt_text], "chosen": [chosen_text], "rejected": [chosen_text]}
            ),
        )
        example = trainer.train_dataset[0]
        assert example["prompt_ids"] + example["chosen_ids"] == tokenizer.apply_chat_template(
            prompt + chosen, return_dict=False
        )

    @pytest.mark.parametrize("model_id", MODEL_IDS)
    def test_orpo(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        prompt = [{"role": "user", "content": "Hi"}]
        chosen = [{"role": "assistant", "content": "Hello!"}]
        trainer = ORPOTrainer(
            model=model_id,
            args=ORPOConfig(output_dir=self.tmp_dir, report_to="none"),
            processing_class=tokenizer,
            train_dataset=Dataset.from_dict({"prompt": [prompt], "chosen": [chosen], "rejected": [chosen]}),
        )
        assert trainer.train_dataset[0]["chosen_input_ids"] == tokenizer.apply_chat_template(
            prompt + chosen, return_dict=False
        )
