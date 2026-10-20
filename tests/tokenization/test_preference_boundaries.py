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

from trl import DPOConfig, DPOTrainer

from ..testing_utils import TrlTestCase


class TestDPOTokenization(TrlTestCase):
    @pytest.mark.parametrize(
        "model_id",
        [
            "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
            "trl-internal-testing/tiny-Qwen3ForCausalLM",
            "trl-internal-testing/tiny-LlamaForCausalLM-3.1",
        ],
    )
    def test_prompt_and_responses_match_chat_template(self, model_id):
        prompt = [{"role": "user", "content": "What color is the sky?"}]
        chosen = [{"role": "assistant", "content": "It is blue."}]
        rejected = [{"role": "assistant", "content": "It is green."}]
        trainer = DPOTrainer(
            model=model_id,
            args=DPOConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": [prompt], "chosen": [chosen], "rejected": [rejected]}),
        )
        tokenizer = trainer.processing_class
        example = trainer.train_dataset[0]
        assert example["prompt_ids"] == tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, return_dict=False
        )
        for key, response in [("chosen_ids", chosen), ("rejected_ids", rejected)]:
            full_ids = tokenizer.apply_chat_template(prompt + response, return_dict=False)
            assert example["prompt_ids"] + example[key] == full_ids
