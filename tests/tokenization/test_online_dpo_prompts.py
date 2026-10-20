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

from contextlib import nullcontext

from datasets import Dataset
from transformers import AutoTokenizer

from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer

from ..testing_utils import TrlTestCase


class FakeVLLMClient:
    """Stands in for the remote vLLM server: records the prompts it receives and returns fixed completions."""

    def weight_update(self):
        return nullcontext()

    def update_named_param(self, name, weights):
        pass

    def reset_prefix_cache(self):
        pass

    def generate(self, prompts, n, **kwargs):
        self.prompts = prompts
        return {"completion_ids": [[11, 12, 13]] * (len(prompts) * n)}


class TestOnlineDPOPrompts(TrlTestCase):
    def test_vllm_server_trains_on_the_sampled_prompt(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        prompts = [
            [{"role": "user", "content": "Hi"}],
            [{"role": "user", "content": "Write a much longer question about the history of the Roman empire."}],
        ]
        trainer = OnlineDPOTrainer(
            model=model_id,
            reward_funcs=[lambda completions, **kwargs: [0.0] * len(completions)],
            args=OnlineDPOConfig(output_dir=self.tmp_dir, max_new_tokens=8, report_to="none"),
            train_dataset=Dataset.from_dict({"prompt": prompts}),
            processing_class=AutoTokenizer.from_pretrained(model_id),
        )
        # Server mode can't be built without vLLM and an accelerator, so switch to it and fake only the server
        trainer.vllm_mode = "server"
        trainer.vllm_client = FakeVLLMClient()
        trainer.generation_config.max_tokens = 8
        prompt_ids, prompt_mask, _, _ = trainer._generate_vllm(prompts)

        for i in range(len(prompts)):
            assert prompt_ids[i][prompt_mask[i].bool()].tolist() == trainer.vllm_client.prompts[i]
