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

from datasets import Dataset
from PIL import Image

from trl.experimental.gold import GOLDConfig, GOLDTrainer
from trl.experimental.gold import gold_trainer as gold_trainer_module

from ..testing_utils import TrlTestCase


class TestGOLDVLMvLLM(TrlTestCase):
    # The VLM vLLM path must train on the sampled completion ids, not on a decode/re-encode of them
    def test_trains_on_sampled_completion_ids(self, monkeypatch):
        model_id = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        dataset = Dataset.from_dict(
            {
                "prompt": [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "3 x 4?"}]}]],
                "completion": [[{"role": "assistant", "content": [{"type": "text", "text": "12"}]}]],
                "images": [[Image.new("RGB", (56, 56))]],
            }
        )

        # Stand in for vLLM, which cannot run here: sample a non-canonical tokenization of "The answer is 12"
        class FakeVLLMGeneration:
            def __init__(self, **kwargs):
                pass

            def sync_weights(self):
                pass

            def sleep(self):
                pass

            def generate(self, prompts, images, num_generations):
                tokens = ["The", "Ġans", "wer", "Ġis", "Ġ", "1", "2", "<|im_end|>"]
                return None, [tokenizer.convert_tokens_to_ids(tokens) for _ in prompts], None, None

        monkeypatch.setattr(gold_trainer_module, "is_vllm_available", lambda: True)
        monkeypatch.setattr(gold_trainer_module, "VLLMGeneration", FakeVLLMGeneration)

        trainer = GOLDTrainer(
            model=model_id,
            teacher_model=model_id,
            args=GOLDConfig(
                output_dir=self.tmp_dir, report_to="none", lmbda=1.0, use_vllm=True, max_completion_length=16
            ),
            train_dataset=dataset,
        )
        tokenizer = trainer.processing_class.tokenizer
        trainer.model.train()
        inputs = trainer._prepare_inputs([dict(dataset[0])])

        sampled = FakeVLLMGeneration().generate([None], None, 1)[1][0]
        assert inputs["labels"][0][inputs["labels"][0] != -100].tolist() == sampled
