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

import torch
from datasets import Dataset
from transformers import AutoProcessor
from transformers.utils import is_vision_available

from trl import GRPOConfig, GRPOTrainer

from ..testing_utils import TrlTestCase, require_vision


if is_vision_available():
    from PIL import Image


class TestVLMTokenTypes(TrlTestCase):
    @require_vision
    def test_token_types_mark_the_image_tokens(self):
        # A user processor pads on the right, while the trained prompts are left-padded.
        model_id = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
        image = Image.new("RGB", (56, 56))
        prompts = [
            [{"role": "user", "content": "Describe."}],
            [{"role": "user", "content": "Describe this image in great detail, please."}],
        ]
        trainer = GRPOTrainer(
            model=model_id,
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            args=GRPOConfig(
                output_dir=self.tmp_dir,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=4,
                report_to="none",
            ),
            train_dataset=Dataset.from_dict({"prompt": prompts, "image": [image, image]}),
            processing_class=AutoProcessor.from_pretrained(model_id),
        )
        output = trainer._generate_and_score_completions([{"prompt": p, "image": image} for p in prompts])
        input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        is_image = (input_ids == trainer.processing_class.image_token_id).long()
        assert torch.equal(output["mm_token_type_ids"], is_image)
