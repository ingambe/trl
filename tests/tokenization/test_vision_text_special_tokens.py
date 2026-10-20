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

from transformers import AutoProcessor
from transformers.utils import is_vision_available

from trl.experimental.utils import DataCollatorForVisionLanguageChatML
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
from trl.trainer.kto_trainer import DataCollatorForVisionUnpairedPreference
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

from ..testing_utils import TrlTestCase, require_vision


if is_vision_available():
    from PIL import Image


@require_vision
class TestVisionTextSpecialTokens(TrlTestCase):
    # Standard (non-conversational) vision examples must end with the EOS, like text-only standard examples
    def setup_method(self):
        self.processor = AutoProcessor.from_pretrained("trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration")
        self.image = Image.new("RGB", (16, 16))
        self.prompt = "<|vision_start|><|image_pad|><|vision_end|>What is this?"

    def last_token_id(self, output):
        return output["input_ids"][0][output["attention_mask"][0].bool()][-1].item()

    def test_sft(self):
        collator = DataCollatorForVisionLanguageModeling(self.processor)
        output = collator([{"images": [self.image], "text": self.prompt + " A square."}])
        assert self.last_token_id(output) == self.processor.tokenizer.eos_token_id
        output = collator([{"images": [self.image], "prompt": self.prompt, "completion": " A square."}])
        assert self.last_token_id(output) == self.processor.tokenizer.eos_token_id

    def test_dpo(self):
        collator = DataCollatorForVisionPreference(self.processor)
        output = collator(
            [{"images": [self.image], "prompt": self.prompt, "chosen": " A square.", "rejected": " A cat."}]
        )
        assert self.last_token_id(output) == self.processor.tokenizer.eos_token_id

    def test_kto(self):
        collator = DataCollatorForVisionUnpairedPreference(self.processor)
        output = collator([{"images": [self.image], "prompt": self.prompt, "completion": " A square.", "label": True}])
        assert self.last_token_id(output) == self.processor.tokenizer.eos_token_id

    def test_gold(self):
        collator = DataCollatorForVisionLanguageChatML(self.processor)
        output = collator([{"images": [self.image], "prompt": self.prompt, "completion": " A square."}])
        assert self.last_token_id(output) == self.processor.tokenizer.eos_token_id
