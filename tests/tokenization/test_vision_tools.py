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

import json

from datasets import Dataset
from transformers import AutoProcessor
from transformers.utils import is_vision_available

from trl import SFTConfig, SFTTrainer
from trl.experimental.utils import DataCollatorForVisionLanguageChatML
from trl.trainer.dpo_trainer import DataCollatorForVisionPreference
from trl.trainer.kto_trainer import DataCollatorForVisionUnpairedPreference

from ..testing_utils import TrlTestCase, require_vision


if is_vision_available():
    from PIL import Image


MODEL_ID = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {}}}}]
PROMPT = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Weather?"}]}]
COMPLETION = [{"role": "assistant", "content": [{"type": "text", "text": "Sunny."}]}]


@require_vision
class TestVisionTools(TrlTestCase):
    # Conversational vision examples must render the per-example `tools`, like text-only examples
    def setup_method(self):
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.processor.chat_template = "{% if tools %}{{ tools | tojson }}{% endif %}" + self.processor.chat_template
        self.image = Image.new("RGB", (16, 16))

    def test_sft(self):
        dataset = Dataset.from_dict(
            {"images": [[self.image]], "prompt": [PROMPT], "completion": [COMPLETION], "tools": [json.dumps(TOOLS)]}
        )
        trainer = SFTTrainer(
            model=MODEL_ID,
            args=SFTConfig(output_dir=self.tmp_dir, report_to="none"),
            train_dataset=dataset,
            processing_class=self.processor,
        )
        batch = next(iter(trainer.get_train_dataloader()))
        assert "get_weather" in self.processor.decode(batch["input_ids"][0])

    def test_dpo(self):
        collator = DataCollatorForVisionPreference(self.processor)
        example = {"images": [self.image], "prompt": PROMPT, "chosen": COMPLETION, "rejected": COMPLETION}
        output = collator([{**example, "tools": TOOLS}])
        assert "get_weather" in self.processor.decode(output["input_ids"][0])

    def test_kto(self):
        collator = DataCollatorForVisionUnpairedPreference(self.processor)
        example = {"images": [self.image], "prompt": PROMPT, "completion": COMPLETION, "label": True}
        output = collator([{**example, "tools": TOOLS}])
        assert "get_weather" in self.processor.decode(output["input_ids"][0])

    def test_gold(self):
        collator = DataCollatorForVisionLanguageChatML(self.processor)
        example = {"images": [self.image], "prompt": PROMPT, "completion": COMPLETION}
        output = collator([{**example, "tools": TOOLS}])
        assert "get_weather" in self.processor.decode(output["input_ids"][0])
