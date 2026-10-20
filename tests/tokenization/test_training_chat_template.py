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
from transformers import AutoTokenizer

from trl.chat_template_utils import (
    deepseek_r1_distill_chat_template,
    deepseekv3_chat_template,
    get_training_chat_template,
)

from ..testing_utils import TrlTestCase


class TestTrainingChatTemplate(TrlTestCase):
    # Only what the model samples after the generation prompt is trained, not the end of the generation prompt
    @pytest.mark.parametrize(
        "chat_template, assistant_message, expected",
        [
            (None, {"content": "It is blue."}, "It is blue.<|im_end|>\n"),
            (deepseekv3_chat_template, {"content": "It is blue."}, "It is blue.<｜end▁of▁sentence｜>"),
            (
                deepseekv3_chat_template,
                {"content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {}}}]},
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json\n{}\n```<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜><｜end▁of▁sentence｜>",
            ),
            (deepseek_r1_distill_chat_template, {"content": "It is blue."}, "It is blue.<｜end▁of▁sentence｜>"),
            (
                deepseek_r1_distill_chat_template,
                {"content": None, "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>f\n```json\n{}\n```<｜tool▁call▁end｜>",
            ),
        ],
        ids=["qwen2_5", "deepseekv3", "deepseekv3_tool_call", "deepseek_r1_distill", "deepseek_r1_distill_tool_call"],
    )
    def test_assistant_mask(self, chat_template, assistant_message, expected):
        tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-Qwen2ForCausalLM-2.5")
        if chat_template is not None:
            tokenizer.chat_template = chat_template
            tokenizer.add_tokens(["<｜Assistant｜>"], special_tokens=True)  # as in the DeepSeek tokenizer
        messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", **assistant_message}]
        output = tokenizer.apply_chat_template(
            messages,
            chat_template=get_training_chat_template(tokenizer),
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        trained = [t for t, m in zip(output["input_ids"], output["assistant_masks"], strict=True) if m]
        assert tokenizer.decode(trained) == expected
