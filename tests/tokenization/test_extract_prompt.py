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

from trl.data_utils import extract_prompt

from ..testing_utils import TrlTestCase


class TestExtractPrompt(TrlTestCase):
    # When one completion is a prefix of the other, the whole shorter one is the prompt
    def test_prefix_completion(self):
        user = {"role": "user", "content": "What color is the sky?"}
        assistant = {"role": "assistant", "content": "It is blue."}
        follow_up = [{"role": "user", "content": "Are you sure?"}, {"role": "assistant", "content": "Yes."}]
        example = {"chosen": [user, assistant], "rejected": [user, assistant, *follow_up]}
        assert extract_prompt(example) == {"prompt": [user, assistant], "chosen": [], "rejected": follow_up}
