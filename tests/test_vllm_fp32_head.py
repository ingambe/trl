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

from unittest.mock import MagicMock, patch

from accelerate import Accelerator

from trl.generation.vllm_generation import VLLMGeneration


def test_colocate_cast_lm_head_to_fp32_sets_vllm_head_dtype():
    with (
        patch("trl.generation.vllm_generation.is_vllm_available", return_value=True),
        patch("trl.generation.vllm_generation.LLM", create=True) as llm,
    ):
        VLLMGeneration(MagicMock(), Accelerator(cpu=True), MagicMock(), mode="colocate", cast_lm_head_to_fp32=True)

    assert llm.call_args.kwargs["hf_overrides"] == {"head_dtype": "float32"}
