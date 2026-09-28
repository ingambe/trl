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
import torch
from transformers.utils import is_peft_available

from .testing_utils import require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@require_peft
def test_failed_transfer_unmerges_adapters(vllm_generation):
    vllm_generation.model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2)), LoraConfig(r=1, target_modules=["0"])
    )
    loader = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = RuntimeError("transfer failed")

    with pytest.raises(RuntimeError, match="transfer failed"):
        vllm_generation.sync_weights()
    assert not vllm_generation.model.base_model.model[0].merged
