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

from .testing_utils import require_peft


@require_peft
def test_single_loader_call_reads_merged_weights(peft_vllm_generation):
    published = {}

    def load(weights):
        pending = list(weights)  # vLLM loaders may consume the iterable before reading
        published.update((name, value.clone()) for name, value in pending)

    loader = peft_vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = load
    peft_vllm_generation.sync_weights()

    loader.assert_called_once()
    # 0.25 + 0.25 * 0.25; biases are not adapted
    merged = torch.full((2, 2), 0.3125)
    bias = torch.full((2,), 0.25)
    expected = {"0.weight": merged, "0.bias": bias, "1.weight": merged, "1.bias": bias}
    torch.testing.assert_close(published, expected, rtol=0, atol=0)
    assert not any(layer.merged for layer in peft_vllm_generation.model.base_model.model)
