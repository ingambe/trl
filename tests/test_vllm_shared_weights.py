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


def test_shared_weights_are_updated_in_place_and_never_published(vllm_generation):
    model = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(2, 3), "up_proj": torch.nn.Linear(2, 3)})
    # vLLM fuses both projections into one layer
    vllm_model = torch.nn.ModuleDict({"gate_up_proj": torch.nn.Linear(2, 6)})
    vllm_model.gate_up_proj.output_sizes = [3, 3]
    vllm_model.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    expected = torch.cat([model.gate_proj.weight, model.up_proj.weight]).detach().clone()

    vllm_generation._shared_views = vllm_generation._share_weights()
    with torch.no_grad():
        model.up_proj.weight.add_(1.0)  # an optimizer step
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    expected[3:] += 1.0
    torch.testing.assert_close(vllm_model.gate_up_proj.weight, expected, rtol=0, atol=0)
    assert published == []


def test_replaced_storage_is_published(vllm_generation):
    model = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(2, 3), "up_proj": torch.nn.Linear(2, 3)})
    vllm_model = torch.nn.ModuleDict({"gate_proj": torch.nn.Linear(2, 3), "up_proj": torch.nn.Linear(2, 3)})
    vllm_model.packed_modules_mapping = {}
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model

    vllm_generation._shared_views = vllm_generation._share_weights()
    model.up_proj.weight.data = torch.ones(3, 2)  # e.g. a checkpoint load
    published = []
    vllm_model.load_weights = lambda weights: published.extend(name for name, _ in weights)
    vllm_generation.sync_weights()

    assert published == ["up_proj.weight"]
