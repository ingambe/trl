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

from types import SimpleNamespace

import pytest
import torch


def test_failed_sync_is_retried_before_generating(vllm_generation, monkeypatch):
    monkeypatch.setattr("trl.generation.vllm_generation.SamplingParams", dict, raising=False)
    vllm_generation.llm.generate.return_value = [
        SimpleNamespace(prompt_token_ids=[1], outputs=[SimpleNamespace(token_ids=[2], logprobs=None)])
    ]
    published = {}

    def fail_on_bias(weights):
        for name, value in weights:
            if name == "bias":
                raise RuntimeError("bias transfer failed")
            published[name] = value.clone()

    loader = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
    loader.side_effect = fail_on_bias
    with pytest.raises(RuntimeError, match="bias transfer failed"):
        vllm_generation.sync_weights()
    with pytest.raises(RuntimeError, match="bias transfer failed"):
        vllm_generation.generate([[1]], images=None, num_generations=1)
    vllm_generation.llm.generate.assert_not_called()

    loader.side_effect = lambda weights: published.update((name, value.clone()) for name, value in weights)
    vllm_generation.generate([[1]], images=None, num_generations=1)
    vllm_generation.llm.generate.assert_called_once()
    torch.testing.assert_close(published, vllm_generation.model.state_dict(), rtol=0, atol=0)
