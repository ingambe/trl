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

from trl.models.fp8 import FP8Linear, quantize_rowwise
from trl.trainer.callbacks import SyncRefModelCallback


@pytest.mark.parametrize("trainable", [False, True])
def test_fp8_linear_matches_its_dequantized_weight_forward_and_backward(trainable):
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 48, dtype=torch.bfloat16).requires_grad_(trainable)
    weight_fp8, weight_scale = quantize_rowwise(linear.weight.detach())
    weight = weight_fp8.float() * weight_scale
    layer = FP8Linear(linear)
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(2, 5, 48)

    out = layer(x)
    out.backward(grad.bfloat16())

    # Only the per-token FP8 rounding of the activations and gradients remains
    expected_out = x.detach().float() @ weight.T + linear.bias.float()
    assert (out.float() - expected_out).norm() / expected_out.norm() < 0.05
    expected_grad = grad @ weight
    assert (x.grad.float() - expected_grad).norm() / expected_grad.norm() < 0.05
    if trainable:
        # The weight gradient is computed in high precision
        expected_weight_grad = grad.reshape(-1, 48).T @ x.detach().reshape(-1, 64).float()
        torch.testing.assert_close(linear.weight.grad.float(), expected_weight_grad, rtol=2e-2, atol=2e-2)


def test_fp8_linear_requantizes_its_weight_once_per_update():
    layer = FP8Linear(torch.nn.Linear(64, 48))
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    for _ in range(2):  # gradient accumulation
        layer(torch.randn(5, 64)).sum().backward()
    assert layer.weight_fp8._version == 0
    optimizer.step()
    for _ in range(2):
        layer(torch.randn(5, 64)).sum().backward()

    assert layer.weight_fp8._version == 1
    weight_fp8, _ = quantize_rowwise(layer.weight.detach())
    assert torch.equal(layer.weight_fp8.view(torch.uint8), weight_fp8.view(torch.uint8))


def test_fp8_reference_model_sees_synced_weights():
    model, ref_model = torch.nn.Linear(64, 48), FP8Linear(torch.nn.Linear(64, 48))

    SyncRefModelCallback._sync_target_model(model, ref_model, alpha=1.0)
    ref_model(torch.randn(5, 64))

    weight_fp8, _ = quantize_rowwise(model.weight.detach())
    assert torch.equal(ref_model.weight_fp8.view(torch.uint8), weight_fp8.view(torch.uint8))
