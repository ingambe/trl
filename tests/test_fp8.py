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
