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

from trl.models.fp8 import FP8Linear


def test_fp8_linear_matches_its_dequantized_weight_forward_and_backward():
    torch.manual_seed(0)
    layer = FP8Linear(torch.nn.Linear(64, 48, dtype=torch.bfloat16))
    weight = layer.weight_fp8.float() * layer.weight_scale
    x = torch.randn(2, 5, 64, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(2, 5, 48)

    out = layer(x)
    out.backward(grad.bfloat16())

    # Only the per-token FP8 rounding of the activations and gradients remains
    expected_out = x.detach().float() @ weight.T + layer.bias.float()
    assert (out.float() - expected_out).norm() / expected_out.norm() < 0.05
    expected_grad = grad @ weight
    assert (x.grad.float() - expected_grad).norm() / expected_grad.norm() < 0.05
