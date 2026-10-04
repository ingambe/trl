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

from trl.models.fp8 import BlockFP8Linear, FP8Linear, dequantize_fp8_layers, quantize_rowwise
from trl.trainer.callbacks import SyncRefModelCallback

from .testing_utils import quantize_like_fp8_checkpoint, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


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


@pytest.mark.parametrize("trainable", [False, True])
def test_fp8_linear_does_not_flush_tiny_gradients_to_zero(trainable):
    linear = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16).requires_grad_(trainable)
    torch.nn.init.constant_(linear.weight, 1 / 16)
    x = torch.randn(3, 64, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.tensor([[2.0**-20], [2.0**-40], [0.0]]).expand(3, 64)

    FP8Linear(linear)(x).backward(grad.bfloat16())

    torch.testing.assert_close(x.grad.float(), 4 * grad, rtol=0, atol=0)


def test_fp8_layers_propagate_small_gradients_to_earlier_layers():
    torch.manual_seed(0)
    layers = [torch.nn.Linear(64, 64, dtype=torch.bfloat16) for _ in range(4)]
    for layer in layers[1:]:
        layer.requires_grad_(False)
    fp8_model = torch.nn.Sequential(layers[0], *[FP8Linear(layer) for layer in layers[1:]])
    x = torch.randn(512, 64, dtype=torch.bfloat16)

    # A mean over many tokens gives the small per-token gradients of real training
    fp8_model(x).float().mean().mul(1e-4).backward()
    fp8_grad = layers[0].weight.grad.float()
    layers[0].weight.grad = None
    torch.nn.Sequential(*layers)(x).float().mean().mul(1e-4).backward()

    expected_grad = layers[0].weight.grad.float()
    assert (fp8_grad - expected_grad).norm() / expected_grad.norm() < 0.05


def test_block_fp8_linear_matches_its_dequantized_weight_forward_and_backward():
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 384, dtype=torch.bfloat16)
    quantize_like_fp8_checkpoint(linear)
    scale = linear.weight_scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    weight = linear.weight.float() * scale
    layer = BlockFP8Linear(linear)
    x = torch.randn(2, 5, 256, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(2, 5, 384) * 2.0**-20

    out = layer(x)
    out.backward(grad.bfloat16())

    # Only the FP8 rounding of the activations and gradients, per group of 128 values, remains
    expected_out = x.detach().float() @ weight.T + linear.bias.float()
    assert (out.float() - expected_out).norm() / expected_out.norm() < 0.05
    expected_grad = grad @ weight
    assert (x.grad.float() - expected_grad).norm() / expected_grad.norm() < 0.05


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


@require_peft
def test_fp8_lora_merges_into_the_dequantized_base():
    model = get_peft_model(
        torch.nn.ModuleDict({"proj": torch.nn.Linear(64, 48)}),
        LoraConfig(r=2, target_modules=["proj"], init_lora_weights=False),
    )
    layer = model.base_model.model["proj"]
    layer.base_layer = FP8Linear(layer.base_layer)
    x = torch.randn(5, 64)
    expected = layer(x)

    dequantize_fp8_layers(model, torch.float32)
    merged = model.merge_and_unload()["proj"]

    assert type(merged) is torch.nn.Linear
    assert (merged(x) - expected).norm() / expected.norm() < 0.05
