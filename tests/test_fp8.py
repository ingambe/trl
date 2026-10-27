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

import weakref

import pytest
import torch
from transformers.utils import is_peft_available

from trl.models.fp8 import BlockFP8Linear, FP8Linear, dequantize_fp8_layers, quantize_rowwise

from .testing_utils import quantize_like_fp8_checkpoint, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model


@pytest.mark.parametrize("trainable, cache_frozen_transpose", [(False, False), (False, True), (True, False)])
def test_fp8_linear_matches_its_dequantized_weight_forward_and_backward(trainable, cache_frozen_transpose):
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 48, dtype=torch.bfloat16).requires_grad_(trainable)
    weight_fp8, weight_scale = quantize_rowwise(linear.weight.detach())
    weight = weight_fp8.float() * weight_scale
    layer = FP8Linear(linear, cache_frozen_transpose=cache_frozen_transpose)
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


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("cache_frozen_transpose", [False, True])
def test_fp8_linear_backward_accepts_column_major_gradients(compiled, cache_frozen_transpose):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("requires native FP8 CUDA matmul")
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 64, bias=False, device="cuda", dtype=torch.bfloat16).requires_grad_(False)
    layer = FP8Linear(linear, cache_frozen_transpose=cache_frozen_transpose)
    weight = layer.weight_fp8.float() * layer.weight_scale
    forward = torch.compile(layer, dynamic=True, fullgraph=True) if compiled else layer

    for tokens in (32, 48):
        x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        # Transposed upstream gradients are valid autograd inputs and occur in
        # the compiled decoder backward even when the forward input is dense.
        grad = torch.randn(64, tokens, device="cuda", dtype=torch.bfloat16).t()
        forward(x).backward(grad)

        expected = grad.float() @ weight
        assert torch.isfinite(x.grad).all()
        assert (x.grad.float() - expected).norm() / expected.norm() < 0.05


@pytest.mark.parametrize("trainable, code", [(False, 3.5), (True, 3.25)])
def test_fp8_linear_rounds_a_frozen_base_like_llm_compressor(trainable, code):
    linear = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16).requires_grad_(trainable)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 0.00750732421875]]))

    layer = FP8Linear(linear)

    scale = 1 / 448 if trainable else torch.tensor(1 / 448, dtype=torch.bfloat16).item()
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_scale.item() == pytest.approx(scale, rel=1e-7)
    assert layer.weight_fp8.float().tolist() == [[448.0, code]]


@pytest.mark.parametrize("trainable", [False, True])
def test_fp8_linear_does_not_flush_tiny_gradients_to_zero(trainable):
    linear = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16).requires_grad_(trainable)
    torch.nn.init.constant_(linear.weight, 1 / 16)
    x = torch.randn(3, 64, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.tensor([[2.0**-20], [2.0**-40], [0.0]]).expand(3, 64)

    FP8Linear(linear)(x).backward(grad.bfloat16())

    torch.testing.assert_close(x.grad.float(), 4 * grad, rtol=0, atol=0)


def test_fp8_linear_requantizes_a_changed_input_and_releases_it():
    layer = FP8Linear(torch.nn.Linear(64, 32, dtype=torch.bfloat16))
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    layer(x)
    with torch.no_grad():
        x.zero_()
        torch.testing.assert_close(layer(x), layer.bias.expand(4, -1), rtol=0, atol=0)

    with torch.inference_mode():
        torch.testing.assert_close(layer(torch.zeros(4, 64, dtype=torch.bfloat16)), layer.bias.expand(4, -1))

    x_ref = weakref.ref(x)
    del x
    assert x_ref() is None


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


@pytest.mark.parametrize("cache_frozen_transpose", [False, True])
def test_block_fp8_linear_matches_its_dequantized_weight_forward_and_backward(cache_frozen_transpose):
    torch.manual_seed(0)
    linear = torch.nn.Linear(256, 384, dtype=torch.bfloat16)
    quantize_like_fp8_checkpoint(linear)
    scale = linear.weight_scale_inv.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    weight = linear.weight.float() * scale
    layer = BlockFP8Linear(linear, cache_frozen_transpose)
    x = torch.randn(2, 5, 256, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(2, 5, 384) * 2.0**-20

    out = layer(x)
    out.backward(grad.bfloat16())

    # Only the FP8 rounding of the activations and gradients, per group of 128 values, remains
    expected_out = x.detach().float() @ weight.T + linear.bias.float()
    assert (out.float() - expected_out).norm() / expected_out.norm() < 0.05
    expected_grad = grad @ weight
    assert (x.grad.float() - expected_grad).norm() / expected_grad.norm() < 0.05


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
