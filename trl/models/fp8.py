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

import dataclasses
from fnmatch import fnmatch

import torch
from torch import nn
from transformers.utils import is_torchao_available


if is_torchao_available():
    from torchao.float8 import Float8GemmConfig, Float8LinearConfig, convert_to_float8_training
    from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
    from torchao.quantization.quantize_.common import KernelPreference


def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row of `x` to float8_e4m3fn with a float32 scale, as vLLM's per-channel FP8 does."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / fp8_max).clamp(min=1 / (fp8_max * 512))
    return (x.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn), scale


class _FP8Matmul(torch.autograd.Function):
    """`x @ (weight_scale * weight_fp8).T` with FP8 GEMMs and per-token scales for `x` and its gradient."""

    @staticmethod
    def forward(ctx, x, weight_fp8, weight_scale):
        ctx.save_for_backward(weight_fp8, weight_scale)
        x_fp8, x_scale = quantize_rowwise(x.reshape(-1, x.shape[-1]))
        out = torch._scaled_mm(x_fp8, weight_fp8.t(), x_scale, weight_scale.t(), out_dtype=torch.bfloat16)
        return out.to(x.dtype).view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        weight_fp8, weight_scale = ctx.saved_tensors
        # The weight scales run along the reduced dimension, so they are folded into the gradient
        grad = grad_output.reshape(-1, grad_output.shape[-1]).float() * weight_scale.t()
        grad_fp8, grad_scale = quantize_rowwise(grad)
        ones = weight_scale.new_ones(1, weight_fp8.shape[1])
        grad_input = torch._scaled_mm(
            grad_fp8, weight_fp8.t().contiguous().t(), grad_scale, ones, out_dtype=torch.bfloat16
        )
        return grad_input.to(grad_output.dtype).view(*grad_output.shape[:-1], -1), None, None


class FP8Linear(nn.Module):
    """
    Frozen linear layer stored as FP8 with per-output-channel scales, computed with FP8 GEMMs.

    Args:
        linear (`nn.Linear`):
            Layer to quantize. Its high-precision weight is not kept.
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        weight_fp8, weight_scale = quantize_rowwise(linear.weight.detach())
        self.register_buffer("weight_fp8", weight_fp8)
        self.register_buffer("weight_scale", weight_scale)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _FP8Matmul.apply(x, self.weight_fp8, self.weight_scale)
        return out if self.bias is None else out + self.bias


def convert_to_fp8_training(model: nn.Module, recipe: str, skip_modules: list[str] | None = None) -> None:
    """
    Run the linear layers of a model with FP8 GEMMs, keeping their parameters in their original precision.

    With the rowwise recipes, frozen layers (such as the base of a PEFT model) are replaced by [`FP8Linear`], which
    drops their high-precision weight. The LM head, PEFT adapter layers, layers whose dimensions aren't multiples of
    16 (32 for MXFP8), and layers matching `skip_modules` are left unchanged. TorchAO's FP8 is emulated off CUDA.

    Args:
        model (`nn.Module`):
            Model to convert in place.
        recipe (`str`):
            TorchAO float8 recipe, e.g. `"rowwise_with_gw_hp"`, or `"mxfp8_with_gw_hp"` for MXFP8 with
            high-precision weight gradients.
        skip_modules (`list[str]`, *optional*):
            Glob patterns of the module names to keep in high precision, e.g. `"model.layers.0.*"`. PEFT prefixes and
            `.base_layer` are removed from the names before matching.
    """
    emulate = next(model.parameters()).device.type != "cuda"
    mxfp8 = recipe == "mxfp8_with_gw_hp"
    multiple = 32 if mxfp8 else 16
    head = model.get_output_embeddings()

    def is_eligible(module: nn.Module, name: str) -> bool:
        return (
            isinstance(module, nn.Linear)
            and module.weight is not head.weight
            and "lora_" not in name
            and module.in_features % multiple == 0
            and module.out_features % multiple == 0
            and not any(
                fnmatch(name.removeprefix("base_model.model.").replace(".base_layer", ""), pattern)
                for pattern in skip_modules or []
            )
        )

    for name, module in list(model.named_modules()):
        if not is_eligible(module, name):
            continue
        if mxfp8:
            fp8_module = MXFP8Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device="meta",
                kernel_preference=KernelPreference.EMULATED if emulate else KernelPreference.AUTO,
                wgrad_with_hp=True,
            )
            fp8_module.weight, fp8_module.bias = module.weight, module.bias
        elif recipe.startswith("rowwise") and not module.weight.requires_grad:
            fp8_module = FP8Linear(module)
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, fp8_module)

    if not mxfp8:
        config = dataclasses.replace(
            Float8LinearConfig.from_recipe_name(recipe),
            # Accurate accumulation in every GEMM
            gemm_config_output=Float8GemmConfig(use_fast_accum=False),
            emulate=emulate,
        )
        convert_to_float8_training(model, module_filter_fn=is_eligible, config=config)
