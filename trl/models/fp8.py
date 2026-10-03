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


if is_torchao_available("0.18.0"):
    from torchao.float8 import Float8GemmConfig, Float8LinearConfig, convert_to_float8_training
    from torchao.prototype.moe_training.mxfp8_linear import MXFP8Linear
    from torchao.quantization.quantize_.common import KernelPreference

    class TokenPaddedMXFP8Linear(MXFP8Linear):
        """MXFP8 layer padding the tokens to a multiple of 32, which its weight-gradient GEMM scales in blocks."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            tokens = x.reshape(-1, x.shape[-1])
            out = super().forward(nn.functional.pad(tokens, (0, 0, 0, -len(tokens) % 32)))
            return out[: len(tokens)].view(*x.shape[:-1], -1)


def quantize_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each row of `x` to float8_e4m3fn with a float32 scale, as vLLM's per-channel FP8 does."""
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = (x.abs().amax(dim=-1, keepdim=True).float() / fp8_max).clamp(min=1 / (fp8_max * 512))
    return (x.float() / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn), scale


class _FP8Matmul(torch.autograd.Function):
    """
    `x @ (weight_scale * weight_fp8).T` with FP8 GEMMs and per-token scales for `x` and its gradient. The gradient of
    the high-precision `weight` it was quantized from, if given, is computed in high precision.
    """

    @staticmethod
    def forward(ctx, x, weight_fp8, weight_scale, weight, fast_accum):
        ctx.save_for_backward(x if weight is not None else None, weight_fp8, weight_scale)
        ctx.input_shape, ctx.weight_dtype, ctx.fast_accum = (
            x.shape,
            None if weight is None else weight.dtype,
            fast_accum,
        )
        x_fp8, x_scale = quantize_rowwise(x.reshape(-1, x.shape[-1]))
        out = torch._scaled_mm(
            x_fp8,
            weight_fp8.t(),
            x_scale,
            weight_scale.t(),
            out_dtype=torch.bfloat16,
            use_fast_accum="output" in fast_accum,
        )
        return out.to(x.dtype).view(*x.shape[:-1], -1)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight_fp8, weight_scale = ctx.saved_tensors
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        # The weight scales run along the reduced dimension, so they are folded into the gradient
        grad_fp8, grad_scale = quantize_rowwise(grad_output.float() * weight_scale.t())
        ones = weight_scale.new_ones(1, weight_fp8.shape[1])
        grad_input = torch._scaled_mm(
            grad_fp8,
            weight_fp8.t().contiguous().t(),
            grad_scale,
            ones,
            out_dtype=torch.bfloat16,
            use_fast_accum="grad_input" in ctx.fast_accum,
        )
        grad_weight = None
        if x is not None:
            x = x.reshape(-1, x.shape[-1])
            grad_weight = (grad_output.t().to(x.dtype) @ x).to(ctx.weight_dtype)
        return grad_input.to(grad_output.dtype).view(ctx.input_shape), None, None, grad_weight, None


class FP8Linear(nn.Module):
    """
    Linear layer computed with FP8 GEMMs from an FP8 copy of its weight with per-output-channel scales, the layout of
    vLLM's per-channel FP8.

    A frozen layer keeps only the FP8 copy. A trainable one keeps its high-precision weight, requantizes it when it
    changes, and computes its gradient in high precision.

    Args:
        linear (`nn.Linear`):
            Layer to convert. Its high-precision weight is kept only if it requires gradients.
        fast_accum (`list[str]`, *optional*):
            GEMMs (`"output"`, `"grad_input"`) that use the faster, less accurate FP8 accumulation.
    """

    def __init__(self, linear: nn.Linear, fast_accum: list[str] | None = None):
        super().__init__()
        self.fast_accum = fast_accum or []
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight if linear.weight.requires_grad else None
        weight_fp8, weight_scale = quantize_rowwise(linear.weight.detach())
        self.register_buffer("weight_fp8", weight_fp8, persistent=self.weight is None)
        self.register_buffer("weight_scale", weight_scale, persistent=self.weight is None)
        self.bias = linear.bias
        self._quantized_version = linear.weight._version

    @torch.no_grad()
    def quantize_weight(self) -> None:
        """Requantize the FP8 copy in place if the high-precision weight changed since the last quantization."""
        if self.weight is not None and self.weight._version != self._quantized_version:
            weight_fp8, weight_scale = quantize_rowwise(self.weight)
            self.weight_fp8.copy_(weight_fp8)
            self.weight_scale.copy_(weight_scale)
            self._quantized_version = self.weight._version

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.quantize_weight()
        out = _FP8Matmul.apply(x, self.weight_fp8, self.weight_scale, self.weight, self.fast_accum)
        return out if self.bias is None else out + self.bias.to(out.dtype)


def convert_to_fp8_training(
    model: nn.Module, recipe: str, skip_modules: list[str] | None = None, fast_accum: list[str] | None = None
) -> None:
    """
    Run the linear layers of a model with FP8 GEMMs, keeping their parameters in their original precision.

    `"rowwise_with_gw_hp"` uses [`FP8Linear`], whose FP8 weights vLLM can share, and so does `"rowwise"` for frozen
    layers. Frozen [`FP8Linear`] layers, such as the base of a PEFT model, drop their high-precision weight. The LM
    head, PEFT adapter layers, layers whose dimensions aren't multiples of 16 (32 for MXFP8), and layers matching
    `skip_modules` are left unchanged. TorchAO's FP8 is emulated without CUDA.

    Args:
        model (`nn.Module`):
            Model to convert in place.
        recipe (`str`):
            `"rowwise_with_gw_hp"`, TorchAO float8 recipes (`"rowwise"` or `"tensorwise"`), or MXFP8 (`"mxfp8"`, or
            `"mxfp8_with_gw_hp"` with high-precision weight gradients).
        skip_modules (`list[str]`, *optional*):
            Glob patterns of the module names to keep in high precision, e.g. `"model.layers.0.*"`. PEFT prefixes and
            `.base_layer` are removed from the names before matching.
        fast_accum (`list[str]`, *optional*):
            FP8 GEMMs (`"output"`, `"grad_input"`, `"grad_weight"`) that use the faster, less accurate accumulation.
            Not available with MXFP8.
    """
    if recipe not in ("rowwise_with_gw_hp", "rowwise", "tensorwise", "mxfp8_with_gw_hp", "mxfp8"):
        raise ValueError(f"Unknown FP8 recipe: {recipe!r}.")
    if recipe != "rowwise_with_gw_hp" and not is_torchao_available("0.18.0"):
        raise ImportError(f"The {recipe!r} FP8 recipe requires torchao 0.18.0 or later: `pip install torchao`.")
    if any("lora_magnitude_vector" in name for name, _ in model.named_parameters()):
        raise ValueError("FP8 training doesn't support DoRA, which needs the high-precision base weights.")
    emulate = not torch.cuda.is_available()
    mxfp8 = recipe.startswith("mxfp8")
    fast_accum = fast_accum or []
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
            fp8_module = (MXFP8Linear if recipe == "mxfp8_with_gw_hp" else TokenPaddedMXFP8Linear)(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device="meta",
                kernel_preference=KernelPreference.EMULATED if emulate else KernelPreference.AUTO,
                wgrad_with_hp=recipe == "mxfp8_with_gw_hp",
            )
            fp8_module.weight, fp8_module.bias = module.weight, module.bias
        elif recipe == "rowwise_with_gw_hp" or (recipe == "rowwise" and not module.weight.requires_grad):
            fp8_module = FP8Linear(module, fast_accum)
        else:
            continue
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, fp8_module)

    if recipe in ("rowwise", "tensorwise"):
        config = dataclasses.replace(
            Float8LinearConfig.from_recipe_name(recipe),
            gemm_config_output=Float8GemmConfig(use_fast_accum="output" in fast_accum),
            gemm_config_grad_input=Float8GemmConfig(use_fast_accum="grad_input" in fast_accum),
            gemm_config_grad_weight=Float8GemmConfig(use_fast_accum="grad_weight" in fast_accum),
            emulate=emulate,
        )
        convert_to_float8_training(model, module_filter_fn=is_eligible, config=config)
