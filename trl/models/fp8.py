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

from torch import nn
from transformers.utils import is_torchao_available


if is_torchao_available():
    from torchao.float8 import Float8GemmConfig, Float8LinearConfig, convert_to_float8_training


def convert_to_fp8_training(model: nn.Module, recipe: str) -> None:
    """
    Run the linear layers of a model with FP8 GEMMs, keeping their parameters in their original precision.

    The LM head, PEFT adapter layers, and layers whose dimensions aren't multiples of 16 are left unchanged. FP8 is
    emulated off CUDA.

    Args:
        model (`nn.Module`):
            Model to convert in place.
        recipe (`str`):
            TorchAO float8 recipe, e.g. `"rowwise_with_gw_hp"`.
    """
    config = dataclasses.replace(
        Float8LinearConfig.from_recipe_name(recipe),
        # Accurate accumulation in every GEMM
        gemm_config_output=Float8GemmConfig(use_fast_accum=False),
        emulate=next(model.parameters()).device.type != "cuda",
    )
    head = model.get_output_embeddings()

    def is_eligible(module: nn.Module, name: str) -> bool:
        return (
            isinstance(module, nn.Linear)
            and module.weight is not head.weight
            and "lora_" not in name
            and module.in_features % 16 == 0
            and module.out_features % 16 == 0
        )

    convert_to_float8_training(model, module_filter_fn=is_eligible, config=config)
