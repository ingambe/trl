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

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import load_file
from transformers.pytorch_utils import Conv1D
from transformers.utils import is_peft_available

import trl.generation.vllm_generation as vllm_generation_module
from trl.models.fp8 import BlockFP8Linear, FP8Linear, quantize_rowwise

from .testing_utils import quantize_like_fp8_checkpoint, require_peft


if is_peft_available():
    from peft import LoraConfig, get_peft_model

# The `vllm_generation` fixture stubs it out
init_vllm = vllm_generation_module.VLLMGeneration._init_vllm


def load_by_name(model):
    """Weight loader that resolves names like vLLM's, e.g. `Qwen2ForCausalLM.load_weights`."""

    def load_weights(weights):
        params = dict(model.named_parameters())
        for name, weight in weights:
            params[name].data.copy_(weight)

    return load_weights


def test_shared_weights_are_updated_in_place_and_never_published(vllm_generation):
    model = torch.nn.ModuleDict(
        {
            "gate_proj": torch.nn.Linear(2, 3),
            "up_proj": torch.nn.Linear(2, 3),
            "qkv_proj": torch.nn.Linear(2, 6),
            "c_proj": Conv1D(3, 3),
        }
    )
    # vLLM fuses gate/up, keeps Phi-3's fused qkv, and transposes GPT-2's Conv1D, even when zero
    vllm_model = torch.nn.ModuleDict(
        {"gate_up_proj": torch.nn.Linear(2, 6), "qkv_proj": torch.nn.Linear(2, 6), "c_proj": torch.nn.Linear(3, 3)}
    )
    torch.nn.init.zeros_(model.c_proj.weight)

    def load_weights(weights):
        for name, weight in weights:
            module, _, kind = name.rpartition(".")
            if module in ("gate_proj", "up_proj"):
                fused = vllm_model.gate_up_proj.get_parameter(kind).data
                (fused[:3] if module == "gate_proj" else fused[3:]).copy_(weight)
            else:
                vllm_model[module].get_parameter(kind).data.copy_(weight.T if name == "c_proj.weight" else weight)

    vllm_model.load_weights = load_weights
    load_weights(model.named_parameters())  # vLLM loads the same checkpoint
    vllm_model.gate_up_proj.output_sizes = [3, 3]
    vllm_model.qkv_proj.output_sizes = [2, 2, 2]
    vllm_model.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"], "qkv_proj": ["qkv_proj"]}
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True
    expected = torch.cat([model.gate_proj.weight, model.up_proj.weight]).detach().clone()

    vllm_generation._share_weights()
    with torch.no_grad():
        model.up_proj.weight.add_(1.0)  # an optimizer step
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    expected[3:] += 1.0
    torch.testing.assert_close(vllm_model.gate_up_proj.weight, expected, rtol=0, atol=0)
    assert [name for name, _ in published] == ["c_proj.weight"]


def test_replaced_storage_is_shared_again(vllm_generation):
    model = torch.nn.ModuleDict({"proj": torch.nn.Linear(2, 3)})
    vllm_model = copy.deepcopy(model)
    vllm_model.load_weights = load_by_name(vllm_model)
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True

    vllm_generation._share_weights()
    model.proj.weight.data = torch.ones(3, 2)  # e.g. a checkpoint load
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    torch.testing.assert_close(vllm_model.proj.weight, torch.ones(3, 2), rtol=0, atol=0)
    assert model.proj.weight.data_ptr() == vllm_model.proj.weight.data_ptr()
    assert published == []


@require_peft
def test_lora_shares_the_base_and_publishes_only_the_adapter(vllm_generation, monkeypatch):
    model = get_peft_model(
        torch.nn.ModuleDict({"proj": torch.nn.Linear(2, 3)}), LoraConfig(r=1, target_modules=["proj"])
    )
    # vLLM's LoRA layers wrap the original layer
    vllm_model = torch.nn.ModuleDict({"proj": torch.nn.Module()})
    vllm_model.proj.base_layer = copy.deepcopy(model.base_model.model.proj.base_layer)
    vllm_model.load_weights = load_by_name(vllm_model)
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True
    vllm_generation.native_lora = True
    monkeypatch.setattr(
        vllm_generation_module,
        "LoRARequest",
        lambda name, lora_id, path: SimpleNamespace(lora_int_id=lora_id, lora_path=path),
        raising=False,
    )
    adapters = []
    vllm_generation.llm.llm_engine.add_lora = lambda request: adapters.append(
        load_file(f"{request.lora_path}/adapter_model.safetensors")
    )

    vllm_generation._share_weights()
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    assert model.base_model.model.proj.base_layer.weight.data_ptr() == vllm_model.proj.base_layer.weight.data_ptr()
    assert published == []
    assert set(adapters[0]) == {"base_model.model.proj.lora_A.weight", "base_model.model.proj.lora_B.weight"}


@require_peft
def test_lora_is_merged_for_generation_only(vllm_generation):
    model = get_peft_model(
        torch.nn.ModuleDict(
            {"proj": torch.nn.Linear(2, 3), "other": torch.nn.Linear(2, 3), "embed": torch.nn.Embedding(4, 2)}
        ),
        LoraConfig(r=1, target_modules=["proj", "embed"], init_lora_weights=False),
    )
    # vLLM pads the vocabulary
    vllm_model = torch.nn.ModuleDict(
        {
            "proj": copy.deepcopy(model.base_model.model.proj.base_layer),
            "other": copy.deepcopy(model.base_model.model.other),
            "embed": torch.nn.Embedding(8, 2),
        }
    )
    vllm_model.load_weights = load_by_name(vllm_model)
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True
    proj, embed = model.base_model.model.proj, model.base_model.model.embed
    base = proj.base_layer.weight.detach().clone()

    vllm_generation._share_weights()
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    torch.testing.assert_close(vllm_model.proj.weight, base + proj.get_delta_weight("default"))
    vllm_generation.sleep()
    torch.testing.assert_close(proj.base_layer.weight, base, rtol=0, atol=0)
    assert proj.base_layer.weight.data_ptr() == vllm_model.proj.weight.data_ptr()
    assert model.base_model.model.other.weight.data_ptr() == vllm_model.other.weight.data_ptr()
    [(name, weight)] = published
    assert name == "embed.weight"
    torch.testing.assert_close(weight, embed.base_layer.weight + embed.get_delta_weight("default"))
    vllm_generation.llm.llm_engine.add_lora.assert_not_called()


def fp8_vllm_model(weights):
    """vLLM model that loaded `weights` of q, k (fused, transposed by the kernel) and o projections in FP8."""

    def fp8_layer(out_features, transposed):
        layer = torch.nn.Module()
        weight = torch.zeros(out_features, 16, dtype=torch.float8_e4m3fn)
        layer.weight = torch.nn.Parameter(weight.t() if transposed else weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(torch.zeros(out_features, 1))
        layer.bias = torch.nn.Parameter(torch.zeros(out_features))
        return layer

    vllm_model = torch.nn.ModuleDict({"qk_proj": fp8_layer(32, transposed=True), "o_proj": fp8_layer(16, False)})
    vllm_model.qk_proj.output_sizes = [16, 16]
    vllm_model.packed_modules_mapping = {"qk_proj": ["q_proj", "k_proj"]}

    def load_weights(weights):
        for name, weight in weights:
            module, _, kind = name.rpartition(".")
            if module in ("q_proj", "k_proj"):
                fused = vllm_model.qk_proj.get_parameter(kind).data
                (fused[:16] if module == "q_proj" else fused[16:]).copy_(weight)
            else:
                vllm_model[module].get_parameter(kind).data.copy_(weight)

    qk_rows, qk_scales = vllm_model.qk_proj.weight.data.t(), vllm_model.qk_proj.weight_scale.data
    views = {
        "q_proj": (qk_rows[:16], qk_scales[:16]),
        "k_proj": (qk_rows[16:], qk_scales[16:]),
        "o_proj": (vllm_model.o_proj.weight.data, vllm_model.o_proj.weight_scale.data),
    }
    for name, weight in weights.items():
        weight_fp8, weight_scale = quantize_rowwise(weight.detach())
        views[name][0].copy_(weight_fp8)
        views[name][1].copy_(weight_scale)
    vllm_model.load_weights = load_weights
    return vllm_model


@require_peft
def test_lora_shares_the_fp8_base_in_any_vllm_layout(vllm_generation):
    model = get_peft_model(
        torch.nn.ModuleDict({name: torch.nn.Linear(16, 16) for name in ("q_proj", "k_proj", "o_proj")}),
        LoraConfig(r=1, target_modules=["q_proj", "k_proj", "o_proj"]),
    )
    layers = {name: model.base_model.model[name] for name in ("q_proj", "k_proj", "o_proj")}
    vllm_model = fp8_vllm_model({name: layer.base_layer.weight for name, layer in layers.items()})
    for layer in layers.values():
        layer.base_layer = FP8Linear(layer.base_layer)
    x = torch.randn(3, 16)
    expected = {name: layer(x) for name, layer in layers.items()}
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True
    vllm_generation.native_lora = True

    unshared = vllm_generation._share_weights()

    assert unshared == []
    for name, layer in layers.items():
        torch.testing.assert_close(layer(x), expected[name], rtol=0, atol=0)
    rows = vllm_model.qk_proj.weight.t()
    assert layers["k_proj"].base_layer.weight_fp8.data_ptr() == rows[16:].data_ptr()
    assert layers["k_proj"].base_layer.weight_scale.data_ptr() == vllm_model.qk_proj.weight_scale[16:].data_ptr()
    assert layers["o_proj"].base_layer.weight_fp8.data_ptr() == vllm_model.o_proj.weight.data_ptr()
    assert layers["q_proj"].base_layer.bias.data_ptr() == vllm_model.qk_proj.bias.data_ptr()


def test_fp8_copies_of_updated_weights_are_written_into_vllm(vllm_generation):
    model = torch.nn.ModuleDict({name: FP8Linear(torch.nn.Linear(16, 16)) for name in ("q_proj", "k_proj", "o_proj")})
    vllm_model = fp8_vllm_model({name: layer.weight for name, layer in model.items()})
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model = model
    vllm_generation.share_weights = True

    vllm_generation._share_weights()
    with torch.no_grad():
        model.k_proj.weight.add_(1.0)  # an optimizer step
    published = []
    vllm_model.load_weights = lambda weights: published.extend(weights)
    vllm_generation.sync_weights()

    weight_fp8, weight_scale = quantize_rowwise(model.k_proj.weight.detach())
    assert torch.equal(vllm_model.qk_proj.weight.t()[16:].view(torch.uint8), weight_fp8.view(torch.uint8))
    torch.testing.assert_close(vllm_model.qk_proj.weight_scale[16:], weight_scale, rtol=0, atol=0)
    assert model.k_proj.bias.data_ptr() == vllm_model.qk_proj.bias[16:].data_ptr()
    assert published == []


def test_fp8_sharing_rejects_a_vllm_loader_that_reorders_rows(vllm_generation):
    model = torch.nn.ModuleDict({name: FP8Linear(torch.nn.Linear(16, 16)) for name in ("q_proj", "k_proj", "o_proj")})
    # Same shapes, but vLLM's loader swapped the q and k rows, as GPT-NeoX's QKV repacking does
    weights = {"q_proj": model.k_proj.weight, "k_proj": model.q_proj.weight, "o_proj": model.o_proj.weight}
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = fp8_vllm_model(weights)
    vllm_generation.model = model
    vllm_generation.share_weights = True

    with pytest.raises(ValueError, match="layout of `q_proj`"):
        vllm_generation._share_weights()


def test_fp8_sharing_rejects_skipping_part_of_a_fused_projection(vllm_generation, monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.setenv(name, "0")  # restored after the test
    monkeypatch.setattr(vllm_generation_module, "is_vllm_available", lambda min_version=None: True)
    # q_proj and k_proj are one layer in vLLM
    model = torch.nn.ModuleDict({"q_proj": FP8Linear(torch.nn.Linear(16, 16)), "k_proj": torch.nn.Linear(16, 16)})
    vllm_generation.model, vllm_generation.mode, vllm_generation.share_weights = model, "colocate", True

    with pytest.raises(ValueError, match="`k_proj`"):
        init_vllm(vllm_generation)


def test_fp8_server_must_keep_the_trainer_layers_in_high_precision(vllm_generation, monkeypatch):
    client = Mock()
    client.get_model_config.return_value = {"quantization": "fp8_per_channel", "quantization_config": {"ignore": []}}
    monkeypatch.setattr(vllm_generation_module, "VLLMClient", lambda **kwargs: client)
    monkeypatch.setattr(vllm_generation_module, "is_vllm_available", lambda min_version=None: True)
    model = torch.nn.ModuleDict({"q_proj": FP8Linear(torch.nn.Linear(16, 16)), "lm_head": torch.nn.Linear(16, 16)})
    vllm_generation.model, vllm_generation.mode = model, "server"

    with pytest.raises(ValueError, match="--quantization-config"):
        init_vllm(vllm_generation)
    client.get_model_config.return_value["quantization_config"]["ignore"] = ["lm_head", "*.experts"]
    init_vllm(vllm_generation)


@require_peft
def test_lora_over_a_blockwise_fp8_checkpoint_leaves_vllm_its_own_fp8_base(vllm_generation):
    linears = {name: torch.nn.Linear(128, 128, bias=False) for name in ("q_proj", "k_proj")}
    for linear in linears.values():
        quantize_like_fp8_checkpoint(linear)
    # vLLM loaded the same checkpoint into one fused layer
    vllm_model = torch.nn.ModuleDict({"qk_proj": torch.nn.Module()})
    for name in ("weight", "weight_scale_inv"):
        fused = torch.cat([linear.get_parameter(name) for linear in linears.values()])
        vllm_model.qk_proj.register_parameter(name, torch.nn.Parameter(fused, requires_grad=False))
    vllm_model.qk_proj.output_sizes = [128, 128]
    vllm_model.packed_modules_mapping = {"qk_proj": ["q_proj", "k_proj"]}
    model = get_peft_model(torch.nn.ModuleDict(linears), LoraConfig(r=1, target_modules=["q_proj", "k_proj"]))
    for name in linears:
        layer = model.base_model.model[name]
        layer.base_layer = BlockFP8Linear(layer.base_layer)
    runner = vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner
    runner.model = vllm_model
    vllm_generation.model, vllm_generation.share_weights, vllm_generation.native_lora = model, True, True

    assert vllm_generation._share_weights() == []
