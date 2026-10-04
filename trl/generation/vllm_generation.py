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

"""vLLM-based generation backend for TRL trainers."""

import json
import logging
import math
import os
from collections import Counter
from contextlib import closing, contextmanager, nullcontext
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import torch
from accelerate.utils import broadcast_object_list, gather_object, is_peft_model
from packaging.version import Version
from safetensors.torch import save_file
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedModel, PreTrainedTokenizerBase, ProcessorMixin, is_bitsandbytes_available
from transformers.utils import (
    is_peft_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

from ..distributed import DistributedBackend
from ..extras.profiling import ProfilingContext
from ..import_utils import is_vllm_available
from ..models.fp8 import BlockFP8Linear, FP8Linear, fused_layer_name
from ..trainer.utils import ensure_master_addr_port
from .vllm_client import VLLMClient


if is_vllm_available():
    from vllm import LLM, RequestOutput, SamplingParams
    from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.sampling_params import StructuredOutputsParams


logger = logging.getLogger(__name__)


def empty_cache() -> None:
    """Empties the cache of the available torch device.

    This function checks for the availability of different torch devices (CUDA, MLU, MPS, NPU, XPU) and empties the
    cache of the first available device it finds.

    If none of the specific devices are available, it defaults to emptying the CUDA cache.
    """
    if is_torch_mlu_available():
        torch.mlu.empty_cache()
    elif is_torch_mps_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch.npu.empty_cache()
    elif is_torch_xpu_available():
        torch.xpu.empty_cache()
    else:
        torch.cuda.empty_cache()


def extract_logprobs(all_outputs: list["RequestOutput"]):
    """
    Extract logprobs and token IDs from vLLM generation outputs.

    Returns logprobs and token IDs sorted by rank (most probable first). Each returned list has shape (num_sequences,
    seq_len, num_logprobs), where num_logprobs is determined by the `logprobs` parameter passed to vLLM (1 when
    `logprobs=0`, up to N+1 when `logprobs=N`). NaN logprob values are replaced with `None`.

    Args:
        all_outputs (list of `RequestOutput`):
            List of vLLM `RequestOutput` objects from generation.

    Returns:
        Tuple of (logprobs, logprob_token_ids), each of shape (num_sequences, seq_len, num_logprobs).
    """
    all_logprobs = []
    all_token_ids = []
    for outputs in all_outputs:
        for output in outputs.outputs:
            if output.logprobs is None:
                return None, None
            seq_logprobs = []
            seq_token_ids = []
            for lp in output.logprobs:
                sorted_items = sorted(lp.items(), key=lambda x: x[1].rank)
                seq_token_ids.append([token_id for token_id, _ in sorted_items])
                seq_logprobs.append([None if math.isnan(item.logprob) else item.logprob for _, item in sorted_items])
            all_logprobs.append(seq_logprobs)
            all_token_ids.append(seq_token_ids)
    return all_logprobs, all_token_ids


if TYPE_CHECKING:
    from accelerate import Accelerator
    from peft import PeftModel


if is_bitsandbytes_available():
    import bitsandbytes as bnb

if is_peft_available():
    import peft
    from peft import LoraConfig, get_peft_model_state_dict
    from peft.tuners.tuners_utils import BaseTunerLayer


class VLLMGeneration:
    """Handles vLLM-based generation for trainers.

    Extracts all vLLM-specific logic (initialization, generation, weight sync) from trainers into a separate, testable
    class.

    Args:
        model ([`~transformers.PreTrainedModel`] or [`~peft.PeftModel`]):
            Model to use for generation.
        accelerator ([`~accelerate.Accelerator`]):
            Accelerator for distributed training.
        processing_class ([`~transformers.PreTrainedTokenizerBase`] or [`~transformers.ProcessorMixin`]):
            Tokenizer or processor for the model.

        > Parameters for vLLM:

        mode (`str`, *optional*, defaults to `"colocate"`):
            vLLM mode. Must be one of `"colocate"` or `"server"`.

            - `"colocate"`: vLLM will run in the same process and share the training GPUs. This avoids the need for a
              separate server but may cause resource contention with training.
            - `"server"`: The trainer will send generation requests to a separate vLLM server. Make sure a vLLM server
              is running (start with `vllm serve`).

        structured_outputs_regex (`str`, *optional*):
            Regex for vLLM structured outputs. If `None` (default), structured outputs is disabled.

        > Parameters for "server" vLLM mode:

        server_base_url (`str`, *optional*):
            Base URL for the vLLM server (e.g., `"http://localhost:8000"`). If provided, `server_host` and
            `server_port` are ignored.
        server_host (`str`, *optional*, defaults to `"0.0.0.0"`):
            Host of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_port (`int`, *optional*, defaults to `8000`):
            Port of the vLLM server to connect to. Ignored if `server_base_url` is provided.
        server_timeout (`float`, *optional*, defaults to `240.0`):
            Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
            timeout, a `ConnectionError` is raised.
        group_port (`int`, *optional*, defaults to `51216`):
            Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
            is occupied, there is no need to change it.

        > Parameters for "colocate" vLLM mode:

        tensor_parallel_size (`int`, *optional*, defaults to `1`):
            The number of GPUs to use for distributed execution with tensor parallelism. This setting only applies when
            `mode` is set to `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately
            when launching the vLLM server via the `--vllm_tensor_parallel_size` flag.
        gpu_memory_utilization (`float`, *optional*, defaults to `0.9`):
            Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's throughput. However, if the value is
            too high, it may cause out-of- memory (OOM) errors. This setting only applies when `mode` is set to
            `"colocate"`. If you are using `mode="server"`, this parameter must be passed separately when launching the
            vLLM server via the `--vllm_gpu_memory_utilization` flag.
        max_model_length (`int`, *optional*):
            Model context length (prompt and completion). Set it to at least the maximum prompt length in the dataset
            plus `max_completion_length`; if omitted, it is inferred from the model config.
        max_num_seqs (`int`, *optional*):
            Maximum number of sequences to process in parallel, effectively capping the batch size.
        max_num_batched_tokens (`int`, *optional*, defaults to `4096`):
            Maximum number of tokens processed per engine step. vLLM's larger default misleads its memory profiler.
        kv_cache_dtype (`str`, *optional*, defaults to `"auto"`):
            Data type of the KV cache, e.g. `"fp8_per_token_head"`.
        kv_cache_dtype_skip_layers (`list[str]`, *optional*):
            Layer indices or attention types whose KV cache keeps the model dtype. Requires vLLM 0.30.0 or later.
        enable_sleep_mode (`bool`, *optional*, defaults to `False`):
            Whether to enable sleep mode for the engine to offload weights/cache during the optimizer step. Keeps GPU
            memory usage low, but waking the engine adds host–device transfer latency.
        share_weights (`bool`, *optional*, defaults to `False`):
            Whether the model's parameters use vLLM's weight memory, so no weights are published and sleep only
            releases the KV cache. With PEFT, the adapter is merged for generation and removed by `sleep()`.
        native_lora (`bool`, *optional*, defaults to `False`):
            With `share_weights=True` and PEFT, serve the adapter as a native vLLM LoRA instead of merging it.
        model_impl (`str`, *optional*, defaults to `"auto"`):
            Model implementation to use for vLLM.
            - "auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers
              implementation if no vLLM implementation is available.
            - "vllm" will use the vLLM model implementation.
            - "transformers" will use the Transformers model implementation.
            - "terratorch" will use the TerraTorch model implementation.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.
        cast_lm_head_to_fp32 (`bool`, *optional*, defaults to `False`):
            Whether to compute the language modeling head in float32 in colocate mode. Requires vLLM 0.26.0 or later.

        > Parameters for generation:

        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Parameter for repetition penalty. It penalizes new tokens based on whether they appear in the prompt and
            the generated text so far. Values > 1 encourage the model to use new tokens, while values < 1 encourage the
            model to repeat tokens. Default `1.0` means no penalty.
        temperature (`float`, *optional*, defaults to `1.0`):
            Sampling temperature. It controls the randomness of the sampling. Lower values make the model more
            deterministic, while higher values make the model more random and increase diversity.
        top_p (`float`, *optional*, defaults to `1.0`):
            Top-p sampling parameter. It controls the cumulative probability of the top tokens to consider. Defaults to
            `1.0` to consider all tokens.
        top_k (`int`, *optional*, defaults to `0`):
            Top-k sampling parameter. It controls the number of top tokens to consider. Defaults to `0` to consider all
            tokens.
        min_p (`float`, *optional*, defaults to `0.0`):
            Min-p sampling parameter. It represents the minimum probability for a token to be considered, relative to
            the probability of the most likely token. Default `0.0` means min-p is disabled.
        max_completion_length (`int`, *optional*, defaults to `16`):
            Maximum number of tokens to generate for each prompt.
        logprobs (`int` or `None`, *optional*, defaults to `0`):
            Number of top logprobs to return per token. When 0 (default), only the sampled token's logprob is returned
            (inner dimension = 1). When N>0, returns up to N+1 logprobs sorted by descending probability, because vLLM
            always includes the sampled token's logprob alongside the top-N (the sampled token may or may not already
            be in the top-N).
        generation_kwargs (`dict`, *optional*):
            Additional generation parameters to pass to the vLLM `SamplingParams`. This can include parameters like
            `seed`, `frequency_penalty`, etc. If it contains keys that conflict with the other parameters, they will
            override them.

    """

    def __init__(
        self,
        model: "PreTrainedModel | PeftModel",
        accelerator: "Accelerator",
        processing_class: PreTrainedTokenizerBase | ProcessorMixin,
        # vLLM configuration
        mode: str = "colocate",
        structured_outputs_regex: str | None = None,
        # Server mode configuration
        server_base_url: str | None = None,
        server_host: str = "0.0.0.0",
        server_port: int = 8000,
        server_timeout: float = 240.0,
        group_port: int = 51216,
        # Colocate mode configuration
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_length: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int = 4096,
        kv_cache_dtype: str = "auto",
        kv_cache_dtype_skip_layers: list[str] | None = None,
        enable_sleep_mode: bool = False,
        share_weights: bool = False,
        native_lora: bool = False,
        model_impl: str = "auto",
        trust_remote_code: bool = False,
        cast_lm_head_to_fp32: bool = False,
        # Generation configuration
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        max_completion_length: int = 16,
        logprobs: int | None = 0,
        generation_kwargs: dict | None = None,
    ):
        self.model = model
        self.accelerator = accelerator
        self._dist = DistributedBackend(accelerator)
        self.processing_class = processing_class

        # vLLM configuration
        self.mode = mode
        self.structured_outputs_regex = structured_outputs_regex

        # Server mode configuration
        self.server_base_url = server_base_url
        self.server_host = server_host
        self.server_port = server_port
        self.group_port = group_port
        self.server_timeout = server_timeout

        # Colocate mode configuration
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_length = max_model_length
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_cache_dtype_skip_layers = kv_cache_dtype_skip_layers
        self.enable_sleep_mode = enable_sleep_mode
        self.share_weights = share_weights
        self.native_lora = native_lora
        self._lora_request = None
        self._merged_bases = []
        self._unshareable = None
        self.model_impl = model_impl
        self.trust_remote_code = trust_remote_code
        self.cast_lm_head_to_fp32 = cast_lm_head_to_fp32

        # Generation configuration
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_completion_length = max_completion_length
        self.logprobs = logprobs
        self.generation_kwargs = generation_kwargs or {}

        # Tensor names, dtypes and shapes streamed to the server on each weight sync. Collected on the first sync, as
        # it requires gathering the parameters, and constant afterwards.
        self._weight_metadata = None
        # Set during a weight sync, so a failed one is retried before generating
        self._weights_dirty = False

        self._init_vllm()

    def _init_vllm(self):
        """Initialize vLLM in server or colocate mode."""
        model = self.model
        accelerator = self.accelerator

        if not is_vllm_available():
            raise ImportError(
                "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                "`pip install trl[vllm]` to use it."
            )

        fp8_layers = [module for module in model.modules() if isinstance(module, FP8Linear)]
        frozen_fp8 = any(layer.weight is None for layer in fp8_layers) or any(
            isinstance(module, BlockFP8Linear) for module in model.modules()
        )
        if frozen_fp8 and not (
            self.mode == "colocate" and self.share_weights and is_peft_model(model) and self.native_lora
        ):
            raise ValueError(
                "vLLM can only serve a PEFT adapter over an FP8 base in colocate mode with `share_weights=True` and "
                "`native_lora=True`: merging the adapter into the base would change the policy."
            )

        if self.mode == "server":
            if accelerator.is_main_process:
                if self.server_base_url is not None:
                    base_url = self.server_base_url
                else:
                    base_url = f"http://{self.server_host}:{self.server_port}"
                self.vllm_client = VLLMClient(
                    base_url=base_url, group_port=self.group_port, connection_timeout=self.server_timeout
                )
                self.vllm_client.init_communicator(device=accelerator.device)
                if fp8_layers:
                    # The server requantizes the high-precision weights it receives on each update
                    server_config = self.vllm_client.get_model_config()
                    quantization = server_config["quantization"]
                    if quantization == "fp8_per_channel":
                        ignore = self._fp8_ignore_list(model)
                        if sorted(server_config["quantization_config"]["ignore"]) != sorted(ignore):
                            raise ValueError(
                                "The vLLM server must keep the same layers in high precision as the trainer: start it "
                                f"with `--quantization-config '{json.dumps({'ignore': ignore})}'`."
                            )
                    elif quantization is not None:
                        raise ValueError(
                            f"The vLLM server runs `{quantization}` quantization, unlike `fp8_recipe`: serve the "
                            "high-precision checkpoint, with `--quantization fp8_per_channel` for FP8 inference."
                        )

        elif self.mode == "colocate":
            # Make sure tensor_parallel_size group size evenly divides the world size - each group should have
            # the same number of ranks
            if not accelerator.num_processes % self.tensor_parallel_size == 0:
                raise ValueError(
                    f"tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )

            if self.tensor_parallel_size > 1:
                # Create subgroups of ranks for TP, each group with `tensor_parallel_size` ranks.
                # For example, if world_size=8 and tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                    [
                        list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
                        for i in range(accelerator.num_processes // self.tensor_parallel_size)
                    ]
                )

            # vLLM requires the environment variables to be set for distributed training.
            os.environ["RANK"] = str(accelerator.process_index)
            os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
            os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
            # Ensure distributed rendezvous variables are set without colliding across concurrent runs
            ensure_master_addr_port()

            quantization = None
            if is_bitsandbytes_available():
                for _, module in model.named_modules():
                    if isinstance(module, bnb.nn.Linear4bit):
                        quantization = "bitsandbytes"
                        break
                    elif isinstance(module, bnb.nn.Linear8bitLt):
                        raise ValueError("vLLM does not support in-flight 8-bit quantization.")

            if self.share_weights and (
                self.tensor_parallel_size > 1 or self._dist.is_fsdp or self._dist.is_zero3 or quantization is not None
            ):
                raise ValueError(
                    "`share_weights=True` requires `tensor_parallel_size=1`, no FSDP or DeepSpeed ZeRO-3, and no "
                    "quantization."
                )
            fp8_kwargs = {}
            if self.kv_cache_dtype_skip_layers:
                fp8_kwargs["kv_cache_dtype_skip_layers"] = self.kv_cache_dtype_skip_layers
            if self.share_weights and fp8_layers:
                if not is_vllm_available(min_version="0.30.0"):
                    raise ImportError("Sharing FP8 weights with vLLM requires vLLM 0.30.0 or later.")
                # vLLM quantizes the same layers as the trainer, whose FP8 weights it then shares
                quantization = "fp8_per_channel"
                fp8_kwargs["quantization_config"] = {"ignore": self._fp8_ignore_list(model)}
            lora_kwargs = {}
            if self.share_weights and is_peft_model(model):
                config = model.peft_config[model.active_adapters[0]]
                tied = Counter(id(param) for _, param in model.named_parameters(remove_duplicate=False))
                if (
                    len(model.active_adapters) != 1
                    or not isinstance(config, LoraConfig)
                    or config.use_dora
                    or config.bias != "none"
                    or config.modules_to_save
                    # Added in PEFT 0.14.0, 0.17.0 and 0.18.0
                    or (Version(peft.__version__) >= Version("0.14.0") and config.lora_bias)
                    or (Version(peft.__version__) >= Version("0.17.0") and config.target_parameters)
                    or (Version(peft.__version__) >= Version("0.18.0") and config.alora_invocation_tokens)
                    or config.rank_pattern
                    or config.alpha_pattern
                    or any(
                        isinstance(module, BaseTunerLayer) and tied[id(module.get_base_layer().weight)] > 1
                        for module in model.modules()
                    )
                ):
                    raise ValueError(
                        "`share_weights=True` with PEFT requires a single plain LoRA adapter: no DoRA, trained biases, "
                        "`lora_bias`, `modules_to_save`, `target_parameters`, aLoRA, `rank_pattern`, `alpha_pattern` "
                        "or adapted tied weights."
                    )
                if self.native_lora:
                    # The ranks vLLM accepts
                    ranks = (1, 8, 16, 32, 64, 128, 256, 320, 512)
                    max_lora_rank = min((rank for rank in ranks if rank >= config.r), default=config.r)
                    lora_kwargs = {"enable_lora": True, "max_lora_rank": max_lora_rank}

            hf_overrides = None
            if self.cast_lm_head_to_fp32:
                if is_vllm_available(min_version="0.26.0"):
                    hf_overrides = {"head_dtype": "float32"}
                else:
                    logger.warning(
                        "`cast_lm_head_to_fp32=True` requires vLLM 0.26.0 or later to run the vLLM lm_head in float32. "
                        "Generation will use the model dtype for the lm_head."
                    )

            # Build LLM initialization kwargs
            self.llm = LLM(
                model=model.name_or_path,
                tensor_parallel_size=self.tensor_parallel_size,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_length,
                max_num_seqs=self.max_num_seqs,
                enable_sleep_mode=self.enable_sleep_mode,
                model_impl=self.model_impl,
                distributed_executor_backend="external_launcher",
                # Feed identical seed for tp groups to ensure sampling results are the same across workers
                seed=accelerator.process_index // self.tensor_parallel_size,
                max_num_batched_tokens=self.max_num_batched_tokens,
                kv_cache_dtype=self.kv_cache_dtype,
                # Important so temperature scaling/logit tweaking affects the TIS log probs
                logprobs_mode="processed_logprobs",
                quantization=quantization,
                trust_remote_code=self.trust_remote_code,
                hf_overrides=hf_overrides,
                **lora_kwargs,
                **fp8_kwargs,
            )
            unshared = self._share_weights() if self.share_weights else []
            if unshared and lora_kwargs:
                raise ValueError("`native_lora=True` requires every base parameter to be shared with vLLM.")
            self._llm_weights_sleeping = False
            self._kv_cache_sleeping = False
            self.sleep()
        else:
            raise ValueError(f"vllm_mode must be either 'server' or 'colocate', got '{self.mode}'.")

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        accelerator.wait_for_everyone()

    def _fp8_ignore_list(self, model: nn.Module) -> list[str]:
        """Layers vLLM keeps in high precision to quantize the same layers as the trainer."""
        ignored = [
            name for name, module in model.named_modules() if isinstance(module, nn.Linear) and "lora_" not in name
        ]
        fused_fp8 = {fused_layer_name(name) for name, module in model.named_modules() if isinstance(module, FP8Linear)}
        for name in ignored:
            if fused_layer_name(name) in fused_fp8:
                raise ValueError(
                    f"vLLM quantizes `{name}` together with the projections it fuses it with: keep all of them in "
                    "high precision with `fp8_skip_modules`, or none."
                )
        # vLLM's fused MoE experts have no trainer counterpart
        return ["*.experts"] + [
            self._fix_param_name_to_vllm(name.removeprefix("base_model.model.").replace(".base_layer", ""))
            for name in ignored
        ]

    @torch.no_grad()
    def _share_weights(self) -> list[tuple[str, torch.Tensor]]:
        """Point the model's parameters at vLLM's weights, and return the parameters that can't be shared."""
        vllm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        views = {name.replace(".base_layer", ""): param.data for name, param in vllm_model.named_parameters()}
        # Fused layers (e.g. qkv_proj) stack several parameters
        packed = vllm_model.packed_modules_mapping if supports_lora(vllm_model) else {}
        fp8_views = {}
        for module_name, module in vllm_model.named_modules():
            module_name = module_name.replace(".base_layer", "")
            fused = module_name.rpartition(".")[2]
            parts = packed.get(fused)
            prefix = module_name.removesuffix(fused)
            params = dict(module.named_parameters(recurse=False))
            if "weight_scale_inv" in params:
                # vLLM loaded the blockwise FP8 weights of the checkpoint itself, like the trainer
                del params["weight"], params["weight_scale_inv"]
            if "weight" in params and params["weight"].dtype == torch.float8_e4m3fn:
                # Per-channel FP8: one row and one scale per output channel, whatever layout the kernel uses
                weight = params.pop("weight").data
                rows = weight if weight.stride(-1) == 1 else weight.t()
                sizes = module.output_sizes if parts and parts != [fused] else [rows.size(0)]
                for part, row_view, scale_view in zip(
                    parts or [fused],
                    rows.split(sizes),
                    params.pop("weight_scale").data.view(-1, 1).split(sizes),
                    strict=True,
                ):
                    fp8_views[f"{prefix}{part}"] = (row_view, scale_view)
            if parts and parts != [fused]:
                for name, param in params.items():
                    for part, view in zip(parts, param.data.split(module.output_sizes), strict=True):
                        views[f"{prefix}{part}.{name}"] = view

        self._unmerge()
        adapter_prefix = self.model.prefix if is_peft_model(self.model) else None
        lora_layers = {}
        if adapter_prefix and not self.native_lora:
            adapter = self.model.active_adapters[0]
            lora_layers = {
                self._fix_param_name_to_vllm(name.removeprefix("base_model.model.")) + ".weight": module
                for name, module in self.model.named_modules()
                if isinstance(module, BaseTunerLayer)
            }
        fp8_layers = {
            self._fix_param_name_to_vllm(name.removeprefix("base_model.model.").replace(".base_layer", "")): module
            for name, module in self.model.named_modules()
            if isinstance(module, FP8Linear)
        }
        params = [
            (self._fix_param_name_to_vllm(name.removeprefix("base_model.model.").replace(".base_layer", "")), param)
            for name, param in self.model.named_parameters()
            if not (adapter_prefix and adapter_prefix in name)
        ]
        # vLLM gets the FP8 copies of these layers' weights, never the high-precision ones
        params = [(name, param) for name, param in params if name.removesuffix(".weight") not in fp8_layers]
        if self._unshareable is None:
            self._unshareable = set()
            # vLLM's loader can't address layers wrapped by native LoRA
            wrapped = [
                (parent, child_name, child)
                for parent in vllm_model.modules()
                for child_name, child in parent.named_children()
                if "base_layer" in child._modules
            ]
            for parent, child_name, child in wrapped:
                setattr(parent, child_name, child.base_layer)
            for name, param in params:
                view = views.get(name)
                if view is None or view.shape != param.shape or view.dtype != param.dtype:
                    self._unshareable.add(name)
                    continue
                # Share only if vLLM's loader keeps the layout (it transposes GPT-2's Conv1D, reorders GPT-NeoX's QKV)
                original, probe = view.clone(), torch.randn_like(view)
                vllm_model.load_weights([(name, probe)])
                if not torch.equal(view, probe):
                    self._unshareable.add(name)
                view.copy_(original)
            for parent, child_name, child in wrapped:
                setattr(parent, child_name, child)
        for name, module in fp8_layers.items():
            weight, scale = fp8_views.get(name, (None, None))
            if weight is None or weight.shape != (module.out_features, module.in_features):
                raise ValueError(f"vLLM doesn't hold `{name}` in per-channel FP8 like the trainer.")
            if not module.weight_fp8.is_set_to(weight):
                # vLLM loaded the same checkpoint, unless its loader reordered the rows (e.g. GPT-NeoX's QKV)
                vllm_weight, trainer_weight = weight.float() * scale, module.weight_fp8.float() * module.weight_scale
                if (vllm_weight - trainer_weight).norm() > 0.1 * trainer_weight.norm():
                    raise ValueError(
                        f"vLLM's loader changes the layout of `{name}`, so its FP8 weights can't be shared."
                    )
                weight.copy_(module.weight_fp8)
                scale.copy_(module.weight_scale)
                module.weight_fp8, module.weight_scale = weight, scale
            # Updated trainable weights are requantized straight into vLLM's FP8 copy
            module.quantize_weight()
        unshared = []
        for name, param in params:
            if name in self._unshareable:
                if name in lora_layers:
                    param = (param + lora_layers[name].get_delta_weight(adapter)).to(param.dtype)
                unshared.append((name, param.data))
                continue
            view = views[name]
            if not param.data.is_set_to(view):
                view.copy_(param.data)
                param.data = view
            if name in lora_layers:
                self._merged_bases.append((view, view.clone()))
                view.copy_(view + lora_layers[name].get_delta_weight(adapter))
        return unshared

    def _unmerge(self):
        """Restore the frozen base of the layers the adapter was merged into."""
        for view, base in self._merged_bases:
            view.copy_(base)
        self._merged_bases = []

    def _fix_param_name_to_vllm(self, name: str, extra_prefixes: list[str] | None = None) -> str:
        """Fix parameter name for vLLM compatibility."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name

    def _iter_fsdp1_params(self, module: nn.Module, prefix: str = "", visited: set[str] | None = None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters."""
        # For FSDP1, we need to recurse into children and also use summon_full_params
        if visited is None:
            visited = set()
        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            yield from self._iter_fsdp1_params(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    full_name = self._fix_param_name_to_vllm(full_name, extra_prefixes=["_fsdp_wrapped_module."])

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    yield full_name, param.data

    def _iter_fsdp2_params(self, module: nn.Module):
        """FSDP2-specific parameter iteration."""
        # For FSDP2, module.state_dict() already covers all parameters, so no need for recursion
        for name, param in module.state_dict().items():
            # When using PEFT, we need to recover the original parameter name
            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
            # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
            if is_peft_model(module) and module.prefix in name:
                continue
            # When module to save, remove its prefix and discard the original module
            if "original_module" in name:
                continue
            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

            if param.is_cpu:
                param = param.to(self.accelerator.device)
            param = param.full_tensor()

            yield name, param

    def _iter_fsdp_params(self, model: nn.Module):
        """Dispatch FSDP parameter iteration to the version-appropriate method."""
        if self._dist.fsdp_version == 1:
            yield from self._iter_fsdp1_params(model)
        elif self._dist.fsdp_version == 2:
            yield from self._iter_fsdp2_params(model)

    @contextmanager
    def _export_named_params(self):
        """Yield [`_iter_named_params`], with PEFT adapters merged until the caller is done."""
        model = self.model

        if is_peft_model(model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            # TODO: does this work with FSDP?
            with self._dist.gather_params(list(model.parameters())):
                # Unmerging is lossy, so keep exact copies to restore
                originals = [
                    (module.get_base_layer(), name, param, param.data.to("cpu", copy=True))
                    for module in model.modules()
                    if isinstance(module, BaseTunerLayer) and not self._dist.is_zero3
                    for name, param in module.get_base_layer().named_parameters(recurse=False)
                ]
                try:
                    model.merge_adapter()
                    with closing(self._iter_named_params()) as params:
                        yield params
                finally:
                    # Unmerge adapters while parameters are still gathered
                    model.unmerge_adapter()
                    # bitsandbytes merges replace the parameter, so re-register the original
                    for base_layer, name, param, data in originals:
                        param.data.copy_(data)
                        base_layer.register_parameter(name, param)
                # Parameters will automatically be repartitioned when exiting the context
        else:
            with closing(self._iter_named_params()) as params:
                yield params

    def _iter_named_params(self):
        """Iterate over the model parameters, materialized one at a time under the name vLLM expects.

        Handles FSDP, DeepSpeed and PEFT. Gathering a parameter is a collective operation, so every process must
        iterate, even the ones that don't push the weights anywhere.
        """
        model = self.model

        if self._dist.is_fsdp:
            # For PEFT with FSDP we need to use the memory efficient post-order traversal
            yield from self._iter_fsdp_params(model)
        elif is_peft_model(model):
            # DeepSpeed ZeRO-3 with PEFT
            for name, param in model.named_parameters():
                # When using PEFT, we need to recover the original parameter name
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                # Skip PEFT layers: they don't exist in vLLM, and they are merged already.
                if model.prefix in name:
                    continue
                # When module to save, remove its prefix and discard the original module
                if "original_module" in name:
                    continue
                name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                yield name, param.data
        else:
            # For non-PEFT models, simply gather (if needed) and read each parameter individually.
            for name, param in model.named_parameters():
                name = self._fix_param_name_to_vllm(name)
                with self._dist.gather_params([param]):
                    yield name, param.data

    def sync_weights(self):
        """Synchronize model weights to vLLM.

        Handles FSDP, DeepSpeed, PEFT weight synchronization.
        """
        self._weights_dirty = True
        # Wake up vLLM weights before loading to ensure device memory is mapped. Without this, load_weights() writes to
        # freed/unmapped memory when sleep mode is active, which crashes on backends with strict physical memory
        # management (e.g., Ascend NPU). See https://github.com/huggingface/trl/issues/5142
        if self.mode == "colocate" and self.enable_sleep_mode and self._llm_weights_sleeping:
            empty_cache()  # required to avoid OOM in some cases
            self.llm.wake_up(tags=["weights"])
            self._llm_weights_sleeping = False

        accelerator = self.accelerator

        if self.mode == "server":
            # The server must know every tensor it is about to receive before the first one is broadcast, so the
            # parameters are walked once to collect their metadata, and streamed on subsequent passes.
            if self._weight_metadata is None:
                with self._export_named_params() as params:
                    self._weight_metadata = [
                        (name, str(param.dtype).removeprefix("torch."), list(param.shape)) for name, param in params
                    ]
            with self._export_named_params() as params:
                if accelerator.is_main_process:
                    self.vllm_client.update_named_params(self._weight_metadata, params)
                else:
                    for _ in params:  # take part in the gather collectives
                        pass
        elif self.mode == "colocate":
            load_weights = self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights
            if self.share_weights:
                load_weights(self._share_weights())
                if is_peft_model(self.model) and self.native_lora:
                    self._publish_adapter()
            else:
                with self._export_named_params() as params:
                    if self._dist.is_fsdp or (self._dist.is_zero3 and not is_peft_model(self.model)):
                        # Gathered tensors are released as the stream advances
                        for name, param in params:
                            load_weights([(name, param)])
                    else:
                        load_weights(params)

        # Reset cache on vLLM
        if self.mode == "server" and accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.mode == "colocate":
            self.llm.reset_prefix_cache()
        self._weights_dirty = False

    def _publish_adapter(self):
        """Replace the adapter vLLM serves with the current one, under a new ID."""
        previous = self._lora_request
        if previous:
            self.llm.llm_engine.remove_lora(previous.lora_int_id)
        state = get_peft_model_state_dict(self.model, adapter_name=self.model.active_adapters[0])
        with TemporaryDirectory(prefix="trl-lora-") as directory:
            save_file(
                {name: tensor.contiguous() for name, tensor in state.items()}, f"{directory}/adapter_model.safetensors"
            )
            self.model.peft_config[self.model.active_adapters[0]].save_pretrained(directory)
            lora_id = previous.lora_int_id + 1 if previous else 1
            self._lora_request = LoRARequest(f"trl-policy-{lora_id}", lora_id, directory)
            self.llm.llm_engine.add_lora(self._lora_request)

    def sleep(self):
        if self._merged_bases:
            self._unmerge()
            self._weights_dirty = True
        if self.mode == "colocate" and self.enable_sleep_mode and not self._kv_cache_sleeping:
            self.llm.reset_mm_cache()
            if self.share_weights:
                # The model trains on vLLM's weights, so only the KV cache is released
                core = self.llm.llm_engine.engine_core.engine_core
                core.pause_scheduler(clear_cache=True)
                if is_vllm_available(min_version="0.28.0"):
                    CuMemAllocator.get_instance().discard("kv_cache")
                else:
                    for data in CuMemAllocator.get_instance().pointer_to_data.values():
                        if data.tag == "kv_cache":
                            unmap_and_release(data.handle)
                core.model_executor.is_sleeping = True
                core.model_executor.sleeping_tags = {"kv_cache"}
            else:
                # Sleep level 2 discards the weights; track it so that generate() knows it must re-push them
                self.llm.sleep(level=2)
                self._llm_weights_sleeping = True
            self._kv_cache_sleeping = True

    def _place_features(self, features: dict | None, prompt_ids: list[int]) -> dict | None:
        """Point the image features at the image tokens of `prompt_ids`.

        The server reports where the images sit in the throwaway conversation it processed them in, which says nothing
        about the prompt being trained on, so their positions are recomputed from the runs of image tokens in the
        trainer's own token IDs.
        """
        if features is None:
            return None

        image_token_id = self.processing_class.image_token_id
        placeholders = []
        offset = 0
        while offset < len(prompt_ids):
            if prompt_ids[offset] == image_token_id:
                length = 0
                while offset + length < len(prompt_ids) and prompt_ids[offset + length] == image_token_id:
                    length += 1
                placeholders.append({"offset": offset, "length": length})
                offset += length
            else:
                offset += 1

        expected = len(features["mm_placeholders"]["image"])
        if len(placeholders) != expected:
            raise ValueError(
                f"Found {len(placeholders)} runs of image tokens in the prompt but {expected} images were processed. "
                "The prompt must contain one run of image tokens per image."
            )
        return {**features, "mm_placeholders": {**features["mm_placeholders"], "image": placeholders}}

    def generate(
        self,
        prompts: list[list[int]],
        images: list[list | None] | None,
        num_generations: int,
        profiler: ProfilingContext | None = None,
    ) -> tuple:
        """Generate completions using vLLM.

        Args:
            prompts: List of token ID lists, one per prompt (already tokenized).
            images: Optional list of image lists for VLM support. Each element is a list of PIL images for the
                corresponding prompt, or `None` if no images for that prompt. `None` if no images at all.
            num_generations: Number of times each original prompt is repeated in `prompts`. In server mode, when
                every group contains identical inputs, this many completions are requested for each first entry. Pass 1
                after tool calls because histories can diverge.
            profiler: Optional profiler for performance tracking.

        Returns:
            Tuple of (prompt_ids, completion_ids, logprobs, logprob_token_ids).

            - `prompt_ids`: `list[list[int]]` of shape `(batch_size, prompt_len)`.
            - `completion_ids`: `list[list[int]]` of shape `(batch_size, completion_len)`.
            - `logprobs`: `list[list[list[float | None]]]` of shape `(batch_size, completion_len, num_logprobs)`.
            - `logprob_token_ids`: `list[list[list[int]]]` of shape `(batch_size, completion_len, num_logprobs)`.

            `num_logprobs` is 1 when `logprobs=0`, or up to N+1 when `logprobs=N` (the sampled token is always included
            and may fall outside the top-N).
        """
        profiler = profiler or nullcontext()
        accelerator = self.accelerator
        temperature = self.temperature
        top_p = self.top_p
        top_k = self.top_k
        min_p = self.min_p
        repetition_penalty = self.repetition_penalty
        max_completion_length = self.max_completion_length

        # Sleep level 2 discards the weights, so waking up isn't enough: they must be re-pushed from the training
        # model. vLLM's `reload_weights` can't be used here, as it reloads the initial checkpoint from disk rather
        # than the current training weights. See https://github.com/vllm-project/vllm/issues/29341
        # A sync that failed midway left the weights partially updated, so it is retried too
        if self._weights_dirty or (self.mode == "colocate" and self.enable_sleep_mode and self._llm_weights_sleeping):
            self.sync_weights()

        # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
        if self.mode == "server":
            # Ranks can hold different numbers of prompts (e.g. in the tool-calling loop)
            gathered_prompts = gather_object([prompts])
            all_prompts = [p for rank_prompts in gathered_prompts for p in rank_prompts]
            # Always gather images (even when None) to avoid deadlock: images may be None on some ranks
            # and non-None on others in mixed datasets, and gather_object is a collective operation.
            all_images = gather_object(images if images is not None else [None] * len(prompts))
            if all(img is None for img in all_images):
                all_images = None

            # Groups whose inputs differ (e.g. environments sampled per rollout) are generated one prompt at a time
            inputs = all_prompts if all_images is None else list(zip(all_prompts, all_images, strict=True))
            if any(inputs[i] != inputs[i - i % num_generations] for i in range(len(inputs))):
                num_generations = 1

            if accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and
                # generate num_generations outputs for each one. This is faster than generating outputs for each
                # duplicate prompt individually.
                ordered_set_of_prompt_ids = all_prompts[::num_generations]

                # The server generates from either token IDs or images, so images are processed on their own first
                # and the resulting features are paired with the token IDs.
                features = None
                if all_images is not None:
                    features = self.vllm_client.image_features(all_images[::num_generations])
                    features = [
                        self._place_features(prompt_features, prompt_ids)
                        for prompt_features, prompt_ids in zip(features, ordered_set_of_prompt_ids, strict=True)
                    ]

                sampling_params = {
                    "n": num_generations,
                    "repetition_penalty": repetition_penalty,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0 if min_p is None else min_p,
                    "max_tokens": max_completion_length,
                    "logprobs": self.logprobs,
                    "structured_outputs_regex": self.structured_outputs_regex,
                    "generation_kwargs": self.generation_kwargs,
                }
                with profiler:
                    output = self.vllm_client.generate(
                        prompts=ordered_set_of_prompt_ids, features=features, **sampling_params
                    )
                    payload = (
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output.get("logprob_token_ids"),
                    )
            else:
                payload = None

            # Broadcast the completions from the main process to all processes, ensuring each process receives its corresponding slice.
            obj_list = [payload]
            broadcast_object_list(obj_list, from_process=0)
            all_prompt_ids, all_completion_ids, all_logprobs, all_logprob_token_ids = obj_list[0]

            # vllm_client.generate(n=num_generations) returns num_generations completions per prompt.
            # Duplicate prompt_ids to align with per-completion entries.
            all_prompt_ids = [ids for ids in all_prompt_ids for _ in range(num_generations)]

            offset = sum(len(rank_prompts) for rank_prompts in gathered_prompts[: accelerator.process_index])
            process_slice = slice(offset, offset + len(prompts))
            prompt_ids = all_prompt_ids[process_slice]
            completion_ids = all_completion_ids[process_slice]
            logprobs = all_logprobs[process_slice] if all_logprobs is not None else None
            logprob_token_ids = all_logprob_token_ids[process_slice] if all_logprob_token_ids is not None else None

        # Generate completions using colocated vLLM instances: each device holds vLLM copy and work on their own batch of prompts
        elif self.mode == "colocate":
            generation_kwargs = {
                "n": 1,  # vLLM on each GPU generates only 1 in colocate mode
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": 0.0 if min_p is None else min_p,
                "max_tokens": max_completion_length,
                "logprobs": self.logprobs,
            }
            generation_kwargs.update(self.generation_kwargs)

            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(regex=self.structured_outputs_regex)
            elif isinstance(structured_outputs_kwargs := generation_kwargs.get("structured_outputs"), dict):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(**structured_outputs_kwargs)
            sampling_params = SamplingParams(**generation_kwargs)

            if self.tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
                all_prompts = [p for sublist in gathered_prompts for p in sublist]
                # Always gather images (even when None) to avoid deadlock: images may be None on some
                # ranks and non-None on others in mixed datasets, and all_gather_object is collective.
                local_images = images if images is not None else [None] * len(prompts)
                gathered_images = [None for _ in range(self.tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_images, local_images, group=self.tp_group)
                all_images = [img for sublist in gathered_images for img in sublist]
                if all(img is None for img in all_images):
                    all_images = None
            else:
                all_prompts = prompts
                all_images = images

            if self.enable_sleep_mode and self._kv_cache_sleeping:
                self.llm.wake_up(tags=["kv_cache"])
                self._kv_cache_sleeping = False

            # Build vLLM-compatible prompt inputs with token IDs and optional multi-modal data
            vllm_prompts = []
            if all_images is not None:
                for ids, img_list in zip(all_prompts, all_images, strict=True):
                    row = {"prompt_token_ids": ids}
                    if img_list is not None:
                        row["multi_modal_data"] = {"image": img_list if len(img_list) > 1 else img_list[0]}
                    vllm_prompts.append(row)
            else:
                vllm_prompts = [{"prompt_token_ids": ids} for ids in all_prompts]

            # When PEFT is used, DDP gradient all-reduce only covers the small LoRA parameters, so
            # NCCL operations complete very quickly. On non-NVLink hardware (e.g. A40/A100), vLLM's
            # TP NCCL collective can race with NCCL's internal P2P/SHM channel cleanup from that
            # all-reduce, causing llm.generate() to hang. A barrier on the default process group
            # forces NCCL to fully drain before vLLM's TP communication starts. We pass device_ids
            # so NCCL uses this rank's device rather than guessing, which itself risks a hang.
            # See https://github.com/huggingface/trl/issues/3671
            if is_peft_model(self.model) and self.tensor_parallel_size > 1:
                torch.distributed.barrier(device_ids=[accelerator.local_process_index])

            with profiler:
                all_outputs = self.llm.generate(
                    vllm_prompts, sampling_params=sampling_params, use_tqdm=False, lora_request=self._lora_request
                )

            all_prompt_ids = [output.prompt_token_ids for output in all_outputs]
            all_completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
            all_logprobs, all_logprob_token_ids = extract_logprobs(all_outputs)

            if self.tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
                offset = sum(len(rank_prompts) for rank_prompts in gathered_prompts[:local_rank_in_group])
                tp_slice = slice(offset, offset + len(prompts))
                prompt_ids = all_prompt_ids[tp_slice]
                completion_ids = all_completion_ids[tp_slice]
                logprobs = all_logprobs[tp_slice] if all_logprobs is not None else None
                logprob_token_ids = all_logprob_token_ids[tp_slice] if all_logprob_token_ids is not None else None
            else:
                prompt_ids = all_prompt_ids
                completion_ids = all_completion_ids
                logprobs = all_logprobs
                logprob_token_ids = all_logprob_token_ids

        return prompt_ids, completion_ids, logprobs, logprob_token_ids
