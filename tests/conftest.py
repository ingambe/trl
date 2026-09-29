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

import gc
import logging
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from accelerate import Accelerator
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.utils import is_liger_kernel_available, is_peft_available, is_torch_xpu_available

from trl import GRPOConfig, GRPOTrainer
from trl.generation.vllm_generation import VLLMGeneration


if is_peft_available():
    from peft import LoraConfig, get_peft_model


# ============================================================================
# Silence transformers "LOAD REPORT" tables
# ============================================================================
# transformers >= 5 prints a "LOAD REPORT" table whenever a checkpoint has missing/mismatched/unexpected keys
# (transformers/utils/loading_report.py). TRL's tiny test models don't serialize the tied `lm_head.weight`, so
# this fires on nearly every model load and floods the test output. It is emitted through `logging` at WARNING
# level from three loggers; we drop only these records, keeping all other warnings visible.
class _DropLoadReport(logging.Filter):
    def filter(self, record):
        return "LOAD REPORT" not in record.getMessage()


for _logger_name in ("transformers.modeling_utils", "transformers.modeling_layers", "transformers.integrations.peft"):
    logging.getLogger(_logger_name).addFilter(_DropLoadReport())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Clear traceback frame locals after a failed test to release CUDA tensor references.

    When a test fails (especially with OOM), the exception traceback holds references to every local variable in every
    frame on the call stack at the time of failure — including the model, trainer, and all intermediate tensors.
    gc.collect() cannot free objects that are still reachable through a live traceback, so memory accumulates across
    reruns (~2 GiB per rerun for Gemma4, reaching 5 × 2.38 GiB = 11.89 GiB after 5 reruns). Clearing the frame locals
    breaks those reference chains so that the subsequent gc.collect() + empty_cache() in cleanup_gpu can actually
    reclaim the CUDA memory before the next attempt.
    """
    yield
    if call.when == "call" and call.excinfo is not None:
        traceback.clear_frames(call.excinfo.tb)
        # Also clear all reachable chained exception tracebacks (both __context__ and __cause__ at
        # every node): when OOM fires inside a try/except in the trainer, the OOM becomes __context__
        # of the outer exception and its traceback holds frame locals (model, tensors) that prevent gc
        # from releasing CUDA memory even after clear_frames above.
        stack, seen = [call.excinfo.value], set()
        while stack:
            exc = stack.pop()
            if exc is None or id(exc) in seen:
                continue
            seen.add(id(exc))
            if exc.__traceback__ is not None:
                traceback.clear_frames(exc.__traceback__)
                exc.__traceback__ = None
            stack.append(exc.__context__)
            stack.append(exc.__cause__)


def _make_server_generation(accelerator, *, max_completion_length):
    """Keep the real server batching code and replace only the client that sends generation requests."""
    generation = object.__new__(VLLMGeneration)
    generation.accelerator = accelerator
    generation.mode = "server"
    generation._weights_dirty = False
    generation.temperature = 1.0
    generation.top_p = 1.0
    generation.top_k = -1
    generation.min_p = None
    generation.repetition_penalty = 1.0
    generation.max_completion_length = max_completion_length
    generation.logprobs = 0
    generation.structured_outputs_regex = None
    generation.generation_kwargs = {}

    def generate_from_submitted_histories(prompts, n, **sampling_kwargs):
        # Match vLLM's prompt-major ordering and n outputs per submitted history. Answers encode the tool result
        # the server actually saw: 30 -> 130, 35 -> 135. Never infer an answer from the intended recipient.
        tool_results = [prompt[-1] for prompt in prompts for _ in range(n)]
        return {
            "prompt_ids": prompts,
            "completion_ids": [[result + 100] for result in tool_results],
            "logprobs": [[[-result / 100]] for result in tool_results],
            "logprob_token_ids": [[[result + 100]] for result in tool_results],
        }

    generation.vllm_client = SimpleNamespace(generate=Mock(side_effect=generate_from_submitted_histories))
    return generation


@pytest.fixture
def server_generation():
    """Provide single-process server-mode generation with a mocked client."""
    with (
        patch("trl.generation.vllm_generation.gather_object", side_effect=lambda values: values),
        patch("trl.generation.vllm_generation.broadcast_object_list", return_value=None),
    ):
        yield _make_server_generation(SimpleNamespace(is_main_process=True, process_index=0), max_completion_length=32)


def _make_server_tool_trainer(accelerator, num_samples):
    """Build a GRPO trainer whose tool loop and server generation run unchanged, with a mocked vLLM client."""
    trainer = object.__new__(GRPOTrainer)
    trainer.accelerator = accelerator
    trainer.args = SimpleNamespace(report_to=[])
    trainer.model = SimpleNamespace(training=True, config=SimpleNamespace(max_position_embeddings=128))
    trainer.state = SimpleNamespace(global_step=0)
    trainer._last_loaded_step = 0
    trainer.use_vllm = True
    trainer._tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    trainer.vllm_mode = "server"
    trainer.num_generations = 2
    trainer.max_tool_calling_iterations = 1
    trainer.max_completion_length = 32
    trainer._is_vlm = False
    trainer.vllm_generation = _make_server_generation(
        trainer.accelerator, max_completion_length=trainer.max_completion_length
    )

    def calculator(result):
        return result

    trainer._sync_tool_dicts = [{"calculator": calculator} for _ in range(num_samples)]
    trainer._async_tool_dicts = [{} for _ in range(num_samples)]
    return trainer


def _encode_tool_result(messages):
    # Synthetic tokenization: a result of "30" becomes token 30.
    return [int(messages[-1]["content"])]


def _decode_assistant_response(tokenizer, ids, prefix):
    return {"role": "assistant", "content": str(ids[0])}


@pytest.fixture
def server_tool_trainer():
    """Provide a two-sibling tool scenario; restore patched dependencies after the test."""
    accelerator = SimpleNamespace(
        device=torch.device("cpu"), is_main_process=True, process_index=0, gather=lambda tensor: tensor
    )
    trainer = _make_server_tool_trainer(accelerator, num_samples=2)

    def gather_on_single_process(values):
        return values

    # Patch only token formatting and distributed transport. The tool loop, single-turn generation, and server
    # grouping run unchanged. These replacements are active during yield and automatically restored afterward.
    with (
        patch.object(trainer, "_get_tool_suffix_ids", side_effect=_encode_tool_result),
        patch("trl.trainer.grpo_trainer.parse_response", side_effect=_decode_assistant_response),
        patch("trl.generation.vllm_generation.gather_object", side_effect=gather_on_single_process),
        patch("trl.generation.vllm_generation.broadcast_object_list", return_value=None),
    ):
        yield trainer


@pytest.fixture
def two_rank_server_tool_trainers():
    """Run a function on two ranks in threads, with collectives that block until both ranks call them."""
    barrier = threading.Barrier(2, timeout=5)
    slots = [None, None]
    local = threading.local()

    def all_gather(obj):
        slots[local.rank] = obj
        barrier.wait()
        gathered = list(slots)
        barrier.wait()
        return gathered

    def gather_object(values):
        return [value for rank_values in all_gather(values) for value in rank_values]

    def broadcast_object_list(obj_list, from_process=0):
        obj_list[0] = all_gather(obj_list[0])[from_process]

    def run_on_ranks(num_samples_per_rank, fn):
        def run(rank):
            local.rank = rank
            accelerator = SimpleNamespace(
                device=torch.device("cpu"),
                is_main_process=rank == 0,
                process_index=rank,
                gather=lambda tensor: torch.stack(all_gather(tensor)),
            )
            trainer = _make_server_tool_trainer(accelerator, num_samples=num_samples_per_rank[rank])
            with patch.object(trainer, "_get_tool_suffix_ids", side_effect=_encode_tool_result):
                return trainer, fn(rank, trainer)

        with ThreadPoolExecutor(max_workers=2) as executor:
            return [future.result() for future in [executor.submit(run, rank) for rank in range(2)]]

    with (
        patch("trl.trainer.grpo_trainer.parse_response", side_effect=_decode_assistant_response),
        patch("trl.generation.vllm_generation.gather_object", side_effect=gather_object),
        patch("trl.generation.vllm_generation.broadcast_object_list", side_effect=broadcast_object_list),
    ):
        yield run_on_ranks


# ============================================================================
# Model Revision Override
# ============================================================================
# To test a tiny model PR before merging to main:
# 1. Add the full model_id and PR revision to this dict
# 2. Commit and push to trigger CI
# 3. Once CI is green, merge the tiny model PR on HF Hub
# 4. Remove the entry from this dict and commit
#
# Example:
#   MODEL_REVISIONS = {
#       "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5": "refs/pr/3",
#       "trl-internal-testing/tiny-LlavaForConditionalGeneration": "refs/pr/5",
#   }
# ============================================================================

MODEL_REVISIONS = {
    # Add model_id: revision mappings here to test PRs
}


@pytest.fixture(autouse=True)
def apply_model_revisions(monkeypatch):
    """Auto-inject revision parameter for models defined in MODEL_REVISIONS."""
    if not MODEL_REVISIONS:
        return

    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

    def create_classmethod_wrapper(original_classmethod):
        # Extract the underlying function from the classmethod
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, pretrained_model_name_or_path, *args, **kwargs):
            # Direct lookup: only inject if model_id is in the override dict
            if pretrained_model_name_or_path in MODEL_REVISIONS:
                if "revision" not in kwargs:
                    kwargs["revision"] = MODEL_REVISIONS[pretrained_model_name_or_path]
                    # Clear _commit_hash: Auto classes resolve it from the default branch before calling
                    # sub-loaders, so the cached hash points to main. If we don't clear it, it silently
                    # overrides the injected revision for the config load while the weight loader uses the
                    # revision, producing a config/weights shape mismatch.
                    kwargs.pop("_commit_hash", None)

            return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

        # Re-wrap as classmethod
        return classmethod(wrapper)

    def create_method_wrapper(original_method):
        @wraps(original_method)
        def wrapper(self, model_id, *args, **kwargs):
            if model_id in MODEL_REVISIONS and "revision" not in kwargs:
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_method(self, model_id, *args, **kwargs)

        return wrapper

    # Patch the transformers Auto* classes and the base classes they dispatch to
    for cls in [
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    ]:
        monkeypatch.setattr(cls, "from_pretrained", create_classmethod_wrapper(cls.from_pretrained))

    def create_peft_classmethod_wrapper(original_classmethod):
        original_func = original_classmethod.__func__

        @wraps(original_func)
        def wrapper(cls, model, model_id, *args, **kwargs):
            if model_id in MODEL_REVISIONS and "revision" not in kwargs:
                kwargs["revision"] = MODEL_REVISIONS[model_id]
            return original_func(cls, model, model_id, *args, **kwargs)

        return classmethod(wrapper)

    # PEFT adapters never reach the loaders above.
    if is_peft_available():
        from peft import AutoPeftModelForCausalLM, PeftModel

        monkeypatch.setattr(
            AutoPeftModelForCausalLM,
            "from_pretrained",
            create_classmethod_wrapper(AutoPeftModelForCausalLM.from_pretrained),
        )
        # `PeftModel.from_pretrained` takes the base model first and the adapter id second.
        monkeypatch.setattr(PeftModel, "from_pretrained", create_peft_classmethod_wrapper(PeftModel.from_pretrained))
        # `load_adapter` is an instance method, not a classmethod.
        monkeypatch.setattr(PeftModel, "load_adapter", create_method_wrapper(PeftModel.load_adapter))


@pytest.fixture(autouse=True)
def force_use_cpu_without_accelerator(monkeypatch):
    """Force `use_cpu=True` on all TRL configs when no accelerator is available.

    TRL configs default `bf16` to `True` (see `_BaseConfig.__post_init__`), which makes
    `transformers.TrainingArguments.__post_init__` raise `ValueError: Your setup doesn't support bf16/gpu. You need to
    assign use_cpu ...` on machines without a GPU. Setting `use_cpu=True` before that validation runs lets the whole
    suite run on CPU without editing every individual config in the tests. On a machine with an accelerator (CUDA, XPU,
    NPU, ...) this fixture is a no-op, so on-device tests and explicit `use_cpu=False` are left untouched.
    """
    from transformers.testing_utils import torch_device

    if torch_device is not None and torch_device != "cpu":
        return

    from trl.trainer.base_config import _BaseConfig

    original_post_init = _BaseConfig.__post_init__

    def patched_post_init(self):
        self.use_cpu = True
        original_post_init(self)

    monkeypatch.setattr(_BaseConfig, "__post_init__", patched_post_init)


@pytest.fixture(autouse=True)
def restore_mixed_precision_env(monkeypatch):
    """
    Restore the mixed precision that `transformers.TrainingArguments` publishes process-wide.

    On transformers < 5, `TrainingArguments.__post_init__` writes the mixed precision to `ACCELERATE_MIXED_PRECISION`
    and reads that same variable back as the default for the next instantiation, where a `bf16=False` flag is
    indistinguishable from an unset one. TRL configs default `bf16` to `True`, so the first trainer built in a worker
    sets the variable to `bf16` for the whole process, and no config built later can clear it. transformers < 5 also
    builds the `Accelerator` without passing the mixed precision, so accelerate falls back to the leaked variable and a
    test that asks for `bf16=False` silently runs in bf16: with `test_chunked_logps_match_full_logits`, the streamed
    projection enters autocast and the full-logits path does not, and the two no longer match. Setting the variable to
    its current value arms monkeypatch's restore, so each test sees the value the session started with.

    Only the minimum-versions CI job is affected: transformers >= 5 passes the mixed precision to the `Accelerator`
    explicitly and never writes the variable.
    """
    monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", os.environ.get("ACCELERATE_MIXED_PRECISION", "no"))


@pytest.fixture(autouse=True)
def cleanup_gpu():
    """
    Automatically cleanup accelerator memory after each test.

    This fixture helps prevent accelerator out of memory errors when running tests in parallel with pytest-xdist by
    ensuring models and tensors are properly garbage collected and accelerator memory caches are cleared between tests.
    """
    yield
    # Cleanup after test
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if is_torch_xpu_available():
        torch.xpu.synchronize()
        torch.xpu.empty_cache()


@pytest.fixture(autouse=True)
def cleanup_vllm_dist_env():
    """
    Destroy the torch.distributed process group left behind by vLLM colocate mode after each test.

    vLLM colocate mode initializes `torch.distributed` internally (`distributed_executor_backend="external_launcher"`)
    but the offline `LLM` class never tears it down: neither `LLM` nor `LLMEngine` calls vLLM's own
    `cleanup_dist_env_and_memory()` on exit. Left undestroyed, PyTorch warns at interpreter shutdown (e.g.
    `TestGRPOTrainerSlow::test_vlm_processor_vllm_colocate_mode`). We call vLLM's own cleanup helper rather than a bare
    `torch.distributed.destroy_process_group()` so that vLLM's internal parallel-state globals (`_TP`, `_WORLD`, ...)
    are reset too, letting the next colocate test reinitialize cleanly.
    """
    yield
    if torch.distributed.is_initialized():
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def undo_liger_kernel_patching(monkeypatch):
    """
    Restore the transformers modeling modules that Liger Kernel patches process-wide.

    `use_liger_kernel=True` makes transformers call `liger_kernel.transformers._apply_liger_kernel_to_instance`, which
    patches the model instance but also rebinds module-level names in `transformers.models.<arch>.modeling_<arch>`
    (`Qwen3RMSNorm` -> `LigerRMSNorm`, `Qwen3MLP` -> `LigerSwiGLUMLP`, `apply_rotary_pos_emb` ->
    `liger_rotary_pos_emb`, ...). Liger never restores them, so every model of that architecture instantiated later in
    the same process silently gets Triton kernels and fails on CPU tensors with `ValueError: Pointer argument cannot be
    accessed from Triton (cpu tensor?)`. Which test trips over it depends on how pytest-xdist spreads tests across
    workers, so the failure surfaces in an unrelated test (e.g. `TestUseAdapter`, whose PEFT model has a Qwen3 base)
    and only in some runs. Snapshot the modeling modules when Liger patches, and restore them once the test is over.

    Only the trigger is transformers-version dependent, not the leak itself: transformers < 5.3 applies the kernels in
    `Trainer.__init__`, so a test that merely builds a trainer leaks, while transformers >= 5.3 applies them in
    `Trainer.train()`. Either way they are never reverted, which is why only the minimum-versions CI job fails: the
    Liger tests that use a Qwen3 model build a trainer but never train.
    """
    if not is_liger_kernel_available():
        yield
        return

    import liger_kernel.transformers as liger_transformers

    apply_to_instance = liger_transformers._apply_liger_kernel_to_instance
    snapshots = {}

    def apply_to_instance_and_snapshot(*args, **kwargs):
        # Liger imports the modeling modules it patches inside the patch function, but a module can only be patched
        # if the model being patched already uses it, so it is imported by now.
        for name, module in list(sys.modules.items()):
            if name.startswith("transformers.models.") and ".modeling_" in name:
                snapshots.setdefault(name, (module, vars(module).copy()))
        return apply_to_instance(*args, **kwargs)

    monkeypatch.setattr(liger_transformers, "_apply_liger_kernel_to_instance", apply_to_instance_and_snapshot)
    yield
    for module, snapshot in snapshots.values():
        vars(module).update(snapshot)


@pytest.fixture
def vllm_generation(monkeypatch):
    monkeypatch.setattr(VLLMGeneration, "_init_vllm", lambda self: None)
    # Stand-in vLLM models declare their fused layers directly
    monkeypatch.setattr(
        "trl.generation.vllm_generation.supports_lora",
        lambda model: "packed_modules_mapping" in vars(model),
        raising=False,
    )
    generation = VLLMGeneration(torch.nn.Linear(1, 1), Accelerator(cpu=True), None)
    generation.llm = Mock()
    return generation


@pytest.fixture
def tiny_llama():
    """Tiny untied Llama model and word-level tokenizer, built locally."""
    config = LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "a": 2}, unk_token="a")),
        pad_token="<pad>",
        eos_token="<eos>",
    )
    return LlamaForCausalLM(config), tokenizer


@pytest.fixture
def peft_vllm_generation(vllm_generation):
    vllm_generation.model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)),
        LoraConfig(r=1, lora_alpha=1, target_modules=["0", "1"]),
    )
    # Dyadic values keep merge and unmerge exact
    with torch.no_grad():
        for parameter in vllm_generation.model.parameters():
            parameter.fill_(0.25)
    return vllm_generation


@pytest.fixture
def colocate_grpo_trainer(monkeypatch, tmp_path, tiny_llama):
    """Real GRPO trainer on a local tiny model, with colocated vLLM in sleep mode replaced by a mock `LLM`."""
    monkeypatch.setattr("trl.generation.vllm_generation.is_vllm_available", lambda *args, **kwargs: True)
    monkeypatch.setattr("trl.generation.vllm_generation.LLM", Mock(), raising=False)
    monkeypatch.setattr("trl.generation.vllm_generation.SamplingParams", lambda **kwargs: kwargs, raising=False)
    monkeypatch.setattr("trl.generation.vllm_generation.empty_cache", lambda: None)
    monkeypatch.setenv("TRL_EXPERIMENTAL_SILENCE", "1")
    model, tokenizer = tiny_llama
    with patch.dict(os.environ):  # colocated vLLM sets RANK, LOCAL_RANK and WORLD_SIZE
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            args=GRPOConfig(
                output_dir=str(tmp_path),
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                per_device_train_batch_size=2,
                num_generations=2,
                report_to="none",
            ),
            train_dataset=Dataset.from_dict({"prompt": ["a", "a"]}),
            rollout_func=lambda prompts, trainer: None,
        )
        trainer.vllm_generation.llm.reset_mock()  # forget the sleep at the end of engine init
        trainer.vllm_generation.llm.generate.return_value = []
        yield trainer
