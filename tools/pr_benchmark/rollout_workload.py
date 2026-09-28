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

"""Paired multi-turn GRPO rollouts, including weight transfer and the final memory handoff."""

import argparse
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch


def make_profiler(directory, warmup_steps, active_steps):
    import torch

    directory.mkdir(parents=True, exist_ok=True)

    def export(profiler):
        profiler.export_chrome_trace(str(directory / "trace.json.gz"))
        averages = profiler.key_averages(group_by_input_shape=True)
        for metric in ("self_cpu_time_total", "self_cuda_time_total", "self_cuda_memory_usage"):
            (directory / f"operators-{metric}.txt").write_text(averages.table(sort_by=metric, row_limit=-1))

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=max(0, warmup_steps - 1), warmup=1, active=active_steps, repeat=1),
        on_trace_ready=export,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # Full Python call trees make multi-turn vLLM traces enormous.
        with_flops=True,
    )


def span(name, enabled):
    if enabled:
        from torch.profiler import record_function

        return record_function("benchmark::" + name)
    return nullcontext()


def annotate(method):
    """Label a bound method's calls with its own name, without changing its arguments or return value."""

    def wrapped(*args, **kwargs):
        with span(method.__name__, True):
            return method(*args, **kwargs)

    return wrapped


def policy_parity(trainer, history, directory, seed):
    """Compare full next-token distributions of the local policy and vLLM on one fixed history."""
    import numpy as np
    import torch
    from peft.tuners.tuners_utils import BaseTunerLayer

    engine = trainer.vllm_generation
    model = engine.model
    # Sample one token and return the full-vocabulary distribution, without truncation or penalties.
    engine.temperature, engine.top_p, engine.top_k, engine.min_p = 1.0, 1.0, -1, 0.0
    engine.max_completion_length = 1
    engine.logprobs = -1
    engine.generation_kwargs = {}
    lora_b = [param for name, param in model.named_parameters() if "lora_B" in name]
    ids = torch.tensor([history], device="cuda")
    records, arrays = [], {}

    def local():
        model.eval()
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1).cpu()

    def record(label, reference, actual):
        p, q = reference.exp(), actual.exp()
        log_midpoint = torch.logaddexp(reference, actual) - math.log(2)
        records.append(
            {
                "label": label,
                "total_variation": ((p - q).abs().sum() / 2).item(),
                "kl_local_vllm": (p * (reference - actual)).sum().item(),
                "js_divergence": ((p * (reference - log_midpoint)).sum() + (q * (actual - log_midpoint)).sum()).item()
                / 2,
            }
        )
        arrays[label + "_local"] = reference.numpy()
        arrays[label + "_vllm"] = actual.numpy()

    previous = None
    for stage_index, stage in enumerate(("dense", "lora", "updated_lora")):
        # Keep the timed model's A matrices; only B changes, so every stage is a distinct nonzero adapter.
        generator = torch.Generator(device="cuda").manual_seed(seed + stage_index)
        with torch.no_grad():
            for param in lora_b:
                if stage == "dense":
                    param.zero_()
                else:
                    param.normal_(std=0.03 if stage == "lora" else 0.1, generator=generator)
        reference = local()
        if stage != "dense":
            # Evaluate the BF16 merged inference policy, restoring the exact training weights afterwards.
            originals = {
                param: param.detach().cpu().clone()
                for module in model.modules()
                if isinstance(module, BaseTunerLayer)
                for param in module.get_base_layer().parameters()
            }
            model.merge_adapter()
            merged_reference = local()
            model.unmerge_adapter()
            with torch.no_grad():
                for param, original in originals.items():
                    param.copy_(original)
        # vLLM is asleep after the timed phases, so this standalone call publishes the current policy on every revision.
        _, _, logprobs, token_ids = engine.generate([history], images=None, num_generations=1)
        actual = torch.full((model.config.vocab_size,), float("nan"))
        actual[torch.tensor(token_ids[0][0])] = torch.tensor(logprobs[0][0])
        if not torch.isfinite(actual).all():
            raise ValueError("Incomplete or non-finite full-vocabulary log probabilities")
        record(stage, reference, actual)
        if stage != "dense":
            record(stage + "_merged_reference", merged_reference, actual)
            # Negative control: the previous stage's policy stands in for a missed synchronization.
            record(stage + "_stale_reference", previous, actual)
        # Publishing must not change the training policy itself.
        record(stage + "_local_policy_drift", reference, local())
        previous = reference
    np.savez_compressed(directory / "policy-distributions.npz", **arrays)
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ("model-path", "checkout", "data", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--side", choices=["base", "head"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, set_seed

    import trl.generation.vllm_generation as generation_module
    from trl import GRPOConfig, GRPOTrainer

    manifest = json.loads(args.manifest.read_text())
    config = manifest["workload"].copy()
    profiling = args.profile_dir is not None
    if profiling:
        config["steps"] = min(2, config["steps"])
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    data = json.loads(args.data.read_text())
    prompts = [
        tokenizer.decode(data["train"][(args.seed + index) % len(data["train"])][: config["prompt_tokens"]])
        for index in range(config["batch_size"])
    ]
    tool_ids = tokenizer.encode("\nTool result: 42. Continue.\n", add_special_tokens=False)

    def rollout(prompts, trainer):
        initial = tokenizer(prompts)["input_ids"]
        histories = [ids.copy() for ids in initial]
        completions = [[] for _ in initial]
        for turn in range(config["turns"]):
            with span(f"turn_{turn}", profiling):
                _, tokens, _, _ = trainer.vllm_generation.generate(histories, images=None, num_generations=1)
            for history, completion, ids in zip(histories, completions, tokens, strict=True):
                history.extend(ids)
                completion.extend(ids)
                # Deterministic CPU tool feedback; all turns keep the same policy.
                if turn + 1 < config["turns"]:
                    with span("cpu_tool", profiling):
                        history.extend(tool_ids)
                        completion.extend(tool_ids)
        return {"prompt_ids": initial, "completion_ids": completions, "logprobs": None}

    # The distribution diagnostic needs full-vocabulary logprobs; only the separate profiled process enables them.
    llm = partial(generation_module.LLM, max_logprobs=-1) if profiling else generation_module.LLM
    with patch.object(generation_module, "LLM", llm):
        trainer = GRPOTrainer(
            model=str(args.model_path),
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
            processing_class=tokenizer,
            args=GRPOConfig(
                output_dir=str(args.output.parent / "checkpoints"),
                model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                vllm_gpu_memory_utilization=0.35,
                vllm_max_model_length=config["context_tokens"],
                max_completion_length=config["completion_tokens"],
                generation_kwargs={"ignore_eos": True, "min_tokens": config["completion_tokens"]},
                temperature=0.0,
                num_generations=2,
                per_device_train_batch_size=config["batch_size"],
                bf16=True,
                report_to="none",
                save_strategy="no",
                seed=args.seed,
            ),
            train_dataset=Dataset.from_dict({"prompt": prompts}),
            reward_funcs=lambda completions, **kwargs: [0.0] * len(completions),
            rollout_func=rollout,
        )
    engine = trainer.vllm_generation
    lora_b = [param for name, param in trainer.model.named_parameters() if "lora_B" in name]
    generator = torch.Generator(device="cuda").manual_seed(args.seed + 1000)
    with torch.no_grad():
        for param in lora_b:
            param.normal_(std=0.01, generator=generator)
    frozen = {
        name: param.detach().cpu().clone()
        for name, param in trainer.model.named_parameters()
        if "base_layer.weight" in name
    }

    # Count publications through vLLM's loaders: full tensors via load_weights, native adapters via add_lora.
    counters = {"weight_transfer_bytes": 0, "sync_count": 0, "load_weights_calls": 0, "add_lora_calls": 0}
    sync_weights = engine.sync_weights
    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model
    load_weights = loader.load_weights
    add_lora = engine.llm.llm_engine.add_lora

    def counted_sync():
        counters["sync_count"] += 1
        with span("sync_weights", profiling):
            return sync_weights()

    def counted_load(weights):
        counters["load_weights_calls"] += 1

        def counted():
            for name, param in weights:
                counters["weight_transfer_bytes"] += param.numel() * param.element_size()
                yield name, param

        with span("load_weights", profiling):
            return load_weights(counted())

    def counted_add_lora(request):
        counters["add_lora_calls"] += 1
        # Tensor payload = file size minus the 8-byte length prefix and JSON header; the tensors are not read.
        path = Path(request.lora_path) / "adapter_model.safetensors"
        with path.open("rb") as stream:
            counters["weight_transfer_bytes"] += path.stat().st_size - 8 - int.from_bytes(stream.read(8), "little")
        with span("add_lora", profiling):
            return add_lora(request)

    engine.sync_weights = counted_sync
    loader.load_weights = counted_load
    engine.llm.llm_engine.add_lora = counted_add_lora
    if profiling:
        engine.llm.wake_up = annotate(engine.llm.wake_up)
        engine.llm.sleep = annotate(engine.llm.sleep)
        engine.llm.reset_prefix_cache = annotate(engine.llm.reset_prefix_cache)
    durations, outputs = [], []
    sleeping = True
    torch.cuda.reset_peak_memory_stats()
    profiler = make_profiler(args.profile_dir, config["warmup_steps"], config["steps"]) if profiling else None
    with profiler if profiler else nullcontext():
        for phase in range(config["warmup_steps"] + config["steps"]):
            # A real update between phases must invalidate the previous policy's KV.
            with torch.no_grad():
                for param in lora_b:
                    param.add_(0.001)
            trainer.state.global_step = phase
            if phase == config["warmup_steps"]:
                counters.update(dict.fromkeys(counters, 0))
            torch.cuda.synchronize()
            started = time.perf_counter()
            with span("rollout_phase", profiling):
                result = trainer._generate(prompts)
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            sleeping = sleeping and engine.llm.llm_engine.is_sleeping()
            if phase >= config["warmup_steps"]:
                durations.append(duration)
                outputs.append(result[1])
            if profiler:
                profiler.step()
    record = {
        "side": args.side,
        "seed": args.seed,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": config["steps"],
        "rollout_seconds": sum(durations),
        "phase_seconds": durations,
        **counters,
        "sleeping_after_phase": sleeping,
        "output_tokens": outputs,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "frozen_weight_max_abs_drift": max(
            (param.detach().cpu() - frozen[name]).abs().float().max().item()
            for name, param in trainer.model.named_parameters()
            if name in frozen
        ),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        },
    }
    if profiling:
        # Runs after the capture window, so the profile covers only complete rollout phases.
        record["policy_parity"] = policy_parity(
            trainer, tokenizer(prompts[0])["input_ids"], args.profile_dir, args.seed
        )
        metadata = {
            "manifest_id": manifest["id"],
            **{key: record[key] for key in ("side", "sha", "seed", "environment")},
            "workload": manifest["workload"],
            "warmup_steps": config["warmup_steps"],
            "active_steps": config["steps"],
        }
        (args.profile_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
