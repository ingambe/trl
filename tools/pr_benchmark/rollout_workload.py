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
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from profiling import annotate, make_profiler, save_profile_metadata, span


def policy_parity(trainer, histories, side, seed):
    """Compare full next-token distributions on fixed histories, including a stale-adapter control."""
    import numpy as np
    import torch
    from peft import PeftModel
    from vllm import SamplingParams

    engine = trainer.vllm_generation
    native_lora = "_lora_config" in dir(engine) and engine._lora_config is not None
    if not isinstance(engine.model, PeftModel):
        raise ValueError("The policy diagnostic requires the benchmark LoRA model")
    engine.temperature = 1.0
    engine.top_p = 1.0
    engine.top_k = -1
    engine.min_p = 0.0
    engine.max_completion_length = 1
    engine.logprobs = -1
    engine.generation_kwargs = {}
    lora_b = [param for name, param in engine.model.named_parameters() if "lora_B" in name]
    records, arrays = [], {}
    ids = torch.tensor(histories[0], device="cuda").unsqueeze(0)

    def local():
        engine.model.eval()
        with torch.no_grad():
            return engine.model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1).cpu()

    def dense(logprobs, token_ids):
        result = torch.full((engine.model.config.vocab_size,), float("nan"))
        result[torch.tensor(token_ids)] = torch.tensor(logprobs)
        if not torch.isfinite(result).all():
            raise ValueError("Incomplete or non-finite full-vocabulary log probabilities")
        return result

    def record(label, reference, actual):
        p, q = reference.exp(), actual.exp()
        log_midpoint = torch.logaddexp(reference, actual) - math.log(2)
        metrics = {
            "label": label,
            "local_logprobs_sha256": hashlib.sha256(reference.numpy().tobytes()).hexdigest(),
            "total_variation": ((p - q).abs().sum() / 2).item(),
            "kl_local_vllm": (p * (reference - actual)).sum().item(),
            "js_divergence": ((p * (reference - log_midpoint)).sum() + (q * (actual - log_midpoint)).sum()).item() / 2,
            "max_probability_error": (p - q).abs().max().item(),
            "local_top_token": p.argmax().item(),
            "vllm_top_token": q.argmax().item(),
        }
        records.append(metrics)
        arrays[label + "_local"] = reference.numpy()
        arrays[label + "_vllm"] = actual.numpy()
        print(json.dumps(metrics), flush=True)  # noqa: T201

    def raw():
        outputs = engine.llm.generate(
            [{"prompt_token_ids": histories[0]}],
            SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1, logprobs=-1),
            use_tqdm=False,
            **({"lora_request": engine._lora_request} if native_lora else {}),
        )
        logprobs = outputs[0].outputs[0].logprobs[0]
        return dense([item.logprob for item in logprobs.values()], list(logprobs))

    for stage_index, stage in enumerate(("dense", "lora", "updated_lora")):
        # Both sides retain the same initialized A matrices from the timed model. Recreating only the merged
        # side's adapter would make cross-backend distribution comparisons use different policies.
        generator = torch.Generator(device="cuda").manual_seed(seed + stage_index)
        with torch.no_grad():
            for param in lora_b:
                if stage == "dense":
                    param.zero_()
                else:
                    param.normal_(std=0.03 if stage == "lora" else 0.1, generator=generator)
        reference = local()
        merged_reference = reference
        if stage != "dense":
            # Evaluate the actual BF16 merged inference policy without perturbing training weights.
            from peft.tuners.tuners_utils import BaseTunerLayer

            originals = {
                param: param.detach().cpu().clone()
                for module in engine.model.modules()
                if isinstance(module, BaseTunerLayer)
                for param in module.get_base_layer().parameters()
            }
            try:
                engine.model.merge_adapter()
                merged_reference = local()
            finally:
                try:
                    engine.model.unmerge_adapter()
                finally:
                    with torch.no_grad():
                        for param, original in originals.items():
                            param.copy_(original)
        # Snapshot only the frozen tensors that PEFT merges, to detect merge/unmerge rounding drift.
        frozen = {
            name: param.detach().clone()
            for name, param in engine.model.named_parameters()
            if "base_layer.weight" in name
        }
        context = engine.rollout_phase() if "rollout_phase" in dir(engine) else nullcontext()
        with context:
            engine.sync_weights()
            for turn in range(4):
                if turn == 3:
                    engine.llm.reset_prefix_cache()
                _, _, logprobs, token_ids = engine.generate([histories[0]], images=None, num_generations=1)
                actual = dense(logprobs[0][0], token_ids[0][0])
                record(f"{stage}_turn{turn}", reference, actual)
                record(f"{stage}_turn{turn}_merged_reference", merged_reference, actual)
                record(f"{stage}_turn{turn}_local_after_sync", local(), actual)
        record(stage + "_local_policy_drift", reference, local())
        deltas = [
            (param - frozen[name]).abs().float().max().item()
            for name, param in engine.model.named_parameters()
            if name in frozen
        ]
        records.append({"label": stage + "_frozen_weight_drift", "max_abs": max(deltas, default=0.0)})

    # Keep the old adapter in vLLM, change the local adapter, and prove the diagnostic detects the missed sync.
    engine.sync_weights()
    engine.llm.wake_up(tags=["kv_cache"])
    try:
        record("negative_control_before_update", local(), raw())
        generator = torch.Generator(device="cuda").manual_seed(seed + 3)
        with torch.no_grad():
            for param in lora_b:
                param.normal_(std=0.3, generator=generator)
        record("negative_control_missing_sync", local(), raw())
        engine.sync_weights()
        record("negative_control_after_sync", local(), raw())
    finally:
        if "sleep" in dir(engine):
            engine.sleep()
        else:
            engine.llm.sleep(level=2)
    np.savez_compressed(Path(__file__).parent / f"{side}-{seed}-policy-distributions.npz", **arrays)
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
    from transformers import AutoTokenizer, set_seed

    from trl import GRPOConfig, GRPOTrainer

    manifest = json.loads(args.manifest.read_text())
    config = manifest["workload"].copy()
    if args.profile_dir:
        config["steps"] = min(2, config["steps"])
    # Fix CPU parallelism across both sides, including unprofiled runs.
    torch.set_num_threads(config.get("cpu_threads", 1))
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    data = json.loads(args.data.read_text())
    prompts = [
        tokenizer.decode(data["train"][(args.seed + index // 2) % len(data["train"])][: config["prompt_tokens"]])
        for index in range(config["batch_size"])
    ]
    tool_ids = tokenizer.encode("\nTool result: 42. Continue.\n", add_special_tokens=False)

    def rollout(prompts, trainer):
        initial = tokenizer(prompts)["input_ids"]
        histories = [ids.copy() for ids in initial]
        completions = [[] for _ in initial]
        for turn in range(config["turns"]):
            if config.get("reset_prefix_between_turns", False):
                trainer.vllm_generation.llm.reset_prefix_cache()
            with span(f"turn_{turn}", args.profile_dir is not None):
                _, tokens, _, _ = trainer.vllm_generation.generate(histories, images=None, num_generations=1)
            for history, completion, ids in zip(histories, completions, tokens, strict=True):
                history.extend(ids)
                completion.extend(ids)
                # Deterministic CPU tool feedback; all turns keep the same policy.
                if turn + 1 < config["turns"]:
                    with span("cpu_tool", args.profile_dir is not None):
                        history.extend(tool_ids)
                        completion.extend(tool_ids)
        return {"prompt_ids": initial, "completion_ids": completions, "logprobs": None}

    # Full-vocabulary logprobs are enabled only for the diagnostic, without changing the production API.
    import trl.generation.vllm_generation as generation_module

    llm = generation_module.LLM

    def diagnostic_llm(**kwargs):
        return llm(**kwargs, max_logprobs=-1)

    from peft import LoraConfig

    initialization_started = time.perf_counter()
    with patch.object(generation_module, "LLM", diagnostic_llm if config.get("policy_parity") else llm):
        trainer = GRPOTrainer(
            model=str(args.model_path),
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
            if config["lora"]
            else None,
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
    torch.cuda.synchronize()
    initialization_seconds = time.perf_counter() - initialization_started
    engine = trainer.vllm_generation
    # The same harness also runs revisions predating native adapter publication.
    native_lora = "_lora_config" in dir(engine) and engine._lora_config is not None
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
    counters = {"weight_transfer_bytes": 0, "sync_count": 0, "load_weights_calls": 0}
    sync_weights = engine.sync_weights
    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model
    load_weights = loader.load_weights

    def counted_load(weights):
        counters["load_weights_calls"] += 1

        def counted():
            for name, param in weights:
                counters["weight_transfer_bytes"] += param.numel() * param.element_size()
                yield name, param

        return load_weights(counted())

    def counted_sync():
        counters["sync_count"] += 1
        return sync_weights()

    if native_lora:
        add_lora = engine.llm.llm_engine.add_lora

        def counted_adapter(request):
            from safetensors import safe_open

            with safe_open(str(Path(request.lora_path) / "adapter_model.safetensors"), framework="pt") as adapter:
                for name in adapter.keys():
                    param = adapter.get_tensor(name)
                    counters["weight_transfer_bytes"] += param.numel() * param.element_size()
            return add_lora(request)

        engine.llm.llm_engine.add_lora = annotate(counted_adapter) if args.profile_dir else counted_adapter

    engine.sync_weights = counted_sync
    loader.load_weights = counted_load
    if args.profile_dir:
        engine.llm.wake_up = annotate(engine.llm.wake_up)
        engine.llm.sleep = annotate(engine.llm.sleep)
        engine.llm.reset_prefix_cache = annotate(engine.llm.reset_prefix_cache)
        engine.sync_weights = annotate(engine.sync_weights)
        loader.load_weights = annotate(loader.load_weights)
    durations, outputs = [], []
    sleeping = True
    torch.cuda.reset_peak_memory_stats()
    profiler = make_profiler(args.profile_dir, config["warmup_steps"], config["steps"]) if args.profile_dir else None
    with profiler if profiler else nullcontext():
        for phase in range(config["warmup_steps"] + config["steps"]):
            # A real update between phases must invalidate the previous policy's KV.
            with torch.no_grad():
                if config["lora"]:
                    for param in lora_b:
                        param.add_(0.001)
                else:
                    next(trainer.model.parameters()).add_(0.01)
            trainer.state.global_step = phase
            if phase == config["warmup_steps"]:
                counters.update(weight_transfer_bytes=0, sync_count=0, load_weights_calls=0)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with span("rollout_phase", args.profile_dir is not None):
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
        "publication": "native_lora" if native_lora else "merged",
        "side": args.side,
        "seed": args.seed,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": config["steps"],
        "trainer_initialization_seconds": initialization_seconds,
        "rollout_seconds": sum(durations),
        "phase_seconds": durations,
        **counters,
        "sleeping_after_phase": sleeping,
        "output_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
        "output_tokens": outputs,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "frozen_weight_max_abs_drift": max(
            (
                (param.detach().cpu() - frozen[name]).abs().float().max().item()
                for name, param in trainer.model.named_parameters()
                if name in frozen
            ),
            default=0.0,
        ),
        "prompt_lengths": [len(ids) for ids in tokenizer(prompts)["input_ids"]],
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        },
    }
    if config.get("policy_parity") and not args.profile_dir and args.seed == config["seeds"][0]:
        histories = tokenizer(prompts)["input_ids"]
        record["policy_parity_prompt_ids"] = histories[0]
        record["policy_parity"] = policy_parity(trainer, histories, args.side, args.seed)
    if args.profile_dir:
        save_profile_metadata(args.profile_dir, record, manifest, config["warmup_steps"], config["steps"])
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
