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


def policy_parity(trainer, histories, side, seed):
    """Compare full next-token distributions on fixed histories, including a stale-adapter control."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from vllm import SamplingParams

    engine = trainer.vllm_generation
    engine.temperature = 1.0
    engine.top_p = 1.0
    engine.top_k = -1
    engine.min_p = 0.0
    engine.max_completion_length = 1
    engine.logprobs = -1
    engine.generation_kwargs = {}
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
        out = (
            engine.llm.generate(
                [{"prompt_token_ids": histories[0]}],
                SamplingParams(temperature=1.0, top_p=1.0, top_k=-1, max_tokens=1, logprobs=-1),
                use_tqdm=False,
            )[0]
            .outputs[0]
            .logprobs[0]
        )
        return dense([item.logprob for item in out.values()], list(out))

    for stage_index, stage in enumerate(("dense", "lora", "updated_lora")):
        if stage == "lora":
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(seed)
                engine.model = get_peft_model(
                    engine.model,
                    LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
                )
        if stage != "dense":
            generator = torch.Generator(device="cuda").manual_seed(seed + stage_index)
            with torch.no_grad():
                for name, param in engine.model.named_parameters():
                    if "lora_B" in name:
                        param.normal_(std=0.03 if stage == "lora" else 0.1, generator=generator)
        reference = local()
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
                record(f"{stage}_turn{turn}_local_after_sync", local(), actual)
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
        old = local()
        old_vllm = raw()
        record("negative_control_before_update", old, old_vllm)
        generator = torch.Generator(device="cuda").manual_seed(seed + 3)
        with torch.no_grad():
            for name, param in engine.model.named_parameters():
                if "lora_B" in name:
                    param.normal_(std=0.3, generator=generator)
        changed = local()
        record("negative_control_missing_sync", changed, raw())
        engine.sync_weights()
        record("negative_control_after_sync", local(), raw())
    finally:
        engine.llm.sleep(level=2)
    np.savez_compressed(Path(__file__).parent / f"{side}-{seed}-policy-distributions.npz", **arrays)
    return records


def main():
    parser = argparse.ArgumentParser()
    for name in ("model-path", "checkout", "data", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--side", choices=["base", "head"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed

    from trl import GRPOConfig, GRPOTrainer

    config = json.loads(args.manifest.read_text())["workload"]
    if config.get("policy_parity"):
        # The diagnostic performs many small CPU tensor reductions on a worker with a limited CPU quota.
        torch.set_num_threads(1)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    data = json.loads(args.data.read_text())
    prompt = tokenizer.decode(data["train"][args.seed % len(data["train"])][:48])
    prompts = [prompt, prompt]
    tool_ids = tokenizer.encode("\nTool result: 42. Continue.\n", add_special_tokens=False)

    def rollout(prompts, trainer):
        initial = tokenizer(prompts)["input_ids"]
        histories = [ids.copy() for ids in initial]
        completions = [[] for _ in initial]
        for turn in range(config["turns"]):
            if config.get("reset_prefix_between_turns", False):
                trainer.vllm_generation.llm.reset_prefix_cache()
            _, tokens, _, _ = trainer.vllm_generation.generate(histories, images=None, num_generations=1)
            for history, completion, ids in zip(histories, completions, tokens, strict=True):
                history.extend(ids)
                completion.extend(ids)
                # Deterministic CPU tool feedback; all turns keep the same policy.
                if turn + 1 < config["turns"]:
                    history.extend(tool_ids)
                    completion.extend(tool_ids)
        return {"prompt_ids": initial, "completion_ids": completions, "logprobs": None}

    # Full-vocabulary logprobs are enabled only for the diagnostic, without changing the production API.
    import trl.generation.vllm_generation as generation_module

    llm = generation_module.LLM

    def diagnostic_llm(**kwargs):
        return llm(**kwargs, max_logprobs=-1)

    with patch.object(generation_module, "LLM", diagnostic_llm if config.get("policy_parity") else llm):
        trainer = GRPOTrainer(
            model=str(args.model_path),
            processing_class=tokenizer,
            args=GRPOConfig(
                output_dir=str(args.output.parent / "checkpoints"),
                model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                vllm_gpu_memory_utilization=0.35,
                vllm_max_model_length=512,
                max_completion_length=16,
                generation_kwargs={"ignore_eos": True, "min_tokens": 16},
                temperature=0.0,
                num_generations=2,
                per_device_train_batch_size=2,
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
    counters = {"weight_transfer_bytes": 0, "sync_count": 0}
    sync_weights = engine.sync_weights
    loader = engine.llm.llm_engine.model_executor.driver_worker.model_runner.model
    load_weights = loader.load_weights

    def counted_load(weights):
        weights = list(weights)
        counters["weight_transfer_bytes"] += sum(param.numel() * param.element_size() for _, param in weights)
        return load_weights(weights)

    def counted_sync():
        counters["sync_count"] += 1
        return sync_weights()

    engine.sync_weights = counted_sync
    loader.load_weights = counted_load
    durations, outputs = [], []
    sleeping = True
    torch.cuda.reset_peak_memory_stats()
    for phase in range(config["warmup_steps"] + config["steps"]):
        # A real update between phases must invalidate the previous policy's KV.
        with torch.no_grad():
            next(trainer.model.parameters()).add_(0.01)
        trainer.state.global_step = phase
        if phase == config["warmup_steps"]:
            counters.update(weight_transfer_bytes=0, sync_count=0)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = trainer._generate(prompts)
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        sleeping = sleeping and engine.llm.llm_engine.is_sleeping()
        if phase >= config["warmup_steps"]:
            durations.append(duration)
            outputs.append(result[1])
    record = {
        "side": args.side,
        "seed": args.seed,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": config["steps"],
        "rollout_seconds": sum(durations),
        "phase_seconds": durations,
        **counters,
        "sleeping_after_phase": sleeping,
        "output_sha256": hashlib.sha256(json.dumps(outputs).encode()).hexdigest(),
        "output_tokens": outputs,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        },
    }
    if config.get("policy_parity"):
        histories = tokenizer(prompts)["input_ids"]
        record["policy_parity_prompt_ids"] = histories[0]
        record["policy_parity"] = policy_parity(trainer, histories, args.side, args.seed)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
