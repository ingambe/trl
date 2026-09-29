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

"""One GRPO training run with colocated vLLM. Invoked in a fresh process for each commit/seed on the GPU worker."""

import argparse
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path


# Metrics compared step by step between base and PR
TRAJECTORY = ("loss", "reward", "grad_norm", "sampling/sampling_logp_difference/mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["base", "head"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from vllm.device_allocator.cumem import CuMemAllocator

    from trl import GRPOConfig, GRPOTrainer

    config = json.loads(args.manifest.read_text())["workload"]
    rank = int(os.environ.get("RANK", "0"))
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    data = json.loads(args.data.read_text())
    train_prompts = [tokenizer.decode(ids[: config["prompt_tokens"]]) for ids in data["train"]]
    eval_prompts = [tokenizer.decode(ids[: config["prompt_tokens"]]) for ids in data["eval"]]

    # Deterministic reward with a learnable target: completions of `target_chars` characters score 1
    def reward(completions, **kwargs):
        if config.get("binary_reward"):
            return [
                float(abs(len(text) - config["target_chars"]) <= config["target_chars"] / 4) for text in completions
            ]
        return [max(0.0, 1 - abs(len(text) - config["target_chars"]) / config["target_chars"]) for text in completions]

    class Timer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            if state.global_step > config["warmup_steps"]:
                timings["steady_seconds"] += time.perf_counter() - self.step_started

    class MemoryBreakdown(TrainerCallback):
        # Training peaks after backward
        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(int(os.environ.get("LOCAL_RANK", "0")))
            if total - free <= breakdown.get("device_bytes", 0):
                return
            executor = trainer.vllm_generation.llm.llm_engine.engine_core.engine_core.model_executor
            sleeping = executor.sleeping_tags if executor.is_sleeping else set()
            vllm = {}
            for data in CuMemAllocator.get_instance().pointer_to_data.values():
                key = f"vllm_{data.tag}_{'released' if data.tag in sleeping else 'mapped'}_bytes"
                vllm[key] = vllm.get(key, 0) + data.handle[1]
            breakdown.clear()
            breakdown.update(
                device_bytes=total - free,
                torch_reserved_bytes=torch.cuda.memory_reserved(),
                torch_allocated_bytes=torch.cuda.memory_allocated(),
                **vllm,
            )

    @contextmanager
    def device_peak(key):
        # Device-wide, so vLLM's allocator counts too
        done = threading.Event()

        def sample():
            while not done.wait(0.005):
                free, total = torch.cuda.mem_get_info(int(os.environ.get("LOCAL_RANK", "0")))
                device[key] = max(device[key], total - free)

        device[key] = 0
        thread = threading.Thread(target=sample)
        thread.start()
        try:
            yield
        finally:
            done.set()
            thread.join()

    timings = {"steady_seconds": 0.0, "generation_seconds": 0.0}
    device = {}
    breakdown = {}
    share = "vllm_share_weights" in GRPOConfig.__dataclass_fields__
    with device_peak("init_peak_device_bytes"):
        trainer = GRPOTrainer(
            model=str(args.model_path),
            reward_funcs=reward,
            processing_class=tokenizer,
            args=GRPOConfig(
                output_dir=str(args.output.parent / "checkpoints"),
                model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
                use_vllm=True,
                vllm_mode="colocate",
                vllm_enable_sleep_mode=True,
                **({"vllm_share_weights": True} if share else {}),
                **({"vllm_native_lora": True} if share and config.get("native_lora") else {}),
                vllm_gpu_memory_utilization=0.35,
                vllm_max_model_length=config["context_tokens"],
                max_completion_length=config["completion_tokens"],
                num_generations=config["num_generations"],
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=1,
                max_steps=config["steps"],
                learning_rate=config["learning_rate"],
                lr_scheduler_type="constant",
                warmup_steps=0,
                weight_decay=0.0,
                optim="adamw_torch",
                bf16=True,
                seed=args.seed,
                data_seed=args.seed,
                logging_steps=1,
                report_to="none",
                save_strategy="no",
                disable_tqdm=True,
            ),
            train_dataset=Dataset.from_dict({"prompt": train_prompts}),
            peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
            if config.get("peft")
            else None,
            callbacks=[Timer(), MemoryBreakdown()],
        )
    # Colocated vLLM reseeds the process with a fixed seed, so reseed for sampling to vary across seeds
    set_seed(args.seed)
    generate_and_score = trainer._generate_and_score_completions

    def timed_generate_and_score(inputs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = generate_and_score(inputs)
        torch.cuda.synchronize()
        if trainer.state.global_step >= config["warmup_steps"]:
            timings["generation_seconds"] += time.perf_counter() - started
        return output

    trainer._generate_and_score_completions = timed_generate_and_score
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with device_peak("train_peak_device_bytes"):
        trainer.train()
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    if trainer.state.global_step != config["steps"]:
        raise RuntimeError("Training stopped before the requested step count")
    trajectory = [{key: r.get(key) for key in TRAJECTORY} for r in trainer.state.log_history if "loss" in r]
    if any(not math.isfinite(value) for step in trajectory for value in step.values() if value is not None):
        raise RuntimeError("Non-finite training metric")
    if rank > 0:
        return

    # Held-out quality: greedy completions outside vLLM
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    batch = tokenizer(eval_prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.inference_mode():
        generated = model.generate(
            **batch, max_new_tokens=config["completion_tokens"], do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    completions = tokenizer.batch_decode(generated[:, batch["input_ids"].shape[1] :], skip_special_tokens=True)
    with torch.no_grad():
        fingerprint = sum(param.double().sum().item() for param in model.parameters())
    record = {
        "side": args.side,
        "seed": args.seed,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": trainer.state.global_step,
        "train_seconds": train_seconds,
        **timings,
        "update_seconds": timings["steady_seconds"] - timings["generation_seconds"],
        "peak_memory_bytes": peak,
        **device,
        "train_peak_breakdown": breakdown,
        "eval_reward": sum(reward(completions)) / len(completions),
        "parameter_sum": fingerprint,
        "trajectory": trajectory,
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        },
    }
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
