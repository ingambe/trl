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

"""One GRPO training run on a seeded tool-calling game with colocated vLLM. Invoked in a fresh process per commit."""

import argparse
import importlib.metadata
import itertools
import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from games import GAMES


# Metrics compared step by step between base and PR
TRAJECTORY = ("loss", "reward", "grad_norm", "completions/mean_length", "sampling/sampling_logp_difference/mean")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["base", "head"], required=True)
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
    set_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    game, prompt = GAMES[config["game"]]
    # Held-out games never appear in training
    eval_seeds = list(range(10**6, 10**6 + config["eval_games"]))

    def games(seeds):
        return Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * len(seeds), "seed": seeds})

    def reward(environments, **kwargs):
        return [env.reward for env in environments]

    class Timer(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.step_started = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            step_seconds[state.global_step] = time.perf_counter() - self.step_started
            # Every rank stops at the same step once the time budget is spent
            over = torch.tensor(time.perf_counter() - started > 60 * config["train_minutes"], device="cuda")
            control.should_training_stop = bool(trainer.accelerator.gather(over).any())

    class FrozenStart(TrainerCallback):
        # The first steps run every code path with a zero learning rate, so both commits time the same policy
        def on_pre_optimizer_step(self, args, state, control, optimizer, **kwargs):
            if state.global_step < config["warmup_steps"] + config["speed_steps"]:
                for group in optimizer.param_groups:
                    group["lr"] = 0.0

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

    step_seconds, generation_seconds = {}, {}
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
                max_tool_calling_iterations=7,
                num_generations=config["num_generations"],
                num_generations_eval=1,
                per_device_train_batch_size=config["batch_size"],
                per_device_eval_batch_size=config["eval_games"] // config.get("gpus", 1),
                gradient_accumulation_steps=1,
                max_steps=config["max_steps"],
                learning_rate=config["learning_rate"],
                lr_scheduler_type="constant",
                warmup_steps=0,
                weight_decay=0.0,
                optim="adamw_torch",
                bf16=True,
                seed=config["seed"],
                data_seed=config["seed"],
                logging_steps=1,
                report_to="none",
                save_strategy="no",
                disable_tqdm=True,
            ),
            train_dataset=games(list(range(config["max_steps"] * config["batch_size"]))),
            eval_dataset=games(eval_seeds),
            environment_factory=game,
            peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
            if config.get("peft")
            else None,
            callbacks=[Timer(), FrozenStart(), MemoryBreakdown()],
        )
    # Per-request sampling seeds, so a token flipped by rounding changes one completion instead of the whole batch
    llm = trainer.vllm_generation.llm
    generate = llm.generate
    calls = itertools.count()

    def seeded_generate(prompts, sampling_params, **kwargs):
        call = next(calls)
        requests = [sampling_params.clone() for _ in prompts]
        for index, request in enumerate(requests):
            request.seed = hash((config["seed"], rank, call, index)) % 2**31
        return generate(prompts, sampling_params=requests, **kwargs)

    llm.generate = seeded_generate
    generate_and_score = trainer._generate_and_score_completions

    def timed_generate_and_score(inputs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = generate_and_score(inputs)
        torch.cuda.synchronize()
        if trainer.model.training:
            generation_seconds[trainer.state.global_step + 1] = time.perf_counter() - started
        return output

    trainer._generate_and_score_completions = timed_generate_and_score
    initial_eval_reward = trainer.evaluate()["eval_reward"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with device_peak("train_peak_device_bytes"):
        trainer.train()
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    trajectory = [{key: r.get(key) for key in TRAJECTORY} for r in trainer.state.log_history if "loss" in r]
    if any(not math.isfinite(value) for step in trajectory for value in step.values() if value is not None):
        raise RuntimeError("Non-finite training metric")
    eval_reward = trainer.evaluate()["eval_reward"]
    if rank > 0:
        return

    steps = range(1, trainer.state.global_step + 1)
    record = {
        "side": args.side,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": trainer.state.global_step,
        "train_seconds": train_seconds,
        "step_seconds": [step_seconds[step] for step in steps],
        "generation_seconds": [generation_seconds[step] for step in steps],
        "peak_memory_bytes": peak,
        **device,
        "train_peak_breakdown": breakdown,
        "initial_eval_reward": initial_eval_reward,
        "eval_reward": eval_reward,
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
