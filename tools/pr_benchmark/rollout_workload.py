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
import subprocess
import sys
import time
from pathlib import Path


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
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
