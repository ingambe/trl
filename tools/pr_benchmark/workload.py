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

"""One SFT run. Invoked in a fresh process for each commit/seed on the GPU worker."""

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

from profiling import make_profiler, save_profile_metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--side", choices=["base", "head"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout))

    import torch
    import torch.nn.functional as F
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, set_seed

    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This workload requires a CUDA GPU with BF16 support")
    manifest = json.loads(args.manifest.read_text())
    config = manifest["workload"].copy()
    active_steps = min(2, config["steps"] - config["warmup_steps"])
    if args.profile_dir:
        config["steps"] = config["warmup_steps"] + active_steps
    profiler = make_profiler(args.profile_dir, config["warmup_steps"], active_steps) if args.profile_dir else None
    data = json.loads(args.data.read_text())
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=False,
        token=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=False,
        token=False,
    )

    class Timer(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.started = time.perf_counter()
            self.steady_started = None

        def on_step_end(self, args, state, control, **kwargs):
            if profiler:
                profiler.step()
            if state.global_step == config["warmup_steps"]:
                torch.cuda.synchronize()
                self.steady_started = time.perf_counter()

        def on_train_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            ended = time.perf_counter()
            self.total = ended - self.started
            self.steady = ended - self.steady_started

    timer = Timer()
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=SFTConfig(
            output_dir=str(args.output.parent / "checkpoints"),
            max_steps=config["steps"],
            per_device_train_batch_size=config["batch_size"],
            gradient_accumulation_steps=config["gradient_accumulation_steps"],
            max_length=config["max_length"],
            learning_rate=2e-5,
            lr_scheduler_type="constant",
            warmup_steps=0,
            weight_decay=0.0,
            optim="adamw_torch",
            bf16=True,
            fp16=False,
            tf32=False,
            gradient_checkpointing=True,
            seed=args.seed,
            data_seed=args.seed,
            dataloader_num_workers=0,
            report_to="none",
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            disable_tqdm=True,
            packing=False,
            padding_free=False,
            loss_type="chunked_nll",
            dataset_kwargs={"skip_prepare_dataset": True},
        ),
        train_dataset=Dataset.from_dict({"input_ids": data["train"]}),
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    with profiler if profiler else nullcontext():
        trained = trainer.train()
    peak = torch.cuda.max_memory_allocated()
    if trainer.state.global_step != config["steps"]:
        raise RuntimeError("Training stopped before the requested step count")
    for record in trainer.state.log_history:
        for metric in ("loss", "grad_norm"):
            if metric in record and not math.isfinite(record[metric]):
                raise RuntimeError(f"Non-finite training {metric}")

    # Evaluate token-weighted next-token NLL independently of the trainer's eval/loss implementation.
    model.eval()
    total_nll, tokens = 0.0, 0
    with torch.inference_mode():
        for ids in data["eval"]:
            ids = torch.tensor([ids], device="cuda")
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            total_nll += F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1), reduction="sum"
            ).item()
            tokens += ids.size(1) - 1

    environment = {
        "gpu": torch.cuda.get_device_name(),
        "gpu_memory": torch.cuda.get_device_properties(0).total_memory,
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "python": platform.python_version(),
        "cuda": torch.version.cuda,
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
    }
    record = {
        "side": args.side,
        "seed": args.seed,
        "sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.checkout, text=True).strip(),
        "steps": trainer.state.global_step,
        "train_seconds": timer.total,
        "steady_seconds": timer.steady,
        "train_loss": trained.training_loss,
        "eval_loss": total_nll / tokens,
        "peak_memory_bytes": peak,
        "environment": environment,
        "trajectory": [r for r in trainer.state.log_history if "loss" in r],
    }
    if args.profile_dir:
        save_profile_metadata(args.profile_dir, record, manifest, config["warmup_steps"], active_steps)
    args.output.write_text(json.dumps(record, allow_nan=False))


if __name__ == "__main__":
    main()
