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

"""Replay frozen GRPO rollouts through a reference and a candidate precision config, and compare the updates.

Rollouts are generated once by the reference trainer, then both trainers take the same optimizer steps on them:

```sh
python tools/fp8_parity/replay.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/DeepMath-103K \
    --candidate '{"fp8_recipe": "rowwise_with_gw_hp"}' --steps 100 --output fp8_replay.json
```

With `--target-kl`, it instead measures the policy KL each linear layer causes alone in FP8 and lists the layers to
keep in high precision, for `fp8_skip_modules`.
"""

import argparse
import copy
import json
import tempfile
from collections import defaultdict
from functools import partial

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import set_seed

from trl import GRPOConfig, GRPOTrainer
from trl.models.fp8 import convert_to_fp8_training


# Smallest positive float8_e4m3fn value
E4M3_MIN = 2.0**-9


def tensor_stats(x: torch.Tensor, max_samples: int = 1 << 20) -> dict[str, float]:
    """Magnitude percentiles of `x`, and the fraction a tensorwise E4M3 cast would flush to zero."""
    x = x.detach().flatten().abs().float()
    amax = x.max()
    index = torch.randint(x.numel(), (min(x.numel(), max_samples),), generator=torch.Generator().manual_seed(0))
    sample = x if x.numel() <= max_samples else x[index.to(x.device)]
    p99, p999, p9999 = torch.quantile(sample, torch.tensor([0.99, 0.999, 0.9999], device=x.device)).tolist()
    scale = amax / torch.finfo(torch.float8_e4m3fn).max
    return {
        "max": amax.item(),
        "p99": p99,
        "p99.9": p999,
        "p99.99": p9999,
        "max_over_p99": (amax / max(p99, 1e-30)).item(),
        "underflow": ((x > 0) & (x < E4M3_MIN * scale)).float().mean().item(),
    }


def compare_logps(
    ref: torch.Tensor,
    cand: torch.Tensor,
    old: torch.Tensor,
    mask: torch.Tensor,
    epsilon_low: float,
    epsilon_high: float,
) -> dict[str, float]:
    """Log-prob error and disagreement of the GRPO clipping decision between two policies."""
    mask = mask.bool()
    diff = (cand - ref)[mask]
    clipped = [
        ((ratio < 1 - epsilon_low) | (ratio > 1 + epsilon_high))[mask]
        for ratio in (torch.exp(ref - old), torch.exp(cand - old))
    ]
    ratio = torch.exp(cand - old)[mask]
    return {
        "logp_rms": diff.pow(2).mean().sqrt().item(),
        "logp_max": diff.abs().max().item(),
        "clip_disagreement": (clipped[0] != clipped[1]).float().mean().item(),
        "ratio_p50": ratio.quantile(0.5).item(),
        "ratio_p99": ratio.quantile(0.99).item(),
        "ratio_max": ratio.max().item(),
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


@torch.no_grad()
def next_token_logps(trainer, batch, rows: int) -> torch.Tensor:
    """Full next-token log-distribution of `trainer.model` over the completions of the first `rows` rows."""
    input_ids = torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1)[:rows]
    attention_mask = torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1)[:rows]
    length = batch["completion_ids"].size(1)
    with trainer.accelerator.autocast():
        logits = trainer.model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -length - 1 : -1]
    return torch.log_softmax(logits.float() / trainer.temperature, dim=-1)


def kl_divergence(ref_logps: torch.Tensor, cand_logps: torch.Tensor, batch, rows: int) -> float:
    """Mean KL(ref || cand) over the completion tokens."""
    kl = (ref_logps.exp() * (ref_logps - cand_logps)).sum(-1)
    return kl[batch["completion_mask"][:rows].bool()].mean().item()


def sensitivity(trainer, batch, recipe: str, rows: int, target_kl: float) -> dict:
    """Rank the linear layers by the policy KL each one causes alone in FP8, then keep the worst in high precision
    until the KL of the whole model is below `target_kl`."""
    model = trainer.model
    fp8_model = copy.deepcopy(model)
    convert_to_fp8_training(fp8_model, recipe)
    modules = dict(model.named_modules())
    fp8_modules = {
        name: m
        for name, m in fp8_model.named_modules()
        if isinstance(modules.get(name), torch.nn.Linear) and type(m) is not type(modules[name])
    }

    def swap(name, module):
        parent, _, child = name.rpartition(".")
        setattr(model.get_submodule(parent), child, module)

    # Output error of each layer in FP8 on its high-precision input
    local_error = {}

    def hook(module, args, output, name):
        fp8_output = fp8_modules[name](args[0])
        local_error[name] = ((fp8_output - output).float().norm() / output.float().norm()).item()

    handles = [modules[name].register_forward_hook(partial(hook, name=name)) for name in fp8_modules]
    ref_logps = next_token_logps(trainer, batch, rows)
    for handle in handles:
        handle.remove()

    kl = {}
    for name, fp8_module in fp8_modules.items():
        swap(name, fp8_module)
        kl[name] = kl_divergence(ref_logps, next_token_logps(trainer, batch, rows), batch, rows)
        swap(name, modules[name])

    # Greedy fallback: all layers in FP8, then the most sensitive ones back to high precision
    for name, fp8_module in fp8_modules.items():
        swap(name, fp8_module)
    skip, ranked = [], sorted(kl, key=kl.get, reverse=True)
    model_kl = kl_divergence(ref_logps, next_token_logps(trainer, batch, rows), batch, rows)
    while model_kl > target_kl and ranked:
        skip.append(ranked.pop(0))
        swap(skip[-1], modules[skip[-1]])
        model_kl = kl_divergence(ref_logps, next_token_logps(trainer, batch, rows), batch, rows)
    for name in fp8_modules:
        swap(name, modules[name])
    return {
        "kl": kl,
        "local_error": local_error,
        "model_kl": model_kl,
        "fp8_skip_modules": [name.removeprefix("base_model.model.").replace(".base_layer", "") for name in skip],
    }


def per_token_logps(trainer, batch) -> torch.Tensor:
    with torch.no_grad():
        logps, _, _ = trainer._get_per_token_logps_and_entropies(
            trainer.model,
            torch.cat([batch["prompt_ids"], batch["completion_ids"]], dim=1),
            torch.cat([batch["prompt_mask"], batch["completion_mask"]], dim=1),
            batch["completion_ids"].size(1),
        )
    return logps


def record_activations(model) -> tuple[dict, list]:
    """Hook every linear layer and activation function to record forward and backward magnitude statistics."""
    stats = defaultdict(dict)
    handles = []
    for name, module in model.named_modules():
        if not (isinstance(module, torch.nn.Linear) or name.endswith("act_fn")):
            continue

        def forward_hook(module, args, output, name=name):
            stats[name]["input"] = tensor_stats(args[0])
            stats[name]["output"] = tensor_stats(output)

        def backward_hook(module, grad_input, grad_output, name=name):
            stats[name]["grad_output"] = tensor_stats(grad_output[0])

        handles += [module.register_forward_hook(forward_hook), module.register_full_backward_hook(backward_hook)]
    return stats, handles


def build_trainer(args, overrides: dict) -> GRPOTrainer:
    set_seed(args.seed)

    # Deterministic reward with a learnable target, so groups have non-zero advantages
    def reward(completions, **kwargs):
        lengths = [len(c[-1]["content"] if isinstance(c, list) else c) for c in completions]
        return [max(0.0, 1 - abs(length - args.target_chars) / args.target_chars) for length in lengths]

    config = GRPOConfig(
        output_dir=tempfile.mkdtemp(),
        model_init_kwargs={"dtype": "bfloat16"},
        bf16=True,
        disable_dropout=True,
        per_device_train_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to="none",
        **{**json.loads(args.config), **overrides},
    )
    dataset = load_dataset(args.dataset, split="train").select(
        range(args.batch_size // args.num_generations * args.batches)
    )
    peft_config = LoraConfig(r=args.lora_rank, target_modules="all-linear") if args.lora_rank else None
    trainer = GRPOTrainer(
        model=args.model, reward_funcs=reward, args=config, train_dataset=dataset, peft_config=peft_config
    )
    trainer.create_optimizer()
    trainer.current_gradient_accumulation_steps = 1  # set by the training loop
    return trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, help="Dataset with a `prompt` column")
    parser.add_argument("--candidate", required=True, help="JSON of GRPOConfig overrides for the candidate")
    parser.add_argument("--config", default="{}", help="JSON of GRPOConfig overrides for both sides")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batches", type=int, default=4, help="Number of frozen rollout batches, replayed in turn")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, help="Train a LoRA adapter of this rank on both sides")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--target-chars", type=int, default=200)
    parser.add_argument("--kl-rows", type=int, default=2, help="Rows whose full next-token distribution is compared")
    parser.add_argument(
        "--target-kl", type=float, help="Instead of replaying, find the layers to keep in high precision for this KL"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ref = build_trainer(args, {})

    # Freeze the rollouts, with the reference policy as the behavior policy
    if ref.beta != 0.0:
        raise ValueError("The replay compares updates without a KL term: set `beta=0.0`.")
    # Each prompt repeated `num_generations` times, as the trainer's sampler lays out a group
    prompts = [row for row in ref.train_dataset.to_list() for _ in range(args.num_generations)]
    batches = []
    for start in range(0, len(prompts), args.batch_size):
        batch = ref._generate_and_score_completions(prompts[start : start + args.batch_size])
        batch["old_per_token_logps"] = per_token_logps(ref, batch)
        batches.append(batch)

    if args.target_kl is not None:
        recipe = json.loads(args.candidate)["fp8_recipe"]
        report = sensitivity(ref, batches[0], recipe, args.kl_rows, args.target_kl)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        return

    cand = build_trainer(args, json.loads(args.candidate))

    initial = {name: p.detach().clone() for name, p in ref.model.named_parameters() if p.requires_grad}
    activations, handles = record_activations(ref.model)
    report = {"args": vars(args), "steps": []}
    for step in range(args.steps):
        batch = batches[step % len(batches)]
        mask = batch["completion_mask"]
        logps = [per_token_logps(trainer, batch) for trainer in (ref, cand)]
        metrics = compare_logps(*logps, batch["old_per_token_logps"], mask, ref.epsilon_low, ref.epsilon_high)
        metrics["kl"] = kl_divergence(
            *(next_token_logps(t, batch, args.kl_rows) for t in (ref, cand)), batch, args.kl_rows
        )

        updates = []
        for trainer in (ref, cand):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            trainer.compute_loss(trainer.model, batch).backward()
            before = {name: p.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
            grads = {name: p.grad.detach().clone() for name, p in trainer.model.named_parameters() if p.requires_grad}
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.args.max_grad_norm)
            trainer.optimizer.step()
            deltas = {
                name: p.detach() - before[name] for name, p in trainer.model.named_parameters() if p.requires_grad
            }
            updates.append((grads, deltas))
        if step == 0:
            for handle in handles:
                handle.remove()

        (ref_grads, ref_deltas), (cand_grads, cand_deltas) = updates
        grad_cosines = {name: cosine(ref_grads[name], cand_grads[name]) for name in ref_grads}
        cand_params = dict(cand.model.named_parameters())
        drift = sum(
            (cand_params[name].detach() - p.detach()).float().norm() ** 2
            for name, p in ref.model.named_parameters()
            if name in initial
        )
        travel = sum(
            (p.detach() - initial[name]).float().norm() ** 2
            for name, p in ref.model.named_parameters()
            if name in initial
        )
        metrics.update(
            grad_cosine_min=min(grad_cosines.values()),
            grad_cosine_worst=sorted(grad_cosines, key=grad_cosines.get)[:5],
            grad_norm_ratio={
                name: (cand_grads[name].float().norm() / ref_grads[name].float().norm().clamp(min=1e-30)).item()
                for name in sorted(grad_cosines, key=grad_cosines.get)[:5]
            },
            update_cosine=cosine(
                *(torch.cat([d.flatten() for d in deltas.values()]) for deltas in (ref_deltas, cand_deltas))
            ),
            param_drift=(drift.sqrt() / travel.sqrt().clamp(min=1e-30)).item(),
        )
        report["steps"].append(metrics)
        print(f"step {step}: " + ", ".join(f"{k}={v:.3g}" for k, v in metrics.items() if isinstance(v, float)))  # noqa: T201

    report["activations"] = activations
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
