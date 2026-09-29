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

"""Compare paired GPU measurements without requiring the training dependencies locally."""

import math
import statistics


def interval(values):
    # Conservative two-sided 95% Student t critical values. Round df down between entries.
    critical = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 15: 2.132, 20: 2.086, 30: 2.042}
    n = len(values)
    mean = statistics.mean(values)
    t = critical[max(df for df in critical if df <= n - 1)]
    radius = t * statistics.stdev(values) / math.sqrt(n)
    return [mean - radius, mean + radius]


def compare(result, manifest):
    if result["manifest_id"] != manifest["id"]:
        raise ValueError("Results belong to a different benchmark request")
    config = manifest["workload"]
    records = result["records"]
    expected = {(side, seed) for side in ("base", "head") for seed in config["seeds"]}
    indexed = {(r["side"], r["seed"]): r for r in records}
    if set(indexed) != expected or len(records) != len(expected):
        raise ValueError("Missing, duplicate, or unexpected measurements")
    rollout = config.get("kind") == "vllm-rollout"
    grpo = config.get("kind") == "grpo-train"
    if rollout:
        metrics, diagnostics = ("rollout_seconds", "weight_transfer_bytes"), ()
    elif grpo:
        metrics, diagnostics = ("train_seconds", "steady_seconds"), ("generation_seconds", "update_seconds")
    else:
        metrics, diagnostics = ("train_seconds", "steady_seconds", "eval_loss"), ("train_loss",)
    environments = []
    for (side, seed), record in indexed.items():
        if record["sha"] != manifest[f"{side}_sha"] or record["steps"] != config["steps"]:
            raise ValueError("Wrong commit or incomplete training")
        for metric in (*metrics, *diagnostics, "peak_memory_bytes", "workload_seconds"):
            value = record[metric]
            # Sharing weights with vLLM legitimately publishes nothing
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or (value == 0 and metric != "weight_transfer_bytes")
            ):
                raise ValueError(f"Invalid {metric} for {side}, seed {seed}")
        environments.append(record["environment"])
    if any(env != environments[0] for env in environments):
        raise ValueError("GPU or dependency environment differs between runs")

    summary = {
        "outcome": "no_regression",
        "metrics": {},
        "pairs": len(config["seeds"]),
        "gpu": environments[0]["gpu"],
    }
    for metric in metrics:
        base = [indexed["base", seed][metric] for seed in config["seeds"]]
        head = [indexed["head", seed][metric] for seed in config["seeds"]]
        # A base that shares weights transfers nothing
        changes = [100 * (h / b - 1) if b else 100.0 * (h > 0) for b, h in zip(base, head, strict=True)]
        margin = manifest["thresholds"][metric]
        ci = interval(changes) if len(changes) >= 5 else None
        if ci is None:
            outcome = "inconclusive"
        elif ci[0] > margin:
            outcome = "regression"
        elif ci[1] > margin:
            outcome = "inconclusive"
        elif ci[1] < -margin:
            outcome = "improved"
        else:
            outcome = "no_regression"
        summary["metrics"][metric] = {
            "base_mean": statistics.mean(base),
            "head_mean": statistics.mean(head),
            "change_pct": statistics.mean(changes),
            "interval_pct": ci,
            "margin_pct": margin,
            "outcome": outcome,
        }
    outcomes = {m["outcome"] for m in summary["metrics"].values()}
    for outcome in ("regression", "inconclusive", "improved"):
        if outcome in outcomes:
            summary["outcome"] = outcome
            break
    if rollout:
        # Fewer transferred bytes alone are not a rollout speedup.
        if summary["outcome"] == "improved" and summary["metrics"]["rollout_seconds"]["outcome"] != "improved":
            summary["outcome"] = "no_regression"
        runs = {side: [indexed[side, seed] for seed in config["seeds"]] for side in ("base", "head")}
        drift = {side: max(r["frozen_weight_max_abs_drift"] for r in runs[side]) for side in runs}
        parity = {side: {item["label"]: item for item in result["policy_parity"][side]} for side in runs}
        summary["quality"] = {
            "Greedy rollout tokens match base": all(
                b["output_tokens"] == h["output_tokens"] for b, h in zip(runs["base"], runs["head"], strict=True)
            ),
            "vLLM asleep after every phase": all(r["sleeping_after_phase"] for r in records),
            "Every phase synchronized the updated policy": all(r["sync_count"] >= config["steps"] for r in records),
            "Frozen-weight drift no worse than base": drift["head"] <= drift["base"],
            "Stale-policy control detected (TV above twice the synced TV)": all(
                p["updated_lora_stale_reference"]["total_variation"] > 2 * p["updated_lora"]["total_variation"]
                for p in parity.values()
            ),
        }
        summary["syncs_per_phase"] = {
            side: statistics.mean(r["sync_count"] for r in runs[side]) / config["steps"] for side in runs
        }
        summary["frozen_weight_drift"] = drift
        summary["policy_parity"] = parity
        if not all(summary["quality"].values()) and summary["outcome"] in ("improved", "no_regression"):
            summary["outcome"] = "quality_failed"
    if grpo:
        runs = {side: [indexed[side, seed] for seed in config["seeds"]] for side in ("base", "head")}

        def gap(key):
            return max(
                abs(b[key] - h[key])
                for base, head in zip(runs["base"], runs["head"], strict=True)
                for b, h in zip(base["trajectory"], head["trajectory"], strict=True)
                if b[key] is not None and h[key] is not None
            )

        means = {
            key: {side: statistics.mean(r[key] for r in runs[side]) for side in runs}
            for key in (
                "generation_seconds",
                "update_seconds",
                "peak_memory_bytes",
                "init_peak_device_bytes",
                "train_peak_device_bytes",
                "eval_reward",
            )
        }
        logp = "sampling/sampling_logp_difference/mean"
        parameters = max(
            abs(h["parameter_sum"] - b["parameter_sum"]) / abs(b["parameter_sum"])
            for b, h in zip(runs["base"], runs["head"], strict=True)
        )
        # Trajectories diverge even main against main, so quality is compared per seed
        gaps = {side: [[step[logp] for step in r["trajectory"]] for r in runs[side]] for side in runs}
        if any(
            not isinstance(gap, float) or not math.isfinite(gap)
            for side in gaps.values()
            for run in side
            for gap in run
        ):
            raise ValueError("Missing or invalid vLLM/trainer logprob gap")
        seed_gaps = {side: [statistics.mean(run) for run in gaps[side]] for side in runs}
        reward_interval = interval(
            [h["eval_reward"] - b["eval_reward"] for b, h in zip(runs["base"], runs["head"], strict=True)]
        )
        seed_rewards = {
            side: [statistics.mean(step["reward"] for step in r["trajectory"]) for r in runs[side]] for side in runs
        }
        train_interval = interval([h - b for b, h in zip(seed_rewards["base"], seed_rewards["head"], strict=True)])
        logp_interval = interval(
            [100 * (h / b - 1) for b, h in zip(seed_gaps["base"], seed_gaps["head"], strict=True)]
        )
        summary["training"] = {
            "means": means,
            "max_step_gap": {key: gap(key) for key in ("loss", "reward", "grad_norm")},
            "logprob_gap": {side: statistics.mean(seed_gaps[side]) for side in runs},
            "eval_reward_interval": reward_interval,
            "train_reward_interval": train_interval,
            "logprob_gap_interval_pct": logp_interval,
            "parameter_sum_relative_gap": parameters,
        }
        train_check = "Training reward not worse than base by over 0.05"
        reward_check = "Held-out reward not clearly worse than base by over 0.02"
        logp_check = "vLLM/trainer logprob gap not worse than base by over 10%"
        # Held-out reward varies too much across seeds to show it is within 0.02, so it only fails when clearly worse
        summary["quality"] = {
            train_check: train_interval[0] >= -0.05,
            reward_check: reward_interval[1] >= -0.02,
            logp_check: logp_interval[1] <= 10,
        }
        # A check passes only when its whole interval clears the margin, and is inconclusive when it crosses it
        summary["quality_inconclusive"] = [
            check
            for check, crossing in ((train_check, train_interval[1] >= -0.05), (logp_check, logp_interval[0] <= 10))
            if crossing and not summary["quality"][check]
        ]
        failed = [check for check, passed in summary["quality"].items() if not passed]
        if failed and summary["outcome"] in ("improved", "no_regression"):
            inconclusive = set(failed) <= set(summary["quality_inconclusive"])
            summary["outcome"] = "inconclusive" if inconclusive else "quality_failed"
    return summary


def markdown(summary, manifest):
    lines = [
        "<!-- trl-pr-benchmark -->",
        f"### GPU benchmark: {summary['outcome'].replace('_', ' ')}",
        "",
        f"Base `{manifest['base_sha']}` → PR `{manifest['head_sha']}`.",
        f"Profile `{manifest['profile']}`; {summary['pairs']} paired seeds; request `{manifest['id']}`.",
        f"GPU: {summary['gpu']}.",
        "",
        "| Metric (lower is better) | Base mean | PR mean | Change | 95% interval | Result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, metric in summary["metrics"].items():
        ci = metric["interval_pct"]
        uncertainty = f"[{ci[0]:+.2f}%, {ci[1]:+.2f}%]" if ci else "Too few pairs"
        lines.append(
            f"| {name} | {metric['base_mean']:.4f} | {metric['head_mean']:.4f} | "
            f"{metric['change_pct']:+.2f}% | {uncertainty} | {metric['outcome']} |"
        )
    rollout = manifest["workload"].get("kind") == "vllm-rollout"
    if rollout:
        base, head = summary["policy_parity"]["base"], summary["policy_parity"]["head"]
        syncs, drift = summary["syncs_per_phase"], summary["frozen_weight_drift"]
        lines += ["", "#### Quality checks (reported separately; any failure blocks a passing verdict)", ""]
        lines += [f"- {'pass' if passed else '**FAIL**'}: {check}" for check, passed in summary["quality"].items()]
        lines += [
            "",
            f"Weight synchronizations per phase: base {syncs['base']:.2f}, PR {syncs['head']:.2f}. "
            f"Maximum frozen-weight drift: base {drift['base']:.3g}, PR {drift['head']:.3g}.",
            "",
            "| Local vs vLLM next-token distribution (first seed) | Base TV | PR TV | PR KL | PR JS |",
            "|---|---:|---:|---:|---:|",
        ]
        lines += [
            f"| {label} | {base[label]['total_variation']:.4g} | {item['total_variation']:.4g} | "
            f"{item['kl_local_vllm']:.4g} | {item['js_divergence']:.4g} |"
            for label, item in head.items()
        ]
    if "training" in summary:
        training = summary["training"]
        means, steps = training["means"], training["max_step_gap"]
        lines += ["", "#### Quality checks (reported separately; any failure blocks a passing verdict)", ""]
        uncertain = summary["quality_inconclusive"]
        lines += [
            f"- {'pass' if passed else 'inconclusive' if check in uncertain else '**FAIL**'}: {check}"
            for check, passed in summary["quality"].items()
        ]
        lines += [
            "",
            "| Mean per run | Base | PR |",
            "|---|---:|---:|",
            f"| Steady generation + scoring seconds | {means['generation_seconds']['base']:.3f} | "
            f"{means['generation_seconds']['head']:.3f} |",
            f"| Steady backward + optimizer seconds | {means['update_seconds']['base']:.3f} | "
            f"{means['update_seconds']['head']:.3f} |",
            f"| Peak torch-allocated memory (GB) | {means['peak_memory_bytes']['base'] / 1e9:.3f} | "
            f"{means['peak_memory_bytes']['head'] / 1e9:.3f} |",
            f"| Peak device memory during construction (GB) | {means['init_peak_device_bytes']['base'] / 1e9:.3f} | "
            f"{means['init_peak_device_bytes']['head'] / 1e9:.3f} |",
            f"| Peak device memory during training (GB) | {means['train_peak_device_bytes']['base'] / 1e9:.3f} | "
            f"{means['train_peak_device_bytes']['head'] / 1e9:.3f} |",
            f"| Held-out greedy reward | {means['eval_reward']['base']:.4f} | {means['eval_reward']['head']:.4f} |",
            f"| Mean vLLM/trainer logprob gap | {training['logprob_gap']['base']:.3g} | "
            f"{training['logprob_gap']['head']:.3g} |",
            "",
            f"Largest per-step gap to base: loss {steps['loss']:.3g}, reward {steps['reward']:.3g}, "
            f"grad norm {steps['grad_norm']:.3g}. Training reward change 95% interval: "
            f"[{training['train_reward_interval'][0]:+.3f}, {training['train_reward_interval'][1]:+.3f}]. "
            "Held-out reward change 95% interval: "
            f"[{training['eval_reward_interval'][0]:+.3f}, {training['eval_reward_interval'][1]:+.3f}]. "
            "Logprob gap change 95% interval: "
            f"[{training['logprob_gap_interval_pct'][0]:+.2f}%, {training['logprob_gap_interval_pct'][1]:+.2f}%]. "
            "Final parameter-sum relative gap: "
            f"{training['parameter_sum_relative_gap']:.3g}.",
        ]
    lines += [
        "",
        "Margins: " + ", ".join(f"{key} +{value}%" for key, value in manifest["thresholds"].items()) + ".",
        (
            "Transfer bytes count tensor payloads passed to vLLM's loaders, not hardware bus traffic."
            if rollout
            else "Held-out reward uses the synthetic length reward; it is not a scored task benchmark."
            if "training" in summary
            else "Held-out SFT token loss is a quality proxy; this does not certify other trainers or downstream tasks."
        ),
        "Smoke runs cannot establish no regression. Missing/non-finite measurements are errors, never passes.",
    ]
    return "\n".join(lines) + "\n"
