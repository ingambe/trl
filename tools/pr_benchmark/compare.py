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

"""Compare one base run with one PR run without requiring the training dependencies locally."""

import math
import statistics


LOGP = "sampling/sampling_logp_difference/mean"


def interval(values):
    # Conservative two-sided 95% Student t critical values. Round df down between entries.
    critical = {4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 15: 2.132, 20: 2.086, 30: 2.042}
    n = len(values)
    mean = statistics.mean(values)
    t = critical[max(df for df in critical if df <= n - 1)]
    radius = t * statistics.stdev(values) / math.sqrt(n)
    return [mean - radius, mean + radius]


def verdict(ci, margin):
    # Lower is better; a verdict needs the whole interval on one side of the margin
    if ci[0] > margin:
        return "regression"
    if ci[1] > margin:
        return "inconclusive"
    if ci[1] < -margin:
        return "improved"
    return "no_regression"


def compare(result, manifest):
    if result["manifest_id"] != manifest["id"]:
        raise ValueError("Results belong to a different benchmark request")
    config = manifest["workload"]
    records = result["records"]
    if sorted(r["side"] for r in records) != ["base", "head"]:
        raise ValueError("Missing, duplicate, or unexpected measurements")
    runs = {r["side"]: r for r in records}
    for side, run in runs.items():
        if run["sha"] != manifest[f"{side}_sha"] or run["steps"] != len(run["step_seconds"]):
            raise ValueError("Wrong commit or incomplete measurements")
        values = [
            *run["step_seconds"],
            *run["generation_seconds"],
            run["initial_eval_reward"],
            run["eval_reward"],
            *(step[key] for step in run["trajectory"] for key in (LOGP,)),
        ]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError(f"Invalid measurement for {side}")
        if len(run["trajectory"]) != run["steps"]:
            raise ValueError(f"Missing measurements for {side}")
    if runs["base"]["environment"] != runs["head"]["environment"]:
        raise ValueError("GPU or dependency environment differs between runs")
    base, head = runs["base"], runs["head"]

    # Both runs keep the initial policy for these steps, so they are paired by index
    steps = range(config["warmup_steps"], config["warmup_steps"] + config["speed_steps"])
    if min(base["steps"], head["steps"]) < steps.stop:
        raise ValueError("Too few steady steps to compare")
    summary = {"outcome": "no_regression", "metrics": {}, "steps": len(steps), "gpu": base["environment"]["gpu"]}
    for metric in ("step_seconds", "generation_seconds"):
        values = {side: [run[metric][i] for i in steps] for side, run in runs.items()}
        changes = [100 * (h / b - 1) for b, h in zip(values["base"], values["head"], strict=True)]
        ci = interval(changes)
        summary["metrics"][metric] = {
            "base_mean": statistics.mean(values["base"]),
            "head_mean": statistics.mean(values["head"]),
            "change_pct": statistics.mean(changes),
            "interval_pct": ci,
            "margin_pct": manifest["thresholds"][metric],
            "outcome": verdict(ci, manifest["thresholds"][metric]),
        }
    outcomes = {m["outcome"] for m in summary["metrics"].values()}
    for outcome in ("regression", "inconclusive", "improved"):
        if outcome in outcomes:
            summary["outcome"] = outcome
            break

    summary["training"] = {
        "steps": {side: run["steps"] for side, run in runs.items()},
        **{key: {side: run[key] for side, run in runs.items()} for key in ("initial_eval_reward", "eval_reward")},
        "logprob_gap": {
            side: statistics.mean(run["trajectory"][i][LOGP] for i in steps) for side, run in runs.items()
        },
        **{
            key: {side: run[key] for side, run in runs.items()}
            for key in ("peak_memory_bytes", "init_peak_device_bytes", "train_peak_device_bytes")
        },
    }
    # The gap is noisy even between identical commits, so only a large increase fails. Runs diverge too much for
    # held-out reward to gate, so it is only reported.
    gap = summary["training"]["logprob_gap"]
    checks = {"vLLM/trainer logprob gap at most 1.5x base": "pass" if gap["head"] <= 1.5 * gap["base"] else "fail"}
    summary["quality"] = checks
    if summary["outcome"] in ("improved", "no_regression") and "fail" in checks.values():
        summary["outcome"] = "quality_failed"
    return summary


def markdown(summary, manifest):
    training = summary["training"]
    lines = [
        "<!-- trl-pr-benchmark -->",
        f"### GPU benchmark: {summary['outcome'].replace('_', ' ')}",
        "",
        f"Base `{manifest['base_sha']}` → PR `{manifest['head_sha']}`.",
        f"Profile `{manifest['profile']}`; one {manifest['workload']['train_minutes']}-minute run per side, "
        f"speed and logprob gap over {summary['steps']} steady steps of the frozen initial policy; request `{manifest['id']}`.",
        f"GPU: {summary['gpu']}.",
        "",
        "| Seconds per step (lower is better) | Base mean | PR mean | Change | 95% interval | Result |",
        "|---|---:|---:|---:|---|---|",
    ]
    for name, metric in summary["metrics"].items():
        ci = metric["interval_pct"]
        lines.append(
            f"| {name} | {metric['base_mean']:.3f} | {metric['head_mean']:.3f} | "
            f"{metric['change_pct']:+.2f}% | [{ci[0]:+.2f}%, {ci[1]:+.2f}%] | {metric['outcome']} |"
        )
    lines += ["", "#### Quality checks (any failure blocks a passing verdict)", ""]
    lines += [
        f"- {'**FAIL**' if result == 'fail' else result}: {check}" for check, result in summary["quality"].items()
    ]
    rows = [
        ("Steps in the time budget", "steps", "{}"),
        ("Held-out reward before training", "initial_eval_reward", "{:.4f}"),
        ("Held-out reward after training", "eval_reward", "{:.4f}"),
        ("Early vLLM/trainer logprob gap", "logprob_gap", "{:.3g}"),
    ]
    lines += ["", "| Per run | Base | PR |", "|---|---:|---:|"]
    lines += [
        f"| {label} | {fmt.format(training[key]['base'])} | {fmt.format(training[key]['head'])} |"
        for label, key, fmt in rows
    ]
    lines += [
        f"| {label} (GB) | {training[key]['base'] / 1e9:.3f} | {training[key]['head'] / 1e9:.3f} |"
        for label, key in (
            ("Peak torch-allocated memory", "peak_memory_bytes"),
            ("Peak device memory during construction", "init_peak_device_bytes"),
            ("Peak device memory during training", "train_peak_device_bytes"),
        )
    ]
    lines += [
        "",
        "Margins: " + ", ".join(f"{key} +{value}%" for key, value in manifest["thresholds"].items()) + ".",
        "Missing or non-finite measurements are errors, never passes.",
    ]
    return "\n".join(lines) + "\n"
