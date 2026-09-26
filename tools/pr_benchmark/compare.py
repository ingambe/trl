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
    metrics = ("train_seconds", "steady_seconds", "eval_loss")
    environments = []
    for (side, seed), record in indexed.items():
        if record["sha"] != manifest[f"{side}_sha"] or record["steps"] != config["steps"]:
            raise ValueError("Wrong commit or incomplete training")
        for metric in (*metrics, "train_loss", "peak_memory_bytes", "workload_seconds"):
            value = record[metric]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
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
        changes = [100 * (h / b - 1) for b, h in zip(base, head, strict=True)]
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
    lines += [
        "",
        "Margins: " + ", ".join(f"{key} +{value}%" for key, value in manifest["thresholds"].items()) + ".",
        "Held-out SFT token loss is a quality proxy; this does not certify other trainers or downstream tasks.",
        "Smoke runs cannot establish no regression. Missing/non-finite measurements are errors, never passes.",
    ]
    return "\n".join(lines) + "\n"
