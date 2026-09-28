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

"""Check regression verdicts using measurements; no cloud services or mocks."""

import copy
import importlib
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
comparison = importlib.import_module("compare")


@pytest.fixture
def measurements():
    manifest = {
        "id": "test-request",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "workload": {"seeds": [42, 43, 44, 45, 46], "steps": 120},
        "thresholds": {"train_seconds": 5.0, "steady_seconds": 5.0, "eval_loss": 1.0},
    }
    records = [
        {
            "side": side,
            "seed": seed,
            "sha": manifest[f"{side}_sha"],
            "steps": 120,
            "train_seconds": 100.0,
            "steady_seconds": 80.0,
            "eval_loss": 2.0,
            "train_loss": 2.5,
            "peak_memory_bytes": 1_000_000,
            "workload_seconds": 110.0,
            "environment": {"gpu": "RTX 3090", "packages": ["torch==2.11.0+cu128"]},
        }
        for seed in manifest["workload"]["seeds"]
        for side in ("base", "head")
    ]
    return manifest, {"manifest_id": manifest["id"], "records": records}


@pytest.mark.parametrize(
    "runtime,loss,outcome",
    [
        (1.0, 1.0, "no_regression"),
        (0.8, 1.0, "improved"),
        (1.0, 0.95, "improved"),
        (1.1, 1.0, "regression"),
        (0.8, 1.02, "regression"),
        (1.1, 0.9, "regression"),
    ],
)
def test_paired_decisions(measurements, runtime, loss, outcome):
    manifest, result = measurements
    for record in result["records"]:
        if record["side"] == "head":
            record["train_seconds"] *= runtime
            record["steady_seconds"] *= runtime
            record["eval_loss"] *= loss
    summary = comparison.compare(result, manifest)
    assert summary["outcome"] == outcome


def test_noise_cannot_pass_noninferiority(measurements):
    manifest, result = measurements
    for record, seconds in zip(result["records"][1::2], [80, 100, 130, 90, 130], strict=True):
        record["train_seconds"] = seconds
    assert comparison.compare(result, manifest)["outcome"] == "inconclusive"


def test_smoke_cannot_be_green(measurements):
    manifest, result = measurements
    manifest["workload"]["seeds"] = [42]
    result["records"] = result["records"][:2]
    assert comparison.compare(result, manifest)["outcome"] == "inconclusive"


@pytest.mark.parametrize("damage", ["nan", "missing", "duplicate", "commit", "steps", "environment", "identity"])
def test_invalid_results_fail_closed(measurements, damage):
    manifest, result = measurements
    if damage == "nan":
        result["records"][0]["eval_loss"] = float("nan")
    elif damage == "missing":
        result["records"].pop()
    elif damage == "duplicate":
        result["records"].append(copy.deepcopy(result["records"][0]))
    elif damage == "commit":
        result["records"][0]["sha"] = "c" * 40
    elif damage == "steps":
        result["records"][0]["steps"] -= 1
    elif damage == "environment":
        result["records"][0]["environment"]["gpu"] = "RTX 5090"
    else:
        result["manifest_id"] = "old-request"
    with pytest.raises(ValueError):
        comparison.compare(result, manifest)


@pytest.fixture
def rollout_measurements(measurements):
    manifest, result = measurements
    manifest["workload"].update(kind="vllm-rollout", steps=6)
    manifest["thresholds"] = {"rollout_seconds": 5.0, "weight_transfer_bytes": 0.0}
    for record in result["records"]:
        head = record["side"] == "head"
        record.update(
            steps=6,
            rollout_seconds=8.0 if head else 10.0,
            weight_transfer_bytes=100 if head else 400,
            sync_count=6 if head else 24,
            sleeping_after_phase=True,
            output_tokens=[[[1, 2]]],
            frozen_weight_max_abs_drift=0.0,
        )
    parity = [
        {"label": "updated_lora", "total_variation": 0.01, "kl_local_vllm": 0.001, "js_divergence": 0.001},
        {"label": "updated_lora_stale_reference", "total_variation": 0.3, "kl_local_vllm": 0.5, "js_divergence": 0.1},
    ]
    result["policy_parity"] = {"base": parity, "head": copy.deepcopy(parity)}
    return manifest, result


@pytest.mark.parametrize(
    "damage,outcome",
    [
        (None, "improved"),
        ("tokens", "quality_failed"),
        ("drift", "quality_failed"),
        ("stale_control", "quality_failed"),
        ("slower", "regression"),
        ("same_latency", "no_regression"),
    ],
)
def test_rollout_quality_is_reported_beside_latency(rollout_measurements, damage, outcome):
    manifest, result = rollout_measurements
    head = result["records"][1]
    if damage == "tokens":
        head["output_tokens"] = [[[1, 3]]]
    elif damage == "drift":
        head["frozen_weight_max_abs_drift"] = 0.001953125
    elif damage == "stale_control":
        result["policy_parity"]["head"][1]["total_variation"] = 0.01
    elif damage == "slower":
        for record in result["records"][1::2]:
            record["rollout_seconds"] = 12.0
    elif damage == "same_latency":
        # Fewer transferred bytes alone are not a rollout speedup.
        for record in result["records"]:
            record["rollout_seconds"] = 10.0
    summary = comparison.compare(result, manifest)
    assert summary["outcome"] == outcome
    report = comparison.markdown(summary, {**manifest, "profile": "vllm-rollout"})
    assert "rollout_seconds" in report
    assert ("**FAIL**" in report) == (outcome == "quality_failed")
