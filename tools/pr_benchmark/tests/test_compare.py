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

import importlib
import random
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
comparison = importlib.import_module("compare")


def run(side, steps=60, seconds=1.0, gap=0.03):
    noise = random.Random(side)
    return {
        "side": side,
        "sha": ("a" if side == "base" else "b") * 40,
        "steps": steps,
        "step_seconds": [seconds * noise.uniform(0.99, 1.01) for _ in range(steps)],
        "generation_seconds": [seconds / 2 * noise.uniform(0.99, 1.01) for _ in range(steps)],
        "initial_eval_reward": 0.2,
        "eval_reward": 0.5,
        "trajectory": [{comparison.LOGP: gap * noise.uniform(0.99, 1.01)} for _ in range(steps)],
        "peak_memory_bytes": 1,
        "init_peak_device_bytes": 1,
        "train_peak_device_bytes": 1,
        "environment": {"gpu": "RTX 3090"},
    }


MANIFEST = {
    "id": "test-request",
    "profile": "wordle",
    "base_sha": "a" * 40,
    "head_sha": "b" * 40,
    "workload": {"warmup_steps": 3, "speed_steps": 50, "eval_games": 32, "train_minutes": 10},
    "thresholds": {"step_seconds": 5.0, "generation_seconds": 5.0},
}


@pytest.mark.parametrize(
    "head,outcome",
    [
        ({}, "no_regression"),
        ({"seconds": 0.8, "steps": 80}, "improved"),
        ({"seconds": 1.1}, "regression"),
        ({"gap": 0.05}, "quality_failed"),
    ],
)
def test_single_run_decisions(head, outcome):
    result = {"manifest_id": "test-request", "records": [run("base"), run("head", **head)]}
    summary = comparison.compare(result, MANIFEST)
    assert summary["outcome"] == outcome
    assert "### GPU benchmark" in comparison.markdown(summary, MANIFEST)


@pytest.mark.parametrize("damage", ["nan", "missing", "commit", "steps", "environment", "identity"])
def test_invalid_results_fail_closed(damage):
    result = {"manifest_id": "test-request", "records": [run("base"), run("head")]}
    base = result["records"][0]
    if damage == "nan":
        base["eval_reward"] = float("nan")
    elif damage == "missing":
        result["records"].pop()
    elif damage == "commit":
        base["sha"] = "c" * 40
    elif damage == "steps":
        base["steps"] += 1
    elif damage == "environment":
        base["environment"]["gpu"] = "RTX 5090"
    else:
        result["manifest_id"] = "old-request"
    with pytest.raises(ValueError):
        comparison.compare(result, MANIFEST)
