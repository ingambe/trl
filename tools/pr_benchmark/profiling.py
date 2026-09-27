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

"""Bounded PyTorch captures, used only in separate diagnostic subprocesses."""

import gzip
import json
from contextlib import nullcontext
from pathlib import Path


def make_profiler(directory, warmup_steps, active_steps):
    import torch

    directory.mkdir(parents=True, exist_ok=True)

    def export(profiler):
        profiler.export_chrome_trace(str(directory / "trace.json.gz"))
        averages = profiler.key_averages(group_by_input_shape=True)
        for metric in ("self_cpu_time_total", "self_cuda_time_total", "self_cuda_memory_usage"):
            (directory / f"operators-{metric}.txt").write_text(averages.table(sort_by=metric, row_limit=-1))
        # Full Python call-tree capture makes multi-turn vLLM traces enormous. Allocation events
        # already contain timestamps, addresses, sizes, and device allocation/reservation totals.
        with gzip.open(directory / "trace.json.gz", "rt") as stream:
            events = json.load(stream)["traceEvents"]
        with gzip.open(directory / "memory-events.json.gz", "wt") as stream:
            json.dump([event for event in events if event.get("name") == "[memory]"], stream)

    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=max(0, warmup_steps - 1), warmup=1, active=active_steps, repeat=1),
        on_trace_ready=export,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    )


def span(name, enabled):
    if enabled:
        from torch.profiler import record_function

        return record_function("benchmark::" + name)
    return nullcontext()


def annotate(method):
    """Annotate the real method without changing its arguments or return value."""

    def wrapped(*args, **kwargs):
        with span(method.__name__, True):
            return method(*args, **kwargs)

    return wrapped


def save_profile_metadata(directory, record, manifest, warmup_steps, active_steps):
    metadata = {
        "manifest_id": manifest["id"],
        "side": record["side"],
        "sha": record["sha"],
        "seed": record["seed"],
        "environment": record["environment"],
        "workload": manifest["workload"],
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "scope": "complete rollout phases"
        if manifest["workload"].get("kind") == "vllm-rollout"
        else "optimizer steps",
        "timing_use": "diagnostic only; excluded from performance verdict",
        "activities": ["CPU", "CUDA"],
        "record_shapes": True,
        "with_stack": False,
        "profile_memory": True,
    }
    (Path(directory) / "metadata.json").write_text(json.dumps(metadata, indent=2))
