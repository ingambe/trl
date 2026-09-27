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

"""Analyze downloaded before/after traces with HTA; rerunnable without GPU access."""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import zipfile
from pathlib import Path


def extract_profile(archive, directory, descriptor):
    if (
        archive.stat().st_size != descriptor["bytes"]
        or hashlib.sha256(archive.read_bytes()).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("Profiler artifact checksum/size mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if sum(info.file_size for info in bundle.infolist()) > 2_000_000_000:
            raise ValueError("Expanded profile exceeds 2 GB")
        # Workers produce only flat files; never trust archive paths from a remote job.
        if any(Path(info.filename).name != info.filename or info.is_dir() for info in bundle.infolist()):
            raise ValueError("Invalid profiler artifact path")
        bundle.extractall(directory)


def analyze(directory):
    import pandas as pd
    from hta.trace_analysis import TraceAnalysis

    directory = directory.resolve()
    result = json.loads((directory / "result.json").read_text())
    manifest = json.loads((directory / "bundle/manifest.json").read_text())
    summaries = {}
    captures = {}
    coverage = {}
    for side in ("base", "head"):
        destination = directory / "profiles" / side
        extract_profile(directory / f"{side}-profile.zip", destination, result["profiles"][side])
        metadata = json.loads((destination / "metadata.json").read_text())
        if (metadata["manifest_id"], metadata["sha"], metadata["side"], metadata["seed"]) != (
            manifest["id"],
            manifest[f"{side}_sha"],
            side,
            manifest["workload"]["seeds"][0],
        ):
            raise ValueError("Profiler provenance does not match the comparison")
        captures[side] = metadata
        with gzip.open(destination / "trace.json.gz", "rt") as stream:
            events = json.load(stream)["traceEvents"]
        if not any(event.get("cat") == "kernel" for event in events):
            raise ValueError(f"{side} profile has no CUDA kernels; check CUPTI availability")
        steps = [
            event
            for event in events
            if event.get("cat") == "user_annotation" and event.get("name", "").startswith("ProfilerStep#")
        ]
        if len(steps) != metadata["active_steps"]:
            raise ValueError(f"{side} capture has an incomplete profiler step window")
        copies = {}
        for event in events:
            if event.get("cat") == "gpu_memcpy":
                name = event["name"]
                copies[name] = copies.get(name, 0) + event["args"]["bytes"]
        coverage[side] = {
            "cuda_graph_launch_events": sum("GraphLaunch" in event.get("name", "") for event in events),
            "kernel_events": sum(event.get("cat") == "kernel" for event in events),
            "memcpy_bytes_by_kind": copies,
        }
        del events
        analysis = TraceAnalysis(trace_files={0: str(destination / "trace.json.gz")}, include_last_profiler_step=True)
        tables = {"temporal": analysis.get_temporal_breakdown(visualize=False)}
        tables["kernel-types"], tables["kernels"] = analysis.get_gpu_kernel_breakdown(
            visualize=False, duration_ratio=1.0, num_kernels=100000
        )
        tables["idle"], tables["idle-intervals"] = analysis.get_idle_time_breakdown(
            visualize=False, show_idle_interval_stats=True
        )
        tables["launches"] = analysis.get_cuda_kernel_launch_stats(visualize=False)[0]
        tables["memory-bandwidth"] = analysis.get_memory_bw_summary()
        tables["queue-length"] = analysis.get_queue_length_summary()
        for name, table in tables.items():
            if table is not None:
                table.to_csv(destination / f"hta-{name}.csv", index=False)
        analysis.generate_trace_with_counters()
        summaries[side] = {
            name: None if table is None else json.loads(table.to_json(orient="records"))
            for name, table in tables.items()
        }

    for key in ("warmup_steps", "active_steps", "workload", "environment"):
        if captures["base"][key] != captures["head"][key]:
            raise ValueError(f"Before/after profiler {key} differs")

    # Full tables stay machine-readable, including empty/unsupported memory-copy categories.
    summary = {
        "manifest_id": manifest["id"],
        "captures": captures,
        "coverage": coverage,
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
        "sides": summaries,
    }
    (directory / "profiles/summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    kernels = pd.DataFrame(summaries["base"]["kernels"]).merge(
        pd.DataFrame(summaries["head"]["kernels"]),
        on=["name", "kernel_type", "rank"],
        how="outer",
        suffixes=("_base", "_head"),
    )
    kernels["delta_sum_us"] = kernels["sum (us)_head"].fillna(0) - kernels["sum (us)_base"].fillna(0)
    kernels.sort_values("delta_sum_us").to_csv(directory / "profiles/kernel-deltas.csv", index=False)
    lines = [
        "# Before/after profiler diagnostics",
        "",
        "Separate instrumented runs; these durations are not benchmark speedup measurements.",
        "",
        "| HTA temporal metric | Base | Head | Head minus base |",
        "|---|---:|---:|---:|",
    ]
    before, after = summaries["base"]["temporal"][0], summaries["head"]["temporal"][0]
    for metric, value in before.items():
        if metric != "rank" and isinstance(value, (float, int)):
            lines.append(f"| {metric} | {value:.4f} | {after[metric]:.4f} | {after[metric] - value:+.4f} |")
    lines += [
        "",
        "[Full kernel before/after deltas](kernel-deltas.csv) · [Analysis metadata and all tables](summary.json)",
    ]
    if any(side["cuda_graph_launch_events"] for side in coverage.values()):
        lines += [
            "",
            "CUDA Graph replays are present. HTA 0.5 does not recognize graph launches in its launch/queue analysis; those tables and idle-cause attribution have incomplete coverage. Use the raw CUDA timeline and temporal/kernel totals for graph execution.",
        ]
    lines += ["", "Raw Chrome traces and HTA traces with queue-length/memcpy-bandwidth counters:", ""]
    for side in ("base", "head"):
        for path in sorted((directory / "profiles" / side).iterdir()):
            lines.append(f"- [{side}/{path.name}]({side}/{path.name})")
    lines += [
        "",
        "Open trace JSON/gzip files in Perfetto or chrome://tracing. CSV durations follow HTA's column units; memory bandwidth is GB/s.",
        "Memory-copy bandwidth excludes bandwidth inside compute kernels. Payload-byte counters in the benchmark remain a separate metric.",
    ]
    report = "\n".join(lines) + "\n"
    (directory / "profiles/report.md").write_text(report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Controller run directory containing downloaded profile archives")
    analyze(parser.parse_args().directory)
