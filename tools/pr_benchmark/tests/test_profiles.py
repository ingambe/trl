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

"""Exercise actual HTA analysis and fail-closed artifact validation without a GPU."""

import gzip
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_profiles import analyze, extract_profile


def descriptor(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_reject_corrupt_archive(tmp_path):
    archive = tmp_path / "profile.zip"
    archive.write_bytes(b"before")
    expected = descriptor(archive)
    archive.write_bytes(b"after!")
    with pytest.raises(ValueError, match="checksum"):
        extract_profile(archive, tmp_path / "out", expected)


def test_reject_archive_traversal(tmp_path):
    archive = tmp_path / "profile.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside", b"bad")
    with pytest.raises(ValueError, match="path"):
        extract_profile(archive, tmp_path / "out", descriptor(archive))
    assert not (tmp_path / "outside").exists()


@pytest.fixture
def profile_run(tmp_path):
    manifest = {"id": "test", "base_sha": "a" * 40, "head_sha": "b" * 40, "workload": {"seeds": [42]}}
    (tmp_path / "bundle").mkdir()
    (tmp_path / "bundle/manifest.json").write_text(json.dumps(manifest))
    events = [
        {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#1", "pid": 1, "tid": 1, "ts": 1000, "dur": 200}
    ]
    for i in range(4):
        memory = i % 2 == 0
        events += [
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaMemcpyAsync" if memory else "cudaLaunchKernel",
                "pid": 1,
                "tid": 1,
                "ts": 1010 + i * 40,
                "dur": 3,
                "args": {"correlation": i + 1, "External id": i + 1},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy" if memory else "kernel",
                "name": "Memcpy HtoD (Pageable -> Device)" if memory else "matmul",
                "pid": 0,
                "tid": 7,
                "ts": 1020 + i * 40,
                "dur": 10,
                "args": {
                    "correlation": i + 1,
                    "External id": i + 1,
                    "stream": 7,
                    "device": 0,
                    "bytes": 1000,
                    "memory bandwidth (GB/s)": 0.1,
                },
            },
        ]
    result = {"profiles": {}}
    for side in ("base", "head"):
        source = tmp_path / side
        source.mkdir()
        with gzip.open(source / "trace.json.gz", "wt") as stream:
            json.dump({"schemaVersion": 1, "traceEvents": events}, stream)
        (source / "metadata.json").write_text(
            json.dumps(
                {
                    "manifest_id": "test",
                    "side": side,
                    "sha": manifest[f"{side}_sha"],
                    "seed": 42,
                    "warmup_steps": 1,
                    "active_steps": 1,
                    "workload": {},
                    "environment": {},
                }
            )
        )
        archive = Path(shutil.make_archive(str(tmp_path / f"{side}-profile"), "zip", source))
        result["profiles"][side] = descriptor(archive)
    (tmp_path / "result.json").write_text(json.dumps(result))
    return tmp_path


def test_hta_identical_traces_and_counter_export(profile_run):
    pytest.importorskip("hta")
    analyze(profile_run)
    summary = json.loads((profile_run / "profiles/summary.json").read_text())
    assert summary["sides"]["base"] == summary["sides"]["head"]
    assert summary["sides"]["base"]["memory-bandwidth"]
    with gzip.open(profile_run / "profiles/base/trace_with_counters.json.gz", "rt") as stream:
        events = json.load(stream)["traceEvents"]
    assert any(event["ph"] == "C" for event in events)
    assert "Head minus base" in (profile_run / "profiles/report.md").read_text()


@pytest.mark.parametrize("failure", ["cpu-only", "wrong-sha", "partial-window"])
def test_reject_incomplete_or_wrong_profile(profile_run, failure):
    pytest.importorskip("hta")
    source = profile_run / "base"
    if failure == "cpu-only":
        with gzip.open(source / "trace.json.gz", "wt") as stream:
            json.dump({"traceEvents": []}, stream)
    else:
        metadata = json.loads((source / "metadata.json").read_text())
        if failure == "wrong-sha":
            metadata["sha"] = "c" * 40
        else:
            metadata["active_steps"] = 2
        (source / "metadata.json").write_text(json.dumps(metadata))
    archive = Path(shutil.make_archive(str(profile_run / "base-profile"), "zip", source))
    result = json.loads((profile_run / "result.json").read_text())
    result["profiles"]["base"] = descriptor(archive)
    (profile_run / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="CUDA kernels|provenance|step window"):
        analyze(profile_run)
