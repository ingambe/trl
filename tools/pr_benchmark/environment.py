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

"""Prepare a reusable environment and pinned model/data assets, without needing a GPU."""

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def environment_spec(manifest):
    config = manifest["workload"]
    return {
        "builder_sha256": manifest["harness"]["environment.py"],
        "requirements_sha256": manifest["harness"]["requirements-gpu.txt"],
        **{key: config[key] for key in ("model", "model_revision", "dataset", "dataset_revision")},
    }


def validate_environment(directory, manifest):
    marker = json.loads((directory / "environment.json").read_text())
    if marker["spec"] != environment_spec(manifest):
        raise ValueError("Prepared environment does not match dependencies/model/data; run prepare again")
    if marker["python"] != platform.python_version():
        raise ValueError("Prepared environment requires the same Python runtime; run prepare again")
    return marker


def prepare_assets(manifest):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    config = manifest["workload"]
    snapshot_download(
        config["model"],
        revision=config["model_revision"],
        token=False,
        local_dir=ROOT / "model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    load_dataset(
        config["dataset"],
        revision=config["dataset_revision"],
        split="train",
        token=False,
    ).save_to_disk(ROOT / "dataset")
    marker = {
        "kind": "prepared_environment",
        "manifest_id": manifest["id"],
        "spec": environment_spec(manifest),
        "python": platform.python_version(),
        "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions()),
    }
    # A marker is written only after installation and both asset downloads have succeeded.
    (ROOT / "environment.json").write_text(json.dumps(marker, indent=2))
    (ROOT / "result.json").write_text(json.dumps(marker, indent=2))


if __name__ == "__main__":
    manifest = json.loads((ROOT / "manifest.json").read_text())
    if "--assets" in sys.argv:
        prepare_assets(manifest)
    else:
        environment = ROOT / "env"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = str(environment / "bin/python")
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--index-strategy",
                "unsafe-best-match",
                "-r",
                str(ROOT / "requirements-gpu.txt"),
            ],
            check=True,
        )
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HF_HOME"] = tempfile.mkdtemp(prefix="trl-prepare-hf-")
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run([python, str(Path(__file__).resolve()), "--assets"], env=env, check=True)
