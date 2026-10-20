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

"""Provider-independent GPU job entry point; the controller uploads only trusted harness files."""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from environment import validate_environment


ROOT = Path(__file__).resolve().parent


def run(environment):
    manifest = json.loads((ROOT / "manifest.json").read_text())
    config = manifest["workload"]
    work = Path(tempfile.mkdtemp(prefix="trl-benchmark-"))
    validate_environment(environment, manifest)
    for side in ("base", "head"):
        checkout = work / side
        subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "fetch",
                "--quiet",
                "--depth=1",
                f"https://github.com/{manifest['repo']}.git",
                manifest[f"{side}_sha"],
            ],
            cwd=checkout,
            check=True,
        )
        subprocess.run(["git", "checkout", "--quiet", "--detach", "FETCH_HEAD"], cwd=checkout, check=True)

    # Do not forward platform/controller tokens, user-site packages, or injected PYTHONPATH to PR code.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LD_LIBRARY_PATH", "SSL_CERT_FILE") if key in os.environ}
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in range(config.get("gpus", 1))),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Some provider hosts have no usable /dev/shm
            "NCCL_SHM_DISABLE": "1",
        }
    )
    # One data-parallel process per GPU, each with its own colocated vLLM engine
    launcher = [sys.executable]
    if config.get("gpus", 1) > 1:
        launcher += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={config['gpus']}"]

    # Heat the GPUs first, so the base run does not get the boost clocks of a cold card
    burn = "import time, torch\nx = torch.randn(8192, 8192, device='cuda')\nend = time.time() + 180\nwhile time.time() < end: x @ x"
    burners = [
        subprocess.Popen([sys.executable, "-c", burn], env={**env, "CUDA_VISIBLE_DEVICES": str(index)})
        for index in range(config.get("gpus", 1))
    ]
    if any(burner.wait() for burner in burners):
        raise RuntimeError("GPU warm-up failed")

    records = []
    started = time.perf_counter()
    for side in ("base", "head"):
        output = work / f"{side}.json"
        # Separate compile caches, so the head run does not reuse kernels built by the base run
        home = work / f"{side}-home"
        side_env = {**env, "HOME": str(home), "TORCHINDUCTOR_CACHE_DIR": str(home / "inductor")}
        with (ROOT / f"{side}.log").open("w") as log:
            subprocess.run(
                [
                    *launcher,
                    str(ROOT / "grpo_workload.py"),
                    "--model-path",
                    str(environment / "model"),
                    "--checkout",
                    str(work / side),
                    "--manifest",
                    str(ROOT / "manifest.json"),
                    "--side",
                    side,
                    "--output",
                    str(output),
                ],
                cwd=work,
                env=side_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        records.append(json.loads(output.read_text()))
        print(f"Completed {side}", flush=True)  # noqa: T201
    result = {
        "manifest_id": manifest["id"],
        "records": records,
        "comparison_seconds": time.perf_counter() - started,
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=Path("/input0"))
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    validate_environment(args.environment, manifest)
    if args.prepared:
        run(args.environment)
    else:
        # The provider mounts a completed preparation job read-only. Never install during a benchmark.
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                str(args.environment / "env/bin/python"),
                str(Path(__file__).resolve()),
                "--prepared",
                "--environment",
                str(args.environment),
            ],
            env=env,
            check=True,
        )
