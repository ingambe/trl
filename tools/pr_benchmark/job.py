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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from environment import validate_environment


ROOT = Path(__file__).resolve().parent


def run(environment):
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    manifest = json.loads((ROOT / "manifest.json").read_text())
    config = manifest["workload"]
    work = Path(tempfile.mkdtemp(prefix="trl-benchmark-"))
    validate_environment(environment, manifest)
    tokenizer = AutoTokenizer.from_pretrained(environment / "model", trust_remote_code=False, local_files_only=True)
    count = config["train_examples"] + config["eval_examples"]
    dataset = load_from_disk(str(environment / "dataset")).shuffle(seed=917, keep_in_memory=True).select(range(count))
    encoded = [
        tokenizer.apply_chat_template(
            row["messages"],
            tokenize=True,
            return_dict=False,
            truncation=True,
            max_length=config["max_length"],
        )
        for row in dataset
    ]
    if any(len(ids) < 2 for ids in encoded):
        raise ValueError("The fixed dataset contains an empty training/evaluation example")
    data = work / "data.json"
    data.write_text(
        json.dumps({"train": encoded[: config["train_examples"]], "eval": encoded[config["train_examples"] :]})
    )
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
    rollout = config.get("kind") == "vllm-rollout"
    # One data-parallel process per GPU, each with its own colocated vLLM engine
    launcher = [sys.executable]
    if config.get("gpus", 1) > 1:
        launcher += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={config['gpus']}"]
    script = {"vllm-rollout": "rollout_workload.py", "grpo-train": "grpo_workload.py"}.get(
        config.get("kind"), "workload.py"
    )

    def run_child(side, stem, *extra):
        output = work / f"{stem}.json"
        with (ROOT / f"{stem}.log").open("w") as log:
            subprocess.run(
                [
                    *launcher,
                    str(ROOT / script),
                    "--model-path",
                    str(environment / "model"),
                    "--checkout",
                    str(work / side),
                    "--manifest",
                    str(ROOT / "manifest.json"),
                    "--data",
                    str(data),
                    "--side",
                    side,
                    "--output",
                    str(output),
                    *extra,
                ],
                cwd=work,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return json.loads(output.read_text())

    records = []
    started = time.perf_counter()
    if rollout:
        # One process per side runs every seed, so vLLM starts once per side instead of once per seed
        for side in ("base", "head"):
            records += run_child(side, side, "--seeds", *map(str, config["seeds"]))
            print(f"Completed {side}", flush=True)  # noqa: T201
    else:
        for index, seed in enumerate(config["seeds"]):
            # Alternate order to reduce drift caused by temperature or competing workloads.
            for side in ("base", "head") if index % 2 == 0 else ("head", "base"):
                before = time.perf_counter()
                record = run_child(side, f"{side}-{seed}", "--seed", str(seed))
                record["workload_seconds"] = time.perf_counter() - before
                records.append(record)
                print(f"Completed {side}, seed {seed}", flush=True)  # noqa: T201
    result = {
        "manifest_id": manifest["id"],
        "records": records,
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "comparison_seconds": time.perf_counter() - started,
    }
    (ROOT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if rollout:
        # Profiled runs come after timing, which is saved first so a failed capture cannot lose it
        result["profiles"], result["policy_parity"] = {}, {}
        seed = config["seeds"][0]
        for side in ("base", "head"):
            directory = ROOT / f"{side}-profile"
            (record,) = run_child(
                side, f"{side}-{seed}-profile", "--seeds", str(seed), "--profile-dir", str(directory)
            )
            result["policy_parity"][side] = record["policy_parity"]
            archive = Path(shutil.make_archive(str(directory), "zip", directory))
            result["profiles"][side] = {
                "bytes": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
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
