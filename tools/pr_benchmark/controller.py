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

"""Local PR watcher: credentials and GitHub writes never enter the GPU job bundle."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path
from urllib.parse import quote

from compare import compare, markdown
from environment import environment_spec
from hyperai import BUNDLE_FILES, TERMINAL, HyperAI


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
COMPARATOR_SHA256 = hashlib.sha256((ROOT / "compare.py").read_bytes()).hexdigest()
ENV_KEYS = {
    "OPENBAYES_TOKEN",
    "OPENBAYES_ORG",
    "HYPERAI_ENDPOINT",
    "HYPERAI_RESOURCE",
    "HYPERAI_RUNTIME",
    "HYPERAI_PROJECT_ID",
    "HYPERAI_ENVIRONMENT_JOB",
    "GH_TOKEN",
}


class BudgetExceeded(ValueError):
    pass


def load_env(path):
    env = dict(os.environ)
    if path:
        path = path.expanduser().resolve()
        if path.is_relative_to(REPO_ROOT):
            raise ValueError("Keep the credential file outside the repository")
        if path.stat().st_mode & 0o077:
            raise ValueError("Credential file must be private: chmod 600 <env-file>")
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts and parts[0] == "export":
                parts = parts[1:]
            if not parts:
                continue
            if len(parts) != 1 or "=" not in parts[0]:
                raise ValueError("Expected KEY=value in credential file; shell expansion is not supported")
            key, value = parts[0].split("=", 1)
            if key not in ENV_KEYS:
                raise ValueError(f"Unsupported credential/config key: {key}")
            env[key] = value
    return env


class GitHub:
    def __init__(self, repo, env):
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
            raise ValueError("Use a GitHub repository in owner/name form")
        self.repo = repo
        self.env = {
            key: env[key]
            for key in (
                "PATH",
                "HOME",
                "GH_CONFIG_DIR",
                "XDG_CONFIG_HOME",
                "GH_TOKEN",
                "GITHUB_TOKEN",
                "SSL_CERT_FILE",
            )
            if key in env
        }
        self.user = self.api("user")["login"]

    def api(self, endpoint, data=None):
        command = ["gh", "api", "--hostname", "github.com", endpoint]
        if data is not None:
            command += ["--method", "POST", "--input", "-"]
        result = subprocess.run(
            command,
            input=json.dumps(data) if data is not None else None,
            text=True,
            capture_output=True,
            env=self.env,
            timeout=60,
        )
        if result.returncode:
            raise RuntimeError(f"GitHub API failed for {endpoint}; check gh authentication and repository permissions")
        return json.loads(result.stdout)

    def pages(self, endpoint):
        separator = "&" if "?" in endpoint else "?"
        for page in range(1, 101):
            batch = self.api(f"{endpoint}{separator}per_page=100&page={page}")
            yield from batch
            if len(batch) < 100:
                return
        raise RuntimeError("GitHub pagination limit exceeded")

    def pull(self, number):
        return self.api(f"repos/{self.repo}/pulls/{number}")

    def base_sha(self, pull):
        ref = quote(pull["base"]["ref"], safe="")
        return self.api(f"repos/{self.repo}/commits/{ref}")["sha"]

    def current(self, manifest):
        if manifest["pr"] is None:
            return True  # Calibration and pre-PR comparisons pin immutable commits.
        pull = self.pull(manifest["pr"])
        return (
            pull["state"] == "open"
            and not pull["draft"]
            and pull["head"]["sha"] == manifest["head_sha"]
            and self.base_sha(pull) == manifest["base_sha"]
        )

    def status(self, manifest, state, description, url=""):
        payload = {"state": state, "context": f"gpu-benchmark/{manifest['profile']}", "description": description[:140]}
        if url:
            payload["target_url"] = url
        self.api(f"repos/{self.repo}/statuses/{manifest['head_sha']}", payload)

    def publish(self, manifest, summary, report, url):
        outcome = summary["outcome"]
        state = {"improved": "success", "no_regression": "success", "regression": "failure", "inconclusive": "error"}
        self.status(manifest, state[outcome], f"{outcome}; base {manifest['base_sha'][:12]}", url)
        # One immutable comment per tested pair retains the comparison history.
        self.api(f"repos/{self.repo}/issues/{manifest['pr']}/comments", {"body": report})


def manifest_for(github, pull, profile, environment_job=None, prepare=False):
    config = json.loads((ROOT / "profiles.json").read_text())[profile]
    manifest = {
        "schema": 1,
        "repo": github.repo,
        "pr": pull["number"],
        "base_ref": pull["base"]["ref"],
        "base_sha": github.base_sha(pull),
        "head_sha": pull["head"]["sha"],
        "profile": profile,
        "prepare": prepare,
        "environment_job": None if prepare else environment_job,
        "workload": config,
        "thresholds": (
            {"rollout_seconds": 5.0, "weight_transfer_bytes": 0.0}
            if config.get("kind") == "vllm-rollout"
            else {"train_seconds": 5.0, "steady_seconds": 5.0, "eval_loss": 1.0}
        ),
        "harness": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (*BUNDLE_FILES[:-1], "compare.py")
        },
    }
    if config.get("kind") == "vllm-rollout":
        manifest["harness"]["requirements-gpu.txt"] = hashlib.sha256(
            (ROOT / "requirements-vllm.txt").read_bytes()
        ).hexdigest()
    for key in ("base_sha", "head_sha"):
        if not re.fullmatch("[a-f0-9]{40}", manifest[key]):
            raise ValueError("GitHub returned an invalid commit SHA")
    manifest["id"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    return manifest


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def reserve(state, manifest, minutes, budget):
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    used = sum(run["reserved_minutes"] for run in state["runs"] if run["day"] == today)
    if used + minutes > budget:
        raise BudgetExceeded(f"Daily compute reservation limit reached ({used}/{budget} minutes)")
    record = {
        "run_id": uuid.uuid4().hex,
        "manifest_id": manifest["id"],
        "day": today,
        "reserved_minutes": minutes,
        "started": time.time(),
        "status": "submitting",
    }
    state["runs"].append(record)
    return record


def wait_for_job(provider, github, manifest, record, deadline, poll=30):
    while True:
        if time.time() >= deadline:
            raise TimeoutError("Compute job exceeded its allocation/queue deadline")
        if not github.current(manifest):
            raise InterruptedError("PR closed, became draft, or its base/head changed")
        status = provider.status(record["job"]["id"])
        if status in TERMINAL:
            if status != "SUCCEEDED":
                raise RuntimeError(f"Compute job ended with {status}")
            return
        time.sleep(min(poll, max(0, deadline - time.time())))


def stop_and_confirm(provider, job_id):
    if provider.status(job_id) in TERMINAL:
        return
    provider.cancel(job_id)
    for _ in range(12):
        if provider.status(job_id) in TERMINAL:
            return
        time.sleep(10)
    raise RuntimeError("Compute stop was requested but not confirmed; check the Hyper.ai console")


def execute(args, github, provider, manifest, state, state_file, record=None):
    if not manifest.get("prepare") and manifest["harness"]["compare.py"] != COMPARATOR_SHA256:
        if record is not None:
            record["status"] = "cancel_pending"
            save(state_file, state)
            stop_and_confirm(provider, record["job"]["id"])
            record["status"] = "error"
            save(state_file, state)
        raise ValueError("Comparator changed; restart the controller and create a new comparison")
    if record is None:
        provider.validate(manifest)
        record = reserve(state, manifest, args.timeout_minutes, args.daily_compute_minutes)
        record["publish"] = args.publish and manifest["pr"] is not None
        directory = args.state_dir / "runs" / record["run_id"]
        directory.mkdir(parents=True)
        bundle = directory / "bundle"
        bundle.mkdir()
        for name in BUNDLE_FILES[:-1]:
            source = (
                "requirements-vllm.txt"
                if (name == "requirements-gpu.txt" and manifest["workload"].get("kind") == "vllm-rollout")
                else name
            )
            shutil.copyfile(ROOT / source, bundle / name)
            if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != manifest["harness"][name]:
                raise ValueError("Harness changed while preparing the bundle; restart the controller")
        save(bundle / "manifest.json", manifest)
        # Persist before submitting. An ambiguous submission is never retried automatically.
        save(state_file, state)
        record["job"] = provider.submit(bundle, args.timeout_minutes * 60)
        record["status"] = "running"
        save(state_file, state)
    directory = args.state_dir / "runs" / record["run_id"]
    job = record["job"]
    publish = record.get("publish", False) and manifest["pr"] is not None
    print(f"Compute job {job['id']}: {job['url']}", flush=True)  # noqa: T201
    try:
        if publish:
            github.status(manifest, "pending", f"Comparing against {manifest['base_sha'][:12]}", job["url"])
        wait_for_job(provider, github, manifest, record, record["started"] + record["reserved_minutes"] * 60)
        result = provider.result(job["id"])
        save(directory / "result.json", result)
        if manifest.get("prepare"):
            if (
                result["manifest_id"] != manifest["id"]
                or result["kind"] != "prepared_environment"
                or result["spec"] != environment_spec(manifest)
            ):
                raise ValueError("Preparation result does not match the requested environment")
            record["status"] = "completed"
            record["outcome"] = "prepared"
            print(f"Environment ready. Set HYPERAI_ENVIRONMENT_JOB={job['id']} in your local env file.", flush=True)  # noqa: T201
            return "prepared"
        summary = compare(result, manifest)
        report = markdown(summary, manifest)
        save(directory / "summary.json", summary)
        (directory / "report.md").write_text(report)
        # Recheck after download/analysis so an obsolete comparison cannot publish a green status.
        if not github.current(manifest):
            raise InterruptedError("Comparison became stale before publication")
        if publish:
            github.publish(manifest, summary, report, job["url"])
        record["status"] = "completed"
        record["outcome"] = summary["outcome"]
        print(report, flush=True)  # noqa: T201
        return summary["outcome"]
    except BaseException:
        record["status"] = "cancel_pending"
        save(state_file, state)
        stop_and_confirm(provider, job["id"])
        record["status"] = "error"
        save(state_file, state)
        if publish:
            github.status(manifest, "error", "Comparison failed, was interrupted, or became stale", job["url"])
        raise
    finally:
        save(state_file, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["doctor", "prepare", "calibrate", "compare", "run", "watch"])
    parser.add_argument("--repo", help="GitHub owner/repo containing the PR")
    parser.add_argument("--pr", type=int)
    parser.add_argument(
        "--ref", default="main", help="Commit/ref for calibration or the candidate in a pre-PR comparison"
    )
    parser.add_argument("--base-ref", default="main", help="Base commit/ref for a pre-PR comparison")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--profile", choices=["sft-3090", "smoke", "vllm-rollout"], default="sft-3090")
    parser.add_argument(
        "--author", action="append", help="Allowed PR author; default is your authenticated GitHub user"
    )
    parser.add_argument("--publish", action="store_true", help="Publish commit statuses and PR reports")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve commits/config; do not provision GPUs or publish"
    )
    parser.add_argument("--rerun", action="store_true", help="Repeat a previously completed manual comparison")
    parser.add_argument("--timeout-minutes", type=int, default=60)
    parser.add_argument("--daily-compute-minutes", type=int, default=120)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--state-dir", type=Path, default=Path.home() / ".local/state/trl-pr-benchmark")
    args = parser.parse_args()
    if args.timeout_minutes <= 0 or args.daily_compute_minutes <= 0 or args.poll_seconds < 10:
        parser.error("Timeout/budget must be positive; polling must be at least 10 seconds")
    if args.command == "run" and not args.pr:
        parser.error("run requires --pr")
    if args.command == "watch" and args.rerun:
        parser.error("--rerun is only supported for manual runs")
    if args.command in {"prepare", "calibrate", "compare"} and args.publish:
        parser.error("Preparation, calibration, and pre-PR comparisons do not publish PR statuses")
    env = load_env(args.env_file)
    if args.command == "doctor":
        print(json.dumps(HyperAI(env).inventory(), indent=2))  # noqa: T201
        return 0
    if not args.repo:
        parser.error("--repo is required")
    github = GitHub(args.repo, env)
    authors = args.author or [github.user]
    provider = None if args.dry_run else HyperAI(env, prepare=args.command == "prepare")
    args.state_dir = args.state_dir.expanduser().resolve()
    if args.state_dir.is_relative_to(REPO_ROOT):
        parser.error("Keep controller state outside the repository")
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    state_file = args.state_dir / "state.json"
    with (args.state_dir / "controller.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another controller is already using this state directory") from error
        state = json.loads(state_file.read_text()) if state_file.exists() else {"runs": []}
        if not args.dry_run:
            for record in state["runs"]:
                if record["status"] == "submitting":
                    raise RuntimeError(
                        f"Submission {record['run_id']} is uncertain. Inspect the provider console before resolving state."
                    )
                if record["status"] == "cancel_pending":
                    stop_and_confirm(provider, record["job"]["id"])
                    record["status"] = "error"
                    save(state_file, state)
                if record["status"] == "running":
                    manifest = json.loads(
                        (args.state_dir / "runs" / record["run_id"] / "bundle/manifest.json").read_text()
                    )
                    if manifest["repo"] != args.repo:
                        raise ValueError("Resume this state directory with its original --repo")
                    execute(args, github, provider, manifest, state, state_file, record)
        while True:
            if args.command in {"prepare", "calibrate", "compare"}:
                sha = github.api(f"repos/{args.repo}/commits/{quote(args.ref, safe='')}")["sha"]
                pulls = [
                    {
                        "number": None,
                        "state": "open",
                        "draft": False,
                        "user": {"login": github.user},
                        "base": {"ref": args.base_ref if args.command == "compare" else sha},
                        "head": {"sha": sha},
                    }
                ]
            elif args.command == "run":
                pulls = [github.pull(args.pr)]
            else:
                pulls = github.pages(f"repos/{args.repo}/pulls?state=open")
            for pull in pulls:
                if pull["user"]["login"] not in authors or pull["draft"] or pull["state"] != "open":
                    if args.command != "watch":
                        raise ValueError("PR must be open, ready for review, and from an explicitly allowed author")
                    continue
                manifest = manifest_for(
                    github, pull, args.profile, env.get("HYPERAI_ENVIRONMENT_JOB"), args.command == "prepare"
                )
                if args.dry_run:
                    print(json.dumps(manifest, indent=2))  # noqa: T201
                    continue
                previous = [r for r in state["runs"] if r["manifest_id"] == manifest["id"]]
                if not args.rerun and previous:
                    if args.command != "watch":
                        record = previous[-1]
                        print(f"Already tested: {record['run_id']} ({record['status']}); use --rerun to repeat")  # noqa: T201
                        return 0 if record.get("outcome") in {"prepared", "improved", "no_regression"} else 1
                    continue
                try:
                    outcome = execute(args, github, provider, manifest, state, state_file)
                except BudgetExceeded:
                    if args.command != "watch":
                        raise
                    break
                except InterruptedError:
                    if args.command != "watch":
                        raise
                    continue
                except Exception:
                    # A completed/confirmed-stopped failed job must not prevent future PRs from being tested.
                    # Uncertain submissions or unconfirmed cleanup stop the watcher to avoid duplicate spending.
                    if args.command != "watch" or any(
                        r["status"] in {"submitting", "running", "cancel_pending"} for r in state["runs"]
                    ):
                        raise
                    print(f"Comparison failed for PR #{pull['number']}; see local state and the GPU job logs")  # noqa: T201
                    continue
                if args.command != "watch":
                    return 0 if outcome in {"prepared", "improved", "no_regression"} else 1
            if args.command != "watch" or args.dry_run:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(
            "Controller stopped; active job cleanup was attempted. Check state.json if interrupted twice."
        ) from None
