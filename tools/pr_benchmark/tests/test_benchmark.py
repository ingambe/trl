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

"""Controller tests run on CPU without importing TRL or provisioning cloud resources."""

import copy
import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
controller = importlib.import_module("controller")
comparison = importlib.import_module("compare")
hyperai = importlib.import_module("hyperai")


@pytest.fixture
def measurements():
    manifest = {
        "id": "test-request",
        "repo": "owner/repo",
        "pr": 1,
        "profile": "sft-3090",
        "base_sha": "a" * 40,
        "head_sha": "b" * 40,
        "workload": json.loads((controller.ROOT / "profiles.json").read_text())["sft-3090"],
        "thresholds": {"train_seconds": 5.0, "steady_seconds": 5.0, "eval_loss": 1.0},
        "harness": {
            name: hashlib.sha256((controller.ROOT / name).read_bytes()).hexdigest()
            for name in (*hyperai.BUNDLE_FILES[:-1], "compare.py")
        },
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


@pytest.fixture
def execution(tmp_path, measurements):
    _, result = measurements
    provider, github = Mock(spec=hyperai.HyperAI), Mock(spec=controller.GitHub)
    provider.submit.return_value = {"id": "job", "url": "https://example.com/job"}
    provider.status.return_value = "SUCCEEDED"
    provider.result.return_value = result
    github.current.return_value = True
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=True)
    return args, github, provider, {"runs": []}, tmp_path / "state.json"


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


def test_env_is_private_local_and_not_shell_code(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "REPO_ROOT", tmp_path / "repo")
    path = tmp_path / "env"
    path.write_text("export OPENBAYES_TOKEN='literal$(do-not-execute)'\nHYPERAI_RESOURCE=rtx-3090\n")
    path.chmod(0o600)
    assert controller.load_env(path)["OPENBAYES_TOKEN"] == "literal$(do-not-execute)"
    path.chmod(0o644)
    with pytest.raises(ValueError, match="chmod 600"):
        controller.load_env(path)
    (tmp_path / "repo").mkdir()
    path.rename(tmp_path / "repo/env")
    with pytest.raises(ValueError, match="outside"):
        controller.load_env(tmp_path / "repo/env")


def test_budget_limit_prevents_submission(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    args.daily_compute_minutes = 1
    controller.reserve(state, manifest, 1, 1)
    with pytest.raises(controller.BudgetExceeded):
        controller.execute(args, github, provider, manifest, state, state_file)
    provider.submit.assert_not_called()
    assert len(state["runs"]) == 1


@pytest.mark.parametrize("reason", ["stale_pr", "deadline", "interrupt"])
def test_interrupted_run_cancels_and_confirms_stop(measurements, execution, monkeypatch, reason):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    if reason == "deadline":
        monkeypatch.setattr(controller.time, "time", Mock(side_effect=[0, 61]))
        expected = TimeoutError
    elif reason == "interrupt":
        github.current.side_effect = KeyboardInterrupt
        expected = KeyboardInterrupt
    else:
        github.current.return_value = False
        expected = InterruptedError
    # Cancellation is asynchronous: the first poll after the stop request still reports RUNNING.
    provider.status.side_effect = ["RUNNING", "RUNNING", "CANCELLED"]

    def cancel(job_id):
        assert json.loads(state_file.read_text())["runs"][0]["status"] == "cancel_pending"

    provider.cancel.side_effect = cancel
    with pytest.raises(expected):
        controller.execute(args, github, provider, manifest, state, state_file)
    provider.cancel.assert_called_once_with("job")
    assert provider.status.call_count == 3
    assert json.loads(state_file.read_text())["runs"][0]["status"] == "error"
    provider.result.assert_not_called()
    github.publish.assert_not_called()
    assert github.status.call_args.args[1] == "error"


def test_unconfirmed_stop_keeps_cleanup_pending(measurements, execution, monkeypatch):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    github.current.return_value = False
    provider.status.return_value = "RUNNING"
    monkeypatch.setattr(controller.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="not confirmed"):
        controller.execute(args, github, provider, manifest, state, state_file)
    provider.cancel.assert_called_once_with("job")
    assert json.loads(state_file.read_text())["runs"][0]["status"] == "cancel_pending"
    github.publish.assert_not_called()


def test_failed_job_never_published_green(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    provider.status.return_value = "FAILED"
    with pytest.raises(RuntimeError, match="FAILED"):
        controller.execute(args, github, provider, manifest, state, state_file)
    assert state["runs"][0]["status"] == "error"
    provider.cancel.assert_not_called()
    github.publish.assert_not_called()
    assert github.status.call_args.args[1] == "error"


def test_submission_ambiguity_remains_durable(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    provider.submit.side_effect = TimeoutError("lost response after createJob")
    with pytest.raises(TimeoutError):
        controller.execute(args, github, provider, manifest, state, state_file)
    persisted = json.loads(state_file.read_text())
    assert persisted["runs"][0]["status"] == "submitting"


def test_current_base_tip_is_used_and_secrets_not_sent_to_github(monkeypatch):
    run = Mock(return_value=SimpleNamespace(returncode=0, stdout='{"login":"owner"}'))
    monkeypatch.setattr(controller.subprocess, "run", run)
    github = controller.GitHub("owner/repo", {"PATH": "/usr/bin", "GH_TOKEN": "github", "OPENBAYES_TOKEN": "private"})
    assert "OPENBAYES_TOKEN" not in run.call_args.kwargs["env"]
    github.api = Mock(return_value={"sha": "c" * 40})
    pull = {"base": {"ref": "release/test", "sha": "a" * 40}}
    assert github.base_sha(pull) == "c" * 40
    github.api.assert_called_once_with("repos/owner/repo/commits/release%2Ftest")


def test_hyperai_token_stays_out_of_job(monkeypatch, tmp_path):
    calls = []

    def query(self, text, variables=None):
        calls.append((text, variables))
        if "me {" in text:
            return {"me": {"username": "owner"}}
        if "createSourceCodePolicy" in text:
            return {
                "createSourceCodePolicy": {
                    "id": "code",
                    "endpoint": "https://storage.example.com",
                    "accessKey": "upload-key",
                    "secretKey": "upload-secret",
                    "path": "/bucket/prefix",
                }
            }
        if "createJob" in text:
            return {"createJob": {"id": "job", "links": []}}
        raise AssertionError(text)

    monkeypatch.setattr(hyperai.HyperAI, "query", query)
    import boto3

    storage = Mock()
    monkeypatch.setattr(boto3, "client", Mock(return_value=storage))
    provider = hyperai.HyperAI(
        {
            "OPENBAYES_TOKEN": "local-secret",
            "HYPERAI_PROJECT_ID": "project",
            "HYPERAI_RUNTIME": "pytorch-2.11.0-2404",
            "HYPERAI_ENVIRONMENT_JOB": "prepared-job",
        }
    )
    (tmp_path / ".env").write_text("OPENBAYES_TOKEN=local-secret")
    provider.submit(tmp_path, 60)
    assert "local-secret" not in json.dumps(calls)
    assert {Path(c.args[0]).name for c in storage.upload_file.call_args_list} == set(hyperai.BUNDLE_FILES)
    job = calls[-1][1]["input"]
    assert job["parameters"] == []
    assert job["resource"] == "rtx-3090"
    assert job["dataBindings"] == [
        {"name": "owner/jobs/prepared-job/output", "path": "/input0", "bindingAuth": "READ_ONLY"}
    ]
    assert "60s" in job["newTask"]["command"]


def test_forbidden_auth_never_creates_compute_or_echoes_credentials(monkeypatch):
    session = Mock()
    session.post.return_value = SimpleNamespace(
        status_code=200,
        json=lambda: {"errors": [{"message": "private-token", "extensions": {"code": "FORBIDDEN"}}]},
    )
    monkeypatch.setattr(hyperai.requests, "Session", lambda: session)
    with pytest.raises(RuntimeError, match="credential scope") as error:
        hyperai.HyperAI({"OPENBAYES_TOKEN": "private-token"})
    assert "private-token" not in str(error.value)
    session.post.assert_called_once()
    assert "createJob" not in session.post.call_args.kwargs["json"]["query"]


@pytest.mark.parametrize("calibration", [False, True])
def test_resume_preserves_original_publication_choice(measurements, execution, calibration):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    if calibration:
        manifest["pr"] = None
    args.publish = calibration
    controller.execute(args, github, provider, manifest, state, state_file)
    record = state["runs"][0]
    args.publish = True
    controller.execute(args, github, provider, manifest, state, state_file, record)
    github.publish.assert_not_called()
    github.status.assert_not_called()


def test_invalid_provider_config_does_not_reserve_budget_or_submit(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    provider.validate.side_effect = ValueError("Unsupported image")
    with pytest.raises(ValueError, match="Unsupported image"):
        controller.execute(args, github, provider, manifest, state, state_file)
    assert state["runs"] == []
    provider.submit.assert_not_called()


def test_changed_comparator_cannot_judge_a_recorded_run(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    manifest["harness"]["compare.py"] = "old-comparator"
    with pytest.raises(ValueError, match="Comparator changed"):
        controller.execute(args, github, provider, manifest, state, state_file)
    provider.submit.assert_not_called()
    record = {"status": "running", "job": {"id": "existing-job"}}
    state["runs"].append(record)
    provider.status.side_effect = ["RUNNING", "CANCELLED"]
    with pytest.raises(ValueError, match="Comparator changed"):
        controller.execute(args, github, provider, manifest, state, state_file, record)
    provider.cancel.assert_called_once_with("existing-job")
    assert record["status"] == "error"


def test_prepare_records_ready_artifact_without_publishing(measurements, execution):
    manifest, _ = measurements
    args, github, provider, state, state_file = execution
    manifest.update(prepare=True, pr=None)
    manifest["harness"]["compare.py"] = "comparator-not-used-during-preparation"
    provider.result.return_value = {
        "manifest_id": manifest["id"],
        "kind": "prepared_environment",
        "spec": controller.environment_spec(manifest),
    }
    assert controller.execute(args, github, provider, manifest, state, state_file) == "prepared"
    assert state["runs"][0]["outcome"] == "prepared"
    github.publish.assert_not_called()


def test_environment_identity_rejects_changed_dependencies_or_python(tmp_path, measurements):
    environment = importlib.import_module("environment")
    manifest, _ = measurements
    marker = {"spec": environment.environment_spec(manifest), "python": environment.platform.python_version()}
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(marker))
    assert environment.validate_environment(tmp_path, manifest) == marker
    marker["spec"]["requirements_sha256"] = "different-dependencies"
    path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="does not match"):
        environment.validate_environment(tmp_path, manifest)
    marker["spec"] = environment.environment_spec(manifest)
    marker["python"] = "different-python"
    path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="same Python"):
        environment.validate_environment(tmp_path, manifest)
