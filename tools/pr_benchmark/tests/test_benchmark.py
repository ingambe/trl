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
    assert manifest["head_sha"] in comparison.markdown(summary, manifest)


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


def test_budget_reserved_before_launch(measurements):
    manifest, _ = measurements
    state = {"runs": []}
    controller.reserve(state, manifest, 60, 60)
    with pytest.raises(controller.BudgetExceeded):
        controller.reserve(state, manifest, 60, 60)
    assert len(state["runs"]) == 1


def test_wait_cancels_on_new_head_and_checks_failed_status(measurements, monkeypatch):
    manifest, _ = measurements
    provider, github = Mock(), Mock()
    record = {"job": {"id": "job"}}
    monkeypatch.setattr(controller.time, "time", lambda: 10)
    github.current.return_value = False
    with pytest.raises(InterruptedError):
        controller.wait_for_job(provider, github, manifest, record, 100)
    provider.status.assert_not_called()
    github.current.return_value = True
    provider.status.return_value = "FAILED"
    with pytest.raises(RuntimeError, match="FAILED"):
        controller.wait_for_job(provider, github, manifest, record, 100)
    with pytest.raises(TimeoutError):
        controller.wait_for_job(provider, github, manifest, record, 5)


def test_failed_job_stopped_and_never_published_green(tmp_path, measurements, monkeypatch):
    manifest, _ = measurements
    provider, github = Mock(), Mock()
    provider.submit.return_value = {"id": "job", "url": "https://example.com/job"}
    provider.status.side_effect = ["FAILED", "FAILED"]
    github.current.return_value = True
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=True)
    with pytest.raises(RuntimeError, match="FAILED"):
        controller.execute(args, github, provider, manifest, state, tmp_path / "state.json")
    assert state["runs"][0]["status"] == "error"
    github.publish.assert_not_called()
    assert github.status.call_args.args[1] == "error"


def test_successful_local_round_trip(tmp_path, measurements):
    manifest, result = measurements
    provider, github = Mock(), Mock()
    provider.submit.return_value = {"id": "job", "url": "https://example.com/job"}
    provider.status.return_value = "SUCCEEDED"
    provider.result.return_value = result
    github.current.return_value = True
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=True)
    assert controller.execute(args, github, provider, manifest, state, tmp_path / "state.json") == "no_regression"
    record = state["runs"][0]
    assert record["status"] == "completed"
    directory = tmp_path / "runs" / record["run_id"]
    assert {p.name for p in (directory / "bundle").iterdir()} == set(hyperai.BUNDLE_FILES)
    assert (directory / "report.md").exists()
    assert json.loads((directory / "result.json").read_text()) == result
    github.publish.assert_called_once()


def test_submission_ambiguity_remains_durable(tmp_path, measurements):
    manifest, _ = measurements
    provider, github = Mock(), Mock()
    provider.submit.side_effect = TimeoutError("lost response after createJob")
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=False)
    with pytest.raises(TimeoutError):
        controller.execute(args, github, provider, manifest, state, tmp_path / "state.json")
    persisted = json.loads((tmp_path / "state.json").read_text())
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


def test_hyperai_token_stays_out_of_job(monkeypatch, tmp_path, measurements):
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
    provider.inventory = Mock(
        return_value={
            "resources": [{"name": "rtx-3090", "gpu": {"count": 1}}],
            "runtimes": [
                {
                    "id": "pytorch-2.11.0-2404",
                    "deprecated": False,
                    "device": "GPU",
                    "labels": ["uv", "python-3.12"],
                },
            ],
        }
    )
    (tmp_path / ".env").write_text("OPENBAYES_TOKEN=local-secret")
    manifest, _ = measurements
    provider.status = Mock(return_value="SUCCEEDED")
    provider.result = Mock(
        return_value={"kind": "prepared_environment", "spec": controller.environment_spec(manifest)}
    )
    provider.validate(manifest)
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
def test_resume_preserves_original_publication_choice(tmp_path, measurements, calibration):
    manifest, result = measurements
    if calibration:
        manifest["pr"] = None
    provider, github = Mock(), Mock()
    provider.submit.return_value = {"id": "job", "url": "https://example.com/job"}
    provider.status.return_value = "SUCCEEDED"
    provider.result.return_value = result
    github.current.return_value = True
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=calibration)
    controller.execute(args, github, provider, manifest, state, tmp_path / "state.json")
    record = state["runs"][0]
    args.publish = True
    controller.execute(args, github, provider, manifest, state, tmp_path / "state.json", record)
    github.publish.assert_not_called()
    github.status.assert_not_called()


def test_invalid_provider_config_does_not_reserve_budget_or_submit(tmp_path, measurements):
    manifest, _ = measurements
    provider = Mock()
    provider.validate.side_effect = ValueError("Unsupported image")
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=False)
    with pytest.raises(ValueError, match="Unsupported image"):
        controller.execute(args, Mock(), provider, manifest, state, tmp_path / "state.json")
    assert state["runs"] == []
    provider.submit.assert_not_called()


def test_changed_comparator_cannot_judge_a_recorded_run(tmp_path, measurements):
    manifest, _ = measurements
    manifest["harness"]["compare.py"] = "old-comparator"
    provider = Mock()
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=False)
    with pytest.raises(ValueError, match="Comparator changed"):
        controller.execute(args, Mock(), provider, manifest, state, tmp_path / "state.json")
    provider.submit.assert_not_called()
    record = {"status": "running", "job": {"id": "existing-job"}}
    state["runs"].append(record)
    provider.status.side_effect = ["RUNNING", "CANCELLED"]
    with pytest.raises(ValueError, match="Comparator changed"):
        controller.execute(args, Mock(), provider, manifest, state, tmp_path / "state.json", record)
    provider.cancel.assert_called_once_with("existing-job")
    assert record["status"] == "error"


def test_prepare_records_ready_artifact_without_publishing(tmp_path, measurements):
    manifest, _ = measurements
    manifest.update(prepare=True, pr=None)
    manifest["harness"]["compare.py"] = "comparator-not-used-during-preparation"
    provider, github = Mock(), Mock()
    provider.submit.return_value = {"id": "environment-job", "url": "https://example.com/job"}
    provider.status.return_value = "SUCCEEDED"
    provider.result.return_value = {
        "manifest_id": manifest["id"],
        "kind": "prepared_environment",
        "spec": controller.environment_spec(manifest),
    }
    github.current.return_value = True
    state = {"runs": []}
    args = SimpleNamespace(state_dir=tmp_path, timeout_minutes=1, daily_compute_minutes=10, publish=False)
    assert controller.execute(args, github, provider, manifest, state, tmp_path / "state.json") == "prepared"
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
