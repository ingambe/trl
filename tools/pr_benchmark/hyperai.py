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

"""Hyper.ai adapter. Wire operations match openbayes-cli 0.28.4's GraphQL clients.

Credentials stay in memory; unlike the CLI login path, this does not save a token to disk.
"""

import json
from pathlib import Path
from urllib.parse import urlparse

import requests
from environment import environment_spec


TERMINAL = {"SUCCEEDED", "FAILED", "CANCELLED"}
BUNDLE_FILES = ("job.py", "workload.py", "environment.py", "requirements-gpu.txt", "manifest.json")


class HyperAI:
    def __init__(self, env, prepare=False):
        token = env.get("OPENBAYES_TOKEN")
        if not token:
            raise ValueError("Set OPENBAYES_TOKEN locally, or use --env-file outside the repository")
        self.endpoint = env.get("HYPERAI_ENDPOINT", "https://app.hyper.ai").rstrip("/")
        if urlparse(self.endpoint).scheme != "https":
            raise ValueError("HYPERAI_ENDPOINT must use HTTPS")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {token}", "Origin": self.endpoint})
        self.user = self.query("query { me { username } }")["me"]["username"]
        self.party = env.get("OPENBAYES_ORG") or self.user
        self.prepare = prepare
        self.resource = "standard-cpu" if prepare else env.get("HYPERAI_RESOURCE", "rtx-3090")
        self.runtime = env.get("HYPERAI_RUNTIME")
        self.project = env.get("HYPERAI_PROJECT_ID")
        self.environment_job = env.get("HYPERAI_ENVIRONMENT_JOB")

    def query(self, query, variables=None):
        response = self.session.post(
            self.endpoint + "/gateway",
            json={"query": query, "variables": variables or {}},
            timeout=60,
            allow_redirects=False,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Hyper.ai API returned HTTP {response.status_code}")
        payload = response.json()
        if payload.get("errors"):
            # Do not echo provider responses, which may contain authentication or signed URL details.
            if any(error.get("extensions", {}).get("code") == "FORBIDDEN" for error in payload["errors"]):
                raise RuntimeError(
                    "Hyper.ai denied account/job API access (403). Check credential scope, expiration, and permissions. "
                    "The Account Settings API Key page documents keys for model deployments, including user-level "
                    "keys; it does not document compute access. See README.md for official authentication routes."
                )
            raise RuntimeError(
                "Hyper.ai GraphQL request failed; check endpoint, account access, and API compatibility"
            )
        return payload["data"]

    def inventory(self):
        resources = self.query(
            """query($partyId: String) {
              normalClusterResources(partyId: $partyId) { name gpu { name count memory } }
            }""",
            {"partyId": self.party},
        )["normalClusterResources"]
        runtimes = self.query(
            """query($partyId: String) {
              normalClusterRuntimes(partyId: $partyId) { name framework version device labels deprecated }
            }""",
            {"partyId": self.party},
        )["normalClusterRuntimes"]
        for runtime in runtimes:
            runtime["id"] = f"{runtime['framework']}-{runtime['version']}"
        return {"user": self.user, "party": self.party, "resources": resources, "runtimes": runtimes}

    def ensure_project(self):
        if self.project:
            return
        name = "trl-pr-benchmark"
        projects = self.query(
            """query($partyId: ID!, $q: String!) {
              party(id: $partyId) { projects(q: $q, page: 1, perPage: 100) { data { id name } } }
            }""",
            {"partyId": self.party, "q": name},
        )["party"]["projects"]["data"]
        exact = [p for p in projects if p["name"] == name]
        if exact:
            self.project = exact[0]["id"]
        else:
            self.project = self.query(
                """mutation($userId: String!, $name: String!) {
                  createProject(userId: $userId, name: $name, tagNames: [{name: "BUSINESS_CHANNEL_ML"}]) { id }
                }""",
                {"userId": self.party, "name": name},
            )["createProject"]["id"]

    def validate(self, manifest):
        inventory = self.inventory()
        selected = [r for r in inventory["resources"] if r["name"] == self.resource]
        if not selected:
            raise ValueError("Requested GPU resource unavailable; run doctor and set HYPERAI_RESOURCE")
        expected_gpus = 0 if self.prepare else 1
        if not selected[0]["gpu"] or selected[0]["gpu"]["count"] != expected_gpus:
            raise ValueError(f"This operation requires {expected_gpus} GPUs")
        supported = {
            r["id"]
            for r in inventory["runtimes"]
            if not r["deprecated"]
            and r["device"] == ("CPU" if self.prepare else "GPU")
            and {"uv", "python-3.12"} <= set(r["labels"])
        }
        if not self.runtime or self.runtime not in supported:
            raise ValueError("Run doctor and set HYPERAI_RUNTIME to a Python 3.12 CUDA image with uv")
        if not self.prepare:
            if not self.environment_job or self.status(self.environment_job) != "SUCCEEDED":
                raise ValueError("Run prepare, then set HYPERAI_ENVIRONMENT_JOB to the successful preparation job")
            marker = self.result(self.environment_job)
            if marker.get("kind") != "prepared_environment" or marker.get("spec") != environment_spec(manifest):
                raise ValueError("Prepared environment does not match dependencies/model/data; run prepare again")

    def submit(self, bundle: Path, timeout_seconds: int):
        import boto3
        from botocore.config import Config

        self.ensure_project()
        policy = self.query(
            """mutation($userId: String!, $storageType: StorageType!) {
              createSourceCodePolicy(userId: $userId, storageType: $storageType) {
                id endpoint accessKey secretKey path
              }
            }""",
            {"userId": self.party, "storageType": "TEMPORARY"},
        )["createSourceCodePolicy"]
        storage = boto3.client(
            "s3",
            endpoint_url=policy["endpoint"],
            aws_access_key_id=policy["accessKey"],
            aws_secret_access_key=policy["secretKey"],
            config=Config(
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
                connect_timeout=30,
                read_timeout=60,
                retries={"max_attempts": 2},
            ),
        )
        bucket, prefix = policy["path"].lstrip("/").split("/", 1)
        for name in BUNDLE_FILES:
            storage.upload_file(str(bundle / name), bucket, f"{prefix.rstrip('/')}/{name}")
        # This remote deadline continues to work if the local watcher disconnects.
        script = "environment.py" if self.prepare else "job.py"
        command = f"timeout --signal=TERM --kill-after=30s {timeout_seconds}s python {script}"
        bindings = (
            []
            if self.prepare
            else [
                {
                    "name": f"{self.party}/jobs/{self.environment_job}/output",
                    "path": "/input0",
                    "bindingAuth": "READ_ONLY",
                }
            ]
        )
        job = self.query(
            """mutation($userId: String!, $input: CreateJobInput) {
              createJob(userId: $userId, input: $input) { id links { name value } }
            }""",
            {
                "userId": self.party,
                "input": {
                    "mode": "TASK",
                    "projectId": self.project,
                    "runtime": self.runtime,
                    "resource": self.resource,
                    "newTask": {"command": command, "code": policy["id"]},
                    "parameters": [],
                    "dataBindings": bindings,
                    "tagNames": [{"name": "BUSINESS_CHANNEL_ML"}],
                },
            },
        )["createJob"]
        return {"id": job["id"], "url": next((x["value"] for x in job["links"] if x["name"] == "frontend"), "")}

    def status(self, job_id):
        return self.query(
            """query($userId: String!, $jobId: String!) {
              job(userId: $userId, jobId: $jobId) { status }
            }""",
            {"userId": self.party, "jobId": job_id},
        )["job"]["status"]

    def cancel(self, job_id):
        self.query(
            """mutation($userId: String!, $jobId: String!) {
              stopJob(userId: $userId, jobId: $jobId) { id }
            }""",
            {"userId": self.party, "jobId": job_id},
        )

    def result(self, job_id):
        output = self.query(
            """mutation($userId: String!, $jobId: String!, $key: String!) {
              createJobOutputDownloadUrl(userId: $userId, jobId: $jobId, key: $key) { url type name }
            }""",
            {"userId": self.party, "jobId": job_id, "key": "result.json"},
        )["createJobOutputDownloadUrl"]
        # The signed download URL must not receive the account Authorization header.
        with requests.get(output["url"], stream=True, timeout=60) as response:
            response.raise_for_status()
            content = bytearray()
            for chunk in response.iter_content(65536):
                content.extend(chunk)
                if len(content) > 10_000_000:
                    raise ValueError("Benchmark output exceeds the 10 MB limit")
        return json.loads(content)
