# Local PR GPU benchmarks

A local controller compares the **current PR target branch tip** against the **exact PR head**, on one Hyper.ai GPU
allocation. It can watch open PRs, run one comparison, or calibrate a commit against itself. No GitHub Actions runner or
GitHub-hosted provider secret is needed. The controller machine must be awake and connected.

## Install the controller

Requires Python 3.10+, `gh`, and Linux or macOS (the local process lock uses `fcntl`). From the repository root:

```bash
python3 -m venv ~/.local/share/trl-pr-benchmark/venv
~/.local/share/trl-pr-benchmark/venv/bin/pip install -r tools/pr_benchmark/requirements-controller.txt
gh auth login
```

Create a file **outside this repository**, for example `~/.config/trl-bench/env`, with permissions `600`:

```dotenv
OPENBAYES_TOKEN=your-account-token
HYPERAI_RESOURCE=rtx-3090
# Set this after checking the inventory below:
HYPERAI_RUNTIME=your-selected-runtime-id
# Set this after the one-time preparation below:
# HYPERAI_ENVIRONMENT_JOB=successful-preparation-job-id
# Optional account/organization settings:
# OPENBAYES_ORG=organization-name
# HYPERAI_PROJECT_ID=existing-project-id
# HYPERAI_ENDPOINT=https://app.hyper.ai
```

Hyper.ai's [CLI authentication documentation](https://app.hyper.ai/docs/cli/login/) describes `OPENBAYES_TOKEN` for
account automation. The **user-level API keys** on Account Settings → API Key are documented as credentials for all
of your **model deployments**, not for creating GPU containers. See the official
[API key scope documentation](https://hyper.ai/en/docs/serving/05-api-key-management).
`Bearer` is the HTTP authorization scheme, not a separate kind of key; Hyper.ai also uses it for serving API keys.

For SSO accounts, Hyper.ai documents an [official OAuth connection](https://hyper.ai/en/docs/ai-features/connect) and
an MCP [`user_create_personal_access_token` tool](https://hyper.ai/en/docs/ai-features/tools#user_create_personal_access_token).
That tool creates an account-wide personal access token with a default 90-day expiry. Creating one requires explicit
approval of its name and expiry. Use the supported sign-in flow; do not extract browser session credentials. The
documented clients are Claude Code and Cursor, with a pre-registered OAuth client ID. The PAT returned by the official
MCP tool has been verified with this adapter's account and GPU/image inventory queries. Put the **PAT** in
`OPENBAYES_TOKEN`; the MCP OAuth access token is a separate credential for the MCP server.

Never put credentials in a PR, command argument, or uploaded job configuration.
`--env-file` accepts literal `KEY=value` assignments, optional quotes and `export`, without shell expansion. Environment
variables work without a file. `GH_TOKEN` is optional when `gh` is already authenticated.

List account resources/images without creating a job:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py doctor \
  --env-file ~/.config/trl-bench/env
```

Select an available CUDA image with Python 3.12, `uv`, `git`, `venv`, and GNU `timeout`. Preparation installs the pinned GPU
requirements once in a reusable environment. Benchmarks only read that completed environment. `rtx-3090` is the default;
the controller never silently upgrades to a 5090 or changes resource when capacity is unavailable. A missing runtime or
unavailable resource fails before compute submission.

The adapter uses the GraphQL operations exercised by `openbayes-cli==0.28.4`, with an explicit allowlist of uploaded files.
It keeps the account token in memory rather than saving a CLI credential cache. API compatibility and runtime support
must be checked against the actual account with `doctor` and the smoke run.

## Prepare once, then smoke and calibrate

Hyper.ai's [runtime FAQ](https://hyper.ai/en/docs/runtimes/faq) says regular accounts cannot supply custom Docker
images. The supported equivalent here is a persistent environment prepared on `standard-cpu`, then mounted read-only
at `/input0` for each GPU comparison. This also avoids copying the environment into every job's output directory.

Prepare dependencies and download the pinned model/dataset once (no GPU is allocated):

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py prepare \
  --repo OWNER/REPO --env-file ~/.config/trl-bench/env --timeout-minutes 30
```

After successful preparation, add `HYPERAI_ENVIRONMENT_JOB=<printed-job-id>` to your local env file. The same artifact
supports the smoke and full profiles because it contains the original dataset and model, not profile-specific samples.
New benchmark jobs run offline for model/data access and never install packages. Only the two Git commit fetches need
network access. Changing the dependency pins, model/data revisions or environment builder requires preparing again;
a mismatch fails before a GPU is submitted. The worker also checks that the Python runtime matches.

The stopped preparation job's output occupies persistent storage; preserve it while it is referenced by benchmarks.
Preparation itself is not a quality or performance result and cannot publish a passing PR status.

A provider supporting custom images can bake the same environment layout into an image and invoke
`python job.py --environment /path/to/prepared-environment`. Hyper.ai's adapter selects a registered runtime; it does not
pretend to accept arbitrary Docker image references.

Preview the smoke configuration and resolved commits without allocating compute:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py calibrate \
  --repo OWNER/REPO --profile smoke --dry-run
```

Run a short base-versus-itself comparison on the configured GPU:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py calibrate \
  --repo OWNER/REPO --ref main --profile smoke \
  --env-file ~/.config/trl-bench/env --timeout-minutes 10
```

The smoke profile uses one paired seed and four training steps. Its expected outcome is **inconclusive**, with exit code
`1`, even when the entire pipeline works. It cannot produce a green no-regression result. Dependency/model downloads
happen in the separate CPU preparation task and are not repeated by GPU runs.

Next run the full `sft-3090` profile against itself to estimate measurement noise before relying on its thresholds:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py calibrate \
  --repo OWNER/REPO --ref main --env-file ~/.config/trl-bench/env
```

`--ref` accepts an exact commit SHA. Calibration pins that SHA for both sides and never posts PR statuses.

## Compare a PR or watch for changes

```bash
# One comparison; results stay local unless --publish is supplied.
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py run \
  --repo OWNER/REPO --pr 123 --env-file ~/.config/trl-bench/env

# Watch PRs and post commit statuses plus a report on each tested PR.
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py watch \
  --repo OWNER/REPO --env-file ~/.config/trl-bench/env --publish
```

By default, only ready-for-review PRs authored by the authenticated GitHub user are eligible. `--author USER` adds an
explicit author allowlist (repeat for multiple authors). This is designed for **your own trusted development PRs**.
Do not enable arbitrary public contributors: the GPU job executes PR Python code, and the benchmark harness is not an
adversarial sandbox. Although local provider/GitHub tokens are not uploaded or forwarded, provider-managed containers
can have their own credentials and services. Run untrusted code only after designing provider-specific isolation.

Watching starts with currently open eligible PRs, then polls every 60 seconds. Changes to either the base tip, PR head,
profile, harness or thresholds create a new comparison identity. Obsolete running comparisons are cancelled when the PR
changes, closes, or becomes a draft. The base is the current branch tip, **not the merge base or GitHub's synthetic merge
commit**. The watcher rechecks identities before publishing results.

Posting requires permission to write commit statuses and PR comments on the repository containing the PR. Upstream PRs
may therefore need maintainer authorization. Running locally without `--publish` does not need those write permissions.
The status context is `gpu-benchmark/sft-3090`. A successful comparison returns `0`; regression or inconclusive returns
`1`. Failed/inconclusive comparisons never receive a successful GitHub status.

## Workload and interpretation

The initial profile tests SFT only:

- Qwen2.5-0.5B-Instruct and Capybara, both pinned to immutable Hub revisions.
- Fixed seeded partition: 1,024 training examples and 128 held-out examples, tokenized once for both commits.
- BF16, SDPA, sequence limit 256, batch size 1, accumulation 4, gradient checkpointing.
- 120 optimizer steps; first 20 excluded from steady-state timing but included in total training time.
- Five paired seeds, with base/head order alternating. Every run starts from the original model in a fresh process.
- CUDA synchronization at timing boundaries; model/data downloads precede measurements.
- Independent token-weighted held-out next-token loss, separate from the trainer's loss/evaluation implementation.

Both commits share exactly one read-only dependency environment and GPU allocation. The PR's dependency files, build hooks, and
benchmark definitions are not installed or used. This isolates TRL code changes; testing a dependency change requires a
separate, explicitly designed experiment. The actual GPU/driver, complete installed package versions, commits, data hash,
raw trajectories, memory peak, per-run process wall time, and measurements are retained.

The core GPU packages are pinned in `requirements-gpu.txt`; transitive versions are recorded rather than fully locked.
Prepare again when updating the image or dependencies; do not treat measurements from different allocations as
interchangeable. Total allocation time includes startup and is not a performance gate. `train_seconds` includes
training warm-up, while `steady_seconds` excludes it. `workload_seconds` additionally includes process startup, model
loading and evaluation; it is retained as a diagnostic, not used as a gate.

For each metric, the comparator uses the mean paired percentage change and a two-sided 95% Student t interval across
seeds (approximate normality is assumed; five seeds can be insufficient). Positive changes are degradations:

| Metric | Initial allowed degradation |
|---|---:|
| Total training seconds | 5% |
| Steady-state training seconds | 5% |
| Held-out token loss | 1% |

- **Regression:** the entire interval exceeds the margin for any metric.
- **Inconclusive:** an interval crosses the margin, or fewer than five pairs were collected.
- **Improved:** no metric regresses/is inconclusive, and one interval is entirely below the negative margin.
- **No regression:** all intervals remain within the allowed upper margins.

The intervals are per-metric, not a family-wide statistical guarantee. These margins are initial policy choices, not
established TRL guarantees. Calibrate them using repeat base-versus-itself runs. Finite metrics, complete paired seeds,
identical environments and expected SHAs/step counts are mandatory. Missing/invalid results are errors.

Held-out SFT loss is a limited quality proxy. This workload does not establish downstream task accuracy, long-run
convergence, DPO/GRPO quality, distributed correctness, or the absence of all regressions. Existing numerical invariant
tests in `tests/invariant/` remain a separate check; this runner does not replace or automatically execute them.

## State, spending limits, and recovery

State lives in `~/.local/state/trl-pr-benchmark` (override with `--state-dir` outside the repo):

```text
state.json
controller.lock
runs/<run-id>/bundle/manifest.json
runs/<run-id>/result.json
runs/<run-id>/summary.json
runs/<run-id>/report.md
```

One job runs at a time per state directory. The default job deadline is 60 minutes and the daily reservation limit is
120 compute minutes (`--daily-compute-minutes`), including CPU preparation conservatively. The full deadline is
charged to the local budget before submission, even if the job finishes early. This is a conservative compute-minute cap,
not a currency/billing guarantee. Use one state directory for the account so separate watchers cannot independently spend
the same budget. Hyper.ai storage charges are outside this limit.

The remote `timeout` bounds a running task even if the controller goes offline. It cannot bound time queued before the
command starts; the local controller separately bounds queue/allocation time. Ctrl-C/SIGTERM attempts to stop the active
job and confirms terminal status. Never infer that a stopped local process means remote compute is stopped.

Restarting the same command resumes a recorded running job. Failed comparisons are not automatically rerun for the same
comparison identity; use manual `run --rerun` or `calibrate --rerun` after resolving the problem. Infrastructure/API errors
can stop the watcher; inspect the local state and provider console before restarting it.

An ambiguous submission (for example, losing the response to job creation) stays `submitting` and blocks additional
launches. Check the provider console, stop any allocated job, then mark that record `error` in `state.json` to acknowledge
resolution. A `cancel_pending` record is retried on restart. No result from either state is published as passing.

## Another provider

`environment.py`, `job.py`, `workload.py`, the manifest and `result.json` format are provider-independent. `hyperai.py` is the only cloud
adapter. A future provider needs `validate(manifest)` for read-only configuration checks before reserving budget, plus
`submit(bundle, timeout_seconds)`, `status(job_id)`,
`result(job_id)`, and `cancel(job_id)`, with a provider-side deadline and normalized terminal states. Keep authentication
and upload logic in the adapter; add explicit provider selection only when a second provider is implemented.

## Local tests

The comparator tests need only pytest. They check regression decisions, uncertainty, and invalid measurements without
mocking the cloud API. Provider behavior and GPU execution require the live smoke run described above.

```bash
python -m pip install pytest
python -m pytest tools/pr_benchmark/tests -q
```
