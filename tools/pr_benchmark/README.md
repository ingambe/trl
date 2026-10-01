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
must be checked against the actual account with `doctor` and a calibration run.

## Prepare once, then calibrate

Hyper.ai's [runtime FAQ](https://hyper.ai/en/docs/runtimes/faq) says regular accounts cannot supply custom Docker
images. The supported equivalent here is a persistent environment prepared on `standard-cpu`, then mounted read-only
at `/input0` for each GPU comparison. This also avoids copying the environment into every job's output directory.

Prepare dependencies and download the pinned model once (no GPU is allocated):

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py prepare \
  --repo OWNER/REPO --env-file ~/.config/trl-bench/env --timeout-minutes 30
```

After successful preparation, add `HYPERAI_ENVIRONMENT_JOB=<printed-job-id>` to your local env file. Every profile on
the same model shares it. New benchmark jobs run offline for model access and never install packages. Only the two Git
commit fetches need network access. Changing the dependency pins, model/data revisions or environment builder requires
preparing again; a mismatch fails before a GPU is submitted. The worker also checks that the Python runtime matches.
Preparation also downloads the Capybara dataset pinned in `profiles.json`; the games do not use it, and it stays pinned
only so the prepared environments remain valid.

The stopped preparation job's output occupies persistent storage; preserve it while it is referenced by benchmarks.
Preparation itself is not a quality or performance result and cannot publish a passing PR status.

A provider supporting custom images can bake the same environment layout into an image and invoke
`python job.py --environment /path/to/prepared-environment`. Hyper.ai's adapter selects a registered runtime; it does not
pretend to accept arbitrary Docker image references.

Run a commit against itself to check the pipeline and the noise before relying on the margins:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py calibrate \
  --repo OWNER/REPO --ref main --profile wordle --env-file ~/.config/trl-bench/env
```

`--ref` accepts an exact commit SHA. Calibration pins that SHA for both sides and never posts PR statuses. `--dry-run`
prints the resolved configuration without allocating compute.

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
The status context is `gpu-benchmark/<profile>`. A successful comparison returns `0`; regression or inconclusive returns
`1`. Failed/inconclusive comparisons never receive a successful GitHub status.

## Workload and interpretation

Each profile trains GRPO with colocated vLLM on one of two local tool-calling games from `games.py`:

- `wordle`: find a hidden 5-letter word in 6 guesses with a `guess` tool that marks each letter G, Y or X. The reward
  is 1 when solved, else `(2 * greens + yellows) / 20` for the best guess.
- `number`: find a hidden number from 1 to 100 in 7 guesses with a `guess` tool that answers higher, lower or correct.
  The reward is 1 when solved, else up to 0.5 for the closest guess.

Every dataset row carries a game seed that `reset` uses, so the rollouts of a group play the same game and both commits
see exactly the same games. Each vLLM request also gets its own sampling seed, so a token flipped by rounding changes one
completion instead of the whole batch. Each commit trains **once**, in a fresh process, for a fixed time budget
(`train_minutes`, 10 minutes). The first `warmup_steps + speed_steps` (3 + 50) steps run every code path with a zero
learning rate, so both commits time the same policy; training then continues for the rest of the budget. The job heats
the GPUs for 3 minutes before the base run, so it does not get the boost clocks of a cold card, and each run has its own
compile caches.

Profiles: `wordle` and `number` train the dense Qwen2.5-0.5B-Instruct; `wordle-lora` and `number-lora` train an
`all-linear` rank-16 LoRA; `wordle-lora-3b` trains the LoRA on Qwen2.5-3B-Instruct, which needs its own prepared
environment. `--gpus 2` trains data-parallel on two GPUs of one allocation (two RTX 3090 when offered, else two RTX
5090); each process has its own colocated vLLM engine and rank 0 records the results. With a LoRA profile,
`--native-lora` has vLLM serve the adapter natively on commits that support it. A checkout whose `GRPOConfig` has
`vllm_share_weights` enables it.

Speed: the 50 frozen steps after the warm-up are paired by index. Once training moves the policies, two runs of the same
commit drift apart (main against itself once ran 655 and 442 steps in the same 10 minutes), so later steps are not
compared. For step time and generation time, the comparator takes the mean paired percentage change and a two-sided 95%
Student t interval across steps. Positive changes are degradations, with a 5% margin:

- **Regression:** the entire interval exceeds the margin for any metric.
- **Inconclusive:** an interval crosses the margin.
- **Improved:** no metric regresses/is inconclusive, and one interval is entirely below the negative margin.
- **No regression:** all intervals remain within the allowed upper margins.

Quality: the mean vLLM/trainer sampling logprob gap over the same 50 frozen steps must be at most 1.5x base, which
catches a broken weight sync or dtype, not small drifts. A failure turns a passing speed verdict into `quality_failed`.
Both commits also play the same 64 held-out games before and after training, but that reward is only reported: main
against itself ended 0.02 and 0.11 on Wordle, so one run per side cannot gate on it.

Peak memory, steps completed and held-out rewards are reported beside the verdict. Both commits share one read-only
dependency environment and GPU allocation; the PR's dependency files are not installed. Missing or non-finite
measurements are errors.

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
command starts; the local controller separately bounds queue/allocation time, extended by `--queue-minutes`. A job whose
Hyper.ai pod failed to start is resubmitted, at most twice. Ctrl-C/SIGTERM attempts to stop the active job and confirms
terminal status. Never infer that a stopped local process means remote compute is stopped.

Restarting the same command resumes a recorded running job. Failed comparisons are not automatically rerun for the same
comparison identity; use manual `run --rerun` or `calibrate --rerun` after resolving the problem. Infrastructure/API errors
can stop the watcher; inspect the local state and provider console before restarting it.

An ambiguous submission (for example, losing the response to job creation) stays `submitting` and blocks additional
launches. Check the provider console, stop any allocated job, then mark that record `error` in `state.json` to acknowledge
resolution. A `cancel_pending` record is retried on restart. No result from either state is published as passing.

## Another provider

`environment.py`, `job.py`, `grpo_workload.py`, `games.py`, the manifest and `result.json` format are provider-independent. `hyperai.py` is the only cloud
adapter. A future provider needs `validate(manifest)` for read-only configuration checks before reserving budget, plus
`submit(bundle, timeout_seconds)`, `status(job_id)`,
`result(job_id)` and `cancel(job_id)`, with a provider-side deadline and normalized terminal states. Keep authentication
and upload logic in the adapter; add explicit provider selection only when a second provider is implemented.

## Local tests

The tests need pytest. They check regression decisions and invalid measurements without mocking the cloud API.
Provider behavior and GPU execution require a live calibration run.

```bash
python -m pip install pytest
python -m pytest tools/pr_benchmark/tests -q
```

## GitHub Actions

The manual **GPU benchmark** workflow compares a branch against a base as seven parallel jobs: `wordle`, `wordle-lora`,
`number` and `number-lora` on one GPU, `wordle` and `number` on two GPUs, and `wordle-lora-3b` on two GPUs. Reports go
to the run summary and run directories are uploaded as artifacts. It needs the `OPENBAYES_TOKEN` secret and the
`HYPERAI_RUNTIME`, `HYPERAI_ENVIRONMENT_JOB` and `HYPERAI_ENVIRONMENT_JOB_3B` (the 3B environment) variables.
