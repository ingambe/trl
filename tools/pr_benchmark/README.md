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

`environment.py`, `job.py`, `workload.py`, the manifest and `result.json` format are provider-independent. `hyperai.py` is the only cloud
adapter. A future provider needs `validate(manifest)` for read-only configuration checks before reserving budget, plus
`submit(bundle, timeout_seconds)`, `status(job_id)`,
`result(job_id)`, `download_profile(job_id, side, destination)` (rollout profiles only), and `cancel(job_id)`, with a provider-side deadline and normalized terminal states. Keep authentication
and upload logic in the adapter; add explicit provider selection only when a second provider is implemented.

## Local tests

The tests need pytest plus the controller requirements (HTA for the profile-analysis tests). They check regression
decisions, uncertainty, invalid measurements and HTA analysis of synthetic traces without mocking the cloud API.
Provider behavior and GPU execution require the live smoke run described above.

```bash
python -m pip install pytest -r tools/pr_benchmark/requirements-controller.txt
python -m pytest tools/pr_benchmark/tests -q
```

## Multi-turn LoRA rollouts with before/after profiles

The `vllm-rollout` profile measures colocated vLLM rollouts with a nonzero rank-8 LoRA adapter. Each phase applies a
deterministic adapter update, then times GRPO's `_generate` through a four-turn `rollout_func` that appends fixed CPU
feedback between turns, and ends with vLLM asleep (the handoff back to training). One warm-up and six measured phases
run for each of five paired seeds; each seed rolls out two distinct prompts. One process per side runs all seeds, base then
head, so model loading and vLLM startup happen once per side. The feedback is a fixed string, not GRPO's
tool-calling loop: this is a systems workload, not a scored task, and reports no task success.

Latency uses the same paired intervals as SFT on `rollout_seconds` (synchronization, all turns, and sleep) and
`weight_transfer_bytes` (logical tensor payload handed to vLLM's `load_weights`, the saved adapter tensors for native
adapters, or the layers an adapter is merged into in place; not bus traffic). A transfer reduction alone is never
reported as an improvement. Quality is reported separately and any failure turns a passing latency verdict into
`quality_failed`; it never hides the timings:

- greedy rollout tokens match base for every seed (native LoRA or a restoration fix can legitimately change them);
- vLLM is asleep after every phase, and every phase synchronized the updated policy (syncs per phase are reported);
- the maximum frozen-weight drift of the PR is no worse than base (both values are reported);
- the stale-policy negative control is detected (see below).

`requirements-gpu.txt` includes vLLM and PEFT, so an environment prepared before this profile existed must be prepared
again. Compare a pushed branch before opening a PR (never publishes a status):

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py compare \
  --repo OWNER/REPO --base-ref BASE_SHA --ref HEAD_SHA --profile vllm-rollout --env-file ~/.config/trl-bench/env
```

`--gpus 2` runs any `vllm-rollout` profile data-parallel on two GPUs of one allocation: two RTX 3090 when the account
offers them, otherwise two RTX 5090 (the report names the GPU). Each process rolls out its own prompts with its own
colocated vLLM engine. A phase lasts as long as its slowest process; rank 0 records the tokens, memory and profile.

`--profile vllm-rollout-dense` runs the same workload without LoRA: each phase updates the norm weights instead of
the adapter, and the frozen-weight check covers every other parameter. On every rollout profile, a checkout whose
`GRPOConfig` has `vllm_share_weights` enables it, so the comparison measures shared weights against weight publication.

`--profile grpo-train-dense` runs real `GRPOTrainer.train()` instead: 30 AdamW steps on the dense model with colocated
vLLM, eight completions of up to 64 tokens per step and a deterministic length reward. It compares training time, with
generation and backward/optimizer time reported separately, and checks quality against base on ten seeds: the training
reward averaged over steps (within 0.05), the vLLM/trainer sampling logprob gap averaged over steps (within 10%), and a
greedy reward on 128 held-out prompts, which varies too much across seeds to show it is within 0.02 and only fails when
clearly worse. Each other check passes only when its whole 95% interval clears the margin, and makes the verdict
inconclusive when the interval crosses it. Per-step gaps and the final parameter-sum gap are reported, not checked:
changing kernel shapes changes sampled tokens, and trajectories diverge from there. Each vLLM request gets its own
sampling seed from the run seed, so a flipped token changes one completion instead of the whole batch. The reward is
synthetic, so this is not a scored task benchmark. `--profile grpo-train-dense-binary` scores 1 only within 25%
of the target length, else 0, so some groups get equal rewards and have zero advantage.

`--profile grpo-train-lora` trains an `all-linear` rank-16 LoRA instead, with completions of up to 256 tokens.
`--profile grpo-train-lora-3b` runs it on Qwen2.5-3B-Instruct, which needs its own prepared environment. With
a LoRA profile, `--native-lora` has vLLM serve the adapter natively on commits that support it.

`--serious` selects a longer workload on one RTX 5090: three warm-up and twelve measured phases, eight distinct
prompts of up to 768 tokens, six turns of 64 generated tokens, 1,536-token context. It is still synthetic and unscored:
longer runs do not establish task quality. Never pool results from different GPUs.

`vllm-rollout-medium` sits between the two: two warm-up and eight measured phases, four prompts of up to 256 tokens,
four turns of 32 generated tokens, 768-token context, on one RTX 3090.

The manual **GPU benchmark** workflow runs these comparisons from GitHub Actions as twelve parallel jobs: small
(`vllm-rollout`), medium and large (`--serious`) rollouts, `grpo-train-dense` and `grpo-train-lora`, each on one and two
GPUs, `grpo-train-lora-3b` on two GPUs, and `sft-long` (2,048-token SFT, several loss chunks per micro-batch) on one GPU. Reports go to the run summary; run directories, profiles included, are
uploaded as artifacts. It needs the `OPENBAYES_TOKEN` secret and the `HYPERAI_RUNTIME`, `HYPERAI_ENVIRONMENT_JOB` and
`HYPERAI_ENVIRONMENT_JOB_3B` (the 3B environment) variables.

### Profiles and the distribution diagnostic

After all timing samples, the job starts one extra process per side with the first seed. It captures the normal
warm-up plus two complete phases with the PyTorch Profiler (CPU and CUDA activities, shapes, memory; no Python stacks),
labelling each phase, turn, CPU feedback, `sync_weights`, `load_weights`, `add_lora`, `wake_up`, `sleep` and
`reset_prefix_cache`. After the capture it compares full next-token distributions of the local policy and vLLM on one
fixed history, for a zero adapter, a nonzero adapter and a second update. Each stage reports total variation, KL and
Jensen–Shannon divergence against the local unmerged model, the BF16-merged model, and the local model after
publication. The negative control compares vLLM with the previous stage's policy, standing in for a missed
synchronization: its total variation must exceed twice the synced one. Only this process enables vLLM's
full-vocabulary logprobs, so timed runs are unaffected. One seed and one history are diagnostic evidence, not a
confidence interval.

The controller downloads both archives and runs HTA locally (`requirements-controller.txt`). Nothing is uploaded: the
report only names the local run directory, which keeps:

```text
runs/<run-id>/{base,head}-profile.zip   # checksummed capture archives, also kept as job outputs
runs/<run-id>/profiles/report.md        # before/after HTA temporal breakdown and artifact links
runs/<run-id>/profiles/summary.json     # all HTA tables, observed memcpy bytes by kind, analysis packages
runs/<run-id>/profiles/kernel-deltas.csv
runs/<run-id>/profiles/{base,head}/     # metadata.json, trace.json.gz, trace_with_counters.json.gz,
                                        # operators-*.txt, hta-*.csv, policy-distributions.npz
```

Observed memcpy bytes come only from these single-seed instrumented captures; timed records report logical payload
bytes only. HTA 0.5 does not recognize CUDA Graph launches in its launch/queue analyses; the report flags this when
replays are present. Rerun the analysis without GPU time with
`python tools/pr_benchmark/analyze_profiles.py ~/.local/state/trl-pr-benchmark/runs/RUN_ID`.
