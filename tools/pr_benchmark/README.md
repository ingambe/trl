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
`result(job_id)`, `download_profile(job_id, side, destination)`, and `cancel(job_id)`, with a provider-side deadline and normalized terminal states. Keep authentication
and upload logic in the adapter; add explicit provider selection only when a second provider is implemented.

## Local tests

The comparator tests need only pytest. They check regression decisions, uncertainty, and invalid measurements without
mocking the cloud API. Provider behavior and GPU execution require the live smoke run described above.

```bash
python -m pip install pytest
python -m pytest tools/pr_benchmark/tests -q
```

## Colocated multi-turn rollout comparison before opening a PR

The `vllm-rollout` profile exercises GRPO's real `_generate` path with a custom four-turn rollout and deterministic
CPU tool feedback. Six measured phases follow one warm-up phase for each of five paired seeds. Each phase changes
nonzero LoRA adapter weights, then times the entire rollout through its final level-2 sleep. Greedy completion tokens must
match between base and head; the head must synchronize at most once per phase and both engines must finish asleep.
`weight_transfer_bytes` measures tensor payload bytes passed to vLLM's `load_weights`, **not measured PCIe traffic**.
`rollout_seconds` includes synchronization, generation, feedback, and sleep; initialization is excluded.

For native LoRA publication, the workload counts the adapter tensors submitted to vLLM's adapter loader as well as
any base-weight copy performed during the measured phases. It consumes full-weight exporters lazily. These are logical publication bytes, not total
device traffic: level-1 sleep also offloads/restores the frozen base, which is visible in the profiler's memory copies.
Native initialization exports the current frozen base once as a sharded checkpoint for vLLM to load normally.
The checkpoint stays in temporary storage for the engine lifetime; its disk writes and initial loading are outside
rollout timing and included in `trainer_initialization_seconds` and `workload_seconds`.
The result's `publication` field distinguishes native adapters from merged exports. The probability diagnostic passes
the published adapter explicitly, including in the deliberately stale-policy control. Native LoRA and BF16-merged
execution can produce different tokens; the existing exact-token gate still reports this as a failed comparison.

This profile needs its own prepared environment (`requirements-vllm.txt`, including vLLM 0.22.0). SFT environments
remain usable for their original profiles. Prepare once with the same command as above, adding `--profile vllm-rollout`,
then select the returned `HYPERAI_ENVIRONMENT_JOB` for this comparison. Respect the controller's existing daily budget.

Push the candidate branch without opening a PR, then run:

```bash
~/.local/share/trl-pr-benchmark/venv/bin/python tools/pr_benchmark/controller.py compare \
  --repo OWNER/REPO --base-ref BASE_SHA --ref HEAD_SHA --profile vllm-rollout \
  --env-file ~/.config/trl-bench/env
```

`compare` resolves and pins both refs and never posts a PR status. `--dry-run` resolves the request without allocating
compute. Require the latency interval to show an improvement before claiming this optimization is faster; fewer
transferred bytes alone are not latency evidence. This bounded synthetic workload does not establish distributed,
GPU-tool performance or downstream task quality; their cleanup behavior also needs the local regression tests.

If the token hash check fails, `vllm-rollout-diagnostic` runs one paired seed with two measured phases, then compares
full-vocabulary vLLM probabilities against the local model on an identical fixed history. It checks dense weights,
nonzero merged LoRA adapters, and a second adapter update across four turns, clearing the prefix cache before the
fourth. Adapter initialization and updates use independent fixed seeds. A negative control deliberately skips an
adapter sync, measures the discrepancy, then synchronizes and measures recovery. Frozen-weight drift is recorded
separately because BF16 merge/unmerge can change the local model itself. The candidate must report exactly zero
frozen-weight drift and zero local-policy total variation after both LoRA stages; missing measurements fail the gate.
The baseline may retain the known rounding defect. CPU backup/restoration adds real transfer work during adapter
exports; both timing and profiling now include that cost. The full profile also runs the fixed-history
distribution diagnostic after timing on the first paired seed. All seeds check frozen-weight preservation.

The result includes total variation, KL and Jensen–Shannon divergence, top tokens, and the exact input tokens.
`SIDE-SEED-policy-distributions.npz` artifacts preserve the full log probabilities. Every rollout workload also
records `output_tokens`. The normal comparator still rejects differing rollout tokens; inspect the saved diagnostic
results even when that gate fails. One fixed history and one paired seed cannot establish downstream quality or
general numerical equivalence, and this diagnostic is not a performance qualification.

### Longer rollout measurements on RTX 5090

Add `--serious` to `--profile vllm-rollout` for the longer workload on **one RTX 5090**. The flag explicitly overrides
`HYPERAI_RESOURCE`; it never falls back to another GPU. The immutable manifest records the selected resource.
It reuses the pinned Qwen 0.5B model, Capybara dataset and prepared environment, while increasing the workload:

| Setting | Standard | `--serious` |
|---|---:|---:|
| Paired seeds | 5 | 5 |
| Warm-up / measured phases per seed | 1 / 6 | 3 / 12 |
| Concurrent histories | 2 | 8 (four distinct prompts, repeated twice) |
| Prompt token limit | 48 | 768 |
| Generated tokens per turn | 16 | 64 |
| Turns per phase | 4 | 6 |
| Context limit | 512 | 1536 |

Both modes use rank-8 nonzero LoRA, real adapter updates between phases, fixed CPU thread counts, alternating side
order, exact token hashes, frozen-weight drift checks, and separately instrumented profiler captures. The first seed
also compares full next-token distributions with local unmerged and BF16-merged references and a stale-adapter control.
This is a more demanding systems workload with synthetic CPU feedback, not a scored reasoning/tool-use environment.
Longer runs and larger batches do not by themselves establish downstream quality or reproducibility across hardware.
Compare base and head **on the same GPU**; never pool 3090 and 5090 results. A baseline rounding defect may produce
different tokens: the comparator still rejects the comparison, even when timing improves and candidate drift is zero.

```bash
python tools/pr_benchmark/controller.py compare \
  --repo OWNER/REPO --base-ref BASE_SHA --ref HEAD_SHA --profile vllm-rollout --serious \
  --env-file ~/.config/trl-bench/env --timeout-minutes 25
```

Deadlines include initialization, timing, diagnostics and two extra profiling starts. Use `--dry-run` to review the
manifest before spending compute; the usual daily reservation limit still applies. Serious mode is restricted to
`vllm-rollout`; it cannot silently turn an SFT or diagnostic-only request into a different experiment.

## Before/after PyTorch and HTA profiles

Every comparison now runs a separate profiling pair after all unprofiled timing samples, using the first paired seed
and the same immutable commits, model, tokenized data, and prepared environment. Each capture follows the workload's
normal warm-up and covers two complete rollout phases (including all turns, weight sync, CPU tool feedback, and final
sleep), or two SFT optimizer steps. Profiles with fewer available steps capture those steps. Initialization, held-out
scoring, and the extra LoRA parity diagnostic are outside the captured window. SFT repeats its normal warm-up training
before the two captured steps. These instrumented runs never enter the latency/quality comparator or its timing totals.
Allow for two additional model starts and profiler overhead inside the existing job deadline and reservation.

Update the local controller dependencies with `pip install -r tools/pr_benchmark/requirements-controller.txt`.
HTA runs **locally**, after downloading the traces; no GPU environment rebuild is needed. PyTorch collects the trace;
HTA analyzes it and writes an augmented trace with queue-length and memcpy-bandwidth counters. It is not a second
independent profiler. The controller records the analysis package versions alongside the pinned HTA version.

The run directory retains:

```text
base-profile.zip / head-profile.zip       # checksummed original capture archives
profiles/report.md                       # before/after temporal breakdown and artifact links
profiles/summary.json                    # complete HTA tables, observed memcpy bytes, analysis environment
profiles/kernel-deltas.csv               # per-kernel time changes (missing kernels remain explicit)
profiles/{base,head}/
  metadata.json                          # commit, seed, workload, capture window, GPU packages
  trace.json.gz                          # CPU/CUDA Chrome trace: operators, shapes, allocations
  trace_with_counters.json.gz             # HTA queue-length and memory-copy bandwidth timeline
  memory-events.json.gz                   # timestamped PyTorch allocation/deallocation events
  operators-*.txt                         # all operators grouped by input shape, CPU/CUDA time and memory
  hta-*.csv                              # kernels, kernel types, temporal/idle breakdown, idle intervals,
                                         # CPU launch/GPU delays, memory bandwidth, queue length
```

Open either Chrome trace in [Perfetto](https://ui.perfetto.dev/) or `chrome://tracing`. The rollout trace annotates
whole phases, each turn, CPU tools, sync, tensor loading, wake, cache reset, and sleep. HTA's memory-copy bandwidth is
for observed memcpy/memset operations; it does not measure bandwidth within compute kernels. The existing
`weight_transfer_bytes` counter still measures logical tensor payload, not PCIe traffic. HTA 0.5 does not recognize
CUDA Graph launches in its launch/queue analyses; the report flags that limitation whenever replays are present.
Use the raw GPU timeline and temporal/kernel totals for graph execution. Full Python call-tree collection is disabled: it produced a 160 MB compressed trace and excessive postprocessing
on this small workload. Operator/shape/memory profiling still adds overhead: use these traces to locate bottlenecks, then validate changes with the unprofiled
paired measurements. One captured seed is diagnostic evidence, not a confidence interval or quality qualification.

Artifacts are downloaded and analyzed before the quality gate, so traces remain available when the gate rejects a
candidate. Missing CUDA kernels (for example, unavailable CUPTI), mismatched provenance, or invalid downloads fail the
run rather than silently claiming a complete profile. Raw archives survive an HTA analysis failure. Downloads are
bounded to 256 MB per archive and 2 GB expanded per side; raw outputs also remain in the Hyper.ai job. Profiling or
analysis failure never erases already completed unprofiled results, but the run cannot publish a passing status.

Rerun HTA locally without spending GPU time:

```bash
python tools/pr_benchmark/analyze_profiles.py ~/.local/state/trl-pr-benchmark/runs/RUN_ID
```

Reference: [PyTorch Profiler](https://docs.pytorch.org/docs/stable/profiler.html) and
[HTA trace analysis](https://hta.readthedocs.io/en/latest/source/api/trace_analysis_api.html).
