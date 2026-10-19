# run_experiment

`run_experiment.py` runs one simulation sweep and writes it to `experiments/<YYYYMMDD-HHMMSS>_<comment>/`.

```bash
python run_experiment.py --comment test
python run_experiment.py --comment azure --experiment trace-rps
# qos-disagg.yaml: a copy of qos.yaml with prediction: prediction-disagg.yaml
python run_experiment.py --comment pd-policy-cmp --experiment trace-rps --qos-config qos-disagg.yaml
python run_experiment.py --replot experiments/20261019-132817_policy-cmp
```

## Configuration

`--qos-config` (default `qos.yaml`) is the experiment's one configuration file. It names the prediction file, the traffic, the policies, and the QoS classes:

```yaml
prediction: prediction.yaml                 # required
traffic:
  trace_dir: traces/azure_conv_poisson_x4   # optional
  rates: {min: 0.5, max: 6.0, step: 0.5}    # optional
policies: [fcfs, edf, sjf]                  # optional, default [fcfs]
seed: 42
classes:
  - name: qos-1
    probability: 0.3
    slo: {ttft_ms: 6000, itl_ms: 50}
```

| Field | Default | Meaning |
|---|---|---|
| `prediction` | Required | Base prediction YAML. Aggregated or disaggregated; see [Disaggregated configs](#disaggregated-configs). `prediction-disagg.yaml` is a disaggregated version of `prediction.yaml`. |
| `traffic.trace_dir` | Unset | Existing Mooncake directory for `trace-rps`. Unset regenerates `traces/azure_conv_poisson` from the Azure CSV. |
| `traffic.rates` | `{min: 0.5, max: 6.0, step: 0.5}` | Load points for `trace-rps`, in requests per second: `min`, `min + step`, … up to `max`. All three are required and positive. |
| `policies` | `[fcfs]` | Waiting-queue policies for `trace-rps`, from `fcfs`, `edf`, and `sjf`. |
| `seed` | `42` | Seed of the class draw when traces are regenerated. |
| `classes` | Required | QoS classes: `name`, `probability`, and optional `slo` (`ttft_ms`, `itl_ms`, `ttlt_ms`). |

Paths are relative to the repository root. Unknown fields are rejected. `max-batched-tokens` uses only `prediction`. The run copies the prediction file to `base.yaml` and the QoS config to `qos.yaml`. The script calls `aisimulate` from `PATH`, or `python/aisimulate/.venv/bin/aisimulate`.

`--replot DIR` redraws an existing experiment's plots with the script's current plot settings. It runs no simulations and recomputes no metrics: it reads the saved `sweep.csv` and, for `trace-rps`, the saved `qos.yaml`, then rewrites the same PDFs. It cannot be combined with any other option. A `trace-rps` `sweep.csv` written before `slo_violation_ratio` existed gets that column once, computed from each point's `requests.jsonl` and the trace that point replayed, and `sweep.csv` is rewritten with the other values unchanged. A disaggregated `sweep.csv` written before the per-pool utilization columns gets those once the same way (see [Disaggregated configs](#disaggregated-configs)). A column the CSV lacks otherwise leaves its subplot empty. `trace-rps` CSVs without a `policy` column are not supported.

## Experiments

`max-batched-tokens` (the default) keeps the base traffic and sweeps `max_batched_tokens` from 100 to 5000 in steps of 500.

`trace-rps` keeps the base engine, including `max_batched_tokens`, and replays one Mooncake file per rate in `traffic.rates`. A rate is written with one decimal when that is exact (`rps_2.0.jsonl`, `points/<policy>/rps-2.0/`) and in its shortest form otherwise (`rps_0.25.jsonl`). Without `traffic.trace_dir`, it rewrites `traces/azure_conv_poisson/rps_<rate>.jsonl` for those rates from `AzureLLMInferenceTrace_conv.csv` and labels every request from the QoS config. One seeded draw assigns each CSV row, so the same request stays in one class at every rate. Each Mooncake row records that class in `qos`. `qos-assignment.json` maps replay's synthetic session id `request_<line>` to the class. Class probabilities must be positive and sum to 1. `slo` is optional, and any of `ttft_ms`, `itl_ms`, and `ttlt_ms` may be omitted.

`traffic.trace_dir` replays that directory as it stands. It must contain a `qos-assignment.json` whose labels are the class names in the QoS config, and at least one `rps_<rate>.jsonl`. The script does not rewrite those files, so a changed QoS seed or class probability does not relabel them. SLOs still come from the QoS config. A rate with no file in the directory is derived from the lowest rate present by rescaling every timestamp by `source_rate / rate`; all rate files share the same unit arrivals, so this matches what the generator would write. Derived files go to the experiment's `traces/` directory and the trace directory is never modified. `traces/azure_conv_poisson_x2` and `traces/azure_conv_poisson_x4` are the same arrivals and classes with input and output lengths multiplied by 2 and 4.

Arrivals are one Poisson process (seed 42), separate from the QoS seed: request `i` at rate `λ` is stamped `1000 * gaps[i] / λ` milliseconds, and the first request arrives at 0. Replay uses Mooncake `trace_timestamps` and runs each file to the end. The full trace estimates at about 3.8 GB, so each point sets `execution.resources.reserve_memory_gb: 0` and `available_memory_fraction: 1` and uses the memory that is free when the run starts. Prediction passes `--capture-per-request`.

## Scheduling policies

`trace-rps` runs the whole rate sweep once per entry in `policies`, in the order given. Every policy replays the same arrivals and classes.

- `fcfs` replays the trace unchanged with the engine's default `queue_policy: fcfs`. Requests are admitted in arrival order, and under KV pressure the newest running request is preempted.
- `edf` is earliest deadline first. A request's deadline is its arrival timestamp plus its class's `slo.ttft_ms`, or `slo.ttlt_ms` when the class sets no TTFT bound. A class with neither is rejected before the run starts.
- `sjf` is shortest job first by prompt length: the key is the request's `input_length`.

For `edf` and `sjf` the script writes the point's trace to `trace.jsonl` with each row's `priority` set to that key, rounded to whole milliseconds for `edf`. It then runs the point with `engine.workers.aggregated.scheduler.queue_policy: priority`, or `engine.workers.prefill.scheduler.queue_policy: priority` for a disaggregated config. The engine admits the lowest key first and keeps arrival order among equal keys. A preempted request rejoins the queue by its key, and the preemption victim is the running request with the largest key. See the [vLLM queue policy](docs/replay/engine/workers.md#vllm-queue-policy). Admission still stops at the first waiting request that does not fit in the KV cache, as in vLLM, so a later request with a worse key cannot overtake a blocked one.

## Metrics

P50 and P99 are computed inside each class from that class's completed requests. Each request's ITL P50 and ITL P99 are theX percentiles of that request's token gaps (`itl_p50_ms`, `itl_p99_ms`). The class P50 is the P50 of those ITL P50 samples, and the class P99 is the P99 of those ITL P99 samples. Requests with one output token have no gaps and are left out of ITL. TTFT is `ttft_ms`. TTLT is arrival to last token (`e2e_latency_ms`). The rank is `round((n - 1) * percentile / 100)`, matching the replay report.

## Outputs

Each run saves `base.yaml` and `sweep.csv`. A `max-batched-tokens` run saves `p99.pdf` and has point directories `points/<max_batched_tokens>/`. A `trace-rps` run also saves `qos.yaml`, one `qos-<class>.pdf` per class, and per policy `points/<policy>/p99.pdf`, with point directories `points/<policy>/rps-<rate>/`. Each point directory has `prediction.yaml`, `prediction.json`, and `predict.log`. A `trace-rps` point also contains `requests.jsonl`, and an `edf` or `sjf` point contains the `trace.jsonl` it replayed.

`max-batched-tokens` records the median and P99 of inter-token latency and time to first token. `p99.pdf` is a 2×2 grid with the median on the first row, P99 on the second, and a logarithmic latency axis.

`trace-rps` writes one CSV row per policy, rate, and class: `policy`, `requests_per_second`, `qos`, `status`, `n_requests`, then the P50 and P99 of ITL, TTFT, and TTLT, then `p50_queue_wait_ms`, `p99_queue_wait_ms`, `preemptions`, and `slo_violation_ratio`, then `gpu_mfu`, `kv_cache_mean_utilization`, `kv_cache_peak_utilization`, and `host_memory_utilization`, then the per-pool columns `gpu_mfu_prefill`, `kv_cache_mean_utilization_prefill`, `kv_cache_peak_utilization_prefill`, `gpu_mfu_decode`, `kv_cache_mean_utilization_decode`, and `kv_cache_peak_utilization_decode`, which are empty for an aggregated run. Queue wait is each request's `first_admit_ms - arrival_time_ms`, the time before the scheduler first admits it, and its P50 and P99 are taken inside the class. `preemptions` is the class's total `readmission_count` over its completed requests; a disaggregated request's admissions to prefill and to decode are not readmissions. `slo_violation_ratio` is the fraction of the class's completed requests that miss any SLO the class sets: `ttft_ms` against the request's TTFT, `ttlt_ms` against its `e2e_latency_ms`, and `itl_ms` against its mean ITL (`itl_ms`). A request with one output token has no ITL, and a class with no SLO has a ratio of 0. The utilization columns are the same on every class row for a policy and rate. `gpu_mfu` is model FLOPs utilization: useful dense-model FLOPs from the completed requests in `requests.jsonl`, divided by the GPU's peak tensor FLOPs (`bfloat16_tc_flops`, or `fp8_tc_flops` for an FP8 checkpoint) times the GPU count times the simulated `duration_ms`. Each request counts its Q, K, V, O, and gated-MLP projections for every prefill token past the reused prefix and every decode step after the first token, causal attention over every earlier position, and the LM head for every sampled token. Preemption recompute is not counted, and mixture-of-experts models leave it empty. The KV cache columns divide resident KV blocks by the total KV blocks of every attention rank. A request holds `ceil(tokens / block_size)` blocks from admission to its last token: its input grows linearly from the reused prefix until the first token, then its output grows linearly until the last. `block_size` is the engine's `kv_cache.block_size` (vLLM default 64), not the trace hash block. The mean is weighted by simulated time, and the peak is not clamped, so a peak above 100% means the reconstruction holds more than the cache could. `host_memory_utilization` is the run's peak process RSS divided by the host's total memory.

`qos-<class>.pdf` compares the policies for one class on a 4×3 grid, one solid line per policy. The first two rows are the P50 and P99 of ITL, TTFT, and TTLT on logarithmic axes. A gray dashed line marks each SLO the class sets. The third row is each policy's GPU MFU and peak KV cache occupancy in percent at each rate, then the class's SLO violation ratio in percent under each policy. In a disaggregated run, the MFU and KV subplots draw each policy twice in its color: the prefill pool solid and the decode pool dashed. Preemptions are not plotted; they stay in `sweep.csv`. The grid is 3×3. `points/<policy>/p99.pdf` uses the same grid for one policy, with one solid line per class and a dashed line in the class color for that class's SLO; its utilization subplots have a single line (solid and dashed in a disaggregated run), and its SLO violation subplot has one line per class. In both layouts, both TTFT rows use a fixed range from 10^2 to 10^5 ms. ITL and TTLT each use one shared logarithmic range for P50 and P99, the powers of ten that contain that metric's samples and SLOs.

## Disaggregated configs

`prediction-disagg.yaml` is `prediction.yaml` split into one prefill and one decode worker with the same model, hardware, backend, and GPU count. Point `prediction` at it to run the disaggregated version. A config with `engine.mode: disaggregated` must define `engine.workers.prefill` and `engine.workers.decode`. Both experiments then change the prefill workers only. `max-batched-tokens` sweeps `workers.prefill.scheduler.max_batched_tokens`, and `trace-rps` sets each policy's `queue_policy` on `workers.prefill.scheduler`. Decode workers keep the config's settings and admit KV handoffs in arrival order. `edf` and `sjf` need `engine.backend: vllm`.

`gpu_mfu` divides by the peak tensor FLOPs of every prefill and decode GPU, using each role's own `hardware`. The `_prefill` and `_decode` columns measure each pool on its own. `gpu_mfu_prefill` counts prefill FLOPs (projections past the reused prefix, causal attention over the input, and the LM head for the first token) over the prefill GPUs' peak; `gpu_mfu_decode` counts the decode steps after the first token over the decode GPUs' peak. The KV cache columns treat the two roles as separate pools, each measured against its own total blocks. The prefill pool holds a request's input from `prefill_admit_ms`, growing from the reused prefix until `source_held_ms`, and keeps it until the transfer releases it at `source_released_ms`. The decode pool holds the input from `destination_reserved_ms`, then the growing output from the first token to the last. The KV columns without a pool suffix report the higher of the two pools, and `gpu_mfu` covers every GPU. `--replot` of a disaggregated experiment whose `sweep.csv` predates the per-pool columns computes them once from the point folders and adds them to `sweep.csv`; this reads every point's `requests.jsonl` and can take minutes.

A failed point is kept in the CSV and the sweep continues. `sweep.csv` and a policy's `p99.pdf` are written as soon as that policy's sweep finishes. The comparison plots are written when at least one point succeeds.