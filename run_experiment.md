# run_experiment

`run_experiment.py` runs one simulation sweep and writes it to `experiments/<YYYYMMDD-HHMMSS>_<comment>/`.

```bash
python run_experiment.py --comment test
python run_experiment.py --comment azure --experiment trace-rps
python run_experiment.py --comment azure --experiment trace-rps --qos-config qos.yaml
python run_experiment.py --comment azure-x4 --experiment trace-rps --trace-dir traces/azure_conv_poisson_x4
```

`--config` defaults to `prediction.yaml`. `--qos-config` defaults to `qos.yaml` and is used by `trace-rps`. `--trace-dir` selects an existing trace directory for `trace-rps`. The script calls `aisimulate` from `PATH`, or `python/aisimulate/.venv/bin/aisimulate`.

## Experiments

`max-batched-tokens` (the default) keeps the base traffic and sweeps `max_batched_tokens` from 100 to 5000 in steps of 500.

`trace-rps` keeps the base engine, including `max_batched_tokens`, and replays one Mooncake file per rate at 0.5, 1.0, …, 6.0 requests per second. Without `--trace-dir`, it rewrites `traces/azure_conv_poisson/rps_<rate>.jsonl` from `AzureLLMInferenceTrace_conv.csv` and labels every request from `--qos-config`. One seeded draw assigns each CSV row, so the same request stays in one class at every rate. Each Mooncake row records that class in `qos`. `qos-assignment.json` maps replay's synthetic session id `request_<line>` to the class. Class probabilities must be positive and sum to 1. `slo` is optional, and any of `ttft_ms`, `itl_ms`, and `ttlt_ms` may be omitted.

`--trace-dir` replays that directory as it stands. It must contain `rps_<rate>.jsonl` for every rate and a `qos-assignment.json` whose labels are the class names in `--qos-config`. The script does not rewrite those files, so a changed QoS seed or class probability does not relabel them. SLOs still come from `--qos-config`. `traces/azure_conv_poisson_x4` is the same arrivals and classes with input and output lengths multiplied by 4.

Arrivals are one Poisson process (seed 42), separate from the QoS seed: request `i` at rate `λ` is stamped `1000 * gaps[i] / λ` milliseconds, and the first request arrives at 0. Replay uses Mooncake `trace_timestamps` and runs each file to the end. The full trace estimates at about 3.8 GB, so each point sets `execution.resources.reserve_memory_gb: 0` and `available_memory_fraction: 1` and uses the memory that is free when the run starts. Prediction passes `--capture-per-request`.

P50 and P99 are computed inside each class from that class's completed requests. Each request's ITL P50 and ITL P99 are theX percentiles of that request's token gaps (`itl_p50_ms`, `itl_p99_ms`). The class P50 is the P50 of those ITL P50 samples, and the class P99 is the P99 of those ITL P99 samples. Requests with one output token have no gaps and are left out of ITL. TTFT is `ttft_ms`. TTLT is arrival to last token (`e2e_latency_ms`). The rank is `round((n - 1) * percentile / 100)`, matching the replay report.

## Outputs

Each run saves `base.yaml`, `sweep.csv`, and `p99.pdf`. A `trace-rps` run also saves `qos.yaml`. Point directories are `points/<max_batched_tokens>/` or `points/rps-<rate>/`, each with `prediction.yaml`, `prediction.json`, and `predict.log`. A `trace-rps` point also contains `requests.jsonl`.

`max-batched-tokens` records the median and P99 of inter-token latency and time to first token. `p99.pdf` is a 2×2 grid with the median on the first row and P99 on the second.

`trace-rps` writes one CSV row per rate and class: `requests_per_second`, `qos`, `status`, `n_requests`, then the P50 and P99 of ITL, TTFT, and TTLT. `p99.pdf` is a 2×3 grid with one solid line per class. A dashed line in the class color marks an SLO on both rows of that metric when that class sets the bound.

A failed point is kept in the CSV and the sweep continues. The plot is written when at least one point succeeds.