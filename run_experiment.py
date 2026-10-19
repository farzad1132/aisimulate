#!/usr/bin/env python3
"""Run a named simulation experiment under a timestamped directory.

Each invocation takes a required comment and writes every config, report, and
plot for that run into experiments/<YYYYMMDD-HHMMSS>_<comment>/.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENTS_DIR = REPO_ROOT / "experiments"

MIN_BATCHED_TOKENS = 100
MAX_BATCHED_TOKENS = 5000
BATCHED_TOKENS_STEP = 500

MIN_REQUESTS_PER_SECOND = 0.5
MAX_REQUESTS_PER_SECOND = 6.0
REQUESTS_PER_SECOND_STEP = 0.5
TRACE_BLOCK_SIZE = 512
POISSON_SEED = 42
AZURE_TRACE_CSV = REPO_ROOT / "AzureLLMInferenceTrace_conv.csv"
POISSON_TRACE_DIR = REPO_ROOT / "traces" / "azure_conv_poisson"

AGGREGATE_LATENCY_FIELDS = (
    ("ITL", "median_itl_ms", "p99_itl_ms"),
    ("TTFT", "median_ttft_ms", "p99_ttft_ms"),
)
QOS_PLOT_FIELDS = (
    ("ITL", "itl_ms", "p50_itl_ms", "p99_itl_ms"),
    ("TTFT", "ttft_ms", "median_ttft_ms", "p99_ttft_ms"),
    ("TTLT", "ttlt_ms", "median_ttlt_ms", "p99_ttlt_ms"),
)
SLO_KEYS = frozenset({"ttft_ms", "itl_ms", "ttlt_ms"})
POLICIES = ("fcfs", "edf", "sjf")
PRIORITY_RANGE = (-(2**31), 2**31 - 1)
QOS_CSV_FIELDS = [
    "policy",
    "requests_per_second",
    "qos",
    "status",
    "n_requests",
    "p50_itl_ms",
    "p99_itl_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "median_ttlt_ms",
    "p99_ttlt_ms",
    "p50_queue_wait_ms",
    "p99_queue_wait_ms",
    "preemptions",
    "slo_violation_ratio",
    "gpu_mfu",
    "kv_cache_mean_utilization",
    "kv_cache_peak_utilization",
    "host_memory_utilization",
    "gpu_mfu_prefill",
    "kv_cache_mean_utilization_prefill",
    "kv_cache_peak_utilization_prefill",
    "gpu_mfu_decode",
    "kv_cache_mean_utilization_decode",
    "kv_cache_peak_utilization_decode",
]
UTILIZATION_PLOT_FIELDS = (
    ("GPU MFU", "gpu_mfu"),
    ("KV cache peak", "kv_cache_peak_utilization"),
)
P99_PLOT = "p99.pdf"
PROBABILITY_SUM_TOLERANCE = 1e-6


def slugify(comment: str) -> str:
    """Turn a comment into a directory-safe suffix."""

    slug = re.sub(r"[^A-Za-z0-9]+", "-", comment.strip()).strip("-")
    if not slug:
        raise SystemExit("comment must contain at least one letter or digit")
    return slug


def batched_token_grid(
    min_value: int = MIN_BATCHED_TOKENS,
    max_value: int = MAX_BATCHED_TOKENS,
    step: int = BATCHED_TOKENS_STEP,
) -> list[int]:
    """Return min, min+step, ... while the value is at most max."""

    if step <= 0:
        raise ValueError("step must be positive")
    values: list[int] = []
    value = min_value
    while value <= max_value:
        values.append(value)
        value += step
    if not values:
        raise ValueError("sweep grid is empty")
    return values


def rps_grid(
    min_value: float = MIN_REQUESTS_PER_SECOND,
    max_value: float = MAX_REQUESTS_PER_SECOND,
    step: float = REQUESTS_PER_SECOND_STEP,
) -> list[float]:
    """Return min, min+step, ... while the value is at most max."""

    if step <= 0 or min_value <= 0 or max_value < min_value:
        raise ValueError("step must be positive and the rate bounds must be positive")
    count = round((max_value - min_value) / step)
    return [round(min_value + index * step, 10) for index in range(count + 1)]


def rate_label(rate: float) -> str:
    """Rate as written in trace and point names: one decimal when exact, else the shortest form."""

    return f"{rate:.1f}" if abs(round(rate, 1) - rate) < 1e-9 else f"{rate:g}"


def read_azure_requests(path: Path) -> list[tuple[int, int]]:
    """Return (context tokens, generated tokens) for every CSV row."""

    if not path.is_file():
        raise SystemExit(f"trace CSV not found: {path}")
    requests: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"ContextTokens", "GeneratedTokens"} <= set(reader.fieldnames):
            raise SystemExit(f"{path} must have ContextTokens and GeneratedTokens columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                context = int(row["ContextTokens"])
                generated = int(row["GeneratedTokens"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"{path}:{line_number} has a non-integer token count") from exc
            if context <= 0 or generated <= 0:
                raise SystemExit(f"{path}:{line_number} token counts must be positive")
            requests.append((context, generated))
    if not requests:
        raise SystemExit(f"{path} did not contain any requests")
    return requests


def cumulative_unit_gaps(count: int, seed: int = POISSON_SEED) -> list[float]:
    """Arrival times of a unit-rate Poisson process, with the first arrival at 0."""

    rng = random.Random(seed)
    arrivals = [0.0]
    clock = 0.0
    for _ in range(count - 1):
        clock += rng.expovariate(1.0)
        arrivals.append(clock)
    return arrivals


def _available_traces(trace_dir: Path) -> dict[float, Path]:
    """Every ``rps_<rate>.jsonl`` in a trace directory, keyed by rate."""

    available: dict[float, Path] = {}
    for path in trace_dir.glob("rps_*.jsonl"):
        try:
            rate = float(path.name[len("rps_") : -len(".jsonl")])
        except ValueError:
            continue
        if rate > 0:
            available[rate] = path
    return available


def _rescale_trace(source: Path, source_rate: float, rate: float, destination: Path) -> None:
    """Copy a Mooncake trace with arrivals rescaled from ``source_rate`` to ``rate``."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    scale = source_rate / rate
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line in reader:
            if not line.strip():
                continue
            record = json.loads(line)
            record["timestamp"] = record["timestamp"] * scale
            writer.write(json.dumps(record, separators=(",", ":")) + "\n")


def load_prepared_traces(
    trace_dir: Path,
    classes: list[QosClass],
    rates: list[float],
    derived_dir: Path | None,
) -> tuple[dict[str, str], list[tuple[float, Path]]]:
    """Replay an existing rate directory instead of rewriting it from the CSV.

    ``qos-assignment.json`` supplies the class of each ``request_<line>``.
    Those labels must be the class names in the QoS config. SLOs still come
    from that config. A rate with no ``rps_<rate>.jsonl`` in the directory is
    derived into ``derived_dir`` by rescaling the arrivals of the lowest rate
    present; every rate file shares the same unit arrivals, so only timestamps
    change. With ``derived_dir`` None, nothing is written and only the checks run.
    """

    if not trace_dir.is_dir():
        raise SystemExit(f"trace directory not found: {trace_dir}")
    assignment_path = trace_dir / "qos-assignment.json"
    if not assignment_path.is_file():
        raise SystemExit(f"{trace_dir} is missing qos-assignment.json")
    raw = json.loads(assignment_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise SystemExit(f"{assignment_path} must be a non-empty JSON object")
    class_names = {item.name for item in classes}
    assignment: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str) or value not in class_names:
            raise SystemExit(f"{assignment_path} has a label outside the QoS config")
        assignment[key] = value
    missing = class_names - set(assignment.values())
    if missing:
        raise SystemExit(f"{assignment_path} has no requests for {', '.join(sorted(missing))}")

    available = _available_traces(trace_dir)
    if not available:
        raise SystemExit(f"{trace_dir} has no rps_<rate>.jsonl file to replay or rescale")
    source_rate = min(available)
    sample = available[source_rate]
    count = 0
    with sample.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle, start=1):
            record = json.loads(line)
            session_id = f"request_{count}"
            label = assignment.get(session_id)
            if label is None or record.get("qos") != label:
                raise SystemExit(f"{sample}:{count} does not match {assignment_path}")
    if count != len(assignment):
        raise SystemExit(f"{sample} has {count} requests but {assignment_path} has {len(assignment)}")

    traces: list[tuple[float, Path]] = []
    for rate in rates:
        path = trace_dir / f"rps_{rate_label(rate)}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                lines = sum(1 for _ in handle)
            if lines != count:
                raise SystemExit(f"{path} has {lines} requests but {sample.name} has {count}")
        elif derived_dir is not None:
            path = derived_dir / path.name
            _rescale_trace(sample, source_rate, rate, path)
            print(f"derived {path.relative_to(REPO_ROOT)} from {sample.name}")
        traces.append((rate, path))
    return assignment, traces


def write_poisson_traces(
    classes: list[QosClass],
    seed: int,
    rates: list[float],
    csv_path: Path = AZURE_TRACE_CSV,
    output_dir: Path = POISSON_TRACE_DIR,
) -> tuple[dict[str, str], list[tuple[float, Path]]]:
    """Write one Mooncake JSONL per rate. Request lengths stay in CSV order.

    Each request is labeled once. ``qos-assignment.json`` maps the synthetic
    session id ``request_<line>`` that replay reports back to that label.
    """

    requests = read_azure_requests(csv_path)
    labels = assign_qos(len(requests), classes, seed)
    assignment = {f"request_{index + 1}": label for index, label in enumerate(labels)}
    arrivals = cumulative_unit_gaps(len(requests))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qos-assignment.json").write_text(json.dumps(assignment) + "\n", encoding="utf-8")
    written = [(rate, output_dir / f"rps_{rate_label(rate)}.jsonl") for rate in rates]
    handles = [path.open("w", encoding="utf-8") for _, path in written]
    try:
        next_hash = 1
        for index, (context, generated) in enumerate(requests):
            block_count = math.ceil(context / TRACE_BLOCK_SIZE)
            hash_ids = list(range(next_hash, next_hash + block_count))
            next_hash += block_count
            request_id = f"azure-{index + 1}"
            for handle, (rate, _) in zip(handles, written, strict=True):
                record = {
                    "request_id": request_id,
                    "timestamp": 1000.0 * arrivals[index] / rate,
                    "input_length": context,
                    "output_length": generated,
                    "hash_ids": hash_ids,
                    "qos": labels[index],
                }
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    finally:
        for handle in handles:
            handle.close()
    return assignment, written


def find_aisimulate() -> Path:
    found = shutil.which("aisimulate")
    if found:
        return Path(found)
    candidate = REPO_ROOT / "python" / "aisimulate" / ".venv" / "bin" / "aisimulate"
    if candidate.is_file():
        return candidate
    raise SystemExit("aisimulate was not found on PATH or at python/aisimulate/.venv/bin/aisimulate")


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a YAML mapping")
    return data


class QosClass(NamedTuple):
    name: str
    probability: float
    slo: dict[str, float]


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SystemExit(f"{label} must be a positive finite number")
    return float(value)


def load_qos_config(path: Path) -> tuple[int, list[QosClass]]:
    """Return the assignment seed and classes. Missing SLO fields stay unset."""

    data = load_config(path)
    seed = data.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SystemExit(f"{path} seed must be an integer")
    raw_classes = data.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise SystemExit(f"{path} classes must be a non-empty list")
    classes: list[QosClass] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_classes, start=1):
        if not isinstance(raw, dict):
            raise SystemExit(f"{path} class {index} must be a mapping")
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(f"{path} class {index} needs a name")
        if name in seen:
            raise SystemExit(f"{path} repeats class {name}")
        seen.add(name)
        probability = _positive_number(raw.get("probability"), f"{path} class {name} probability")
        slo: dict[str, float] = {}
        raw_slo = raw.get("slo", {})
        if raw_slo is None:
            raw_slo = {}
        if not isinstance(raw_slo, dict):
            raise SystemExit(f"{path} class {name} slo must be a mapping")
        unknown = set(raw_slo) - SLO_KEYS
        if unknown:
            raise SystemExit(f"{path} class {name} has unknown slo fields: {', '.join(sorted(unknown))}")
        for key in SLO_KEYS:
            if key in raw_slo:
                slo[key] = _positive_number(raw_slo[key], f"{path} class {name} slo.{key}")
        classes.append(QosClass(name, probability, slo))
    total = sum(item.probability for item in classes)
    if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise SystemExit(f"{path} class probabilities must sum to 1")
    return seed, classes


class ExperimentConfig(NamedTuple):
    prediction: Path
    trace_dir: Path | None
    rates: list[float]
    policies: list[str]
    seed: int
    classes: list[QosClass]


EXPERIMENT_KEYS = frozenset({"prediction", "traffic", "policies", "seed", "classes"})
TRAFFIC_KEYS = frozenset({"trace_dir", "rates"})
RATE_KEYS = ("min", "max", "step")


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"{label} must be a path")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_experiment_config(path: Path) -> ExperimentConfig:
    """The whole experiment from the QoS config: prediction file, traces, rates, policies, and classes."""

    seed, classes = load_qos_config(path)
    data = load_config(path)
    unknown = set(data) - EXPERIMENT_KEYS
    if unknown:
        raise SystemExit(f"{path} has unknown fields: {', '.join(sorted(unknown))}")
    if "prediction" not in data:
        raise SystemExit(f"{path} needs prediction, the base prediction YAML")
    prediction = _repo_path(data["prediction"], f"{path} prediction")
    if not prediction.is_file():
        raise SystemExit(f"prediction file not found: {prediction}")

    traffic = data.get("traffic") or {}
    if not isinstance(traffic, dict):
        raise SystemExit(f"{path} traffic must be a mapping")
    unknown = set(traffic) - TRAFFIC_KEYS
    if unknown:
        raise SystemExit(f"{path} traffic has unknown fields: {', '.join(sorted(unknown))}")
    trace_dir = None
    if traffic.get("trace_dir") is not None:
        trace_dir = _repo_path(traffic["trace_dir"], f"{path} traffic.trace_dir")
        if not trace_dir.is_dir():
            raise SystemExit(f"trace directory not found: {trace_dir}")
    raw_rates = traffic.get("rates")
    if raw_rates is None:
        rates = rps_grid()
    else:
        if not isinstance(raw_rates, dict) or set(raw_rates) != set(RATE_KEYS):
            raise SystemExit(f"{path} traffic.rates must be a mapping with min, max, and step")
        bounds = [_positive_number(raw_rates[key], f"{path} traffic.rates.{key}") for key in RATE_KEYS]
        if bounds[1] < bounds[0]:
            raise SystemExit(f"{path} traffic.rates.max must be at least min")
        rates = rps_grid(*bounds)

    policies = parse_policies(data.get("policies", ["fcfs"]), f"{path} policies")
    return ExperimentConfig(prediction, trace_dir, rates, policies, seed, classes)


def assign_qos(count: int, classes: list[QosClass], seed: int) -> list[str]:
    """Draw one class per request. The same seed keeps a request in one class at every rate."""

    rng = random.Random(seed)
    thresholds: list[float] = []
    total = 0.0
    for item in classes:
        total += item.probability
        thresholds.append(total)
    labels: list[str] = []
    last = classes[-1].name
    for _ in range(count):
        draw = rng.random()
        chosen = last
        for item, threshold in zip(classes, thresholds, strict=True):
            if draw <= threshold:
                chosen = item.name
                break
        labels.append(chosen)
    return labels


def percentile(values: list[float], percent: float) -> float | None:
    """Match the replay report rank: round((n - 1) * percent / 100), half away from zero."""

    if not values:
        return None
    ordered = sorted(values)
    rank = math.floor((len(ordered) - 1) * percent / 100.0 + 0.5)
    return ordered[min(rank, len(ordered) - 1)]


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _policy_role(config: dict) -> str:
    """Worker role whose scheduler an experiment changes: ``aggregated``, or ``prefill`` in P/D."""

    engine = config.get("engine")
    if not isinstance(engine, dict):
        raise SystemExit("config has no engine mapping")
    mode = engine.get("mode", "aggregated")
    if mode == "aggregated":
        return "aggregated"
    if mode != "disaggregated":
        raise SystemExit(f"engine.mode {mode!r} is not supported; use aggregated or disaggregated")
    workers = engine.get("workers")
    if not isinstance(workers, dict) or not all(isinstance(workers.get(role), dict) for role in ("prefill", "decode")):
        raise SystemExit("a disaggregated config needs engine.workers.prefill and engine.workers.decode")
    return "prefill"


def _role_scheduler(config: dict) -> dict:
    role = _policy_role(config)
    worker = config["engine"].setdefault("workers", {}).setdefault(role, {})
    scheduler = worker.setdefault("scheduler", {})
    if not isinstance(scheduler, dict):
        raise SystemExit(f"engine.workers.{role}.scheduler must be a mapping")
    return scheduler


def set_max_batched_tokens(config: dict, value: int) -> None:
    _role_scheduler(config)["max_batched_tokens"] = value


def summary_metrics(report: dict) -> dict:
    summary = report.get("summary")
    if isinstance(summary, dict):
        return summary
    return report


def run_point(
    aisimulate: Path,
    config: dict,
    point_dir: Path,
    *,
    capture_per_request: bool = False,
    summary_fields: tuple[tuple[str, str, str], ...] = AGGREGATE_LATENCY_FIELDS,
) -> tuple[str, dict[str, float | None]]:
    """Predict one concrete config. A failed point does not stop the sweep."""

    point_dir.mkdir(parents=True, exist_ok=True)
    config_path = point_dir / "prediction.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    command = [
        str(aisimulate),
        "predict",
        "--stack",
        "engine",
        "--config",
        str(config_path),
        "--output-dir",
        str(point_dir),
        "--overwrite",
    ]
    if capture_per_request:
        command.append("--capture-per-request")
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    (point_dir / "predict.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    metric_names = [name for _, median_name, p99_name in summary_fields for name in (median_name, p99_name)]
    metrics: dict[str, float | None] = dict.fromkeys(metric_names)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        print(f"  failed: {detail}", file=sys.stderr)
        return "failed", metrics
    report_path = point_dir / "prediction.json"
    if not report_path.is_file():
        print("  failed: prediction.json was not written", file=sys.stderr)
        return "failed", metrics
    if capture_per_request and not (point_dir / "requests.jsonl").is_file():
        print("  failed: requests.jsonl was not written", file=sys.stderr)
        return "failed", metrics
    if not metric_names:
        return "ok", metrics
    summary = summary_metrics(json.loads(report_path.read_text(encoding="utf-8")))
    missing: list[str] = []
    for name in metric_names:
        value = summary.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[name] = float(value)
        else:
            missing.append(name)
    if missing:
        print(f"  failed: missing {', '.join(missing)}", file=sys.stderr)
        return "failed", metrics
    return "ok", metrics


def metric_row(x_column: str, x_value: object, metrics: dict[str, float | None], status: str) -> dict[str, object]:
    row: dict[str, object] = {x_column: x_value, "status": status}
    for _, median_name, p99_name in AGGREGATE_LATENCY_FIELDS:
        for name in (median_name, p99_name):
            row[name] = "" if metrics[name] is None else metrics[name]
    return row


def write_sweep_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_csv_fields(x_column: str) -> list[str]:
    fieldnames = [x_column]
    for _, median_name, p99_name in AGGREGATE_LATENCY_FIELDS:
        fieldnames.extend((median_name, p99_name))
    fieldnames.append("status")
    return fieldnames


def violates_slo(record: dict, slo: dict[str, float]) -> bool:
    """Whether a completed request misses any SLO its class sets.

    TTFT and TTLT compare the request's own values. ITL compares the request's
    mean inter-token latency; a request with one output token has no ITL.
    """

    bounds = (("ttft_ms", "ttft_ms"), ("ttlt_ms", "e2e_latency_ms"), ("itl_ms", "itl_ms"))
    for slo_key, record_key in bounds:
        bound = slo.get(slo_key)
        if bound is None:
            continue
        if slo_key == "itl_ms":
            output_length = _token_count(record.get("output_length"))
            if output_length is None or output_length <= 1:
                continue
        value = _finite_float(record.get(record_key))
        if value is not None and value > bound:
            return True
    return False


def qos_metric_rows(
    point_dir: Path,
    policy: str,
    rate: float,
    status: str,
    classes: list[QosClass],
    assignment: dict[str, str],
) -> list[dict[str, object]]:
    """P50 and P99 inside each class.

    ITL P50 is the P50 of each request's own ITL P50. ITL P99 is the P99 of
    each request's own ITL P99. Requests with one output token have no gaps.
    """

    samples: dict[str, dict[str, list[float]]] = {
        item.name: {"itl_p50_ms": [], "itl_p99_ms": [], "ttft_ms": [], "e2e_latency_ms": [], "queue_wait_ms": []}
        for item in classes
    }
    counts = dict.fromkeys((item.name for item in classes), 0)
    preemptions = dict.fromkeys((item.name for item in classes), 0)
    violations = dict.fromkeys((item.name for item in classes), 0)
    slos = {item.name: item.slo for item in classes}
    if status == "ok":
        unmatched = 0
        requests_path = point_dir / "requests.jsonl"
        with requests_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("terminal_status") != "completed":
                    continue
                session_id = record.get("session_id")
                label = assignment.get(session_id) if isinstance(session_id, str) else None
                bucket = samples.get(label) if label is not None else None
                if bucket is None:
                    unmatched += 1
                    continue
                counts[label] += 1
                # A P/D request is admitted once per pool; only readmissions are preemptions.
                readmissions = _token_count(record.get("readmission_count"))
                if readmissions:
                    preemptions[label] += readmissions
                arrival = _finite_float(record.get("arrival_time_ms"))
                admit = _finite_float(record.get("first_admit_ms"))
                if arrival is not None and admit is not None:
                    bucket["queue_wait_ms"].append(max(admit - arrival, 0.0))
                ttft = _finite_float(record.get("ttft_ms"))
                if ttft is not None:
                    bucket["ttft_ms"].append(ttft)
                ttlt = _finite_float(record.get("e2e_latency_ms"))
                if ttlt is not None:
                    bucket["e2e_latency_ms"].append(ttlt)
                if violates_slo(record, slos[label]):
                    violations[label] += 1
                output_length = record.get("output_length")
                if isinstance(output_length, int) and not isinstance(output_length, bool) and output_length > 1:
                    itl_p50 = _finite_float(record.get("itl_p50_ms"))
                    if itl_p50 is not None:
                        bucket["itl_p50_ms"].append(itl_p50)
                    itl_p99 = _finite_float(record.get("itl_p99_ms"))
                    if itl_p99 is not None:
                        bucket["itl_p99_ms"].append(itl_p99)
        if unmatched:
            raise SystemExit(f"{requests_path} has {unmatched} completed requests with no QoS class")
    utilizations: dict[str, float | None] = dict.fromkeys(
        (
            "gpu_mfu",
            "kv_cache_mean_utilization",
            "kv_cache_peak_utilization",
            "host_memory_utilization",
            *POOL_UTILIZATION_FIELDS,
        )
    )
    if status == "ok":
        utilizations.update(point_utilizations(point_dir))
        utilizations["host_memory_utilization"] = host_memory_utilization(point_dir)
    rows: list[dict[str, object]] = []
    for item in classes:
        bucket = samples[item.name]
        row: dict[str, object] = {
            "policy": policy,
            "requests_per_second": rate,
            "qos": item.name,
            "status": status,
            "n_requests": counts[item.name] if status == "ok" else "",
        }
        for name, value in utilizations.items():
            row[name] = "" if value is None else value
        itl_p50 = percentile(bucket["itl_p50_ms"], 50)
        itl_p99 = percentile(bucket["itl_p99_ms"], 99)
        row["p50_itl_ms"] = "" if itl_p50 is None else itl_p50
        row["p99_itl_ms"] = "" if itl_p99 is None else itl_p99
        for _, _, p50_name, p99_name in QOS_PLOT_FIELDS[1:]:
            sample_name = "ttft_ms" if p50_name == "median_ttft_ms" else "e2e_latency_ms"
            p50 = percentile(bucket[sample_name], 50)
            p99 = percentile(bucket[sample_name], 99)
            row[p50_name] = "" if p50 is None else p50
            row[p99_name] = "" if p99 is None else p99
        queue_p50 = percentile(bucket["queue_wait_ms"], 50)
        queue_p99 = percentile(bucket["queue_wait_ms"], 99)
        row["p50_queue_wait_ms"] = "" if queue_p50 is None else queue_p50
        row["p99_queue_wait_ms"] = "" if queue_p99 is None else queue_p99
        row["preemptions"] = preemptions[item.name] if status == "ok" else ""
        row["slo_violation_ratio"] = (
            violations[item.name] / counts[item.name] if status == "ok" and counts[item.name] else ""
        )
        rows.append(row)
    return rows


class _RoleWorker(NamedTuple):
    worker: dict
    hardware: str


def _engine_workers(config_path: Path) -> tuple[dict, dict[str, _RoleWorker], str]:
    """Engine mapping, each KV-holding role's worker and hardware, and the model.

    The roles are ``aggregated``, or ``prefill`` and ``decode`` in P/D.
    """

    config = load_config(config_path)
    engine = config.get("engine")
    if not isinstance(engine, dict):
        raise SystemExit(f"{config_path} has no engine mapping")
    roles = ("aggregated",) if _policy_role(config) == "aggregated" else ("prefill", "decode")
    workers = _mapping(engine, "workers")
    model = engine.get("model")
    if not isinstance(model, str):
        raise SystemExit(f"{config_path} needs engine.model")
    result: dict[str, _RoleWorker] = {}
    for role in roles:
        worker = _mapping(workers, role)
        hardware = worker.get("hardware") if isinstance(worker.get("hardware"), str) else engine.get("hardware")
        if not isinstance(hardware, str):
            raise SystemExit(f"{config_path} needs engine.hardware or engine.workers.{role}.hardware")
        result[role] = _RoleWorker(worker, hardware)
    return engine, result, model


def _mapping(parent: dict, key: str) -> dict:
    value = parent.get(key)
    return value if isinstance(value, dict) else {}


def _parallel_sizes(worker: dict) -> tuple[int, int]:
    """Total GPUs and KV-holding attention ranks across all replicas."""

    parallelism = _mapping(worker, "parallelism")
    sizes = {name: parallelism.get(name, 1) for name in ("replicas", "tensor", "pipeline", "attention_data")}
    for name, value in sizes.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SystemExit(f"parallelism.{name} must be a positive integer")
    ranks = sizes["replicas"] * sizes["attention_data"]
    return ranks * sizes["tensor"] * sizes["pipeline"], ranks


def _completed_requests(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("terminal_status") == "completed":
                records.append(record)
    return records


def _token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _reused_input_tokens(record: dict) -> int:
    history = record.get("admission_history")
    if isinstance(history, list) and history and isinstance(history[0], dict):
        reused = _token_count(history[0].get("reused_input_tokens"))
        if reused is not None:
            return reused
    return 0


def _position_sum(low: int, high: int) -> int:
    """Sum of the integers from low through high."""

    return (low + high) * (high - low + 1) // 2 if high >= low else 0


class _FlopModel(NamedTuple):
    linear_per_token: float
    attention_per_position: float
    lm_head_per_token: float
    peak_flops: float


_FLOP_MODELS: dict[tuple[str, str], _FlopModel | None] = {}


def _flop_model(model: str, hardware: str) -> _FlopModel | None:
    """Dense decoder FLOPs per token and the GPU's peak tensor FLOPs."""

    key = (model, hardware)
    if key in _FLOP_MODELS:
        return _FLOP_MODELS[key]
    from aisimulate_core.sdk.perf_database import load_system_spec
    from aisimulate_core.sdk.utils import get_model_config_from_model_path

    parsed = get_model_config_from_model_path(model)
    if parsed.get("num_experts"):
        print(f"  MFU unavailable: {model} is a mixture-of-experts model", file=sys.stderr)
        _FLOP_MODELS[key] = None
        return None
    layers, heads, kv_heads, head_dim = (int(parsed[name]) for name in ("layers", "n", "n_kv", "d"))
    hidden, inter, vocab = (int(parsed[name]) for name in ("hidden_size", "inter_size", "vocab"))
    quantization = parsed["raw_config"].get("quantization_config")
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    flops_key = "fp8_tc_flops" if method == "fp8" else "bfloat16_tc_flops"
    peak = _finite_float(load_system_spec(hardware).get("gpu", {}).get(flops_key))
    if peak is None or peak <= 0:
        raise SystemExit(f"system {hardware} has no positive gpu.{flops_key}")
    attention_width = heads * head_dim
    projections = 2 * hidden * attention_width + 2 * hidden * kv_heads * head_dim + 3 * hidden * inter
    flop_model = _FlopModel(
        linear_per_token=2.0 * layers * projections,
        attention_per_position=4.0 * layers * attention_width,
        lm_head_per_token=2.0 * hidden * vocab,
        peak_flops=peak,
    )
    _FLOP_MODELS[key] = flop_model
    return flop_model


def gpu_mfu(point_dir: Path) -> dict[str, float] | None:
    """Useful model FLOPs over peak tensor FLOPs for the simulated duration.

    Prefill skips reused prefix tokens, attends causally and samples the first
    token. Each decode step after the first token attends to every earlier
    position and samples one token. Preemption recompute is not useful work.

    Returns ``total`` over every GPU, plus ``prefill`` and ``decode`` over
    each pool's own GPUs in P/D.
    """

    requests_path = point_dir / "requests.jsonl"
    config_path = point_dir / "prediction.yaml"
    report_path = point_dir / "prediction.json"
    if not (requests_path.is_file() and config_path.is_file() and report_path.is_file()):
        return None
    duration = _finite_float(json.loads(report_path.read_text(encoding="utf-8")).get("duration_ms"))
    if duration is None or duration <= 0:
        return None
    _, roles, model = _engine_workers(config_path)
    flop_models = {role: _flop_model(model, item.hardware) for role, item in roles.items()}
    if any(value is None for value in flop_models.values()):
        return None
    # Per-token FLOPs depend only on the model; peak FLOPs differ by each role's hardware.
    flop_model = next(iter(flop_models.values()))
    peaks = {role: flop_models[role].peak_flops * _parallel_sizes(item.worker)[0] for role, item in roles.items()}
    # Linear tokens, attended positions and sampled tokens for each phase.
    prefill = [0, 0, 0]
    decode = [0, 0, 0]
    for record in _completed_requests(requests_path):
        context = _token_count(record.get("input_length"))
        generated = _token_count(record.get("output_length"))
        if context is None or generated is None:
            continue
        reused = min(_reused_input_tokens(record), context)
        decode_steps = max(generated - 1, 0)
        prefill[0] += context - reused
        prefill[1] += _position_sum(reused + 1, context)
        prefill[2] += min(generated, 1)
        decode[0] += decode_steps
        decode[1] += _position_sum(context + 1, context + decode_steps)
        decode[2] += decode_steps

    def flops(counts: list[int]) -> float:
        linear, attended, sampled = counts
        return (
            linear * flop_model.linear_per_token
            + attended * flop_model.attention_per_position
            + sampled * flop_model.lm_head_per_token
        )

    seconds = duration / 1000.0
    phase_flops = {"prefill": flops(prefill), "decode": flops(decode)}
    result = {"total": sum(phase_flops.values()) / (sum(peaks.values()) * seconds)}
    if "aggregated" not in roles:
        for role, value in phase_flops.items():
            result[role] = value / (peaks[role] * seconds)
    return result


_DEFAULT_BLOCK_SIZES = {"vllm": 64, "sglang": 1, "trtllm": 32}
_KV_CAPACITIES: dict[tuple[object, ...], int] = {}


def _kv_capacity(config_path: Path, role: str) -> tuple[int, int]:
    """KV block size and total KV blocks across all attention ranks of one role."""

    engine, roles, model = _engine_workers(config_path)
    worker, hardware = roles[role]
    scheduler = _mapping(worker, "scheduler")
    cache = _mapping(worker, "kv_cache")
    capacity = _mapping(cache, "capacity")
    backend = engine.get("backend") if isinstance(engine.get("backend"), str) else "vllm"
    version = engine.get("backend_version") if isinstance(engine.get("backend_version"), str) else "current"
    block_size = cache.get("block_size", _DEFAULT_BLOCK_SIZES.get(backend, 64))
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
        raise SystemExit(f"{config_path} kv_cache.block_size must be a positive integer")
    _, ranks = _parallel_sizes(worker)
    if capacity.get("type") == "fixed" and isinstance(capacity.get("blocks"), int):
        return block_size, capacity["blocks"] * ranks
    max_tokens = scheduler.get("max_batched_tokens", 8192)
    max_sequences = scheduler.get("max_sequences", 1 if role == "prefill" else 256)
    fraction = capacity.get("memory_fraction")
    if fraction is None:
        fraction = 0.88 if backend == "sglang" else 0.9
    key = (model, hardware, backend, version, max_tokens, max_sequences, fraction)
    tokens = _KV_CAPACITIES.get(key)
    if tokens is None:
        from aisimulate_core.sdk.memory import estimate_kv_cache

        estimate = estimate_kv_cache(
            model,
            hardware,
            backend,
            version,
            max_num_tokens=int(max_tokens),
            max_batch_size=int(max_sequences),
            memory_fraction_kind="of_free" if backend == "trtllm" else "of_total",
            memory_fraction_value=float(fraction),
        )
        tokens = int(estimate["total_kv_size_tokens"])
        _KV_CAPACITIES[key] = tokens
    return block_size, tokens // block_size * ranks


class _Residency(NamedTuple):
    admit: float
    first: float
    last: float
    reused: int
    context: int
    generated: int


def _residencies(path: Path) -> list[_Residency]:
    """Admission to last token of every completed request."""

    spans: list[_Residency] = []
    for record in _completed_requests(path):
        admit = _finite_float(record.get("first_admit_ms"))
        first = _finite_float(record.get("first_token_ms"))
        last = _finite_float(record.get("last_token_ms"))
        context = _token_count(record.get("input_length"))
        generated = _token_count(record.get("output_length"))
        if admit is None or first is None or last is None or context is None or generated is None:
            continue
        first = max(first, admit)
        last = max(last, first)
        reused = min(_reused_input_tokens(record), context)
        spans.append(_Residency(admit, first, last, reused, context, generated))
    return spans


def _pd_residencies(path: Path) -> tuple[list[_Residency], list[_Residency]]:
    """Prefill-pool and decode-pool residency of every completed P/D request.

    Prefill computes the input from its admission until the source KV is held
    for transfer, then holds it until the transfer releases it. Decode holds the
    input from its reservation until the first token, then grows to the output.
    """

    prefill: list[_Residency] = []
    decode: list[_Residency] = []
    for record in _completed_requests(path):
        times = [
            _finite_float(record.get(name))
            for name in (
                "prefill_admit_ms",
                "source_held_ms",
                "source_released_ms",
                "destination_reserved_ms",
                "first_token_ms",
                "last_token_ms",
            )
        ]
        context = _token_count(record.get("input_length"))
        generated = _token_count(record.get("output_length"))
        if any(value is None for value in times) or context is None or generated is None:
            continue
        admit, held, released, reserved, first, last = times
        held = max(held, admit)
        released = max(released, held)
        first = max(first, reserved)
        last = max(last, first)
        reused = min(_reused_input_tokens(record), context)
        prefill.append(_Residency(admit, held, released, reused, context, 0))
        decode.append(_Residency(reserved, first, last, context, context, generated))
    return prefill, decode


def _resident_tokens(span: _Residency, time: float) -> float:
    """Prefill grows from the reused prefix to the input by the first token, then decode grows to the output."""

    if time < span.first:
        progress = (time - span.admit) / (span.first - span.admit)
        return span.reused + (span.context - span.reused) * progress
    if span.last > span.first:
        return span.context + span.generated * min((time - span.first) / (span.last - span.first), 1.0)
    return float(span.context + span.generated)


def _block_integral(tokens: float, block_size: int) -> float:
    """Integral of ceil(x / block_size) for x from 0 to tokens."""

    if tokens <= 0:
        return 0.0
    blocks = math.ceil(tokens / block_size)
    return block_size * blocks * (blocks - 1) / 2 + blocks * (tokens - (blocks - 1) * block_size)


def _block_time(start_tokens: float, end_tokens: float, elapsed: float, block_size: int) -> float:
    """Block-milliseconds held while tokens grow linearly over ``elapsed``."""

    if elapsed <= 0:
        return 0.0
    if end_tokens <= start_tokens:
        return elapsed * math.ceil(start_tokens / block_size)
    span = _block_integral(end_tokens, block_size) - _block_integral(start_tokens, block_size)
    return elapsed * span / (end_tokens - start_tokens)


def kv_cache_utilization(point_dir: Path) -> dict[str, tuple[float, float]] | None:
    """Time-weighted mean and peak KV blocks resident, over total KV blocks, per pool.

    Each request holds ``ceil(tokens / block_size)`` blocks from admission to
    its last token. The pool is ``aggregated``, or ``prefill`` and ``decode``
    in P/D (see ``_pd_residencies``), each against its own capacity.
    """

    requests_path = point_dir / "requests.jsonl"
    config_path = point_dir / "prediction.yaml"
    report_path = point_dir / "prediction.json"
    if not (requests_path.is_file() and config_path.is_file() and report_path.is_file()):
        return None
    duration = _finite_float(json.loads(report_path.read_text(encoding="utf-8")).get("duration_ms"))
    if duration is None or duration <= 0:
        return None
    _, roles, _ = _engine_workers(config_path)
    if "aggregated" in roles:
        pools = {"aggregated": _residencies(requests_path)}
    else:
        prefill, decode = _pd_residencies(requests_path)
        pools = {"prefill": prefill, "decode": decode}
    if not all(pools.values()):
        return None
    results: dict[str, tuple[float, float]] = {}
    for role, pool_spans in pools.items():
        block_size, capacity = _kv_capacity(config_path, role)
        if capacity <= 0:
            return None
        results[role] = _pool_utilization(pool_spans, duration, block_size, capacity)
    return results


POOL_UTILIZATION_FIELDS = [
    f"{metric}_{role}"
    for role in ("prefill", "decode")
    for metric in ("gpu_mfu", "kv_cache_mean_utilization", "kv_cache_peak_utilization")
]


def point_utilizations(point_dir: Path) -> dict[str, float | None]:
    """Utilization columns of one point. The ``_prefill``/``_decode`` columns are ``None`` unless P/D.

    ``gpu_mfu`` covers every GPU; the KV columns without a pool suffix take the higher pool.
    """

    values: dict[str, float | None] = dict.fromkeys(
        ["gpu_mfu", "kv_cache_mean_utilization", "kv_cache_peak_utilization", *POOL_UTILIZATION_FIELDS]
    )
    mfu = gpu_mfu(point_dir)
    if mfu is not None:
        values["gpu_mfu"] = mfu["total"]
        for role in ("prefill", "decode"):
            values[f"gpu_mfu_{role}"] = mfu.get(role)
    kv_cache = kv_cache_utilization(point_dir)
    if kv_cache is not None:
        values["kv_cache_mean_utilization"] = max(mean for mean, _ in kv_cache.values())
        values["kv_cache_peak_utilization"] = max(peak for _, peak in kv_cache.values())
        for role in ("prefill", "decode"):
            if role in kv_cache:
                mean, peak = kv_cache[role]
                values[f"kv_cache_mean_utilization_{role}"] = mean
                values[f"kv_cache_peak_utilization_{role}"] = peak
    return values


def _pool_utilization(
    spans: list[_Residency], duration: float, block_size: int, capacity: int
) -> tuple[float, float]:
    """Mean and peak resident blocks of one KV pool over its capacity.

    Block counts only fall at a release, so the peak is the total just before
    some release.
    """

    block_time = 0.0
    for span in spans:
        block_time += _block_time(span.reused, span.context, span.first - span.admit, block_size)
        block_time += _block_time(span.context, span.context + span.generated, span.last - span.first, block_size)

    admissions = sorted(range(len(spans)), key=lambda index: spans[index].admit)
    releases = sorted(range(len(spans)), key=lambda index: spans[index].last)
    active: dict[int, _Residency] = {}
    admitted = 0
    peak = 0
    evaluated_at = None
    for index in releases:
        time = spans[index].last
        while admitted < len(admissions) and spans[admissions[admitted]].admit <= time:
            active[admissions[admitted]] = spans[admissions[admitted]]
            admitted += 1
        if time != evaluated_at:
            resident = sum(math.ceil(_resident_tokens(span, time) / block_size) for span in active.values())
            peak = max(peak, resident)
            evaluated_at = time
        active.pop(index, None)
    return block_time / (duration * capacity), peak / capacity


def host_memory_utilization(point_dir: Path) -> float | None:
    """Peak process RSS divided by the host's total memory."""

    runtime_path = point_dir / "resource-runtime.json"
    events_path = point_dir / "execution-events.jsonl"
    if not runtime_path.is_file() or not events_path.is_file():
        return None
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    peak = _finite_float(runtime.get("peak_observed_rss_bytes"))
    total = None
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != "resource_plan":
                continue
            value = event.get("value")
            host = value.get("host") if isinstance(value, dict) else None
            if isinstance(host, dict):
                total = _finite_float(host.get("total_memory_bytes"))
            break
    if peak is None or total is None or total <= 0:
        return None
    return peak / total


def _shared_log_ylim(values: list[float]) -> tuple[float, float] | None:
    """Outward powers of ten that contain every positive sample."""

    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    lower = math.floor(math.log10(min(positive)) + 1e-9)
    upper = math.ceil(math.log10(max(positive)) - 1e-9)
    if upper <= lower:
        upper = lower + 1
    return (10.0**lower, 10.0**upper)


def _flat_ylim(values: list[float]) -> tuple[float, float] | None:
    """Linear range for a series with no spread, which the tick helper cannot scale."""

    if not values or max(values) != min(values):
        return None
    value = values[0]
    return (0.0, 1.0) if value <= 0 else (0.0, 2.0 * value)


def _metric_series(
    rows: list[dict[str, object]],
    x_column: str,
    y_column: str,
) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        value = row[y_column]
        if value == "" or value is None:
            continue
        xs.append(float(row[x_column]))
        ys.append(float(value))
    return xs, ys


def plot_latency(
    path: Path,
    xs: list[float],
    rows: list[dict[str, object]],
    xlabel: str,
    x_type: str = "int",
) -> None:
    """Draw median latency on the first row and P99 latency on the second."""

    import matplotlib

    matplotlib.use("Agg")
    from plotting_primitives import ACM_COMPACT_FULL, SubplotGrid, plot_line

    style = ACM_COMPACT_FULL
    grid = SubplotGrid(style, layout=f"2x{len(AGGREGATE_LATENCY_FIELDS)}")
    for col, (label, median_name, p99_name) in enumerate(AGGREGATE_LATENCY_FIELDS):
        median_ys = [float(row[median_name]) for row in rows]
        p99_ys = [float(row[p99_name]) for row in rows]
        for row_index, ys in enumerate((median_ys, p99_ys)):
            ax = grid.get_ax(row_index, col)
            plot_line(ax, xs, ys, style=style, show_markers=True, color_idx=col)
            grid.configure_ax(
                ax,
                xlabel=xlabel,
                ylabel="median (ms)" if row_index == 0 else "P99 (ms)",
                title=label,
                show_xlabel=row_index == 1,
                show_ylabel=col == 0,
                show_xticklabels=row_index == 1,
                show_title=row_index == 0,
                x_data=xs,
                log_y=True,
                x_type=x_type,
            )
    grid.save(path)


class _QosLine(NamedTuple):
    label: str
    rows: list[dict[str, object]]


def plot_qos_latency(
    path: Path,
    classes: list[QosClass],
    rows: list[dict[str, object]],
    xlabel: str,
) -> None:
    """Draw one solid line per class. A dashed line marks an SLO that class sets.

    The third row is GPU MFU and peak KV cache occupancy at each rate (P/D:
    prefill solid, decode dashed), then each class's SLO violation ratio.
    """

    lines = [_QosLine(item.name, [row for row in rows if row["qos"] == item.name]) for item in classes]
    slos = [(item.slo, index) for index, item in enumerate(classes)]
    _plot_qos_grid(path, lines, slos, [_QosLine("", rows)], xlabel)


def plot_policy_comparison(
    path: Path,
    item: QosClass,
    policies: list[str],
    rows: list[dict[str, object]],
    xlabel: str,
) -> None:
    """Draw one solid line per policy for one class. A gray dashed line marks each SLO the class sets.

    The third row is each policy's GPU MFU and peak KV cache occupancy (P/D:
    prefill solid, decode dashed), then this class's SLO violation ratio
    under each policy.
    """

    lines = [
        _QosLine(policy, [row for row in rows if row["policy"] == policy and row["qos"] == item.name])
        for policy in policies
    ]
    _plot_qos_grid(path, lines, [(item.slo, None)], lines, xlabel)


def _plot_qos_grid(
    path: Path,
    lines: list[_QosLine],
    slos: list[tuple[dict[str, float], int | None]],
    utilization_lines: list[_QosLine],
    xlabel: str,
) -> None:
    """Latency rows, then utilization and SLO violations. A ``None`` SLO color draws gray."""

    import matplotlib

    matplotlib.use("Agg")
    from plotting_primitives import ACM_COMPACT_FULL, SubplotGrid, plot_line

    style = ACM_COMPACT_FULL
    grid = SubplotGrid(style, layout="3x3")
    xs = sorted({float(row["requests_per_second"]) for line in lines for row in line.rows})
    for col, (label, slo_key, p50_name, p99_name) in enumerate(QOS_PLOT_FIELDS):
        series: list[tuple[int, str, str, list[float], list[float]]] = []
        for index, line in enumerate(lines):
            for metric_name in (p50_name, p99_name):
                metric_xs, metric_ys = _metric_series(line.rows, "requests_per_second", metric_name)
                series.append((index, line.label, metric_name, metric_xs, metric_ys))
        shared_ylim = None
        if label in {"ITL", "TTLT"}:
            plotted = [value for *_, metric_ys in series for value in metric_ys]
            plotted.extend(slo[slo_key] for slo, _ in slos if slo_key in slo)
            shared_ylim = _shared_log_ylim(plotted)
        for row_index, metric_name in enumerate((p50_name, p99_name)):
            ax = grid.get_ax(row_index, col)
            for index, name, series_metric, metric_xs, metric_ys in series:
                if series_metric != metric_name or not metric_xs:
                    continue
                plot_line(
                    ax,
                    metric_xs,
                    metric_ys,
                    style=style,
                    show_markers=True,
                    color_idx=index,
                    style_idx=0,
                    marker=style.markers[index % len(style.markers)],
                    label=name,
                )
            for slo_values, color_index in slos:
                slo = slo_values.get(slo_key)
                if slo is None:
                    continue
                ax.axhline(
                    slo,
                    color="0.4" if color_index is None else style.colors[color_index % len(style.colors)],
                    linestyle="--",
                    linewidth=style.line_width,
                )
            if label == "TTFT":
                ylim = (100.0, 100_000.0)
            else:
                ylim = shared_ylim
            grid.configure_ax(
                ax,
                xlabel=xlabel,
                ylabel="P50 (ms)" if row_index == 0 else "P99 (ms)",
                title=label,
                show_xlabel=row_index == 1,
                show_ylabel=col == 0,
                show_xticklabels=row_index == 1,
                show_title=row_index == 0,
                x_data=xs,
                log_y=True,
                x_type="float",
                ylim=ylim,
            )
            if ylim is not None:
                lower = round(math.log10(ylim[0]))
                upper = round(math.log10(ylim[1]))
                ax.set_yticks([10**exponent for exponent in range(lower, upper + 1)])
    split_pools = any(
        row.get(f"{metric_name}_prefill") not in ("", None)
        for line in utilization_lines
        for row in line.rows
        for _, metric_name in UTILIZATION_PLOT_FIELDS
    )
    # P/D draws each pool with its own line style in the line's color: prefill solid, decode dashed.
    pools = ((0, "_prefill"), (1, "_decode")) if split_pools else ((0, ""),)
    for col, (label, metric_name) in enumerate(UTILIZATION_PLOT_FIELDS):
        ax = grid.get_ax(2, col)
        plotted_ys: list[float] = []
        for line_index, line in enumerate(utilization_lines):
            for style_index, suffix in pools:
                util_xs, util_ys = _metric_series(line.rows, "requests_per_second", metric_name + suffix)
                seen: set[float] = set()
                rate_xs: list[float] = []
                rate_ys: list[float] = []
                for rate, utilization in zip(util_xs, util_ys, strict=True):
                    if rate in seen:
                        continue
                    seen.add(rate)
                    rate_xs.append(rate)
                    rate_ys.append(100.0 * utilization)
                if not rate_xs:
                    continue
                plot_line(
                    ax,
                    rate_xs,
                    rate_ys,
                    style=style,
                    show_markers=True,
                    color_idx=line_index if line.label else 4,
                    style_idx=style_index,
                    marker=style.markers[line_index % len(style.markers)],
                    label=(line.label or None) if style_index == 0 else None,
                )
                plotted_ys.extend(rate_ys)
        if split_pools:
            for style_index, pool in enumerate(("prefill", "decode")):
                ax.plot(
                    [],
                    [],
                    color="0.4",
                    linestyle=style.line_styles[style_index],
                    linewidth=style.line_width,
                    label=pool,
                )
        grid.configure_ax(
            ax,
            xlabel=xlabel,
            ylabel="utilization (%)",
            title=label,
            show_xlabel=True,
            show_ylabel=col == 0,
            show_xticklabels=True,
            show_title=True,
            x_data=xs,
            y_data=plotted_ys or None,
            x_type="float",
            y_type="float",
            ylim=(0.0, 100.0) if label == "GPU MFU" else _flat_ylim(plotted_ys),
        )
        if label == "GPU MFU":
            ticks = list(range(0, 101, 20))
            ax.set_yticks(ticks, labels=[str(tick) for tick in ticks])
    ax = grid.get_ax(2, len(UTILIZATION_PLOT_FIELDS))
    for index, line in enumerate(lines):
        metric_xs, metric_ys = _metric_series(line.rows, "requests_per_second", "slo_violation_ratio")
        if not metric_xs:
            continue
        plot_line(
            ax,
            metric_xs,
            [100.0 * ratio for ratio in metric_ys],
            style=style,
            show_markers=True,
            color_idx=index,
            style_idx=0,
            marker=style.markers[index % len(style.markers)],
            label=line.label,
        )
    grid.configure_ax(
        ax,
        xlabel=xlabel,
        ylabel="requests (%)",
        title="SLO violations",
        show_xlabel=True,
        show_ylabel=True,
        show_xticklabels=True,
        show_title=True,
        x_data=xs,
        x_type="float",
        y_type="float",
        ylim=(0.0, 100.0),
    )
    ticks = list(range(0, 101, 20))
    ax.set_yticks(ticks, labels=[str(tick) for tick in ticks])
    grid.add_shared_legend(position="top")
    grid.save(path)


def run_max_batched_tokens(output_dir: Path, config_path: Path) -> None:
    base = load_config(config_path)
    shutil.copyfile(config_path, output_dir / "base.yaml")
    aisimulate = find_aisimulate()
    rows: list[dict[str, object]] = []
    for value in batched_token_grid():
        print(f"predict max_batched_tokens={value}")
        point_config = copy.deepcopy(base)
        set_max_batched_tokens(point_config, value)
        status, metrics = run_point(aisimulate, point_config, output_dir / "points" / str(value))
        rows.append(metric_row("max_batched_tokens", value, metrics, status))
    write_sweep_csv(output_dir / "sweep.csv", rows, aggregate_csv_fields("max_batched_tokens"))
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise SystemExit(f"no successful points; see {output_dir / 'sweep.csv'}")
    xs = [float(row["max_batched_tokens"]) for row in ok_rows]
    plot_latency(output_dir / P99_PLOT, xs, ok_rows, xlabel="max_batched_tokens")


def set_trace_traffic(config: dict, trace_path: Path) -> None:
    relative = trace_path.relative_to(REPO_ROOT).as_posix()
    config["traffic"] = {
        "source": {
            "type": "trace",
            "paths": [relative],
            "format": "mooncake",
            "block_size": TRACE_BLOCK_SIZE,
        },
        "load": {"type": "trace_timestamps"},
    }
    # The full conversation trace estimates at about 3.8 GB. The default budget
    # keeps 1 GB free and only offers 90% of what remains, which refuses it.
    resources = config.setdefault("execution", {}).setdefault("resources", {})
    resources["reserve_memory_gb"] = 0
    resources["available_memory_fraction"] = 1.0


def parse_policies(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(name, str) for name in value):
        raise SystemExit(f"{label} must be a non-empty list of policy names")
    policies = [name.strip() for name in value]
    unknown = [name for name in policies if name not in POLICIES]
    if unknown:
        raise SystemExit(f"{label} has unknown policies {', '.join(unknown)}; choose from {', '.join(POLICIES)}")
    if len(set(policies)) != len(policies):
        raise SystemExit(f"{label} repeats a policy")
    return policies


def edf_deadline_offsets(classes: list[QosClass]) -> dict[str, float]:
    """Milliseconds from arrival to each class's EDF deadline: TTFT SLO, else TTLT SLO."""

    offsets: dict[str, float] = {}
    for item in classes:
        offset = item.slo.get("ttft_ms", item.slo.get("ttlt_ms"))
        if offset is None:
            raise SystemExit(f"edf needs slo.ttft_ms or slo.ttlt_ms for class {item.name}")
        offsets[item.name] = offset
    return offsets


def write_priority_trace(source: Path, destination: Path, policy: str, deadline_offsets: dict[str, float]) -> None:
    """Copy a Mooncake trace with each row's ``priority`` set to its EDF or SJF key. Lower runs first."""

    low, high = PRIORITY_RANGE
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, destination.open("w", encoding="utf-8") as writer:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if policy == "edf":
                key = round(record["timestamp"] + deadline_offsets[record["qos"]])
            else:
                key = record["input_length"]
            if not low <= key <= high:
                raise SystemExit(f"{source}:{line_number} {policy} priority {key} does not fit in 32 bits")
            record["priority"] = key
            writer.write(json.dumps(record, separators=(",", ":")) + "\n")


def set_queue_policy(config: dict, value: str) -> None:
    """Set the waiting-queue policy on aggregated workers, or on prefill workers in P/D."""

    _role_scheduler(config)["queue_policy"] = value


def run_trace_rps(output_dir: Path, experiment: ExperimentConfig, qos_config_path: Path) -> None:
    policies = experiment.policies
    classes = experiment.classes
    base = load_config(experiment.prediction)
    deadline_offsets = edf_deadline_offsets(classes) if "edf" in policies else {}
    shutil.copyfile(experiment.prediction, output_dir / "base.yaml")
    shutil.copyfile(qos_config_path, output_dir / "qos.yaml")
    if experiment.trace_dir is None:
        assignment, traces = write_poisson_traces(classes, experiment.seed, experiment.rates)
    else:
        assignment, traces = load_prepared_traces(
            experiment.trace_dir, classes, experiment.rates, output_dir / "traces"
        )
    aisimulate = find_aisimulate()
    rows: list[dict[str, object]] = []
    for policy in policies:
        policy_rows: list[dict[str, object]] = []
        for rate, trace_path in traces:
            print(f"predict policy={policy} requests_per_second={rate_label(rate)}")
            point_dir = output_dir / "points" / policy / f"rps-{rate_label(rate)}"
            point_config = copy.deepcopy(base)
            if policy == "fcfs":
                set_queue_policy(point_config, "fcfs")
            else:
                point_trace = point_dir / "trace.jsonl"
                write_priority_trace(trace_path, point_trace, policy, deadline_offsets)
                trace_path = point_trace
                set_queue_policy(point_config, "priority")
            set_trace_traffic(point_config, trace_path)
            status, _metrics = run_point(
                aisimulate,
                point_config,
                point_dir,
                capture_per_request=True,
                summary_fields=(),
            )
            policy_rows.extend(qos_metric_rows(point_dir, policy, rate, status, classes, assignment))
        rows.extend(policy_rows)
        write_sweep_csv(output_dir / "sweep.csv", rows, QOS_CSV_FIELDS)
        ok_policy_rows = [row for row in policy_rows if row["status"] == "ok"]
        if ok_policy_rows:
            plot_qos_latency(
                output_dir / "points" / policy / P99_PLOT, classes, ok_policy_rows, xlabel="requests per second"
            )
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise SystemExit(f"no successful points; see {output_dir / 'sweep.csv'}")
    plot_policy_comparisons(output_dir, classes, policies, ok_rows)


def plot_policy_comparisons(
    output_dir: Path, classes: list[QosClass], policies: list[str], ok_rows: list[dict[str, object]]
) -> list[Path]:
    """Write ``qos-<class>.pdf`` for every class."""

    paths = []
    for item in classes:
        path = output_dir / f"qos-{slugify(item.name)}.pdf"
        plot_policy_comparison(path, item, policies, ok_rows, xlabel="requests per second")
        paths.append(path)
    return paths


def plot_trace_rps(
    output_dir: Path, classes: list[QosClass], policies: list[str], ok_rows: list[dict[str, object]]
) -> list[Path]:
    """Write every trace-rps plot: each policy's ``p99.pdf``, then the per-class comparisons."""

    paths = []
    for policy in policies:
        policy_rows = [row for row in ok_rows if row["policy"] == policy]
        if not policy_rows:
            continue
        path = output_dir / "points" / policy / P99_PLOT
        plot_qos_latency(path, classes, policy_rows, xlabel="requests per second")
        paths.append(path)
    return paths + plot_policy_comparisons(output_dir, classes, policies, ok_rows)


def _point_assignment(point_dir: Path) -> dict[str, str] | None:
    """Class of each ``request_<line>`` from the Mooncake trace this point replayed."""

    config_path = point_dir / "prediction.yaml"
    if not config_path.is_file():
        return None
    paths = load_config(config_path).get("traffic", {}).get("source", {}).get("paths") or []
    if not paths:
        return None
    trace_path = Path(paths[0])
    trace_path = trace_path if trace_path.is_absolute() else REPO_ROOT / trace_path
    if not trace_path.is_file():
        return None
    assignment: dict[str, str] = {}
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                assignment[f"request_{line_number}"] = json.loads(line)["qos"]
    return assignment


def _backfill_slo_violations(
    experiment_dir: Path, classes: list[QosClass], rows: list[dict[str, object]]
) -> None:
    """Fill ``slo_violation_ratio`` from each point's ``requests.jsonl`` in one cheap pass."""

    slos = {item.name: item.slo for item in classes}
    points: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        points.setdefault((str(row["policy"]), str(row["requests_per_second"])), []).append(row)
    for (policy, rate), point_rows in points.items():
        for row in point_rows:
            row["slo_violation_ratio"] = ""
        if all(row["status"] != "ok" for row in point_rows):
            continue
        point_dir = experiment_dir / "points" / policy / f"rps-{rate_label(float(rate))}"
        requests_path = point_dir / "requests.jsonl"
        assignment = _point_assignment(point_dir)
        if assignment is None or not requests_path.is_file():
            print(f"warning: cannot compute SLO violations for {point_dir}", file=sys.stderr)
            continue
        counts = dict.fromkeys(slos, 0)
        violations = dict.fromkeys(slos, 0)
        with requests_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("terminal_status") != "completed":
                    continue
                label = assignment.get(record.get("session_id"))
                if label not in slos:
                    continue
                counts[label] += 1
                violations[label] += violates_slo(record, slos[label])
        for row in point_rows:
            label = str(row["qos"])
            if row["status"] == "ok" and counts.get(label):
                row["slo_violation_ratio"] = violations[label] / counts[label]


def _backfill_pool_utilization(experiment_dir: Path, rows: list[dict[str, object]]) -> None:
    """Fill the ``_prefill``/``_decode`` utilization columns from each P/D point's outputs."""

    points: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        for name in POOL_UTILIZATION_FIELDS:
            row[name] = ""
        if row["status"] == "ok":
            points.setdefault((str(row["policy"]), str(row["requests_per_second"])), []).append(row)
    for (policy, rate), point_rows in points.items():
        point_dir = experiment_dir / "points" / policy / f"rps-{rate_label(float(rate))}"
        values = point_utilizations(point_dir)
        if all(values[name] is None for name in POOL_UTILIZATION_FIELDS):
            print(f"warning: cannot compute per-pool utilization for {point_dir}", file=sys.stderr)
            continue
        for row in point_rows:
            for name in POOL_UTILIZATION_FIELDS:
                row[name] = "" if values[name] is None else values[name]


def replot(experiment_dir: Path) -> None:
    """Redraw an experiment's plots from its ``sweep.csv`` without running or recomputing points."""

    sweep_path = experiment_dir / "sweep.csv"
    if not sweep_path.is_file():
        raise SystemExit(f"sweep.csv not found: {sweep_path}")
    with sweep_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows: list[dict[str, object]] = list(reader)
    written: list[Path] = []
    if "max_batched_tokens" in header:
        ok_rows = [row for row in rows if row["status"] == "ok"]
        if not ok_rows:
            raise SystemExit(f"no successful points in {sweep_path}")
        xs = [float(row["max_batched_tokens"]) for row in ok_rows]
        plot_latency(experiment_dir / P99_PLOT, xs, ok_rows, xlabel="max_batched_tokens")
        written.append(experiment_dir / P99_PLOT)
    elif {"policy", "qos", "requests_per_second"} <= set(header):
        qos_path = experiment_dir / "qos.yaml"
        if not qos_path.is_file():
            raise SystemExit(f"qos.yaml not found: {qos_path}")
        _, classes = load_qos_config(qos_path)
        added: list[str] = []
        if "slo_violation_ratio" not in header:
            _backfill_slo_violations(experiment_dir, classes, rows)
            added.append("slo_violation_ratio")
        base_path = experiment_dir / "base.yaml"
        if (
            not set(POOL_UTILIZATION_FIELDS) <= set(header)
            and base_path.is_file()
            and _policy_role(load_config(base_path)) == "prefill"
        ):
            _backfill_pool_utilization(experiment_dir, rows)
            added.extend(POOL_UTILIZATION_FIELDS)
        if added:
            # Keep the script's column order; columns it does not know go last.
            known = [name for name in QOS_CSV_FIELDS if name in header or name in added]
            write_sweep_csv(sweep_path, rows, known + [name for name in header if name not in QOS_CSV_FIELDS])
            print(f"added {', '.join(added)} to {sweep_path}")
        for row in rows:
            for name in QOS_CSV_FIELDS:
                row.setdefault(name, "")
        policies = list(dict.fromkeys(str(row["policy"]) for row in rows))
        ok_rows = [row for row in rows if row["status"] == "ok"]
        if not ok_rows:
            raise SystemExit(f"no successful points in {sweep_path}")
        written.extend(plot_trace_rps(experiment_dir, classes, policies, ok_rows))
    else:
        raise SystemExit(
            f"unsupported sweep.csv format in {sweep_path}: expected a max_batched_tokens column "
            "or policy, requests_per_second and qos columns"
        )
    for path in written:
        print(f"wrote {path}")


EXPERIMENTS = {
    "max-batched-tokens": run_max_batched_tokens,
    "trace-rps": run_trace_rps,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation experiment and save its outputs.")
    parser.add_argument("--comment", help="Short label included in the output directory name. Required unless --replot.")
    parser.add_argument(
        "--replot",
        type=Path,
        metavar="DIR",
        help="Redraw an existing experiment's plots from its sweep.csv without running anything.",
    )
    parser.add_argument(
        "--experiment",
        default="max-batched-tokens",
        choices=sorted(EXPERIMENTS),
        help="Experiment to run. Defaults to max-batched-tokens.",
    )
    parser.add_argument(
        "--qos-config",
        type=Path,
        default=REPO_ROOT / "qos.yaml",
        help=(
            "Experiment YAML: prediction file, traffic (trace_dir, rates), policies, and QoS classes. "
            "Defaults to qos.yaml."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.replot is not None:
        defaults = parse_args(["--replot", str(args.replot)])
        conflicting = [
            flag
            for flag, name in (
                ("--comment", "comment"),
                ("--experiment", "experiment"),
                ("--qos-config", "qos_config"),
            )
            if getattr(args, name) != getattr(defaults, name)
        ]
        if conflicting:
            raise SystemExit(f"--replot cannot be combined with {', '.join(conflicting)}")
        experiment_dir = args.replot if args.replot.is_absolute() else REPO_ROOT / args.replot
        if not experiment_dir.is_dir():
            raise SystemExit(f"experiment directory not found: {experiment_dir}")
        replot(experiment_dir)
        return
    if args.comment is None:
        raise SystemExit("--comment is required unless --replot is given")
    qos_path = args.qos_config if args.qos_config.is_absolute() else REPO_ROOT / args.qos_config
    if not qos_path.is_file():
        raise SystemExit(f"qos config not found: {qos_path}")
    experiment = load_experiment_config(qos_path)
    base_config = load_config(experiment.prediction)
    policy_role = _policy_role(base_config)
    if args.experiment == "trace-rps":
        backend = base_config["engine"].get("backend")
        if any(policy != "fcfs" for policy in experiment.policies) and backend not in (None, "vllm"):
            raise SystemExit(f"edf and sjf need engine.backend vllm, not {backend}")
        if "edf" in experiment.policies:
            edf_deadline_offsets(experiment.classes)
        if experiment.trace_dir is not None:
            load_prepared_traces(experiment.trace_dir, experiment.classes, experiment.rates, None)
    output_dir = EXPERIMENTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{slugify(args.comment)}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"experiment directory: {output_dir}")
    print(f"prediction: {experiment.prediction}")
    if policy_role != "aggregated":
        print(f"disaggregated config: the swept scheduler setting applies to {policy_role} workers")
    if args.experiment == "trace-rps":
        run_trace_rps(output_dir, experiment, qos_path)
        return
    EXPERIMENTS[args.experiment](output_dir, experiment.prediction)


if __name__ == "__main__":
    main()
