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
QOS_CSV_FIELDS = [
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
]
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


def load_prepared_traces(
    trace_dir: Path,
    classes: list[QosClass],
) -> tuple[dict[str, str], list[tuple[float, Path]]]:
    """Replay an existing rate directory instead of rewriting it from the CSV.

    ``qos-assignment.json`` supplies the class of each ``request_<line>``.
    Those labels must be the class names in the QoS config. SLOs still come
    from that config.
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

    traces = [(rate, trace_dir / f"rps_{rate:.1f}.jsonl") for rate in rps_grid()]
    for _, path in traces:
        if not path.is_file():
            raise SystemExit(f"trace file not found: {path}")
    sample = traces[0][1]
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
    for _, path in traces[1:]:
        with path.open(encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        if lines != count:
            raise SystemExit(f"{path} has {lines} requests but {sample.name} has {count}")
    return assignment, traces


def write_poisson_traces(
    classes: list[QosClass],
    seed: int,
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
    written = [(rate, output_dir / f"rps_{rate:.1f}.jsonl") for rate in rps_grid()]
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


def set_max_batched_tokens(config: dict, value: int) -> None:
    engine = config.setdefault("engine", {})
    workers = engine.setdefault("workers", {})
    aggregated = workers.setdefault("aggregated", {})
    scheduler = aggregated.setdefault("scheduler", {})
    if not isinstance(scheduler, dict):
        raise SystemExit("engine.workers.aggregated.scheduler must be a mapping")
    scheduler["max_batched_tokens"] = value


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

    point_dir.mkdir(parents=True, exist_ok=False)
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


def qos_metric_rows(
    point_dir: Path,
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
        item.name: {"itl_p50_ms": [], "itl_p99_ms": [], "ttft_ms": [], "e2e_latency_ms": []} for item in classes
    }
    counts = dict.fromkeys((item.name for item in classes), 0)
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
                ttft = _finite_float(record.get("ttft_ms"))
                if ttft is not None:
                    bucket["ttft_ms"].append(ttft)
                ttlt = _finite_float(record.get("e2e_latency_ms"))
                if ttlt is not None:
                    bucket["e2e_latency_ms"].append(ttlt)
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
    rows: list[dict[str, object]] = []
    for item in classes:
        bucket = samples[item.name]
        row: dict[str, object] = {
            "requests_per_second": rate,
            "qos": item.name,
            "status": status,
            "n_requests": counts[item.name] if status == "ok" else "",
        }
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
        rows.append(row)
    return rows


def y_step_with_min_ticks(lower: float, upper: float, min_ticks: int = 5) -> float:
    """Largest 1-2-5 step that still places at least min_ticks inside [lower, upper]."""

    span = upper - lower
    if span <= 0:
        span = max(abs(upper), 1.0)
    magnitude = 10 ** math.floor(math.log10(span))
    fallback = magnitude
    for exponent in range(2, -6, -1):
        scale = magnitude * 10**exponent
        for multiplier in (5, 2, 1):
            step = multiplier * scale
            start = math.floor(lower / step) * step
            stop = math.ceil(upper / step) * step
            count = sum(
                1
                for index in range(round((stop - start) / step) + 1)
                if lower - 1e-9 <= start + index * step <= upper + 1e-9
            )
            if count >= min_ticks:
                return step
            fallback = step
    return fallback


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
        y_data = median_ys + p99_ys
        y_step = y_step_with_min_ticks(min(y_data), max(y_data))
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
                y_data=y_data,
                y_step=y_step,
                x_type=x_type,
            )
    grid.save(path)


def plot_qos_latency(
    path: Path,
    classes: list[QosClass],
    rows: list[dict[str, object]],
    xlabel: str,
) -> None:
    """Draw one solid line per class. A dashed line marks an SLO that class sets."""

    import matplotlib

    matplotlib.use("Agg")
    from plotting_primitives import ACM_COMPACT_FULL, SubplotGrid, plot_line

    style = ACM_COMPACT_FULL
    grid = SubplotGrid(style, layout="2x3")
    by_class = {item.name: [row for row in rows if row["qos"] == item.name] for item in classes}
    xs = sorted({float(row["requests_per_second"]) for row in rows})
    for col, (label, slo_key, p50_name, p99_name) in enumerate(QOS_PLOT_FIELDS):
        y_data: list[float] = []
        series: list[tuple[int, str, str, list[float], list[float]]] = []
        for index, item in enumerate(classes):
            for metric_name in (p50_name, p99_name):
                metric_xs, metric_ys = _metric_series(by_class[item.name], "requests_per_second", metric_name)
                series.append((index, item.name, metric_name, metric_xs, metric_ys))
                y_data.extend(metric_ys)
            slo = item.slo.get(slo_key)
            if slo is not None:
                y_data.append(slo)
        y_step = y_step_with_min_ticks(min(y_data), max(y_data)) if y_data else None
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
            for index, item in enumerate(classes):
                slo = item.slo.get(slo_key)
                if slo is None:
                    continue
                ax.axhline(
                    slo,
                    color=style.colors[index % len(style.colors)],
                    linestyle="--",
                    linewidth=style.line_width,
                )
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
                y_data=y_data or None,
                y_step=y_step,
                x_type="float",
            )
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


def run_trace_rps(
    output_dir: Path,
    config_path: Path,
    qos_config_path: Path,
    trace_dir: Path | None = None,
) -> None:
    base = load_config(config_path)
    seed, classes = load_qos_config(qos_config_path)
    shutil.copyfile(config_path, output_dir / "base.yaml")
    shutil.copyfile(qos_config_path, output_dir / "qos.yaml")
    if trace_dir is None:
        assignment, traces = write_poisson_traces(classes, seed)
    else:
        assignment, traces = load_prepared_traces(trace_dir, classes)
    aisimulate = find_aisimulate()
    rows: list[dict[str, object]] = []
    for rate, trace_path in traces:
        print(f"predict requests_per_second={rate:.1f}")
        point_config = copy.deepcopy(base)
        set_trace_traffic(point_config, trace_path)
        status, _metrics = run_point(
            aisimulate,
            point_config,
            output_dir / "points" / f"rps-{rate:.1f}",
            capture_per_request=True,
            summary_fields=(),
        )
        rows.extend(qos_metric_rows(output_dir / "points" / f"rps-{rate:.1f}", rate, status, classes, assignment))
    write_sweep_csv(output_dir / "sweep.csv", rows, QOS_CSV_FIELDS)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        raise SystemExit(f"no successful points; see {output_dir / 'sweep.csv'}")
    plot_qos_latency(output_dir / P99_PLOT, classes, ok_rows, xlabel="requests per second")


EXPERIMENTS = {
    "max-batched-tokens": run_max_batched_tokens,
    "trace-rps": run_trace_rps,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one simulation experiment and save its outputs.")
    parser.add_argument("--comment", required=True, help="Short label included in the output directory name.")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "prediction.yaml",
        help="Base prediction YAML. Defaults to prediction.yaml.",
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
        help="QoS class YAML for trace-rps. Defaults to qos.yaml.",
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help=(
            "Existing Mooncake directory for trace-rps, with rps_<rate>.jsonl and "
            "qos-assignment.json. Defaults to regenerating traces/azure_conv_poisson from the Azure CSV."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    if not config_path.is_file():
        raise SystemExit(f"config not found: {config_path}")
    qos_path: Path | None = None
    trace_dir: Path | None = None
    if args.trace_dir is not None and args.experiment != "trace-rps":
        raise SystemExit("--trace-dir applies only to --experiment trace-rps")
    if args.experiment == "trace-rps":
        qos_path = args.qos_config if args.qos_config.is_absolute() else REPO_ROOT / args.qos_config
        if not qos_path.is_file():
            raise SystemExit(f"qos config not found: {qos_path}")
        _, classes = load_qos_config(qos_path)
        if args.trace_dir is not None:
            trace_dir = args.trace_dir if args.trace_dir.is_absolute() else REPO_ROOT / args.trace_dir
            load_prepared_traces(trace_dir, classes)
    output_dir = EXPERIMENTS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{slugify(args.comment)}"
    output_dir.mkdir(parents=True, exist_ok=False)
    print(f"experiment directory: {output_dir}")
    if qos_path is not None:
        run_trace_rps(output_dir, config_path, qos_path, trace_dir)
        return
    EXPERIMENTS[args.experiment](output_dir, config_path)


if __name__ == "__main__":
    main()
