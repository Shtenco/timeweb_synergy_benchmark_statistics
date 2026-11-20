"""Add uncertainty summaries to paired, seed-repeated classification measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from prototypes.scenarios.article15 import minimum_sample_size, wilson_interval


def analyze_binary_benchmark(source: dict) -> dict:
    runs = source.get("runs")
    if not isinstance(runs, list) or len(runs) < 2:
        raise ValueError("at least two paired model runs are required")
    sample_sizes = {row.get("test_rows") for row in runs}
    if len(sample_sizes) != 1:
        raise ValueError("all runs must use the same number of test rows")
    trials = next(iter(sample_sizes))
    if not isinstance(trials, int) or trials < 1:
        raise ValueError("test_rows must be a positive integer")

    per_seed = []
    deltas = []
    float_rates = []
    binary_rates = []
    for row in runs:
        float_rate = float(row["float32_test_accuracy"])
        binary_rate = float(row["binary_packed_test_accuracy"])
        if not (0 <= float_rate <= 1 and 0 <= binary_rate <= 1):
            raise ValueError("accuracy values must be between zero and one")
        float_successes = round(float_rate * trials)
        binary_successes = round(binary_rate * trials)
        if abs(float_successes / trials - float_rate) > 1e-9 or abs(binary_successes / trials - binary_rate) > 1e-9:
            raise ValueError("accuracy must correspond to integer correct predictions")
        float_interval = wilson_interval(float_successes, trials)
        binary_interval = wilson_interval(binary_successes, trials)
        delta = binary_rate - float_rate
        deltas.append(delta)
        float_rates.append(float_rate)
        binary_rates.append(binary_rate)
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "test_rows": trials,
                "fp32_correct": float_successes,
                "fp32_accuracy": float_rate,
                "fp32_wilson_95": list(float_interval),
                "binary_correct": binary_successes,
                "binary_accuracy": binary_rate,
                "binary_wilson_95": list(binary_interval),
                "paired_accuracy_delta_binary_minus_fp32": delta,
            }
        )

    mean_delta = float(np.mean(deltas))
    standard_error = float(np.std(deltas, ddof=1) / math.sqrt(len(deltas)))
    critical = float(student_t.ppf(0.975, df=len(deltas) - 1))
    return {
        "article_id": "15",
        "experiment": "Uncertainty summary for the paired UCI digits FP32/BNN comparison",
        "evidence_type": "computed_from_measured_public_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_experiment": "article-02-digits-binary",
        "test_rows": trials,
        "seed_count": len(deltas),
        "mean_fp32_accuracy": float(np.mean(float_rates)),
        "mean_binary_accuracy": float(np.mean(binary_rates)),
        "binary_minus_fp32_accuracy_mean": mean_delta,
        "paired_seed_difference_sd": float(np.std(deltas, ddof=1)),
        "paired_seed_difference_t95": [mean_delta - critical * standard_error, mean_delta + critical * standard_error],
        "per_seed_wilson_95": per_seed,
        "sample_size_for_5pp_margin_at_50pct": minimum_sample_size(margin=0.05, assumed_rate=0.5),
        "limitations": [
            "Wilson intervals treat test images as independent Bernoulli trials; writer-level clustering is unavailable in this copy.",
            "The t interval uses only three paired initialization seeds and is exploratory, not a stable population interval.",
            "The three seeds use the same fixed test set, so seed variation does not measure test-set sampling variation.",
            "The simple sample-size calculation is a Bernoulli proportion approximation, not power analysis for a paired model comparison.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--out", type=Path, default=Path("evidence/15_real"))
    args = parser.parse_args(argv)
    input_bytes = args.input.read_bytes()
    source = json.loads(input_bytes.decode("utf-8"))
    result = analyze_binary_benchmark(source)
    result["input_file"] = str(args.input)
    result["input_sha256"] = hashlib.sha256(input_bytes).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.out / f"article-15-binary-uncertainty-{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(path)
    print("paired_delta", round(result["binary_minus_fp32_accuracy_mean"], 4))
    print("paired_seed_t95", [round(value, 4) for value in result["paired_seed_difference_t95"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
