"""Wilson confidence intervals and a conservative binomial sample-size estimate."""

from __future__ import annotations

import math


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if not isinstance(trials, int) or isinstance(trials, bool) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if not isinstance(successes, int) or isinstance(successes, bool) or not 0 <= successes <= trials:
        raise ValueError("successes must be an integer between zero and trials")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def minimum_sample_size(margin: float, z: float = 1.96, assumed_rate: float = 0.5) -> int:
    if not 0 < margin < 1 or not math.isfinite(margin):
        raise ValueError("margin must be finite and between zero and one")
    if not math.isfinite(z) or z <= 0 or not 0 < assumed_rate < 1:
        raise ValueError("z must be positive and assumed_rate must be between zero and one")
    return math.ceil(z * z * assumed_rate * (1 - assumed_rate) / (margin * margin))


def run(seed: int = 42) -> dict:
    scenarios = []
    for successes, trials in ((9, 10), (90, 100), (900, 1000)):
        low, high = wilson_interval(successes, trials)
        scenarios.append({
            "successes": successes,
            "trials": trials,
            "observed_rate": successes / trials,
            "wilson_95": [low, high],
        })
    return {
        "evidence_type": "computed",
        "seed": seed,
        "examples": scenarios,
        "sample_size_for_5pp_margin_at_50pct": minimum_sample_size(0.05),
        "limitations": [
            "These are formula examples, not measurements of an AI model.",
            "The Wilson interval assumes independent Bernoulli observations; paired benchmark comparisons need a paired method.",
            "The sample-size calculation is a simple normal approximation, not a power analysis for every experiment.",
        ],
    }
