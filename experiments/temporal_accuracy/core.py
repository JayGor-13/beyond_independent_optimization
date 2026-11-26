import random

import numpy as np


LETTERS = "ABCDE"
PRIMARY_CONDITIONS = ("chronological", "mild", "medium", "severe")
ALL_CONDITIONS = PRIMARY_CONDITIONS + ("reverse",)


def broad_category(question_type):
    if question_type in {"TN", "TP", "TC"}:
        return "temporal"
    if question_type in {"CW", "CH"}:
        return "causal"
    return "descriptive"


def normalized_kendall_distance(order):
    inversions = sum(
        order[left] > order[right]
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )
    maximum = len(order) * (len(order) - 1) / 2
    return float(inversions / maximum) if maximum else 0.0


def adjacency_retention(order):
    if len(order) < 2:
        return 1.0
    retained = sum(abs(order[index] - order[index + 1]) == 1 for index in range(len(order) - 1))
    return float(retained / (len(order) - 1))


def permutation_record(condition, replicate, order):
    kendall = normalized_kendall_distance(order)
    retained = adjacency_retention(order)
    return {
        "key": condition if replicate is None else f"{condition}_r{replicate}",
        "condition": condition,
        "replicate": replicate,
        "severity_rank": {
            "chronological": 0,
            "mild": 1,
            "medium": 2,
            "severe": 3,
            "reverse": 4,
        }[condition],
        "order": list(order),
        "kendall_distance": kendall,
        "adjacency_retention": retained,
        "adjacency_destruction": 1.0 - retained,
    }


def generate_permutation_bank(frame_count, seed, repetitions=2):
    if frame_count < 5:
        raise ValueError("At least five frames are required for graded temporal perturbations.")
    if repetitions < 1:
        raise ValueError("Permutation repetitions must be at least one.")

    # The bounds are defined only from sequence geometry. They must not be
    # changed after observing routing or answer results.
    bounds = {
        "mild": {
            "kendall": (0.03, 0.25),
            "retention": (0.55, 0.86),
        },
        "medium": {
            "kendall": (0.25, 0.60),
            "retention": (0.28, 0.45),
        },
        "severe": {
            "kendall": (0.60, 0.95),
            "retention": (0.00, 0.15),
        },
    }

    identity = tuple(range(frame_count))
    reverse = tuple(reversed(identity))
    used = {identity, reverse}
    rng = random.Random(seed)
    records = [permutation_record("chronological", None, identity)]

    for condition in ("mild", "medium", "severe"):
        accepted = 0
        attempts = 0
        while accepted < repetitions and attempts < 200_000:
            attempts += 1
            candidate = list(identity)
            rng.shuffle(candidate)
            candidate = tuple(candidate)
            if candidate in used:
                continue
            kendall = normalized_kendall_distance(candidate)
            retained = adjacency_retention(candidate)
            low_k, high_k = bounds[condition]["kendall"]
            low_a, high_a = bounds[condition]["retention"]
            if low_k <= kendall <= high_k and low_a <= retained <= high_a:
                records.append(permutation_record(condition, accepted, candidate))
                used.add(candidate)
                accepted += 1
        if accepted != repetitions:
            raise RuntimeError(
                f"Could not construct {repetitions} {condition} permutations "
                f"for {frame_count} frames after {attempts} attempts."
            )

    records.append(permutation_record("reverse", None, reverse))
    return records


def jaccard(left, right):
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def top_m_mask(values, m):
    m = min(int(m), values.size)
    mask = np.zeros(values.size, dtype=bool)
    if m:
        mask[np.argpartition(values, -m)[-m:]] = True
    return mask


def layer_metrics(mass, counts):
    if mass.shape != counts.shape or mass.ndim != 3:
        raise ValueError("mass and counts must have shape [frames, layers, experts].")
    frames, layers, experts = mass.shape
    if frames < 2:
        raise ValueError("TRC requires at least two frames.")

    rows = []
    for layer in range(layers):
        union_values = []
        trc4_values = []
        trc8_values = []
        trc16_values = []
        weighted_values = []
        for frame in range(frames - 1):
            left_mass = mass[frame, layer]
            right_mass = mass[frame + 1, layer]
            union_values.append(jaccard(counts[frame, layer] > 0, counts[frame + 1, layer] > 0))
            trc4_values.append(jaccard(top_m_mask(left_mass, 4), top_m_mask(right_mass, 4)))
            trc8_values.append(jaccard(top_m_mask(left_mass, 8), top_m_mask(right_mass, 8)))
            trc16_values.append(jaccard(top_m_mask(left_mass, 16), top_m_mask(right_mass, 16)))
            denominator = np.maximum(left_mass, right_mass).sum()
            weighted_values.append(
                float(np.minimum(left_mass, right_mass).sum() / denominator) if denominator else 1.0
            )

        positive = np.where(mass[:, layer] > 0, mass[:, layer], 1.0)
        entropy = -(mass[:, layer] * np.log(positive)).sum(axis=1)
        if experts > 1:
            entropy /= np.log(experts)
        aggregate = mass[:, layer].sum(axis=0)
        aggregate_mean = aggregate.mean()
        rows.append(
            {
                "union_trc": float(np.mean(union_values)),
                "trc4": float(np.mean(trc4_values)),
                "trc8": float(np.mean(trc8_values)),
                "trc16": float(np.mean(trc16_values)),
                "weighted_trc": float(np.mean(weighted_values)),
                "router_entropy": float(entropy.mean()),
                "load_cv": float(aggregate.std() / aggregate_mean) if aggregate_mean else 0.0,
            }
        )
    return rows
