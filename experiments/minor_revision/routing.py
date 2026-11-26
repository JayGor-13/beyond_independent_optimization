"""Identity-aligned routing metrics and per-image token selection."""
from __future__ import annotations

import numpy as np

METRICS = ("racr", "top1_change", "exact_topk_change", "load_cv_delta")


def select_visual_ids(norms, image_ids, method, retention, seed):
    norms = np.asarray(norms)
    image_ids = np.asarray(image_ids)
    if norms.ndim != 1 or image_ids.shape != norms.shape or not len(norms):
        raise ValueError("Visual norms and image identities must be nonempty aligned vectors.")
    if not np.isfinite(norms).all() or not 0 < retention <= 1 or method not in {"baseline", "l2", "random"}:
        raise ValueError("Invalid token selection inputs.")
    if method == "baseline" or retention == 1:
        return np.arange(len(norms), dtype=np.int64)
    generator = np.random.default_rng(seed)
    kept = []
    # Preserve at least one token from every image in multi-image questions.
    for image in np.unique(image_ids):
        positions = np.flatnonzero(image_ids == image)
        count = max(1, round(len(positions) * retention))
        if method == "l2":
            order = np.argsort(-norms[positions], kind="stable")
            kept.extend(positions[order[:count]])
        else:
            kept.extend(generator.choice(positions, size=count, replace=False))
    return np.sort(np.asarray(kept, dtype=np.int64))


def validate_routes(routes, experts):
    if routes.ndim != 3 or not all(routes.shape) or routes.shape[-1] > experts:
        raise ValueError("Expected nonempty routes [layers, visual_tokens, top_k].")
    if not np.issubdtype(routes.dtype, np.integer) or (routes < 0).any() or (routes >= experts).any():
        raise ValueError("Expert IDs are outside the valid range.")
    if (np.diff(np.sort(routes, axis=-1), axis=-1) == 0).any():
        raise ValueError("An expert occurs twice in a token's Top-k set.")


def routing_metrics(baseline, compressed, visual_ids, experts):
    validate_routes(baseline, experts)
    validate_routes(compressed, experts)
    visual_ids = np.asarray(visual_ids)
    if (visual_ids.ndim != 1 or not np.issubdtype(visual_ids.dtype, np.integer)
            or not len(visual_ids) or (visual_ids < 0).any() or (visual_ids >= baseline.shape[1]).any()
            or (np.diff(visual_ids) <= 0).any()):
        raise ValueError("Retained IDs must be unique, sorted, original visual-token indices.")
    original = baseline[:, visual_ids, :]
    if original.shape != compressed.shape:
        raise ValueError("Retained routes do not align with baseline token identities.")
    overlap = (original[..., :, None] == compressed[..., None, :]).any(axis=-1).mean(axis=-1)
    layer_racr = 1.0 - overlap.mean(axis=1)
    layer_top1 = (original[..., 0] != compressed[..., 0]).mean(axis=1)
    layer_topk = (np.sort(original, axis=-1) != np.sort(compressed, axis=-1)).any(axis=-1).mean(axis=1)
    cv_delta = []
    for before, after in zip(original, compressed):
        old = np.bincount(before.ravel(), minlength=experts)
        new = np.bincount(after.ravel(), minlength=experts)
        cv_delta.append(float(new.std() / new.mean() - old.std() / old.mean()))
    layer_values = dict(zip(METRICS, (layer_racr, layer_top1, layer_topk, np.asarray(cv_delta))))
    return {**{name: float(value.mean()) for name, value in layer_values.items()},
            "layerwise": {name: value.tolist() for name, value in layer_values.items()}}


def bootstrap_columns(values, samples=2000, seed=20261121):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) < 1 or not np.isfinite(values).all() or samples < 100:
        raise ValueError("Bootstrap needs finite per-example values and at least 100 replicates.")
    generator = np.random.default_rng(seed)
    estimates = []
    # Use bootstrap question multiplicities, not a [replicates, questions, metrics]
    # allocation. Every metric/layer in a row receives exactly the same resample.
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        indices = generator.integers(0, len(values), size=(size, len(values)))
        counts = np.stack([np.bincount(row, minlength=len(values)) for row in indices])
        estimates.append(counts @ values / len(values))
    low, high = np.quantile(np.concatenate(estimates), (0.025, 0.975), axis=0)
    return values.mean(axis=0), low, high


def paired_bootstrap(values, samples=2000, seed=20261121):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Expected one paired value per question.")
    return tuple(float(value[0]) for value in bootstrap_columns(values[:, None], samples, seed))
