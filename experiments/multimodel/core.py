"""Shared experiment logic; no model downloads or CUDA required on import."""
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

MODELS = {
    "internvl35": ("OpenGVLab/InternVL3_5-30B-A3B", "ba1762d830e2aeddb3272b66ba9e0abf7e69f626"),
    "qwen3vl": ("Qwen/Qwen3-VL-30B-A3B-Instruct", "9c4b90e1e4ba969fd3b5378b57d966d725f1b86c"),
}
CONDITIONS = ("chronological", "shuffled", "duplicate", "cross_clip")
LETTERS = "ABCDE"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def lock_json(path, value):
    path = Path(path)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Configuration/manifest mismatch at {path}; use a new output directory.")
    write_json(path, value)


def answer_letter(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def question_text(example):
    return "\n".join([example["question"], *[
        f"({LETTERS[i]}) {choice}" for i, choice in enumerate(example["choices"])
    ], "Answer with only the option letter."])


def science_manifest(dataset, limit, seed):
    indices = [i for i in range(len(dataset)) if dataset[i]["image"] is not None]
    random.Random(seed).shuffle(indices)
    if len(indices) < limit:
        raise ValueError("Not enough ScienceQA image examples.")
    return [{"id": f"scienceqa_test_{i:05d}", "dataset_index": i} for i in indices[:limit]]


def nextqa_manifest(dataset, available, limit, seed):
    # Preserve the original category/seed sampling while failing clearly on exhaustion.
    pools = {name: [] for name in ("temporal", "causal", "descriptive")}
    for i, row in enumerate(dataset):
        if str(row["video"]) in available:
            category = ("temporal" if row["type"] in {"TN", "TP", "TC"} else
                        "causal" if row["type"] in {"CW", "CH"} else "descriptive")
            pools[category].append(i)
    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)
    selected, used = [], set()
    while len(selected) < limit:
        category = list(pools)[len(selected) % 3]
        if not pools[category]:
            raise ValueError(f"Not enough unique available videos in category {category}.")
        i = pools[category].pop()
        row = dataset[i]
        video = str(row["video"])
        if video in used:
            continue
        used.add(video)
        selected.append({"id": f"nextqa_{i:05d}", "dataset_index": i, "video_id": video,
                         "qid": int(row["qid"]), "type": row["type"], "category": category,
                         "question": row["question"], "choices": [row[f"a{j}"] for j in range(5)],
                         "answer": int(row["answer"])})
    return selected


def cross_indices(item_index, size, frames):
    if size < 2:
        raise ValueError("Cross-clip controls require at least two unique videos.")
    # Cycle over other videos only, including for a two-video smoke test.
    return [(item_index + 1 + (f % (size - 1))) % size for f in range(frames)]


def frame_groups(mask, expected):
    positions = torch.where(mask)[0]
    if not len(positions):
        raise ValueError("No visual tokens found.")
    starts = torch.cat((torch.ones(1, dtype=torch.bool, device=mask.device),
                        positions[1:] != positions[:-1] + 1))
    groups = starts.long().cumsum(0) - 1
    if int(groups[-1]) + 1 != expected:
        raise ValueError(f"Expected {expected} separate frame blocks; found {int(groups[-1]) + 1}.")
    result = torch.full_like(mask, -1, dtype=torch.long)
    result[positions] = groups
    return result


def compress(batch, method, retention, seed):
    if method not in {"baseline", "l2", "random"} or not 0 < retention <= 1:
        raise ValueError("Invalid pruning condition.")
    mask = batch["visual_mask"]
    positions = torch.where(mask)[0]
    if not len(positions):
        raise ValueError("Cannot prune a sequence without visual tokens.")
    keep_ids = torch.arange(len(positions), device=mask.device)
    if method != "baseline" and retention < 1:
        count = max(1, round(len(positions) * retention))
        if method == "l2":
            scores = batch["embeddings"][0, positions].float().norm(dim=-1)
        else:
            generator = torch.Generator(device=mask.device).manual_seed(seed)
            scores = torch.rand(len(positions), generator=generator, device=mask.device)
        keep_ids = scores.topk(count).indices.sort().values
    keep = ~mask.clone()
    keep[positions[keep_ids]] = True
    result = dict(batch)
    for key in ("embeddings", "attention_mask", "input_ids"):
        if key in batch:
            result[key] = batch[key][:, keep]
    result["visual_mask"] = mask[keep]
    result["visual_ids"] = keep_ids
    if "frame_ids" in batch:
        result["frame_ids"] = batch["frame_ids"][keep]
    if "position_ids" in batch:
        # Qwen's spatial coordinates must retain their meaning after pruning.
        result["position_ids"] = batch["position_ids"][..., keep]
    if "deepstack" in batch:
        result["deepstack"] = [value[keep_ids.to(value.device)] for value in batch["deepstack"]]
    return result


def aggregate_frames(topk, weights, frame_ids, n_experts, frames):
    if topk.shape != weights.shape or topk.shape[1] != len(frame_ids):
        raise ValueError("Routing and frame identities do not align.")
    if np.any(topk < 0) or np.any(topk >= n_experts) or not np.isfinite(weights).all():
        raise ValueError("Invalid router output.")
    counts = np.zeros((frames, topk.shape[0], n_experts), dtype=np.int32)
    mass = np.zeros_like(counts, dtype=np.float32)
    for frame in range(frames):
        selected = frame_ids == frame
        if not selected.any():
            raise ValueError(f"Frame {frame} has no visual tokens.")
        for layer in range(topk.shape[0]):
            ids = topk[layer, selected].ravel()
            counts[frame, layer] = np.bincount(ids, minlength=n_experts)
            mass[frame, layer] = np.bincount(ids, weights=weights[layer, selected].ravel(), minlength=n_experts)
            total = mass[frame, layer].sum()
            if total <= 0:
                raise ValueError("Router assigned zero probability mass.")
            mass[frame, layer] /= total
    return mass, counts, np.bincount(frame_ids, minlength=frames)
