"""Pinned benchmark adapters. Answers/explanations never enter the prompt."""
from __future__ import annotations

import ast
import random
import re
from pathlib import Path

from protocol import BENCHMARKS, LETTERS, lock_json

IMAGE_REFERENCE = re.compile(r"<image\s*(\d+)>", re.I)


def as_choices(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or len(value) > len(LETTERS):
        raise ValueError("Malformed benchmark choices.")
    return [str(choice) for choice in value]


def normalize_row(benchmark, row, index):
    """Return a serializable problem plus an ordered list of PIL images."""
    item = {"id": f"{benchmark}_{index:06d}", "dataset_index": index, "benchmark": benchmark,
            "question": str(row["question"]), "answer_type": "choice", "precision": None}
    if benchmark == "scienceqa":
        images = [row["image"]] if row["image"] is not None else []
        item.update(source_id=str(index), choices=as_choices(row["choices"]),
                    target_index=int(row["answer"]), target=str(row["answer"]), category=str(row.get("subject", "")))
    elif benchmark == "mmmu_pro":
        images = [row[f"image_{n}"] for n in range(1, 8) if row.get(f"image_{n}") is not None]
        # Image numbers are semantic references; non-contiguous numbering must not be silently renumbered.
        if any(row.get(f"image_{n}") is None for n in range(1, len(images) + 1)):
            raise ValueError(f"Non-contiguous image references in {row['id']}.")
        choices = as_choices(row["options"])
        answer = str(row["answer"]).strip()
        if len(answer) != 1 or answer not in LETTERS[:len(choices)]:
            raise ValueError(f"Invalid MMMU-Pro answer in {row['id']}.")
        item.update(source_id=str(row["id"]), choices=choices, target_index=LETTERS.index(answer),
                    target=answer, category=str(row.get("subject", "")))
    elif benchmark == "mathvista":
        image = row.get("decoded_image")
        if image is None:
            raise ValueError("MathVista must supply decoded_image; no remote image fetching is performed.")
        images = [image]
        choices = as_choices(row.get("choices") or []) if row["question_type"] == "multi_choice" else []
        target = str(row["answer"])
        if choices and target not in choices:
            raise ValueError(f"MathVista gold answer is absent from choices: {row['pid']}.")
        item.update(source_id=str(row["pid"]), choices=choices, target=target,
                    target_index=choices.index(target) if choices else None,
                    answer_type=str(row["answer_type"]), precision=row.get("precision"), unit=row.get("unit"),
                    category=str((row.get("metadata") or {}).get("category", "")))
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if not images:
        raise ValueError(f"No image for {item['id']}.")
    if item["choices"] and not 0 <= item["target_index"] < len(item["choices"]):
        raise ValueError("Answer index is out of range.")
    item["image_count"] = len(images)
    return item, [image.convert("RGB") for image in images]


def prompt_content(item, images):
    text = "Question:\n" + item["question"]
    if item["choices"]:
        text += "\nOptions:\n" + "\n".join(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(item["choices"]))
        text += "\nAnswer with only the option letter. Do not include reasoning."
    else:
        if item.get("unit"):
            text += f"\nAnswer unit: {item['unit']}."
        if item["answer_type"] == "float":
            text += f"\nRound your answer to {int(item['precision'])} decimal places."
        text += "\nAnswer with only the final value or list. Do not include reasoning or units."
    references = {int(match.group(1)) for match in IMAGE_REFERENCE.finditer(text)}
    if references - set(range(1, len(images) + 1)):
        raise ValueError("Question references a missing image.")
    content, seen = [], set()
    for number, image in enumerate(images, 1):
        if number not in references:
            content.extend([{"type": "text", "text": f"Image {number}:"}, {"type": "image", "image": image}])
            seen.add(number)
    cursor = 0
    for match in IMAGE_REFERENCE.finditer(text):
        number = int(match.group(1))
        content.append({"type": "text", "text": text[cursor:match.start()] + f"Image {number}:"})
        if number not in seen:
            content.append({"type": "image", "image": images[number - 1]})
            seen.add(number)
        cursor = match.end()
    content.append({"type": "text", "text": text[cursor:]})
    return content


def load_benchmark(benchmark, revision, limit, seed, output_root, smoke=False):
    from datasets import load_dataset
    specification = BENCHMARKS[benchmark]
    arguments = [specification["repo"]]
    if specification["config"]:
        arguments.append(specification["config"])
    dataset = load_dataset(*arguments, split=specification["split"], revision=revision)
    if benchmark == "scienceqa":
        eligible = [i for i, value in enumerate(dataset["image"]) if value is not None]
    else:
        eligible = list(range(len(dataset)))
    # Full evaluations use dataset order. Subsets use the same seeded order across models.
    if limit:
        random.Random(seed).shuffle(eligible)
        if limit > len(eligible):
            raise ValueError(f"Requested {limit} examples, but only {len(eligible)} are eligible.")
        eligible = eligible[:limit]
    phase = "smoke" if smoke else "full"
    manifest = {"benchmark": benchmark, **specification, "revision": revision, "seed": seed,
                "requested_limit": limit, "indices": eligible, "phase": phase}
    path = Path(output_root) / "manifests" / f"{benchmark}_{phase}_n{len(eligible)}_seed{seed}.json"
    lock_json(path, manifest)
    return dataset, manifest
