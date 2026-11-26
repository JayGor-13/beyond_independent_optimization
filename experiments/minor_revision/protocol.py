"""Experiment definitions, strict answer parsing, and immutable run metadata."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = 1
MODELS = {
    "qwen35_122b": "Qwen/Qwen3.5-122B-A10B",
    "qwen35_35b": "Qwen/Qwen3.5-35B-A3B",
    "qwen36_35b": "Qwen/Qwen3.6-35B-A3B",
}
MODEL_REVISIONS = {
    "qwen35_122b": "dc4d348443bc740c68e2d77492492c11606384d5",
    "qwen35_35b": "2508b2437e382c605eb0fc52b83f5545960f6b4a",
    "qwen36_35b": "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
}
BENCHMARKS = {
    "scienceqa": {"repo": "derek-thomas/ScienceQA", "config": None, "split": "test"},
    "mmmu_pro": {"repo": "MMMU/MMMU_Pro", "config": "standard (10 options)", "split": "test"},
    "mathvista": {"repo": "AI4Math/MathVista", "config": None, "split": "testmini"},
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def lock_json(path, value):
    path = Path(path)
    with file_lock(path.with_name(path.name + ".lock")):
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Configuration or sample mismatch: {path}. Use a new output directory.")
        write_json(path, value)


@contextlib.contextmanager
def file_lock(path):
    """Linux/macOS process lock; released automatically even if a run crashes."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process holds {path}; do not run two writers together.") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def resolve_revisions(output_root, model_keys, benchmark_keys):
    """Use checked model commits; resolve dataset main only on first use."""
    from huggingface_hub import HfApi
    path = Path(output_root) / "revisions.json"
    with file_lock(path.with_suffix(".lock")):
        pins = json.loads(path.read_text()) if path.exists() else {"schema": SCHEMA, "models": {}, "datasets": {}}
        if pins["schema"] != SCHEMA:
            raise ValueError("Unsupported revision manifest.")
        api = HfApi()
        for kind, keys, registry in (("models", model_keys, MODELS), ("datasets", benchmark_keys, BENCHMARKS)):
            for key in keys:
                repo = registry[key] if kind == "models" else registry[key]["repo"]
                if key not in pins[kind]:
                    info = api.model_info(repo, revision=MODEL_REVISIONS[key]) if kind == "models" else api.dataset_info(repo)
                    pins[kind][key] = {"repo": repo, "revision": info.sha}
                if pins[kind][key]["repo"] != repo or not re.fullmatch(r"[0-9a-f]{40}", pins[kind][key]["revision"]):
                    raise ValueError(f"Invalid immutable revision for {kind}/{key}.")
                if kind == "models" and pins[kind][key]["revision"] != MODEL_REVISIONS[key]:
                    raise ValueError(f"Model commit differs from this package's checked revision: {key}.")
        write_json(path, pins)
    return pins


def code_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def example_seed(example_id, seed):
    value = hashlib.sha256(f"tmlr-minor-v1:{example_id}:{seed}".encode()).digest()
    return int.from_bytes(value[:8], "little")


def conditions(retentions, seeds):
    if not retentions or any(not math.isfinite(r) or not 0 < r < 1 for r in retentions):
        raise ValueError("Pruning retentions must be finite and strictly between 0 and 1.")
    tags = [str(round(r * 10000)) for r in retentions]
    if len(set(tags)) != len(tags) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Retention names and random seeds must be unique; provide at least one seed.")
    result = [
        {"name": "baseline", "method": "baseline", "retention": 1.0, "seed": None},
        {"name": "l2_r10000", "method": "l2", "retention": 1.0, "seed": None},
        {"name": "random_r10000_s0", "method": "random", "retention": 1.0, "seed": 0},
    ]
    for retention, tag in zip(retentions, tags):
        result.append({"name": f"l2_r{tag}", "method": "l2", "retention": retention, "seed": None})
        result.extend({"name": f"random_r{tag}_s{seed}", "method": "random", "retention": retention, "seed": seed}
                      for seed in seeds)
    return result


def final_answer(text):
    """No judge model and no guessing from a reasoning trace."""
    text = str(text).strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    elif "<think>" in text:
        return ""
    boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}[.!]?", text)
    if boxed:
        text = boxed.group(1)
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*|:\s*)", "", text, flags=re.I)
    return text.strip()


def option_index(text, choices):
    text = final_answer(text)
    match = re.fullmatch(r"\(?([A-Z])\)?[.!]?", text, flags=re.I)
    if match and LETTERS.index(match.group(1).upper()) < len(choices):
        return LETTERS.index(match.group(1).upper())
    exact = [i for i, choice in enumerate(choices) if str(choice).strip().casefold() == text.casefold()]
    return exact[0] if len(exact) == 1 else None


def score_answer(text, item):
    if item["choices"]:
        index = option_index(text, item["choices"])
        return {"prediction": LETTERS[index] if index is not None else "", "valid_answer": index is not None,
                "correct": index is not None and index == item["target_index"]}
    answer = final_answer(text)
    try:
        kind = item["answer_type"]
        if kind == "integer":
            number = float(answer)
            if not math.isfinite(number):
                raise ValueError("Non-finite answer")
            prediction = str(int(number))  # MathVista's published numeric normalization.
        elif kind == "float":
            number = float(answer)
            if not math.isfinite(number):
                raise ValueError("Non-finite answer")
            prediction = str(round(number, int(item["precision"])))
        elif kind == "list":
            if not isinstance(ast.literal_eval(answer), list):
                raise ValueError("Expected a list")
            prediction = answer
        else:
            raise ValueError(f"Unsupported answer type: {kind}")
        return {"prediction": prediction, "valid_answer": True, "correct": prediction == str(item["target"])}
    except (ValueError, TypeError, OverflowError, SyntaxError):
        return {"prediction": "", "valid_answer": False, "correct": False}
