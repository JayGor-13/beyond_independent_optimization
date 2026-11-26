"""Run only InternVL3.5 and Qwen3-VL replications; never launches DeepSeek."""
import argparse
import importlib.metadata
import json
import os
import random
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")

INTERNVL_RETENTIONS = [1.0, 0.25, 0.50, 0.75, 0.70]

from core import (CONDITIONS, LETTERS, MODELS, aggregate_frames, answer_letter, compress,
                  cross_indices, lock_json, nextqa_manifest, question_text, science_manifest, write_json)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--experiments", choices=[1, 2], type=int, nargs="+", default=[1, 2])
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="Use two examples in separate smoke output directories.")
    parser.add_argument("--retentions", type=float, nargs="+",
                        help="Fractions of visual tokens retained, including 1.0 for 100%%.")
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--precision", choices=["expert-nf4", "float16", "bfloat16"], default="expert-nf4")
    parser.add_argument("--max-pixels", type=int, default=262144, help="Qwen image pixel limit (InternVL uses one 448px tile).")
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/new_model_results"))
    parser.add_argument("--scratch", type=Path, default=Path("/kaggle/working/model_shards"))
    args = parser.parse_args()
    if args.retentions is None:
        args.retentions = list(INTERNVL_RETENTIONS) if args.model == "internvl35" else [0.5]
    if args.smoke:
        args.limit = 2
    if args.limit < 1 or (2 in args.experiments and (args.limit < 2 or args.frames < 2)):
        parser.error("Experiment 2 needs at least two videos and two frames.")
    if any(not 0 < r <= 1 for r in args.retentions) or args.max_new_tokens < 1 or args.max_pixels < 65536:
        parser.error("Use 0 < retention <= 1, max-new-tokens >= 1, and max-pixels >= 65536.")
    names = [f"r{round(r * 100)}" for r in args.retentions]
    if len(set(names)) != len(names) or len(set(args.random_seeds)) != len(args.random_seeds):
        parser.error("Retention condition names and random seeds must be unique.")
    return args


def output_path(args, experiment):
    phase = "smoke" if args.smoke else f"n{args.limit}"
    return args.output_root / args.model / f"experiment_{experiment}_{phase}"


def save_npz(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def config_for(args, experiment, dataset_revision, video_revision=None):
    return {"schema_version": 1, "adapter_version": 1, "model_alias": args.model,
            "model": MODELS[args.model][0], "model_revision": MODELS[args.model][1],
            "experiment": experiment, "limit": args.limit, "smoke": args.smoke,
            "seed": 20260818 if experiment == 1 else 20260819,
            "precision": args.precision, "max_new_tokens": args.max_new_tokens,
            "max_pixels": args.max_pixels if args.model == "qwen3vl" else None,
            "internvl_tiles_per_image": 1 if args.model == "internvl35" else None,
            "dataset": "derek-thomas/ScienceQA" if experiment == 1 else "lmms-eval/NExTQA",
            "dataset_revision": dataset_revision, "video_revision": video_revision,
            "protocol": "visual_token_pruning" if experiment == 1 else "sampled_frames_as_separate_images",
            "position_policy": "preserve_native_mrope" if args.model == "qwen3vl" else "reindex_retained_sequence",
            "retentions": args.retentions if experiment == 1 else None,
            "random_seeds": args.random_seeds if experiment == 1 else None,
            "frames": args.frames if experiment == 2 else None,
            "cross_clip_policy": "one_frame_per_other_video_cycling_excluding_target_v1" if experiment == 2 else None,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "accelerate", "bitsandbytes", "datasets")}}


def prepare_experiment(args, experiment):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    dataset_id = "derek-thomas/ScienceQA" if experiment == 1 else "lmms-eval/NExTQA"
    root = output_path(args, experiment)
    # Reuse pinned data revisions on resume rather than silently following main.
    old = json.loads((root / "run_config.json").read_text()) if (root / "run_config.json").exists() else {}
    revision = old.get("dataset_revision") or api.dataset_info(dataset_id).sha
    video_revision = None
    if experiment == 2:
        video_revision = old.get("video_revision") or api.dataset_info("VLM2Vec/nextqa-rawvideo").sha
    config = config_for(args, experiment, revision, video_revision)
    lock_json(root / "run_config.json", config)
    dataset = load_dataset(dataset_id, *(["MC"] if experiment == 2 else []), split="test", revision=revision)
    files = {}
    if experiment == 1:
        manifest = science_manifest(dataset, args.limit, config["seed"])
    else:
        for filename in api.list_repo_files("VLM2Vec/nextqa-rawvideo", repo_type="dataset", revision=video_revision):
            if filename.endswith(".mp4"):
                key = Path(filename).stem
                if key in files:
                    raise ValueError(f"Duplicate video filename stem: {key}")
                files[key] = filename
        manifest = nextqa_manifest(dataset, set(files), args.limit, config["seed"])
    lock_json(root / "manifests" / "examples.json", manifest)
    # In one output root both models must use identical data and sample identities.
    phase = "smoke" if args.smoke else f"n{args.limit}"
    lock_json(args.output_root / "shared_manifests" / f"experiment_{experiment}_{phase}.json",
              {"dataset_revision": revision, "video_revision": video_revision, "examples": manifest})
    return root, config, dataset, manifest, files


def experiment1(args, adapter, prepared):
    from tqdm import tqdm
    root, config, dataset, manifest, _ = prepared
    conditions = [("baseline", "baseline", 1.0, None)]
    for retention in args.retentions:
        conditions.append((f"l2_r{round(retention * 100)}", "l2", retention, None))
        conditions.extend((f"random_r{round(retention * 100)}_s{seed}", "random", retention, seed)
                          for seed in args.random_seeds)
    checked = False
    for item in tqdm(manifest, desc=f"{args.model} experiment 1"):
        pending = [(name, method, ratio, seed) for name, method, ratio, seed in conditions
                   if not all((root / "conditions" / name / kind / f"{item['id']}.{ext}").exists()
                              for kind, ext in [("predictions", "json"), ("routes", "npz")])]
        if not pending:
            continue
        example = dataset[item["dataset_index"]]
        batch = adapter.prepare([example["image"]], question_text(example))
        if not checked:
            write_json(root / "native_equivalence.json", adapter.check_native(batch))
            checked = True
        batch.pop("native")
        for name, method, retention, seed in pending:
            condition = compress(batch, method, retention, config["seed"] + item["dataset_index"] * 1009 + (seed or 0))
            topk, _, layers, answer = adapter.evaluate(condition, args.max_new_tokens)
            predicted, target = answer_letter(answer), LETTERS[example["answer"]]
            directory = root / "conditions" / name
            save_npz(directory / "routes" / f"{item['id']}.npz", topk=topk,
                     visual_ids=condition["visual_ids"].cpu().numpy(), layer_names=layers,
                     n_experts=np.asarray(adapter.n_experts))
            write_json(directory / "predictions" / f"{item['id']}.json",
                       {**item, "schema_version": 2, "model": config["model"], "model_revision": config["model_revision"],
                        "condition": name, "method": method, "retention": retention, "random_seed": seed,
                        "target": target, "predicted": predicted, "correct": predicted == target,
                        "raw_answer": answer, "visual_tokens_before": int(batch["visual_mask"].sum()),
                        "visual_tokens_after": int(condition["visual_mask"].sum())})
        del batch, condition
        torch.cuda.empty_cache()
    write_json(root / "complete.json", {"examples": len(manifest), "conditions": [c[0] for c in conditions]})


def sample_frames(path, count):
    import cv2
    from PIL import Image
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            raise ValueError(f"Video has no decodable frames: {path}")
        start = int(total * 0.02)
        stop = max(start, total - 1 - int(total * 0.02))
        indices = np.linspace(start, stop, count).round().astype(int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Failed reading frame {index} in {path}")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames, indices
    finally:
        capture.release()


def experiment2(args, adapter, prepared):
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    root, config, _, manifest, files = prepared
    args.scratch.mkdir(parents=True, exist_ok=True)

    @lru_cache(maxsize=12)
    def get_frames(video_id):
        with tempfile.TemporaryDirectory(prefix="video_", dir=args.scratch) as directory:
            path = hf_hub_download("VLM2Vec/nextqa-rawvideo", files[video_id], repo_type="dataset",
                                   revision=config["video_revision"], local_dir=directory)
            return sample_frames(path, args.frames)

    checked = False
    for i, item in enumerate(tqdm(manifest, desc=f"{args.model} experiment 2")):
        if all((root / "routes" / f"{item['id']}__{name}.npz").exists() and
               (root / "condition_predictions" / name / f"{item['id']}.json").exists() for name in CONDITIONS) and (
                   root / "predictions" / f"{item['id']}.json").exists():
            continue
        frames, indices = get_frames(item["video_id"])
        others = cross_indices(i, len(manifest), args.frames)
        for name in CONDITIONS:
            path = root / "routes" / f"{item['id']}__{name}.npz"
            pred_path = root / "condition_predictions" / name / f"{item['id']}.json"
            if path.exists() and pred_path.exists():
                if name == "chronological":
                    write_json(root / "predictions" / f"{item['id']}.json", json.loads(pred_path.read_text()))
                continue
            order = list(range(args.frames))
            source_ids = [item["video_id"]] * args.frames
            if name == "shuffled":
                random.Random(config["seed"] + item["dataset_index"]).shuffle(order)
            elif name == "duplicate":
                order = [args.frames // 2] * args.frames
            if name == "cross_clip":
                selected = [get_frames(manifest[j]["video_id"])[0][f] for f, j in enumerate(others)]
                selected_indices = [int(get_frames(manifest[j]["video_id"])[1][f]) for f, j in enumerate(others)]
                source_ids = [manifest[j]["video_id"] for j in others]
            else:
                selected = [frames[j] for j in order]
                selected_indices = [int(indices[j]) for j in order]
            batch = adapter.prepare(selected, question_text(item), temporal=True)
            if not checked:
                write_json(root / "native_equivalence.json", adapter.check_native(batch))
                checked = True
            batch.pop("native")
            topk, weights, layers, answer = adapter.evaluate(batch, args.max_new_tokens)
            frame_ids = batch["frame_ids"][batch["visual_mask"]].cpu().numpy()
            mass, counts, tokens = aggregate_frames(topk, weights, frame_ids, adapter.n_experts, args.frames)
            save_npz(path, mass=mass, counts=counts, layer_names=layers, tokens_per_frame=tokens,
                     frame_indices=np.asarray(selected_indices), source_video_ids=np.asarray(source_ids),
                     n_experts=np.asarray(adapter.n_experts), router_topk=np.asarray(adapter.top_k))
            predicted, target = answer_letter(answer), LETTERS[item["answer"]]
            prediction = {**item, "model": config["model"], "condition": name,
                          "raw_answer": answer, "predicted": predicted, "target": target,
                          "correct": predicted == target, "frame_indices": selected_indices,
                          "source_video_ids": source_ids}
            write_json(pred_path, prediction)
            if name == "chronological":
                write_json(root / "predictions" / f"{item['id']}.json", prediction)
            del batch
            torch.cuda.empty_cache()
    get_frames.cache_clear()
    write_json(root / "complete.json", {"examples": len(manifest), "conditions": list(CONDITIONS)})


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU in Kaggle; select T4 x2 for the default expert-NF4 mode.")
    if args.precision == "bfloat16" and not all(torch.cuda.get_device_capability(i)[0] >= 8 for i in range(torch.cuda.device_count())):
        raise ValueError("This GPU does not support BF16; use expert-nf4 or float16.")
    random.seed(20260818)
    np.random.seed(20260818)
    torch.manual_seed(20260818)
    prepared = {experiment: prepare_experiment(args, experiment) for experiment in args.experiments}
    from analyze import validate
    for experiment in list(prepared):
        if (prepared[experiment][0] / "complete.json").exists():
            try:
                validate(prepared[experiment][0])
            except (ValueError, FileNotFoundError):
                continue
            print(f"Experiment {experiment} already complete; skipping model execution.")
            del prepared[experiment]
    if not prepared:
        return
    from loading import load_model
    from adapters import Adapter
    dtype = torch.bfloat16 if args.precision == "bfloat16" else torch.float16
    model, loading_info = load_model(args.model, dtype, args.precision == "expert-nf4", args.scratch)
    adapter = Adapter(model, args.model, dtype, args.max_pixels)
    for experiment, data in prepared.items():
        write_json(data[0] / "loading_info.json", {**loading_info, "n_experts": adapter.n_experts,
                                                  "top_k": adapter.top_k, "routed_layers": len(adapter.blocks)})
        (experiment1 if experiment == 1 else experiment2)(args, adapter, data)
        print(f"Completed experiment {experiment}: {data[0]}", flush=True)


if __name__ == "__main__":
    main()
