import argparse
import gc
import json
import os
import random
import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from core import ALL_CONDITIONS, LETTERS, broad_category, generate_permutation_bank


DEFAULT_REMOTE_MODEL = "deepseek-ai/deepseek-vl2-tiny"


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "trc_accuracy_outputs"
    return Path(__file__).resolve().parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Paired temporal-perturbation experiment relating TRC to NExT-QA performance."
    )
    parser.add_argument(
        "--model",
        default="auto",
        help=(
            "Model checkpoint path or Hugging Face ID. 'auto' first searches /kaggle/input "
            f"for an uploaded DeepSeek-VL2 checkpoint, then uses {DEFAULT_REMOTE_MODEL}."
        ),
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--annotations", default="lmms-eval/NExTQA")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-repo", default="VLM2Vec/nextqa-rawvideo")
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--exclude-manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--temporal-count", type=int, default=None)
    parser.add_argument("--causal-count", type=int, default=None)
    parser.add_argument("--descriptive-count", type=int, default=None)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--permutation-repetitions", type=int, default=2)
    parser.add_argument("--conditions", nargs="+", default=["all"])
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def torch_siglip_attention(self, x):
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    x = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=self.attn_drop.p if self.training else 0.0,
    )
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(x))


def category_targets(args):
    explicit = [args.temporal_count, args.causal_count, args.descriptive_count]
    if all(value is None for value in explicit):
        temporal = args.limit // 2
        remaining = args.limit - temporal
        causal = remaining // 2
        return {
            "temporal": temporal,
            "causal": causal,
            "descriptive": args.limit - temporal - causal,
        }
    if any(value is None for value in explicit):
        raise ValueError("Set all three category counts or leave all three unset.")
    if any(value < 0 for value in explicit) or sum(explicit) != args.limit:
        raise ValueError("Category counts must be non-negative and sum to --limit.")
    return dict(zip(("temporal", "causal", "descriptive"), explicit))


def read_excluded_video_ids(path):
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"Exclusion manifest not found: {path}")
    if path.suffix.lower() == ".csv":
        import pandas as pd

        table = pd.read_csv(path)
        for column in ("video_id", "video", "id"):
            if column in table:
                return set(table[column].astype(str))
        raise ValueError(f"No video ID column found in {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("items", payload.get("manifest", []))
    excluded = set()
    for item in payload:
        if isinstance(item, dict):
            value = item.get("video_id", item.get("video"))
        else:
            value = item
        if value is not None:
            excluded.add(str(value))
    return excluded


def build_manifest(dataset, available_videos, targets, excluded_videos, seed, path):
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = sum(targets.values())
        if len(manifest) != expected:
            raise RuntimeError(
                f"Existing manifest has {len(manifest)} items but the run requests {expected}. "
                "Use a new output directory when changing the sample."
            )
        return manifest

    pools = {name: [] for name in targets}
    for index, example in enumerate(dataset):
        video_id = str(example["video"])
        category = broad_category(example["type"])
        if video_id in available_videos and video_id not in excluded_videos:
            pools[category].append(index)

    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)

    remaining = dict(targets)
    used_videos = set()
    manifest = []
    categories = ("temporal", "causal", "descriptive")
    while any(remaining.values()):
        progressed = False
        for category in categories:
            if remaining[category] == 0:
                continue
            selected = None
            while pools[category]:
                index = pools[category].pop()
                example = dataset[index]
                video_id = str(example["video"])
                if video_id not in used_videos:
                    selected = (index, example, video_id)
                    break
            if selected is None:
                continue
            index, example, video_id = selected
            used_videos.add(video_id)
            remaining[category] -= 1
            progressed = True
            manifest.append(
                {
                    "schema_version": 1,
                    "id": f"nextqa_{index:05d}",
                    "dataset_index": index,
                    "video_id": video_id,
                    "qid": int(example["qid"]),
                    "type": example["type"],
                    "category": category,
                    "question": example["question"],
                    "choices": [example[f"a{i}"] for i in range(5)],
                    "answer": int(example["answer"]),
                }
            )
        if not progressed:
            raise RuntimeError(
                "Not enough unique available videos to satisfy category targets. "
                f"Still needed: {remaining}"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def discover_local_model():
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        return None
    candidates = []
    weight_names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    try:
        configs = kaggle_input.rglob("config.json")
        for config_path in configs:
            directory = config_path.parent
            if not any((directory / name).exists() for name in weight_names):
                continue
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            architecture_text = " ".join(config.get("architectures", []))
            identity = f"{directory} {architecture_text} {config.get('model_type', '')}".lower()
            if "deepseek" in identity and ("vl2" in identity or "vl_v2" in identity):
                candidates.append(directory)
    except OSError:
        return None
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0] if candidates else None


def resolve_model_source(requested):
    if requested != "auto":
        return requested
    environment_path = os.environ.get("DEEPSEEK_MODEL_PATH")
    if environment_path:
        return environment_path
    local = discover_local_model()
    if local is not None:
        print(f"Using uploaded local checkpoint: {local}")
        return str(local)
    print(f"No uploaded checkpoint found; using Hugging Face model: {DEFAULT_REMOTE_MODEL}")
    return DEFAULT_REMOTE_MODEL


def list_available_videos(video_repo, video_dir):
    if video_dir is not None:
        paths = {path.stem: path for path in video_dir.rglob("*.mp4")}
        if not paths:
            raise RuntimeError(f"No .mp4 files found under {video_dir}")
        return set(paths), paths

    from huggingface_hub import list_repo_files

    available = {
        Path(filename).stem
        for filename in list_repo_files(video_repo, repo_type="dataset")
        if filename.lower().endswith(".mp4")
    }
    return available, None


def download_video(video_repo, video_id, local_video_paths):
    if local_video_paths is not None:
        return local_video_paths[video_id]
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=video_repo,
            filename=f"{video_id}.mp4",
            repo_type="dataset",
        )
    )


def nearby_frame_indices(target, total):
    """Yield valid frame indices from nearest to farthest, preferring the earlier frame on ties."""
    yield target
    maximum_distance = max(target, total - 1 - target)
    for distance in range(1, maximum_distance + 1):
        earlier = target - distance
        later = target + distance
        if earlier >= 0:
            yield earlier
        if later < total:
            yield later


def decode_frame_at(capture, index, total):
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        return None, None
    reported_position = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    actual_index = reported_position if 0 <= reported_position < total else int(index)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame), actual_index


def sample_frames(video_path, count):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine frame count for {video_path}")
    start = int(total * 0.02)
    stop = max(start, total - 1 - int(total * 0.02))
    requested_indices = np.linspace(start, stop, count).round().astype(int)
    frames = []
    actual_indices = []
    recoveries = []
    used_indices = set()
    for requested_index in requested_indices:
        selected_frame = None
        selected_index = None
        for candidate_index in nearby_frame_indices(int(requested_index), total):
            frame, actual_index = decode_frame_at(capture, candidate_index, total)
            if frame is None or actual_index in used_indices:
                continue
            selected_frame = frame
            selected_index = int(actual_index)
            break
        if selected_frame is None:
            capture.release()
            raise RuntimeError(
                f"Could not find a unique decodable frame near requested frame {requested_index} "
                f"in {video_path} ({total} frames reported)."
            )
        frames.append(selected_frame)
        actual_indices.append(selected_index)
        used_indices.add(selected_index)
        if selected_index != int(requested_index):
            recovery = {
                "requested_index": int(requested_index),
                "actual_index": selected_index,
                "offset": selected_index - int(requested_index),
            }
            recoveries.append(recovery)
            warnings.warn(
                f"{Path(video_path).name}: requested frame {requested_index} was unavailable; "
                f"using nearest decodable frame {selected_index}.",
                RuntimeWarning,
            )
    capture.release()
    return (
        frames,
        np.asarray(actual_indices, dtype=int),
        requested_indices,
        recoveries,
    )


def build_prompt(item, frame_count):
    lines = ["The following images are sampled frames from one video."]
    for index in range(frame_count):
        lines.extend([f"Frame {index + 1}:", "<image>"])
    lines.extend(["Question:", item["question"]])
    lines.extend(f"({LETTERS[index]}) {choice}" for index, choice in enumerate(item["choices"]))
    lines.append("Answer with only the option letter.")
    return "\n".join(lines)


def visual_frame_ids(visual_mask, expected_frames):
    positions = torch.where(visual_mask)[0]
    if positions.numel() == 0:
        raise RuntimeError("Processor did not mark any visual tokens.")
    starts = torch.cat(
        [
            torch.ones(1, dtype=torch.bool, device=positions.device),
            positions[1:] != positions[:-1] + 1,
        ]
    )
    group_ids = starts.cumsum(0) - 1
    found = int(group_ids.max().item()) + 1
    if found != expected_frames:
        raise RuntimeError(f"Expected {expected_frames} visual-token blocks, found {found}")
    frame_ids = torch.full_like(visual_mask, -1, dtype=torch.long)
    frame_ids[positions] = group_ids
    return frame_ids


def prepare_sequence(processor, model, item, frames):
    prompt = build_prompt(item, len(frames))
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [f"frame-{index}.jpg" for index in range(len(frames))],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepared = processor(
        conversations=conversation,
        images=frames,
        force_batchify=True,
        system_prompt="",
    ).to(model.device, dtype=torch.float16)
    with torch.inference_mode():
        prepared_embeddings = model.prepare_inputs_embeds(**prepared)
    if isinstance(prepared_embeddings, tuple):
        embeddings = prepared_embeddings[0]
        visual_mask = prepared_embeddings[2][0].bool()
    else:
        embeddings = prepared_embeddings
        visual_mask = prepared.images_seq_mask[0].bool()
    frame_ids = visual_frame_ids(visual_mask, len(frames))
    return {
        "prompt": prompt,
        "attention_mask": prepared.attention_mask,
        "embeddings": embeddings,
        "frame_ids": frame_ids,
    }


def capture_routes_and_logits(model, batch, frame_count):
    topk_records = []
    weight_records = []
    layer_names = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            hidden = inputs[0]
            topk = output[0].detach().reshape(hidden.shape[0], hidden.shape[1], -1)[0]
            weights = output[1].detach().reshape(hidden.shape[0], hidden.shape[1], -1)[0]
            visual = batch["frame_ids"] >= 0
            topk_records.append(topk[visual].cpu().to(torch.int16).numpy())
            weight_records.append(weights[visual].float().cpu().numpy())
            layer_names.append(name)

        return hook

    gates = [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == "MoEGate"]
    if not gates:
        raise RuntimeError("No MoEGate modules found; this checkpoint cannot produce TRC.")
    for name, gate in gates:
        handles.append(gate.register_forward_hook(make_hook(name)))
    try:
        with torch.inference_mode():
            outputs = model.language(
                inputs_embeds=batch["embeddings"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
        prompt_logits = outputs.logits[0, -1].float().detach()
    finally:
        for handle in handles:
            handle.remove()

    if len(topk_records) != len(gates):
        raise RuntimeError(f"Expected {len(gates)} router records, captured {len(topk_records)}.")
    topk = np.stack(topk_records)
    weights = np.stack(weight_records)
    visual_frame_id_values = batch["frame_ids"][batch["frame_ids"] >= 0].cpu().numpy()
    n_experts = int(gates[0][1].n_routed_experts)
    mass = np.zeros((frame_count, len(gates), n_experts), dtype=np.float32)
    counts = np.zeros((frame_count, len(gates), n_experts), dtype=np.int32)
    for frame in range(frame_count):
        token_mask = visual_frame_id_values == frame
        for layer in range(len(gates)):
            ids = topk[layer, token_mask].reshape(-1).astype(int)
            route_weights = weights[layer, token_mask].reshape(-1)
            mass[frame, layer] = np.bincount(ids, weights=route_weights, minlength=n_experts)
            counts[frame, layer] = np.bincount(ids, minlength=n_experts)
            total_mass = mass[frame, layer].sum()
            if total_mass <= 0:
                raise RuntimeError(f"Zero routed mass for frame {frame}, layer {layer}.")
            mass[frame, layer] /= total_mass
    tokens_per_frame = np.bincount(visual_frame_id_values, minlength=frame_count)
    return mass, counts, np.asarray(layer_names), tokens_per_frame, topk.shape[-1], prompt_logits


def completion_token_ids(tokenizer):
    encoded = {}
    for letter in LETTERS:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if not token_ids:
            raise RuntimeError(f"Tokenizer produced no tokens for answer letter {letter}.")
        encoded[letter] = [int(token_id) for token_id in token_ids]
    return encoded


def continuation_logprob(model, batch, token_ids, first_log_probs):
    score = first_log_probs[token_ids[0]]
    if len(token_ids) == 1:
        return float(score.item())
    prefix_ids = torch.tensor(token_ids[:-1], device=batch["embeddings"].device, dtype=torch.long).unsqueeze(0)
    prefix_embeddings = model.language.get_input_embeddings()(prefix_ids).to(batch["embeddings"].dtype)
    embeddings = torch.cat([batch["embeddings"], prefix_embeddings], dim=1)
    extension_mask = torch.ones(
        (batch["attention_mask"].shape[0], prefix_ids.shape[1]),
        dtype=batch["attention_mask"].dtype,
        device=batch["attention_mask"].device,
    )
    attention_mask = torch.cat([batch["attention_mask"], extension_mask], dim=1)
    with torch.inference_mode():
        outputs = model.language(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    prompt_length = batch["embeddings"].shape[1]
    for index in range(1, len(token_ids)):
        logits = outputs.logits[0, prompt_length + index - 1].float()
        score = score + torch.log_softmax(logits, dim=-1)[token_ids[index]]
    return float(score.item())


def score_answer_choices(model, batch, prompt_logits, token_ids_by_letter):
    first_log_probs = torch.log_softmax(prompt_logits, dim=-1)
    scores = {
        letter: continuation_logprob(model, batch, token_ids_by_letter[letter], first_log_probs)
        for letter in LETTERS
    }
    score_values = np.asarray([scores[letter] for letter in LETTERS], dtype=np.float64)
    probabilities = np.exp(score_values - score_values.max())
    probabilities /= probabilities.sum()
    predicted_index = int(np.argmax(score_values))
    return scores, {letter: float(probabilities[index]) for index, letter in enumerate(LETTERS)}, LETTERS[predicted_index]


def save_json_atomic(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def save_routes_atomic(path, **arrays):
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def selected_permutations(bank, requested_conditions):
    if requested_conditions == ["all"]:
        return bank
    unknown = set(requested_conditions) - set(ALL_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}. Valid values: {list(ALL_CONDITIONS)}")
    return [record for record in bank if record["condition"] in requested_conditions]


def scientific_config(args, targets):
    return {
        "schema_version": 1,
        "requested_model": args.model,
        "model_revision": args.model_revision,
        "annotations": args.annotations,
        "split": args.split,
        "video_repo": args.video_repo,
        "video_dir": str(args.video_dir) if args.video_dir else None,
        "exclude_manifest": str(args.exclude_manifest) if args.exclude_manifest else None,
        "limit": args.limit,
        "category_targets": targets,
        "frames": args.frames,
        "permutation_repetitions": args.permutation_repetitions,
        "seed": args.seed,
        "prompt_version": "neutral_sampled_frames_v1",
        "answer_scoring": "teacher_forced_option_letter_log_likelihood",
    }


def validate_or_save_config(output_dir, config):
    path = output_dir / "run_config.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "This output directory was created with a different scientific configuration. "
                "Use another --output-dir."
            )
    else:
        save_json_atomic(path, config)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select a T4 GPU accelerator in Kaggle.")
    if args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require --num-shards >= 1 and 0 <= --shard-index < --num-shards.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    targets = category_targets(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    route_dir = args.output_dir / "routes"
    prediction_dir = args.output_dir / "predictions"
    manifest_dir = args.output_dir / "manifests"
    for directory in (route_dir, prediction_dir, manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)
    validate_or_save_config(args.output_dir, scientific_config(args, targets))

    from datasets import load_dataset

    print("Loading NExT-QA annotations and checking available videos...")
    dataset = load_dataset(args.annotations, "MC", split=args.split)
    available_videos, local_video_paths = list_available_videos(args.video_repo, args.video_dir)
    excluded_videos = read_excluded_video_ids(args.exclude_manifest)
    manifest = build_manifest(
        dataset,
        available_videos,
        targets,
        excluded_videos,
        args.seed,
        manifest_dir / "nextqa_manifest.json",
    )

    permutation_path = manifest_dir / "permutations.json"
    generated_bank = generate_permutation_bank(args.frames, args.seed + 7919, args.permutation_repetitions)
    if permutation_path.exists():
        bank = json.loads(permutation_path.read_text(encoding="utf-8"))
        if bank != generated_bank:
            raise RuntimeError("Existing permutation bank differs from the requested configuration.")
    else:
        bank = generated_bank
        save_json_atomic(permutation_path, bank)
    permutations = selected_permutations(bank, args.conditions)
    shard_manifest = [
        item for index, item in enumerate(manifest) if index % args.num_shards == args.shard_index
    ]

    pending = []
    for item in shard_manifest:
        for spec in permutations:
            stem = f"{item['id']}__{spec['key']}"
            if not (route_dir / f"{stem}.npz").exists() or not (prediction_dir / f"{stem}.json").exists():
                pending.append((item, spec))
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: {len(shard_manifest)} questions, "
        f"{len(pending)} unfinished question-condition runs."
    )
    if not pending:
        print("Everything requested is already complete; no model load is needed.")
        return

    model_source = resolve_model_source(args.model)
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.models.siglip_vit import Attention as SiglipAttention

    SiglipAttention.forward = torch_siglip_attention
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "local_files_only": args.local_files_only,
    }
    if args.model_revision:
        load_kwargs["revision"] = args.model_revision
    print(f"Loading {model_source} in FP16 on {torch.cuda.get_device_name(0)}...")
    processor = DeepseekVLV2Processor.from_pretrained(
        model_source,
        revision=args.model_revision,
        local_files_only=args.local_files_only,
    )
    tokenizer = processor.tokenizer
    token_ids_by_letter = completion_token_ids(tokenizer)
    print(f"Answer-letter tokenization: {token_ids_by_letter}")
    model = DeepseekVLV2ForCausalLM.from_pretrained(model_source, **load_kwargs).cuda().eval()

    @lru_cache(maxsize=4)
    def load_frames(video_id):
        video_path = download_video(args.video_repo, video_id, local_video_paths)
        frames, indices, requested_indices, recoveries = sample_frames(video_path, args.frames)
        return (
            tuple(frames),
            tuple(int(index) for index in indices),
            tuple(int(index) for index in requested_indices),
            tuple(recoveries),
        )

    pending_keys = {(item["id"], spec["key"]) for item, spec in pending}
    progress = tqdm(total=len(pending), desc="Experiment 3")
    for item in shard_manifest:
        item_specs = [spec for spec in permutations if (item["id"], spec["key"]) in pending_keys]
        if not item_specs:
            continue
        frames, chronological_indices, requested_indices, frame_decode_recoveries = load_frames(
            item["video_id"]
        )
        frames = list(frames)
        chronological_indices = np.asarray(chronological_indices)
        requested_indices = np.asarray(requested_indices)
        for spec in item_specs:
            stem = f"{item['id']}__{spec['key']}"
            route_path = route_dir / f"{stem}.npz"
            prediction_path = prediction_dir / f"{stem}.json"
            order = np.asarray(spec["order"], dtype=int)
            selected_frames = [frames[index] for index in order]
            selected_indices = chronological_indices[order]
            batch = prepare_sequence(processor, model, item, selected_frames)
            mass, counts, layer_names, tokens_per_frame, router_topk, prompt_logits = capture_routes_and_logits(
                model, batch, args.frames
            )
            choice_logprobs, choice_probabilities, predicted = score_answer_choices(
                model, batch, prompt_logits, token_ids_by_letter
            )
            target = LETTERS[item["answer"]]
            wrong_scores = [choice_logprobs[letter] for letter in LETTERS if letter != target]
            margin = float(choice_logprobs[target] - max(wrong_scores))
            prediction = {
                "schema_version": 1,
                **item,
                "condition_key": spec["key"],
                "condition": spec["condition"],
                "replicate": spec["replicate"],
                "severity_rank": spec["severity_rank"],
                "order": spec["order"],
                "kendall_distance": spec["kendall_distance"],
                "adjacency_retention": spec["adjacency_retention"],
                "adjacency_destruction": spec["adjacency_destruction"],
                "target": target,
                "predicted": predicted,
                "correct": predicted == target,
                "correct_option_margin": margin,
                "choice_logprobs": choice_logprobs,
                "choice_probabilities": choice_probabilities,
                "choice_token_ids": token_ids_by_letter,
                "requested_chronological_frame_indices": requested_indices.tolist(),
                "chronological_frame_indices": chronological_indices.tolist(),
                "presented_frame_indices": selected_indices.tolist(),
                "frame_decode_recoveries": list(frame_decode_recoveries),
                "model_source": model_source,
                "inference_mode": "sampled_frames_as_multi_image_sequence",
            }
            save_routes_atomic(
                route_path,
                schema_version=np.asarray(1),
                mass=mass,
                counts=counts,
                layer_names=layer_names,
                tokens_per_frame=tokens_per_frame,
                requested_chronological_frame_indices=requested_indices,
                chronological_frame_indices=chronological_indices,
                presented_frame_indices=selected_indices,
                order=order,
                n_experts=np.asarray(mass.shape[-1]),
                router_topk=np.asarray(router_topk),
            )
            save_json_atomic(prediction_path, prediction)
            del batch, mass, counts, prompt_logits
            progress.update(1)
        gc.collect()
        torch.cuda.empty_cache()
    progress.close()
    print(f"Completed requested runs. Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
