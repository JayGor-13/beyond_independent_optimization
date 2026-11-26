import argparse
import json
import random
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor
from deepseek_vl2.models.siglip_vit import Attention as SiglipAttention


LETTERS = "ABCDE"
CONDITIONS = ("chronological", "shuffled", "duplicate", "cross_clip")


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "trc_outputs"
    return Path(__file__).resolve().parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument("--annotations", default="lmms-eval/NExTQA")
    parser.add_argument("--video-repo", default="VLM2Vec/nextqa-rawvideo")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--max-new-tokens", type=int, default=8)
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


def broad_category(question_type):
    if question_type in {"TN", "TP", "TC"}:
        return "temporal"
    if question_type in {"CW", "CH"}:
        return "causal"
    return "descriptive"


def build_manifest(dataset, available_videos, limit, seed, path):
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if all(item["video_id"] in available_videos for item in manifest):
            return manifest

    pools = {name: [] for name in ("temporal", "causal", "descriptive")}
    for index, example in enumerate(dataset):
        if str(example["video"]) in available_videos:
            pools[broad_category(example["type"])].append(index)

    rng = random.Random(seed)
    for pool in pools.values():
        rng.shuffle(pool)

    manifest = []
    used_videos = set()
    categories = list(pools)
    while len(manifest) < limit:
        category = categories[len(manifest) % len(categories)]
        index = pools[category].pop()
        example = dataset[index]
        video_id = str(example["video"])
        if video_id in used_videos:
            continue
        used_videos.add(video_id)
        manifest.append(
            {
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def download_video(video_repo, video_id):
    return hf_hub_download(
        repo_id=video_repo,
        filename=f"{video_id}.mp4",
        repo_type="dataset",
    )


def sample_frames(video_path, count):
    capture = cv2.VideoCapture(str(video_path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start = int(total * 0.02)
    stop = max(start, total - 1 - int(total * 0.02))
    indices = np.linspace(start, stop, count).round().astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    capture.release()
    return frames, indices


def build_prompt(item, frame_count):
    lines = ["The following images are chronological frames from one video."]
    for index in range(frame_count):
        lines.extend([f"Frame {index + 1}:", "<image>"])
    lines.extend(["Question:", item["question"]])
    lines.extend(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(item["choices"]))
    lines.append("Answer with only the option letter.")
    return "\n".join(lines)


def prepare_sequence(processor, model, item, frames):
    prompt = build_prompt(item, len(frames))
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": [f"frame-{i}.jpg" for i in range(len(frames))],
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
        embeddings = model.prepare_inputs_embeds(**prepared)

    frame_ids = visual_frame_ids(prepared.images_seq_mask[0].bool(), len(frames))
    return {
        "prompt": prompt,
        "input_ids": prepared.input_ids,
        "attention_mask": prepared.attention_mask,
        "embeddings": embeddings,
        "frame_ids": frame_ids,
    }


def visual_frame_ids(visual_mask, expected_frames):
    positions = torch.where(visual_mask)[0]
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


def capture_routes(model, batch, frame_count):
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
    for name, gate in gates:
        handles.append(gate.register_forward_hook(make_hook(name)))

    with torch.inference_mode():
        model.language.model(
            inputs_embeds=batch["embeddings"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
        )

    for handle in handles:
        handle.remove()

    topk = np.stack(topk_records)
    weights = np.stack(weight_records)
    frame_ids = batch["frame_ids"][batch["frame_ids"] >= 0].cpu().numpy()
    n_experts = int(gates[0][1].n_routed_experts)
    mass = np.zeros((frame_count, len(gates), n_experts), dtype=np.float32)
    counts = np.zeros((frame_count, len(gates), n_experts), dtype=np.int32)

    for frame in range(frame_count):
        token_mask = frame_ids == frame
        for layer in range(len(gates)):
            ids = topk[layer, token_mask].reshape(-1).astype(int)
            route_weights = weights[layer, token_mask].reshape(-1)
            mass[frame, layer] = np.bincount(ids, weights=route_weights, minlength=n_experts)
            counts[frame, layer] = np.bincount(ids, minlength=n_experts)
            mass[frame, layer] /= mass[frame, layer].sum()

    tokens_per_frame = np.bincount(frame_ids, minlength=frame_count)
    return mass, counts, np.asarray(layer_names), tokens_per_frame, topk.shape[-1]


def generate_answer(model, tokenizer, batch, max_new_tokens):
    with torch.inference_mode():
        output = model.generate(
            input_ids=batch["input_ids"],
            inputs_embeds=batch["embeddings"],
            attention_mask=batch["attention_mask"],
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = output[0, batch["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_letter(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def condition_frames(name, frames, indices, cross_frames, cross_indices, seed):
    if name == "chronological":
        return frames, indices
    if name == "duplicate":
        middle = len(frames) // 2
        return [frames[middle]] * len(frames), np.repeat(indices[middle], len(frames))
    if name == "cross_clip":
        return cross_frames, cross_indices
    order = list(range(len(frames)))
    random.Random(seed).shuffle(order)
    return [frames[i] for i in order], indices[order]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select a GPU accelerator in Kaggle.")

    route_dir = args.output_dir / "routes"
    prediction_dir = args.output_dir / "predictions"
    manifest_dir = args.output_dir / "manifests"
    route_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    SiglipAttention.forward = torch_siglip_attention
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.annotations, "MC", split="test")
    available_videos = {
        Path(filename).stem
        for filename in list_repo_files(args.video_repo, repo_type="dataset")
        if filename.endswith(".mp4")
    }
    manifest_path = manifest_dir / f"nextqa_{args.limit}_seed{args.seed}.json"
    manifest = build_manifest(dataset, available_videos, args.limit, args.seed, manifest_path)

    processor = DeepseekVLV2Processor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).cuda().eval()

    @lru_cache(maxsize=16)
    def load_frames(video_id):
        video_path = download_video(args.video_repo, video_id)
        frames, indices = sample_frames(video_path, args.frames)
        return tuple(frames), tuple(indices.tolist())

    for item_index, item in enumerate(tqdm(manifest, desc="NExT-QA TRC")):
        frames, frame_indices = load_frames(item["video_id"])
        frames = list(frames)
        frame_indices = np.asarray(frame_indices)

        cross_frames = []
        cross_indices = []
        for frame_index in range(args.frames):
            other = manifest[(item_index + frame_index + 1) % len(manifest)]
            other_frames, other_indices = load_frames(other["video_id"])
            cross_frames.append(other_frames[frame_index])
            cross_indices.append(other_indices[frame_index])
        cross_indices = np.asarray(cross_indices)

        for condition in CONDITIONS:
            route_path = route_dir / f"{item['id']}__{condition}.npz"
            prediction_path = prediction_dir / f"{item['id']}.json"
            need_answer = condition == "chronological" and not prediction_path.exists()
            if route_path.exists() and not need_answer:
                continue

            selected_frames, selected_indices = condition_frames(
                condition,
                frames,
                frame_indices,
                cross_frames,
                cross_indices,
                args.seed + item["dataset_index"],
            )
            batch = prepare_sequence(processor, model, item, selected_frames)

            if not route_path.exists():
                mass, counts, layer_names, tokens_per_frame, router_topk = capture_routes(
                    model, batch, args.frames
                )
                np.savez_compressed(
                    route_path,
                    mass=mass,
                    counts=counts,
                    layer_names=layer_names,
                    tokens_per_frame=tokens_per_frame,
                    frame_indices=np.asarray(selected_indices),
                    n_experts=np.asarray(mass.shape[-1]),
                    router_topk=np.asarray(router_topk),
                )

            if need_answer:
                raw_answer = generate_answer(model, tokenizer, batch, args.max_new_tokens)
                predicted = answer_letter(raw_answer)
                target = LETTERS[item["answer"]]
                prediction = {
                    **item,
                    "target": target,
                    "predicted": predicted,
                    "correct": predicted == target,
                    "raw_answer": raw_answer,
                    "frame_indices": frame_indices.tolist(),
                }
                prediction_path.write_text(
                    json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8"
                )

            del batch
            torch.cuda.empty_cache()

    print(f"Saved routes and predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
