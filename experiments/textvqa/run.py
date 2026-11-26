"""Experiment 1 replication on TextVQA using DeepSeek-VL2-Tiny."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm

from textvqa_metrics import extract_answer, normalize_answer, vqa_soft_score


def locate_base_dir() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        here.parent.parent / "compression_routing",
        here.parent.parent / "Experiment-1" / "compression_routing_kaggle",
        here.parent.parent / "compression_routing_kaggle",
    ]
    for candidate in candidates:
        if (candidate / "run.py").exists() and (candidate / "model_compat.py").exists():
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = sorted(kaggle_input.glob("**/compression_routing_kaggle/run.py"))
        if matches:
            return matches[-1].parent
    raise RuntimeError(
        "Could not find Experiment-1/compression_routing_kaggle. Upload the complete "
        "Experiment-1-TextVQA.zip through Kaggle Inputs > Models."
    )


BASE_DIR = locate_base_dir()
sys.path.insert(0, str(BASE_DIR))
SPEC = importlib.util.spec_from_file_location("experiment1_routing_core", BASE_DIR / "run.py")
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CORE)


def default_output_dir() -> Path:
    root = Path("/kaggle/working")
    return root / "compression_routing_exp1_textvqa" if root.exists() else Path(__file__).parent / "outputs_textvqa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument("--dataset", default="facebook/textvqa")
    parser.add_argument("--dataset-config", default="textvqa")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "l2", "random"], default=["baseline", "l2", "random"])
    ratios = parser.add_mutually_exclusive_group()
    ratios.add_argument("--compression-ratios", nargs="+", type=float, default=None)
    ratios.add_argument("--retentions", nargs="+", type=float, default=None)
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def build_manifest(dataset, limit: int, seed: int, path: Path) -> list[dict]:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if len(manifest) != limit:
            raise RuntimeError(f"Existing manifest contains {len(manifest)} examples, not {limit}. Use a new --output-dir.")
        return manifest
    if limit > len(dataset):
        raise ValueError(f"Requested {limit} examples but the split contains {len(dataset)}.")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = indices[:limit]
    manifest = [
        {
            "id": f"textvqa_val_{int(dataset[index]['question_id']):08d}",
            "dataset_index": index,
            "question_id": int(dataset[index]["question_id"]),
            "image_id": str(dataset[index]["image_id"]),
        }
        for index in selected
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    config = {
        "schema_version": 3,
        "task": "textvqa",
        "model": args.model,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt": "image + question; short answer only; no supplied OCR tokens",
    }
    path = output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"Run configuration does not match {path}. Use a different --output-dir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def prepare_example(processor, model, example):
    prompt = "<image>\n" + example["question"].strip() + "\nAnswer with only a short answer."
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": ["textvqa-image"]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    image = example["image"].convert("RGB")
    prepared = processor(
        conversations=conversation,
        images=[image],
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
    return {
        "prompt": prompt,
        "input_ids": prepared.input_ids,
        "attention_mask": prepared.attention_mask,
        "embeddings": embeddings,
        "visual_mask": visual_mask,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select a GPU accelerator in Kaggle.")
    retentions = CORE.resolve_retentions(args)
    conditions = CORE.condition_names(args.methods, retentions, args.random_seeds)
    print("Compression schedule: " + ", ".join(f"{1-r:.0%} removed -> {r:.0%} retained" for r in retentions))
    validate_run_config(args.output_dir, args)
    CORE.create_output_dirs(args.output_dir, conditions)
    CORE.SiglipAttention.forward = CORE.torch_siglip_attention
    torch.manual_seed(args.seed)

    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    manifest_path = args.output_dir / "manifests" / f"textvqa_{args.split}_{args.limit}_seed{args.seed}.json"
    manifest = build_manifest(dataset, args.limit, args.seed, manifest_path)

    processor = CORE.DeepseekVLV2Processor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = CORE.DeepseekVLV2ForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).cuda().eval()
    CORE.patch_eager_attention(model)

    for item in tqdm(manifest, desc="TextVQA"):
        example = dataset[item["dataset_index"]]
        references = [str(answer) for answer in example["answers"]]
        if not any(answer.strip() for answer in references):
            raise RuntimeError("The selected split has no reference answers. Use --split validation.")
        batch = prepare_example(processor, model, example)

        for method, retention, random_seed in conditions:
            name = CORE.condition_name(method, retention, random_seed)
            prediction_path = args.output_dir / "conditions" / name / "predictions" / f"{item['id']}.json"
            route_path = args.output_dir / "conditions" / name / "routes" / f"{item['id']}.npz"
            if prediction_path.exists() and route_path.exists():
                continue
            condition = CORE.compress_inputs(
                batch, method, retention,
                args.seed + item["dataset_index"] * 1009 + (random_seed or 0),
            )
            routes, layer_names, n_experts = CORE.capture_routes(model, condition)
            raw_answer = CORE.generate_answer(model, tokenizer, condition, args.max_new_tokens)
            predicted = extract_answer(raw_answer)
            score = vqa_soft_score(predicted, references)
            prediction = {
                "schema_version": 3,
                "task": "textvqa",
                "id": item["id"],
                "dataset_index": item["dataset_index"],
                "question_id": item["question_id"],
                "image_id": item["image_id"],
                "question": example["question"],
                "condition": name,
                "method": method,
                "retention": retention,
                "compression_ratio": 0.0 if method == "baseline" else 1.0 - retention,
                "random_seed": random_seed,
                "answers": references,
                "predicted": predicted,
                "normalized_prediction": normalize_answer(predicted),
                "vqa_score": score,
                "raw_answer": raw_answer,
                "visual_tokens_before": int(batch["visual_mask"].sum()),
                "visual_tokens_after": int(condition["visual_mask"].sum()),
            }
            CORE.save_result(args.output_dir, name, item["id"], prediction, routes, condition, layer_names, n_experts)

        del batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
