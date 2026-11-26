import argparse
import json
import os
import random
import re
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm

from model_compat import patch_eager_attention


LETTERS = "ABCDE"
KEEP_SCHEDULES = {
    0.75: "0.91,0.91,0.91",
    0.50: "0.79,0.79,0.79",
    0.25: "0.63,0.63,0.63",
}


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "compression_routing_exp1"
    return Path(__file__).resolve().parent / "outputs_exp1"


def default_fastmmoe_root():
    kaggle_root = Path("/kaggle/working/FastMMoE")
    if kaggle_root.exists():
        return kaggle_root
    return Path(__file__).resolve().parents[2] / "external" / "FastMMoE"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "fastmmoe_prune", "fastmmoe_full"], required=True)
    parser.add_argument("--retention", type=float, choices=sorted(KEEP_SCHEDULES), default=0.5)
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument("--dataset", default="derek-thomas/ScienceQA")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--include-hint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--fastmmoe-root", type=Path, default=default_fastmmoe_root())
    return parser.parse_args()


def configure_fastmmoe(mode, retention):
    os.environ["MERGE_LAYER_LOCS"] = "2,5,8"
    os.environ["TOKEN_MERGE_METHOD"] = "mlerp"
    os.environ["BASE_ALPHA"] = "0.3"
    os.environ["MERGE_RATIO"] = "0.05"
    os.environ["ROUTING_SIMILARITY_WINDOW_SIZE"] = "3"

    if mode == "baseline":
        os.environ["TOKEN_MERGE_STRATEGY"] = "none"
        os.environ["KEEP_TOKEN_RATIO"] = "none"
        os.environ["REDUCTION_LAYER_IDX"] = "999"
        os.environ["VISION_EXPERT_REDUCE_FACTOR"] = "1.0"
    else:
        os.environ["TOKEN_MERGE_STRATEGY"] = "hybrid"
        os.environ["KEEP_TOKEN_RATIO"] = KEEP_SCHEDULES[retention]
        os.environ["REDUCTION_LAYER_IDX"] = "999" if mode == "fastmmoe_prune" else "2"
        os.environ["VISION_EXPERT_REDUCE_FACTOR"] = "1.0" if mode == "fastmmoe_prune" else "0.5"


def condition_name(mode, retention):
    return "baseline" if mode == "baseline" else f"{mode}_r{round(retention * 100)}"


def build_manifest(dataset, limit, seed, path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    indices = [index for index in range(len(dataset)) if dataset[index]["image"] is not None]
    random.Random(seed).shuffle(indices)
    manifest = []
    for index in indices[:limit]:
        example = dataset[index]
        manifest.append(
            {
                "id": f"scienceqa_test_{index:05d}",
                "dataset_index": index,
                "subject": example["subject"],
                "topic": example["topic"],
                "category": example["category"],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_prompt(example, include_hint):
    lines = ["<image>"]
    if include_hint and example["hint"]:
        lines.append(example["hint"])
    lines.append(example["question"])
    lines.extend(f"({LETTERS[index]}) {choice}" for index, choice in enumerate(example["choices"]))
    lines.append("Answer with only the option letter.")
    return "\n".join(lines)


def prepare_example(processor, model, example, include_hint):
    prompt = build_prompt(example, include_hint)
    conversation = [
        {"role": "<|User|>", "content": prompt, "images": ["scienceqa-image"]},
        {"role": "<|Assistant|>", "content": ""},
    ]
    prepared = processor(
        conversations=conversation,
        images=[example["image"].convert("RGB")],
        force_batchify=True,
        system_prompt="",
    ).to(model.device, dtype=torch.float16)
    return prompt, prepared


def answer_letter(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def torch_siglip_attention(self, x):
    batch, tokens, channels = x.shape
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    x = x.transpose(1, 2).reshape(batch, tokens, channels)
    return self.proj_drop(self.proj(x))


def validate_run_config(output_dir, args):
    config = {
        "schema_version": 2,
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "include_hint": args.include_hint,
        "max_new_tokens": args.max_new_tokens,
    }
    path = output_dir / "run_config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise RuntimeError(f"Run configuration does not match {path}. Use a different --output-dir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


class RouteRecorder:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.original_infer = []
        self.current_masks = {}
        self.pending = {}
        self.records = {}

    def start(self):
        for layer_index, layer in enumerate(self.model.language.model.layers):
            mlp = layer.mlp
            if not hasattr(mlp, "gate") or not hasattr(mlp, "moe_infer"):
                continue

            def layer_pre_hook(module, args, kwargs, index=layer_index):
                mask = kwargs.get("images_seq_mask_without_separator")
                self.current_masks[index] = mask

            self.handles.append(layer.register_forward_pre_hook(layer_pre_hook, with_kwargs=True))

            original = mlp.moe_infer
            self.original_infer.append((mlp, original))

            def wrapped_infer(module, x, topk_ids, topk_weight, index=layer_index, original_method=original):
                self.pending[index] = (topk_ids.detach(), topk_weight.detach())
                return original_method(x, topk_ids, topk_weight)

            mlp.moe_infer = types.MethodType(wrapped_infer, mlp)

            def mlp_hook(module, args, kwargs, output, index=layer_index):
                if index not in self.pending:
                    return
                topk, weights = self.pending.pop(index)
                mask = output[1] if isinstance(output, tuple) else self.current_masks.get(index)
                if mask is None:
                    return
                mask = mask.detach().bool().reshape(-1)
                if mask.numel() != topk.shape[0]:
                    return

                topk = topk[mask]
                weights = weights[mask]
                n_experts = module.gate.n_routed_experts
                valid = (topk >= 0) & (topk < n_experts) & (weights > 0)
                counts = torch.zeros(n_experts, dtype=torch.int64, device=topk.device)
                mass = torch.zeros(n_experts, dtype=torch.float32, device=topk.device)
                counts.scatter_add_(0, topk[valid], torch.ones_like(topk[valid], dtype=torch.int64))
                mass.scatter_add_(0, topk[valid], weights[valid].float())
                self.records[index] = {
                    "counts": counts.cpu().numpy(),
                    "mass": mass.cpu().numpy(),
                    "visual_tokens": int(mask.sum()),
                    "active_assignments": int(valid.sum()),
                }

            self.handles.append(mlp.register_forward_hook(mlp_hook, with_kwargs=True))

    def stop(self):
        for handle in self.handles:
            handle.remove()
        for mlp, original in self.original_infer:
            mlp.moe_infer = original

    def arrays(self):
        layer_indices = sorted(self.records)
        if not layer_indices:
            raise RuntimeError("No visual-token MoE routes were captured.")
        return {
            "layer_indices": np.asarray(layer_indices, dtype=np.int16),
            "expert_counts": np.stack([self.records[index]["counts"] for index in layer_indices]),
            "expert_mass": np.stack([self.records[index]["mass"] for index in layer_indices]),
            "visual_tokens": np.asarray([self.records[index]["visual_tokens"] for index in layer_indices]),
            "active_assignments": np.asarray([self.records[index]["active_assignments"] for index in layer_indices]),
        }


def reset_model_state(model):
    language = model.language.model
    language.init_token_total_shape = 0
    language.generate_process_count = 0


def capture_routes(model, prepared):
    reset_model_state(model)
    recorder = RouteRecorder(model)
    recorder.start()
    try:
        with torch.inference_mode():
            model(
                input_ids=prepared.input_ids,
                images=prepared.images,
                images_seq_mask=prepared.images_seq_mask,
                images_spatial_crop=prepared.images_spatial_crop,
                attention_mask=prepared.attention_mask,
                use_cache=False,
                return_dict=True,
            )
    finally:
        recorder.stop()
    return recorder.arrays()


def generate_answer(model, tokenizer, prepared, max_new_tokens):
    reset_model_state(model)
    with torch.inference_mode():
        output = model.generate(
            input_ids=prepared.input_ids,
            images=prepared.images,
            images_seq_mask=prepared.images_seq_mask,
            images_spatial_crop=prepared.images_spatial_crop,
            attention_mask=prepared.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    new_tokens = output[0, prepared.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")

    configure_fastmmoe(args.mode, args.retention)
    deepseek_root = args.fastmmoe_root / "DeepSeek-VL2"
    if not deepseek_root.exists():
        raise RuntimeError(f"Official FastMMoE DeepSeek code not found at {deepseek_root}. Run setup_kaggle.sh first.")
    sys.path.insert(0, str(deepseek_root))

    from deepseek_vl2.models import DeepseekVLV2FastMMoEForCausalLM, DeepseekVLV2Processor
    from deepseek_vl2.models.siglip_vit import Attention as SiglipAttention

    SiglipAttention.forward = torch_siglip_attention
    torch.manual_seed(args.seed)

    condition = condition_name(args.mode, args.retention)
    validate_run_config(args.output_dir, args)
    condition_dir = args.output_dir / "fastmmoe_conditions" / condition
    prediction_dir = condition_dir / "predictions"
    route_dir = condition_dir / "routes"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split="test")
    manifest_path = args.output_dir / "manifests" / f"scienceqa_{args.limit}_seed{args.seed}.json"
    manifest = build_manifest(dataset, args.limit, args.seed, manifest_path)

    processor = DeepseekVLV2Processor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = DeepseekVLV2FastMMoEForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).cuda().eval()
    patch_eager_attention(model)

    for item in tqdm(manifest, desc=condition):
        prediction_path = prediction_dir / f"{item['id']}.json"
        route_path = route_dir / f"{item['id']}.npz"
        if prediction_path.exists() and route_path.exists():
            continue

        example = dataset[item["dataset_index"]]
        prompt, prepared = prepare_example(processor, model, example, args.include_hint)
        routes = capture_routes(model, prepared)
        raw_answer = generate_answer(model, tokenizer, prepared, args.max_new_tokens)
        predicted = answer_letter(raw_answer)
        target = LETTERS[example["answer"]]

        prediction = {
            "schema_version": 2,
            "id": item["id"],
            "dataset_index": item["dataset_index"],
            "condition": condition,
            "mode": args.mode,
            "target_retention": 1.0 if args.mode == "baseline" else args.retention,
            "keep_schedule": "none" if args.mode == "baseline" else KEEP_SCHEDULES[args.retention],
            "target": target,
            "predicted": predicted,
            "correct": predicted == target,
            "raw_answer": raw_answer,
            "visual_tokens_first_layer": int(routes["visual_tokens"][0]),
            "visual_tokens_last_layer": int(routes["visual_tokens"][-1]),
        }
        prediction_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")
        np.savez_compressed(route_path, **routes)

        del prepared
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
