import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from deepseek_vl2.models import DeepseekVLV2ForCausalLM, DeepseekVLV2Processor
from deepseek_vl2.models.siglip_vit import Attention as SiglipAttention

from model_compat import patch_eager_attention


LETTERS = "ABCDE"
DEFAULT_COMPRESSION_RATIOS = [0.25, 0.30, 0.50, 0.75]


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "compression_routing_exp1"
    return Path(__file__).resolve().parent / "outputs_exp1"


def condition_name(method, retention, random_seed=None):
    if method == "baseline":
        return "baseline"
    name = f"{method}_r{round(retention * 100)}"
    if method == "random":
        name += f"_s{random_seed}"
    return name


def create_output_dirs(output_dir, conditions):
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    for method, retention, random_seed in conditions:
        root = output_dir / "conditions" / condition_name(method, retention, random_seed)
        (root / "predictions").mkdir(parents=True, exist_ok=True)
        (root / "routes").mkdir(parents=True, exist_ok=True)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument("--dataset", default="derek-thomas/ScienceQA")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "l2", "random"], default=["baseline", "l2", "random"])
    ratio_group = parser.add_mutually_exclusive_group()
    ratio_group.add_argument(
        "--compression-ratios",
        nargs="+",
        type=float,
        default=None,
        help="Fractions of visual tokens to remove, e.g. 0.25 removes 25%%.",
    )
    ratio_group.add_argument(
        "--retentions",
        nargs="+",
        type=float,
        default=None,
        help="Backward-compatible fractions of visual tokens to keep.",
    )
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--include-hint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def resolve_retentions(args):
    if args.retentions is not None:
        retentions = args.retentions
    else:
        compression_ratios = (
            args.compression_ratios
            if args.compression_ratios is not None
            else DEFAULT_COMPRESSION_RATIOS
        )
        if any(not 0 < ratio < 1 for ratio in compression_ratios):
            raise ValueError("Compression ratios must be between 0 and 1.")
        retentions = [round(1.0 - ratio, 10) for ratio in compression_ratios]

    if any(not 0 < retention < 1 for retention in retentions):
        raise ValueError("Retention values must be between 0 and 1.")
    if len(set(retentions)) != len(retentions):
        raise ValueError("Compression/retention ratios must be unique.")
    return retentions


def build_manifest(dataset, limit, seed, path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    image_indices = [i for i in range(len(dataset)) if dataset[i]["image"] is not None]
    random.Random(seed).shuffle(image_indices)
    selected = image_indices[:limit]

    manifest = []
    for index in selected:
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


def build_prompt(example, include_hint):
    lines = ["<image>"]
    if include_hint and example["hint"]:
        lines.append(example["hint"])
    lines.append(example["question"])
    lines.extend(f"({LETTERS[i]}) {choice}" for i, choice in enumerate(example["choices"]))
    lines.append("Answer with only the option letter.")
    return "\n".join(lines)


def prepare_example(processor, model, example, include_hint):
    prompt = build_prompt(example, include_hint)
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
            "images": ["scienceqa-image"],
        },
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


def condition_names(methods, retentions, random_seeds):
    names = [("baseline", 1.0, None)]
    for method in methods:
        if method == "baseline":
            continue
        for retention in retentions:
            seeds = random_seeds if method == "random" else [None]
            for random_seed in seeds:
                names.append((method, retention, random_seed))
    return names


def compress_inputs(batch, method, retention, seed):
    visual_positions = torch.where(batch["visual_mask"])[0]
    visual_ids = torch.arange(len(visual_positions), device=visual_positions.device)

    if method == "baseline":
        keep_visual_ids = visual_ids
    else:
        keep_count = max(1, round(len(visual_positions) * retention))
        if method == "l2":
            scores = batch["embeddings"][0, visual_positions].float().norm(dim=-1)
        elif method == "random":
            generator = torch.Generator(device=visual_positions.device).manual_seed(seed)
            scores = torch.rand(len(visual_positions), generator=generator, device=visual_positions.device)
        else:
            raise ValueError(f"Unknown method: {method}")
        keep_visual_ids = scores.topk(keep_count).indices.sort().values

    keep_mask = ~batch["visual_mask"].clone()
    kept_positions = visual_positions[keep_visual_ids]
    keep_mask[kept_positions] = True

    return {
        "input_ids": batch["input_ids"][:, keep_mask],
        "attention_mask": batch["attention_mask"][:, keep_mask],
        "embeddings": batch["embeddings"][:, keep_mask],
        "visual_mask": batch["visual_mask"][keep_mask],
        "visual_ids": keep_visual_ids,
    }


def reset_fastmmoe_sequence_state(model, sequence_length):
    """Keep the FastMMoE fork's stateful policy length aligned per condition."""
    decoder = model.language.model
    if hasattr(decoder, "init_token_total_shape"):
        decoder.init_token_total_shape = int(sequence_length)
    if hasattr(decoder, "generate_process_count"):
        decoder.generate_process_count = 0


def capture_routes(model, condition):
    records = []
    layer_names = []
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            hidden = inputs[0]
            topk = output[0].detach().reshape(hidden.shape[0], hidden.shape[1], -1)
            weights = output[1].detach().reshape(hidden.shape[0], hidden.shape[1], -1)
            topk = topk[0, condition["visual_mask"]]
            weights = weights[0, condition["visual_mask"]]
            order = weights.argsort(dim=-1, descending=True)
            topk = topk.gather(-1, order)
            records.append(topk.cpu().to(torch.int16).numpy())
            layer_names.append(name)

        return hook

    gates = [(name, module) for name, module in model.named_modules() if module.__class__.__name__ == "MoEGate"]
    for name, gate in gates:
        handles.append(gate.register_forward_hook(make_hook(name)))

    reset_fastmmoe_sequence_state(model, condition["embeddings"].shape[1])
    with torch.inference_mode():
        model.language.model(
            inputs_embeds=condition["embeddings"],
            attention_mask=condition["attention_mask"],
            use_cache=False,
            return_dict=True,
        )

    for handle in handles:
        handle.remove()

    n_experts = gates[0][1].n_routed_experts
    return np.stack(records), np.asarray(layer_names), n_experts


def generate_answer(model, tokenizer, condition, max_new_tokens):
    decoder = model.language.model
    prompt_length = condition["embeddings"].shape[1]
    reset_fastmmoe_sequence_state(model, prompt_length)

    def sync_cached_policy_length(module, args, kwargs):
        attention_mask = kwargs.get("attention_mask")
        if (
            attention_mask is not None
            and attention_mask.ndim == 2
            and hasattr(module, "generate_process_count")
        ):
            module.generate_process_count = max(
                0,
                int(attention_mask.shape[-1]) - int(prompt_length),
            )
        return args, kwargs

    handle = decoder.register_forward_pre_hook(sync_cached_policy_length, with_kwargs=True)
    try:
        with torch.inference_mode():
            output = model.language.generate(
                input_ids=condition["input_ids"],
                inputs_embeds=condition["embeddings"],
                attention_mask=condition["attention_mask"],
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        handle.remove()

    new_tokens = output[0, condition["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def answer_letter(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def save_result(output_dir, condition_name, item_id, prediction, routes, condition, layer_names, n_experts):
    prediction_dir = output_dir / "conditions" / condition_name / "predictions"
    route_dir = output_dir / "conditions" / condition_name / "routes"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    route_dir.mkdir(parents=True, exist_ok=True)

    (prediction_dir / f"{item_id}.json").write_text(
        json.dumps(prediction, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        route_dir / f"{item_id}.npz",
        topk=routes,
        visual_ids=condition["visual_ids"].cpu().numpy(),
        layer_names=layer_names,
        n_experts=np.asarray(n_experts),
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Select a GPU accelerator in Kaggle.")

    retentions = resolve_retentions(args)
    print(
        "Compression schedule (compression -> retention): "
        + ", ".join(f"{1.0 - retention:.0%} -> {retention:.0%}" for retention in retentions)
    )
    conditions = condition_names(args.methods, retentions, args.random_seeds)
    validate_run_config(args.output_dir, args)
    create_output_dirs(args.output_dir, conditions)
    SiglipAttention.forward = torch_siglip_attention
    torch.manual_seed(args.seed)
    dataset = load_dataset(args.dataset, split="test")
    manifest_path = args.output_dir / "manifests" / f"scienceqa_{args.limit}_seed{args.seed}.json"
    manifest = build_manifest(dataset, args.limit, args.seed, manifest_path)

    processor = DeepseekVLV2Processor.from_pretrained(args.model)
    tokenizer = processor.tokenizer
    model = DeepseekVLV2ForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).cuda().eval()
    patch_eager_attention(model)

    for item in tqdm(manifest, desc="ScienceQA"):
        example = dataset[item["dataset_index"]]
        batch = prepare_example(processor, model, example, args.include_hint)

        for method, retention, random_seed in conditions:
            name = condition_name(method, retention, random_seed)
            prediction_path = args.output_dir / "conditions" / name / "predictions" / f"{item['id']}.json"
            route_path = args.output_dir / "conditions" / name / "routes" / f"{item['id']}.npz"
            if prediction_path.exists() and route_path.exists():
                continue

            condition = compress_inputs(
                batch,
                method,
                retention,
                args.seed + item["dataset_index"] * 1009 + (random_seed or 0),
            )
            routes, layer_names, n_experts = capture_routes(model, condition)
            raw_answer = generate_answer(model, tokenizer, condition, args.max_new_tokens)
            predicted = answer_letter(raw_answer)
            target = LETTERS[example["answer"]]

            prediction = {
                "schema_version": 2,
                "id": item["id"],
                "dataset_index": item["dataset_index"],
                "condition": name,
                "method": method,
                "retention": retention,
                "compression_ratio": 0.0 if method == "baseline" else 1.0 - retention,
                "random_seed": random_seed,
                "target": target,
                "predicted": predicted,
                "correct": predicted == target,
                "raw_answer": raw_answer,
                "visual_tokens_before": int(batch["visual_mask"].sum()),
                "visual_tokens_after": int(condition["visual_mask"].sum()),
            }
            save_result(
                args.output_dir,
                name,
                item["id"],
                prediction,
                routes,
                condition,
                layer_names,
                n_experts,
            )

        del batch
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
