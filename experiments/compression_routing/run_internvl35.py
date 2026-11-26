import argparse
import copy
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torchvision.transforms.functional import InterpolationMode


LETTERS = "ABCDE"
DEFAULT_MODEL = "OpenGVLab/InternVL3_5-30B-A3B"
DEFAULT_REVISION = "ba1762d830e2aeddb3272b66ba9e0abf7e69f626"
IMAGE_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGE_START_TOKEN = "<img>"
IMAGE_END_TOKEN = "</img>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "compression_routing_exp1_internvl35"
    return Path(__file__).resolve().parent / "outputs_exp1_internvl35"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Repeat Experiment 1 on the InternVL3.5 MoE checkpoint."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset", default="derek-thomas/ScienceQA")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["baseline", "l2", "random"],
        default=["baseline", "l2", "random"],
    )
    parser.add_argument("--retentions", nargs="+", type=float, default=[0.5])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--include-hint", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--max-gpu-memory",
        default=None,
        help="Per-GPU limit used by device_map=auto, for example 14GiB on Kaggle T4 x2.",
    )
    parser.add_argument(
        "--max-cpu-memory",
        default="28GiB",
        help="CPU offload limit used when --max-gpu-memory is set.",
    )
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def resolve_dtype(name):
    if name == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16; use --dtype float16.")
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def condition_name(method, retention, random_seed=None):
    if method == "baseline":
        return "baseline"
    name = f"{method}_r{round(retention * 100)}"
    if method == "random":
        name += f"_s{random_seed}"
    return name


def condition_names(methods, retentions, random_seeds):
    conditions = [("baseline", 1.0, None)]
    for method in methods:
        if method == "baseline":
            continue
        for retention in retentions:
            seeds = random_seeds if method == "random" else [None]
            conditions.extend((method, retention, seed) for seed in seeds)
    return conditions


def create_output_dirs(output_dir, conditions):
    (output_dir / "manifests").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    for method, retention, random_seed in conditions:
        root = output_dir / "conditions" / condition_name(method, retention, random_seed)
        (root / "predictions").mkdir(parents=True, exist_ok=True)
        (root / "routes").mkdir(parents=True, exist_ok=True)


def build_manifest(dataset, limit, seed, path):
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if len(manifest) < limit:
            raise RuntimeError(
                f"{path} contains {len(manifest)} examples, fewer than requested {limit}. "
                "Use the full-run command first or remove only this incomplete manifest."
            )
        return manifest[:limit]

    image_indices = [index for index in range(len(dataset)) if dataset[index]["image"] is not None]
    random.Random(seed).shuffle(image_indices)
    manifest = []
    for index in image_indices[:limit]:
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


def validate_run_config(output_dir, args, dtype):
    config = {
        "schema_version": 2,
        "adapter": "internvl3_5",
        "model": args.model,
        "revision": args.revision,
        "dataset": args.dataset,
        "seed": args.seed,
        "include_hint": args.include_hint,
        "max_new_tokens": args.max_new_tokens,
        "max_tiles": args.max_tiles,
        "dtype": str(dtype).removeprefix("torch."),
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
        "max_gpu_memory": args.max_gpu_memory,
        "max_cpu_memory": args.max_cpu_memory,
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
    lines.extend(f"({LETTERS[index]}) {choice}" for index, choice in enumerate(example["choices"]))
    lines.append("Answer with only the option letter.")
    return "\n".join(lines)


def choose_tile_ratio(image, max_tiles, tile_size):
    width, height = image.size
    aspect_ratio = width / height
    candidates = sorted({
        (columns, rows)
        for count in range(1, max_tiles + 1)
        for columns in range(1, count + 1)
        for rows in range(1, count + 1)
        if 1 <= columns * rows <= max_tiles
    }, key=lambda ratio: ratio[0] * ratio[1])
    best_ratio = candidates[0]
    best_difference = math.inf
    for ratio in candidates:
        difference = abs(aspect_ratio - ratio[0] / ratio[1])
        if difference < best_difference:
            best_difference = difference
            best_ratio = ratio
        elif difference == best_difference:
            target_area = tile_size * tile_size * ratio[0] * ratio[1]
            if width * height > 0.5 * target_area:
                best_ratio = ratio
    return best_ratio


def image_to_tiles(image, tile_size, max_tiles):
    image = image.convert("RGB")
    columns, rows = choose_tile_ratio(image, max_tiles, tile_size)
    resized = image.resize((columns * tile_size, rows * tile_size), Image.Resampling.BICUBIC)
    tiles = []
    for row in range(rows):
        for column in range(columns):
            left = column * tile_size
            top = row * tile_size
            tiles.append(resized.crop((left, top, left + tile_size, top + tile_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((tile_size, tile_size), Image.Resampling.BICUBIC))
    return tiles


def preprocess_image(image, tile_size, max_tiles):
    transform = transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return torch.stack([transform(tile) for tile in image_to_tiles(image, tile_size, max_tiles)])


def model_image_size(model):
    return int(model.config.force_image_size or model.config.vision_config.image_size)


def module_device_and_dtype(module, fallback_dtype):
    for parameter in module.parameters():
        if parameter.device.type != "meta":
            dtype = parameter.dtype if parameter.is_floating_point() else fallback_dtype
            return parameter.device, dtype
    return torch.device("cuda:0"), fallback_dtype


def build_model_inputs(model, tokenizer, example, include_hint, max_tiles, dtype):
    prompt = build_prompt(example, include_hint)
    tile_size = model_image_size(model)
    pixel_values = preprocess_image(example["image"], tile_size, max_tiles)
    vision_device, vision_dtype = module_device_and_dtype(model.vision_model, dtype)
    pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)

    with torch.inference_mode():
        visual_features = model.extract_feature(pixel_values)

    image_context_id = tokenizer.convert_tokens_to_ids(IMAGE_CONTEXT_TOKEN)
    if image_context_id is None or image_context_id == tokenizer.unk_token_id:
        raise RuntimeError(f"Tokenizer does not define {IMAGE_CONTEXT_TOKEN}.")
    model.img_context_token_id = image_context_id

    template = copy.deepcopy(model.conv_template)
    template.system_message = model.system_message
    template.append_message(template.roles[0], prompt)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = (
        IMAGE_START_TOKEN
        + IMAGE_CONTEXT_TOKEN * model.num_image_token * pixel_values.shape[0]
        + IMAGE_END_TOKEN
    )
    query = query.replace("<image>", image_tokens, 1)
    if "<image>" in query:
        raise RuntimeError("The single-image ScienceQA prompt contains an unexpected extra image placeholder.")

    encoded = tokenizer(query, return_tensors="pt")
    embedding_layer = model.language_model.get_input_embeddings()
    embedding_device, _ = module_device_and_dtype(embedding_layer, dtype)
    input_ids = encoded["input_ids"].to(embedding_device)
    attention_mask = encoded["attention_mask"].to(embedding_device)
    visual_mask = input_ids[0].eq(image_context_id)

    with torch.inference_mode():
        embeddings = embedding_layer(input_ids).clone()
    flat_features = visual_features.reshape(-1, visual_features.shape[-1]).to(
        device=embeddings.device, dtype=embeddings.dtype
    )
    if int(visual_mask.sum()) != flat_features.shape[0]:
        raise RuntimeError(
            "InternVL visual placeholder count does not match the encoded visual features: "
            f"{int(visual_mask.sum())} placeholders versus {flat_features.shape[0]} features."
        )
    embeddings[0, visual_mask] = flat_features

    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    return {
        "prompt": prompt,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "embeddings": embeddings,
        "visual_mask": visual_mask,
        "eos_token_id": eos_token_id,
        "tiles": int(pixel_values.shape[0]),
    }


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
    keep_mask[visual_positions[keep_visual_ids]] = True
    return {
        "input_ids": batch["input_ids"][:, keep_mask],
        "attention_mask": batch["attention_mask"][:, keep_mask],
        "embeddings": batch["embeddings"][:, keep_mask],
        "visual_mask": batch["visual_mask"][keep_mask],
        "visual_ids": keep_visual_ids,
    }


def block_top_k(block):
    value = getattr(block, "top_k", None)
    if value is None:
        value = getattr(block.gate, "top_k", None)
    return value


def block_n_experts(block):
    value = getattr(block, "num_experts", None)
    if value is None:
        value = getattr(block.gate, "num_experts", None)
    if value is None:
        value = getattr(block.experts, "num_experts", None)
    return value


def find_moe_blocks(language_model):
    blocks = []
    for name, module in language_model.named_modules():
        if not hasattr(module, "gate") or not hasattr(module, "experts"):
            continue
        if block_top_k(module) is not None and block_n_experts(module) is not None:
            blocks.append((name, module))
    if not blocks:
        raise RuntimeError("No routed MoE blocks were found in the InternVL language model.")
    return blocks


def capture_routes(model, condition):
    blocks = find_moe_blocks(model.language_model)
    records = {}
    handles = []

    for name, block in blocks:
        def hook(module, inputs, output, block_name=name, parent=block):
            logits = output[0] if isinstance(output, tuple) else output
            logits = logits.reshape(-1, logits.shape[-1])
            mask = condition["visual_mask"].reshape(-1).to(logits.device)
            if logits.shape[0] != mask.numel():
                raise RuntimeError(
                    f"Router {block_name} saw {logits.shape[0]} tokens; expected {mask.numel()}."
                )
            records[block_name] = logits.float().topk(int(block_top_k(parent)), dim=-1).indices[mask].cpu()

        handles.append(block.gate.register_forward_hook(hook))

    decoder = getattr(model.language_model, "model", model.language_model)
    try:
        with torch.inference_mode():
            decoder(
                inputs_embeds=condition["embeddings"],
                attention_mask=condition["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, _ in blocks if name not in records]
    if missing:
        raise RuntimeError(f"No router output captured for: {', '.join(missing)}")
    layer_names = np.asarray([name for name, _ in blocks])
    routes = np.stack([records[name].to(torch.int16).numpy() for name in layer_names])
    n_experts = int(block_n_experts(blocks[0][1]))
    if any(int(block_n_experts(block)) != n_experts for _, block in blocks):
        raise RuntimeError("InternVL uses different expert counts across routed layers; update the route schema.")
    return routes, layer_names, n_experts


def generate_answer(model, tokenizer, condition, max_new_tokens, eos_token_id):
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.language_model.generate(
            inputs_embeds=condition["embeddings"],
            attention_mask=condition["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def answer_letter(text):
    match = re.search(r"\b([A-E])\b", text.upper())
    return match.group(1) if match else ""


def save_result(output_dir, name, item_id, prediction, routes, condition, layer_names, n_experts):
    prediction_dir = output_dir / "conditions" / name / "predictions"
    route_dir = output_dir / "conditions" / name / "routes"
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
        raise RuntimeError("A CUDA GPU is required.")
    if args.max_tiles < 1:
        raise ValueError("--max-tiles must be at least 1.")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load-in-4bit and --load-in-8bit.")
    if any(not 0 < retention < 1 for retention in args.retentions):
        raise ValueError("Retention values must be between 0 and 1.")

    dtype = resolve_dtype(args.dtype)
    conditions = condition_names(args.methods, args.retentions, args.random_seeds)
    validate_run_config(args.output_dir, args, dtype)
    create_output_dirs(args.output_dir, conditions)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        use_fast=False,
    )
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    max_memory = None
    if args.max_gpu_memory:
        max_memory = {
            index: args.max_gpu_memory for index in range(torch.cuda.device_count())
        }
        max_memory["cpu"] = args.max_cpu_memory

    model = AutoModel.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory=max_memory,
        quantization_config=quantization_config,
        use_flash_attn=args.use_flash_attn,
    ).eval()
    find_moe_blocks(model.language_model)

    dataset = load_dataset(args.dataset, split="test")
    manifest_path = args.output_dir / "manifests" / f"scienceqa_{args.limit}_seed{args.seed}.json"
    manifest = build_manifest(dataset, args.limit, args.seed, manifest_path)

    for item in tqdm(manifest, desc="InternVL3.5 ScienceQA"):
        example = dataset[item["dataset_index"]]
        batch = build_model_inputs(
            model,
            tokenizer,
            example,
            args.include_hint,
            args.max_tiles,
            dtype,
        )

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
            raw_answer = generate_answer(
                model,
                tokenizer,
                condition,
                args.max_new_tokens,
                batch["eos_token_id"],
            )
            predicted = answer_letter(raw_answer)
            target = LETTERS[example["answer"]]
            prediction = {
                "schema_version": 2,
                "model": args.model,
                "model_revision": args.revision,
                "id": item["id"],
                "dataset_index": item["dataset_index"],
                "condition": name,
                "method": method,
                "retention": retention,
                "random_seed": random_seed,
                "target": target,
                "predicted": predicted,
                "correct": predicted == target,
                "raw_answer": raw_answer,
                "image_tiles": batch["tiles"],
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
