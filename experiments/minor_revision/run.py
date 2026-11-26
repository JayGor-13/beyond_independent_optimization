#!/usr/bin/env python3
"""Run the additional Section 4.2 experiments; smoke/plan modes are explicit."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from protocol import (BENCHMARKS, MODELS, SCHEMA, code_digest, conditions, file_lock,
                      lock_json, resolve_revisions, score_answer, write_json)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--benchmarks", choices=BENCHMARKS, nargs="+", default=list(BENCHMARKS))
    parser.add_argument("--precision", choices=["bf16", "expert-nf4"], default="expert-nf4")
    parser.add_argument("--retentions", type=float, nargs="+", default=[0.75, 0.5, 0.25])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--limit", type=int, default=0, help="0 = full new benchmarks; ScienceQA uses --scienceqa-limit.")
    parser.add_argument("--scienceqa-limit", type=int, default=1000)
    parser.add_argument("--smoke", action="store_true", help="Two examples per benchmark in separate smoke directories.")
    parser.add_argument("--plan-only", action="store_true", help="Pin revisions and prepare data; do not load model weights.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144, help="Per-image cap, fixed across conditions.")
    parser.add_argument("--route-storage", choices=["full", "metrics"], default="full")
    parser.add_argument("--reserve-gib", type=float, default=12)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--scratch", type=Path, default=Path(__file__).parent / "model_shards")
    args = parser.parse_args(argv)
    try:
        args.conditions = conditions(args.retentions, args.random_seeds)
    except ValueError as exc:
        parser.error(str(exc))
    if (args.limit < 0 or args.scienceqa_limit < 0 or args.max_new_tokens < 1 or args.max_input_tokens < 64
            or args.min_pixels < 1024 or args.max_pixels < args.min_pixels or args.reserve_gib < 1):
        parser.error("Invalid limit, pixel, token, or memory budget.")
    if len(set(args.benchmarks)) != len(args.benchmarks):
        parser.error("Benchmarks must be unique.")
    return args


def result_path(root, condition, item_id, extension="json"):
    return root / "conditions" / condition / f"{item_id}.{extension}"


def record_complete(root, condition, item_id, route_storage):
    return result_path(root, condition, item_id).exists() and (
        route_storage == "metrics" or result_path(root, condition, item_id, "npz").exists())


def save_routes(path, routes, visual_ids):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, topk=routes, visual_ids=visual_ids)
    temporary.replace(path)


def run_benchmark(args, adapter, root, dataset, manifest, config):
    import numpy as np
    import torch
    from tqdm import tqdm
    from adapter import prune_batch
    from data import normalize_row
    from routing import routing_metrics

    checked = False
    for index in tqdm(manifest["indices"], desc=f"{args.model}/{manifest['benchmark']}"):
        item_id = f"{manifest['benchmark']}_{index:06d}"
        if all(record_complete(root, condition["name"], item_id, args.route_storage) for condition in args.conditions):
            continue
        item, images = normalize_row(manifest["benchmark"], dataset[index], index)
        batch = adapter.prepare(item, images)
        if args.smoke or not checked:
            check = adapter.check_native(batch)
            write_json(root / "native_equivalence" / f"{item_id}.json", {"example_id": item_id, **check})
            checked = True
        del batch["native"]
        baseline = adapter.evaluate(batch, args.max_new_tokens)
        baseline_score = score_answer(baseline["raw_answer"], item)
        baseline_hash = hashlib.sha256(baseline["routes"].tobytes()).hexdigest()
        baseline_path = result_path(root, "baseline", item_id)
        if baseline_path.exists():
            old = json.loads(baseline_path.read_text())
            if old["baseline_route_sha256"] != baseline_hash or old["raw_answer"] != baseline["raw_answer"]:
                raise RuntimeError(f"Baseline changed on resume for {item_id}; use a fresh run directory.")
        for condition in args.conditions:
            if record_complete(root, condition["name"], item_id, args.route_storage):
                continue
            pruned = prune_batch(batch, condition, item_id)
            result = baseline if condition["method"] == "baseline" else adapter.evaluate(pruned, args.max_new_tokens)
            metrics = routing_metrics(baseline["routes"], result["routes"], pruned["visual_ids"], adapter.n_experts)
            if condition["retention"] == 1 and (not np.array_equal(baseline["routes"], result["routes"])
                                                or baseline["raw_answer"] != result["raw_answer"]):
                raise RuntimeError(f"100% identity control failed: {item_id}/{condition['name']}.")
            score = score_answer(result["raw_answer"], item)
            record = {"schema": SCHEMA, "example_id": item_id, "source_id": item["source_id"],
                      "dataset_index": index, "condition": condition, "problem": item,
                      "raw_answer": result["raw_answer"], "generated_tokens": result["generated_tokens"],
                      "generated_token_ids": result["generated_token_ids"],
                      "stop_reason": result["stop_reason"], **score, "baseline_correct": baseline_score["correct"],
                      "baseline_route_sha256": baseline_hash, "visual_tokens_before": len(batch["norms"]),
                      "visual_tokens_after": len(pruned["visual_ids"]), "image_count": len(images),
                      "actual_retention": len(pruned["visual_ids"]) / len(batch["norms"]), **metrics}
            if args.route_storage == "full":
                save_routes(result_path(root, condition["name"], item_id, "npz"), result["routes"], pruned["visual_ids"])
            write_json(result_path(root, condition["name"], item_id), record)
        del batch, baseline, pruned, result
        torch.cuda.empty_cache()
    write_json(root / "complete.json", {"schema": SCHEMA, "examples": len(manifest["indices"]),
                                        "conditions": args.conditions})


def main(argv=None):
    args = parse_args(argv)
    phase = "smoke" if args.smoke else "full"
    run_root = args.output_root / args.model / args.precision / phase
    with file_lock(run_root / ".run.lock"):
        pins = resolve_revisions(args.output_root, [args.model], args.benchmarks)
        from data import load_benchmark
        prepared = []
        versions = {name: importlib.metadata.version(name) for name in
                    ("torch", "torchvision", "transformers", "accelerate", "huggingface-hub", "datasets", "numpy", "pillow", "safetensors")}
        if args.precision == "expert-nf4":
            versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        for benchmark in args.benchmarks:
            limit = 2 if args.smoke else args.limit or (args.scienceqa_limit if benchmark == "scienceqa" else 0)
            dataset, manifest = load_benchmark(benchmark, pins["datasets"][benchmark]["revision"],
                                               limit, args.seed, args.output_root, args.smoke)
            root = run_root / benchmark
            config = {"schema": SCHEMA, "model_alias": args.model, **pins["models"][args.model], "precision": args.precision,
                      "phase": phase, "benchmark": benchmark, "dataset": pins["datasets"][benchmark],
                      "examples": len(manifest["indices"]), "seed": args.seed, "conditions": args.conditions,
                      "min_pixels": args.min_pixels, "max_pixels": args.max_pixels,
                      "max_new_tokens": args.max_new_tokens, "max_input_tokens": args.max_input_tokens,
                      "route_storage": args.route_storage, "versions": versions, "code_sha256": code_digest(),
                      "position_policy": "preserve_original_native_multimodal_coordinates",
                      "pruning_policy": "per_image_budget_at_least_one_token_post_vision_pre_decoder",
                      "generation": "greedy_thinking_disabled", "scoring": "strict_direct_answers_v1",
                      "metrics_unit": "example_macro_average; load_cv_uses_matched_retained_tokens"}
            lock_json(root / "run_config.json", config)
            lock_json(root / "manifest.json", manifest)
            print(f"{benchmark}: {config['examples']} paired examples × {len(args.conditions)} conditions; "
                  f"output: {root}", flush=True)
            if not all(record_complete(root, condition["name"], f"{benchmark}_{index:06d}", args.route_storage)
                       for index in manifest["indices"] for condition in args.conditions):
                prepared.append((root, dataset, manifest, config))
        if args.plan_only or not prepared:
            print("Plan prepared; no model weights loaded." if args.plan_only else "All requested records already exist; run analyze.py to validate them.")
            return
        import torch
        from transformers import AutoProcessor
        from adapter import Adapter
        from loading import load_model
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_pin = pins["models"][args.model]
        model, loading_info = load_model(model_pin["repo"], model_pin["revision"], args.precision,
                                         args.scratch, args.reserve_gib)
        processor = AutoProcessor.from_pretrained(model_pin["repo"], revision=model_pin["revision"],
                                                  min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        adapter = Adapter(model, processor, args.max_input_tokens)
        for root, dataset, manifest, config in prepared:
            write_json(root / "loading_info.json", {**loading_info, "layer_names": adapter.layer_names,
                                                    "n_experts": adapter.n_experts, "top_k": adapter.top_k})
            run_benchmark(args, adapter, root, dataset, manifest, config)


if __name__ == "__main__":
    main()
