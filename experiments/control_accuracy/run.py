"""Generate answers for the original TRC controls, without routing capture."""
import argparse
import csv
import importlib.util
import json
import random
import time
from functools import lru_cache
from pathlib import Path

from core import CONDITIONS, donor_items, ensure_config, fingerprint, summarize, validate_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=32, help="Prefix of the full original manifest; 256 for the paper.")
    parser.add_argument("--manifest", type=Path, help="Original nextqa_256_seed20260819.json if available.")
    parser.add_argument("--reference", type=Path, default=Path(__file__).with_name("reference_chronological.csv"))
    parser.add_argument("--model", default="deepseek-ai/deepseek-vl2-tiny")
    parser.add_argument("--revision", help="Use the original checkpoint revision if known.")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/deepseek_control_accuracy"))
    args = parser.parse_args()
    if not 1 <= args.limit <= 256:
        raise ValueError("--limit must be in [1,256].")
    # Keep the original preprocessing, generation and frame-control implementation.
    helper_path = Path(__file__).with_name("original_trc_run.py")
    spec = importlib.util.spec_from_file_location("original_trc", helper_path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    torch, np = legacy.torch, legacy.np
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a Kaggle GPU accelerator.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    reference = list(csv.DictReader(args.reference.open(encoding="utf-8", newline="")))
    manifest_path = args.output_dir / "full_manifest.json"
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        source = "original_manifest"
    elif manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        source = json.loads((args.output_dir / "run_config.json").read_text())["manifest_source"]
    else:
        print("Reconstructing original seeded sample; exact question/video membership must match saved results.", flush=True)
        dataset = legacy.load_dataset("lmms-eval/NExTQA", "MC", split="test")
        videos = {Path(p).stem for p in legacy.list_repo_files("VLM2Vec/nextqa-rawvideo", repo_type="dataset") if p.endswith(".mp4")}
        candidate = args.output_dir / "candidate_manifest.json"
        manifest = legacy.build_manifest(dataset, videos, 256, args.seed, candidate)
        source = "seeded_reconstruction_membership_checked"
    expected = validate_manifest(manifest, reference)
    config = dict(model=args.model, revision_requested=args.revision, seed=args.seed,
                  frames=8, max_new_tokens=8, precision="float16", conditions=list(CONDITIONS),
                  manifest_source=source, manifest_sha256=fingerprint(manifest),
                  reference_sha256=fingerprint(reference), runner_sha256=fingerprint(Path(__file__).read_text()),
                  core_sha256=fingerprint(Path(__file__).with_name("core.py").read_text()),
                  original_runner_sha256=fingerprint(helper_path.read_text()),
                  prompt_note="Original prompt retained verbatim, including chronological-frame wording in all conditions.")
    ensure_config(args.output_dir / "run_config.json", config)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if source != "original_manifest":
        print("NOTE: Original manifest order is not independently available; inspect reconstruction before combining cross-clip accuracy with old TRC.", flush=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    legacy.SiglipAttention.forward = legacy.torch_siglip_attention
    processor = legacy.DeepseekVLV2Processor.from_pretrained(args.model, revision=args.revision)
    model = legacy.AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True,
        torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").cuda().eval()
    resolved = getattr(model.config, "_commit_hash", None)
    import transformers
    provenance = dict(resolved_model_revision=resolved, torch_version=torch.__version__,
                      transformers_version=transformers.__version__, gpu=torch.cuda.get_device_name(0))
    prov_path = args.output_dir / "runtime_provenance.json"
    if prov_path.exists() and json.loads(prov_path.read_text()) != provenance:
        raise ValueError("Checkpoint or runtime changed during resume; use a new directory.")
    prov_path.write_text(json.dumps(provenance, indent=2))

    @lru_cache(maxsize=16)
    def load_frames(video_id):
        frames, indices = legacy.sample_frames(legacy.download_video("VLM2Vec/nextqa-rawvideo", video_id), 8)
        return frames, indices

    timings = []
    for index, item in enumerate(legacy.tqdm(manifest[:args.limit], desc="Answers only")):
        missing = [c for c in CONDITIONS if not (pred_dir / f"{item['id']}__{c}.json").exists()]
        if not missing:
            continue
        frames, indices = load_frames(item["video_id"])
        cross_frames, cross_indices, donor_ids = [], [], []
        if "cross_clip" in missing:
            for i, donor in enumerate(donor_items(manifest, index)):
                other_frames, other_indices = load_frames(donor["video_id"])
                cross_frames.append(other_frames[i])
                cross_indices.append(int(other_indices[i]))
                donor_ids.append(donor["video_id"])
        for condition in missing:
            started = time.monotonic()
            selected, selected_indices = legacy.condition_frames(
                condition, frames, indices, cross_frames, np.asarray(cross_indices), args.seed+item["dataset_index"])
            batch = legacy.prepare_sequence(processor, model, item, selected)
            raw = legacy.generate_answer(model, processor.tokenizer, batch, 8)
            answer = legacy.answer_letter(raw)
            record = dict(id=item["id"], video_id=item["video_id"], condition=condition,
                          target=legacy.LETTERS[item["answer"]], predicted=answer,
                          correct=answer == legacy.LETTERS[item["answer"]], raw_answer=raw,
                          prompt=batch["prompt"], frame_indices=np.asarray(selected_indices).tolist(),
                          donor_video_ids=donor_ids if condition == "cross_clip" else [],
                          manifest_sha256=config["manifest_sha256"])
            (pred_dir / f"{item['id']}__{condition}.json").write_text(json.dumps(record, indent=2))
            timings.append(time.monotonic()-started)
            del batch
            torch.cuda.empty_cache()
        if index == 1:
            print(f"Mean answer inference time so far: {np.mean(timings):.1f}s; downloads are additional.", flush=True)
    selected_ids = {item["id"] for item in manifest[:args.limit]}
    records = [json.loads(p.read_text()) for p in pred_dir.glob("*.json")]
    records = [r for r in records if r["id"] in selected_ids]
    if any(r["manifest_sha256"] != config["manifest_sha256"] for r in records):
        raise ValueError("Prediction manifest mismatch.")
    summary, checks = summarize(records, expected)
    checks.update(manifest_source=source, resolved_model_revision=resolved,
                  original_checkpoint_revision_known=bool(args.revision),
                  requested_examples=args.limit, completed_answers=len(records))
    (args.output_dir / "checks.json").write_text(json.dumps(checks, indent=2))
    with (args.output_dir / "condition_accuracy.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    for row in summary:
        print(f"{row['condition']}: {row['correct']}/{row['examples']} = {row['accuracy_percent']:.2f}%")
    print(json.dumps(checks, indent=2))
    print("Pilot results cannot fill an n=256 table." if args.limit < 256 else "Review checks, sample provenance and checkpoint before updating the paper.")


if __name__ == "__main__":
    main()
