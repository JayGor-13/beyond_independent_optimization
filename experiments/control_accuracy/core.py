"""Dependency-free checks for the accuracy-only follow-up."""
import hashlib
import json
from pathlib import Path

CONDITIONS = ("chronological", "shuffled", "duplicate", "cross_clip")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_manifest(manifest, reference):
    expected = {row["id"]: row for row in reference}
    if len(expected) != 256 or len(manifest) != 256:
        raise ValueError("Require the complete original 256-question sample, even for a pilot.")
    if len({item["id"] for item in manifest}) != 256:
        raise ValueError("Duplicate question IDs in manifest.")
    if len({str(item["video_id"]) for item in manifest}) != 256:
        raise ValueError("The original experiment requires unique videos.")
    if {item["id"] for item in manifest} != set(expected):
        raise ValueError("Reconstructed sample differs from the saved TRC sample; attach the original manifest.")
    for item in manifest:
        row = expected[item["id"]]
        if str(item["video_id"]) != str(row["video_id"]):
            raise ValueError(f"Video mismatch for {item['id']}")
        if item["type"] != row["question_type"] or item["category"] != row["category"]:
            raise ValueError(f"Question metadata mismatch for {item['id']}")
        if item["id"] != f"nextqa_{int(item['dataset_index']):05d}":
            raise ValueError("Question ID does not match dataset index.")
        if len(item["choices"]) != 5 or not 0 <= item["answer"] < 5:
            raise ValueError("Invalid answer choices.")
    return expected


def donor_items(manifest, item_index, frames=8):
    # Always use the full manifest: a pilot must retain the same cross-clip donors.
    return [manifest[(item_index+i+1) % len(manifest)] for i in range(frames)]


def bool_value(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError("Missing or invalid reference correctness.")
    return str(value).lower() == "true"


def ensure_config(path, config):
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        if old != config:
            raise ValueError("Output configuration changed; use a different output directory.")
    else:
        path.write_text(json.dumps(config, indent=2))


def summarize(records, expected, full_count=256):
    summaries = []
    for condition in CONDITIONS:
        rows = [r for r in records if r["condition"] == condition]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate predictions.")
        correct = sum(r["correct"] for r in rows)
        summaries.append(dict(condition=condition, examples=len(rows), correct=correct,
                              accuracy_percent=100*correct/len(rows) if rows else None,
                              invalid_answers=sum(not r["predicted"] for r in rows)))
    chronological = {r["id"]: r for r in records if r["condition"] == "chronological"}
    mismatches = [key for key, row in chronological.items()
                  if row["correct"] != bool_value(expected[key]["correct"])]
    complete = all(s["examples"] == full_count for s in summaries)
    return summaries, {"full_sample_complete": complete,
                       "chronological_correctness_mismatches": mismatches,
                       "baseline_matches_saved_correctness": not mismatches,
                       "eligible_for_paper_review": complete and not mismatches,
                       "note": "Eligibility is a review check, not proof of matching the original checkpoint or manifest order."}
