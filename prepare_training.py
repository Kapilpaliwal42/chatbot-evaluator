"""Validate human labels and export Colab-ready training/validation data.

This tool deliberately refuses to export incomplete or low-confidence labels.
The resulting JSONL uses a generic instruction/completion format that can be
adapted to SFT libraries such as TRL or Unsloth in Google Colab.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random


LABELS = ("pass", "fail", "review")
INPUT_KEYS = ("question", "answer", "references", "requirements")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}, line {line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}, line {line_number}: expected object")
        rows.append(row)
    return rows


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def build_exports(cases_path: Path, reviews_path: Path, output: Path, seed: int = 20261018) -> dict:
    cases = read_jsonl(cases_path)
    reviews = read_jsonl(reviews_path)
    by_id = {row.get("id"): row for row in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("Review file contains duplicate or missing IDs")
    export = []
    for case in cases:
        case_id = case.get("id")
        review = by_id.get(case_id)
        if review is None:
            raise ValueError(f"Missing human review for {case_id}")
        if review.get("human_verdict") not in LABELS:
            raise ValueError(f"{case_id}: invalid human_verdict")
        if review.get("confidence") != "high":
            raise ValueError(f"{case_id}: only high-confidence labels may be exported")
        rationale = review.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(f"{case_id}: rationale is required")
        payload = {key: case.get(key, []) if key == "requirements" else case[key] for key in INPUT_KEYS}
        export.append({
            "id": case_id,
            "label": review["human_verdict"],
            "confidence": review["confidence"],
            "input": payload,
            "completion": json.dumps({
                "verdict": review["human_verdict"],
                "rationale": rationale,
            }, ensure_ascii=False),
        })
    if len(export) < 30:
        raise ValueError("At least 30 high-confidence human-reviewed cases are required before export")
    rng = random.Random(seed)
    rng.shuffle(export)
    test_count = max(1, round(len(export) * 0.2))
    validation_count = max(1, round(len(export) * 0.1))
    test = export[:test_count]
    validation = export[test_count:test_count + validation_count]
    train = export[test_count + validation_count:]
    if not train or not validation or not test:
        raise ValueError("Dataset split produced an empty partition")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_cases_sha256": digest(cases),
        "source_reviews_sha256": digest(reviews),
        "human_approved": True,
        "label_policy": "Only high-confidence human labels with nonempty rationales are exported.",
        "counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "labels": {label: sum(row["label"] == label for row in export) for label in LABELS},
        "seed": seed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export high-confidence human labels for model training.")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = build_exports(args.cases, args.reviews, args.out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
