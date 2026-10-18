"""Download and normalize public Hugging Face grounding datasets.

Public data is exported as a separate grounding-specialist dataset. It is not
silently converted into project pass/fail labels because public schemas do not
contain reliable instruction-following and completeness judgments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DATASET_DEFAULTS = {
    "wandb/RAGTruth-processed": {"split": "train"},
    "blue-guardrails/ragtruth-plus-plus": {"split": "train"},
}

TEXT_FIELDS = ("prompt", "question", "query", "instruction", "input", "context", "source", "document")
ANSWER_FIELDS = ("response", "answer", "generation", "output", "completion", "text")
LABEL_FIELDS = ("label", "hallucination", "is_hallucination", "faithfulness", "groundedness", "verdict")


def _first_text(row: dict[str, Any], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _render_context(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                identifier = item.get("id") or item.get("title") or "reference"
                text = item.get("text") or item.get("content") or item.get("document") or ""
                if text:
                    parts.append(f"[{identifier}] {text}")
        return "\n\n".join(parts).strip()
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return ""


def _find_context(row: dict[str, Any]) -> str:
    for name in TEXT_FIELDS:
        if name in row:
            rendered = _render_context(row[name])
            if rendered and name in ("context", "source", "document"):
                return rendered
    for name in ("contexts", "documents", "references", "retrieved_context"):
        if name in row:
            rendered = _render_context(row[name])
            if rendered:
                return rendered
    return ""


def _normalize_label(row: dict[str, Any]) -> str:
    for name in LABEL_FIELDS:
        if name not in row:
            continue
        value = row[name]
        if isinstance(value, bool):
            return "unsupported" if value else "supported"
        text = str(value).strip().lower()
        if text in {"1", "true", "hallucinated", "hallucination", "unsupported", "false", "fail"}:
            return "unsupported" if text not in {"false"} else "supported"
        if text in {"0", "supported", "faithful", "grounded", "pass", "entailment"}:
            return "supported"
    return "unknown"


def _find_label(row: dict[str, Any]) -> str:
    """Read simple labels and RAGTruth's processed span-count annotation."""
    processed = row.get("hallucination_labels_processed")
    if isinstance(processed, dict):
        conflict = processed.get("evident_conflict", 0)
        baseless = processed.get("baseless_info", 0)
        try:
            return "unsupported" if int(conflict) > 0 or int(baseless) > 0 else "supported"
        except (TypeError, ValueError):
            pass
    return _normalize_label(row)


def normalize_row(row: dict[str, Any], dataset: str, index: int) -> dict[str, Any] | None:
    prompt = _first_text(row, ("prompt", "question", "query", "instruction", "input"))
    answer = _first_text(row, ANSWER_FIELDS)
    context = _find_context(row)
    label = _find_label(row)
    if not prompt or not answer or not context or label == "unknown":
        return None
    return {
        "id": f"{dataset.replace('/', '__')}__{index:07d}",
        "task": "groundedness",
        "question": prompt,
        "answer": answer,
        "context": context,
        "label": label,
        "source_dataset": dataset,
        "source_row": index,
        "completion": json.dumps({"groundedness": label}, ensure_ascii=False),
    }


def prepare(dataset_name: str, split: str, output: Path, limit: int | None = None) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependencies first: pip install datasets") from exc
    dataset = load_dataset(dataset_name, split=split)
    rows = []
    skipped = 0
    for index, row in enumerate(dataset):
        normalized = normalize_row(dict(row), dataset_name, index)
        if normalized is None:
            skipped += 1
        else:
            rows.append(normalized)
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError("No rows matched the grounding schema; inspect the dataset columns and add a mapping.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "dataset": dataset_name,
        "split": split,
        "rows": len(rows),
        "skipped": skipped,
        "labels": {label: sum(row["label"] == label for row in rows) for label in ("supported", "unsupported")},
        "purpose": "groundedness specialist pretraining; not project verdict labels",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize a Hugging Face grounding dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    split = args.split or DATASET_DEFAULTS.get(args.dataset, {}).get("split", "train")
    try:
        print(json.dumps(prepare(args.dataset, split, args.out, args.limit), indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
