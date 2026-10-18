"""A reproducible synthetic benchmark; labels never enter judge prompts."""

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import statistics
import time

from evaluation.core import (CRITERIA, JUDGMENT_SCHEMA, OllamaJudge, PROMPT_VERSION,
                             RUBRIC, SYSTEM_PROMPT, case_digest, error_judgment,
                             load_cases, parse_judgment, summarize)
from evaluation.report import write_reports


LABELS = ("pass", "fail", "review")
PREDICTIONS = LABELS + ("error",)
INPUT_KEYS = ("id", "question", "answer", "references", "requirements")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def prompt_fingerprint():
    return fingerprint({"system": SYSTEM_PROMPT, "rubric": RUBRIC, "schema": JUDGMENT_SCHEMA})


def core_fingerprint():
    return hashlib.sha256((Path(__file__).parent / "core.py").read_bytes()).hexdigest()


def read_jsonl(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            rows.append(row)
        except ValueError as exc:
            raise ValueError(f"{path}, line {number}: {exc}") from exc
    return rows


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_blind_review(source_paths, directory, seed=20261013):
    """Create sanitized review inputs and privately retain author labels."""
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("Benchmark is already frozen; use a new directory for new data")
    rows = []
    seen_ids, seen_families = set(), set()
    for path in source_paths:
        raw = read_jsonl(path)
        clean = load_cases(Path(path))
        metadata = {row["id"]: row for row in raw}
        for case in clean:
            extra = metadata[case["id"]]
            for field in ("family_id", "category", "gold_rationale"):
                if not isinstance(extra.get(field), str) or not extra[field].strip():
                    raise ValueError(f"{case['id']}: missing {field}")
            if case["id"] in seen_ids or extra["family_id"] in seen_families:
                raise ValueError("Duplicate case IDs or policy families in authored data")
            if "expected_verdict" not in case:
                raise ValueError("Every authored example needs an expected verdict")
            seen_ids.add(case["id"])
            seen_families.add(extra["family_id"])
            rows.append({**case, **{key: extra[key] for key in ("family_id", "category", "gold_rationale")},
                         "original_id": case["id"]})
    random.Random(seed).shuffle(rows)
    for index, row in enumerate(rows, 1):
        row["id"] = f"bench-{index:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "authored_labels.jsonl", rows)
    blinded = [{key: row[key] for key in INPUT_KEYS} for row in rows]
    write_jsonl(directory / "blind_review.jsonl", blinded)
    (directory / "preparation.json").write_text(json.dumps({"blind_inputs_sha256": fingerprint(blinded)}, indent=2), encoding="utf-8")
    return {"cases": len(rows), "author_labels": {label: sum(r["expected_verdict"] == label for r in rows) for label in LABELS},
            "blind_input": str(directory / "blind_review.jsonl")}


def freeze_dataset(directory, review_path, seed=20261013):
    """Accept only independently agreed labels; disagreement needs adjudication."""
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("Dataset is already frozen")
    authored = read_jsonl(directory / "authored_labels.jsonl")
    blinded = read_jsonl(directory / "blind_review.jsonl")
    preparation = json.loads((directory / "preparation.json").read_text(encoding="utf-8"))
    if fingerprint(blinded) != preparation["blind_inputs_sha256"]:
        raise ValueError("Blind review inputs changed after preparation")
    if fingerprint([{key: row[key] for key in INPUT_KEYS} for row in authored]) != fingerprint(blinded):
        raise ValueError("Authored inputs changed after independent review; review the new inputs again")
    reviews = read_jsonl(review_path)
    ids = [r.get("id") for r in reviews]
    if len(ids) != len(set(ids)) or set(ids) != {row["id"] for row in authored}:
        raise ValueError("Independent review must cover every case exactly once")
    review_map = {r["id"]: r for r in reviews}
    disagreements = []
    for row in authored:
        review = review_map[row["id"]]
        if review.get("independent_verdict") not in LABELS or not isinstance(review.get("rationale"), str) or not review["rationale"].strip():
            raise ValueError(f"{row['id']}: invalid independent review")
        if review["independent_verdict"] != row["expected_verdict"]:
            disagreements.append({"id": row["id"], "author": row["expected_verdict"],
                                  "reviewer": review["independent_verdict"], "reviewer_rationale": review["rationale"]})
    if disagreements:
        (directory / "label_disagreements.json").write_text(json.dumps(disagreements, indent=2), encoding="utf-8")
        raise ValueError(f"{len(disagreements)} label disagreements require resolution before freezing (see label_disagreements.json)")
    rng = random.Random(seed)
    splits = {"calibration": [], "holdout": []}
    for label in LABELS:
        group = [row for row in authored if row["expected_verdict"] == label]
        if len(group) < 2:
            raise ValueError("Each verdict needs at least two independent policy families")
        rng.shuffle(group)
        midpoint = len(group) // 2
        splits["calibration"].extend(group[:midpoint])
        splits["holdout"].extend(group[midpoint:])
    gold = []
    manifest = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "label_provenance": "Synthetic cases authored by agents and independently reviewed by another agent; human approval pending.",
                "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_fingerprint(),
                "core_sha256": core_fingerprint(), "splits": {}, "human_approved": False}
    for name, rows in splits.items():
        rng.shuffle(rows)
        inputs = [{key: row[key] for key in INPUT_KEYS} for row in rows]
        write_jsonl(directory / f"{name}.jsonl", inputs)
        for row in rows:
            gold.append({"id": row["id"], "split": name, "family_id": row["family_id"], "category": row["category"],
                         "expected_verdict": row["expected_verdict"], "author_rationale": row["gold_rationale"],
                         "independent_rationale": review_map[row["id"]]["rationale"], "case_sha256": fingerprint({key: row[key] for key in INPUT_KEYS})})
        manifest["splits"][name] = {"count": len(rows), "sha256": fingerprint(inputs),
                                    "ids": [row["id"] for row in rows],
                                    "families": [row["family_id"] for row in rows],
                                    "labels": {label: sum(row["expected_verdict"] == label for row in rows) for label in LABELS}}
    if set(manifest["splits"]["calibration"]["families"]) & set(manifest["splits"]["holdout"]["families"]):
        raise ValueError("Policy families overlap between splits")
    write_jsonl(directory / "gold.jsonl", gold)
    manifest["gold_sha256"] = fingerprint(gold)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (directory / "human_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "split", "agent_consensus_label", "human_label", "notes"])
        writer.writerows([row["id"], row["split"], row["expected_verdict"], "", ""] for row in gold)
    return manifest


def load_frozen(directory, split):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["prompt_sha256"] != prompt_fingerprint():
        raise ValueError("Judge prompt changed after freeze; use a new dataset version for new tuning")
    if manifest["core_sha256"] != core_fingerprint():
        raise ValueError("Judge validation code changed after freeze")
    cases = load_cases(directory / f"{split}.jsonl")
    if any("expected_verdict" in case for case in cases):
        raise ValueError("Benchmark inputs must not include labels")
    if fingerprint(cases) != manifest["splits"][split]["sha256"]:
        raise ValueError("Frozen inputs changed")
    gold = read_jsonl(directory / "gold.jsonl")
    if fingerprint(gold) != manifest["gold_sha256"]:
        raise ValueError("Frozen gold labels changed")
    labels = {row["id"]: row for row in gold if row["split"] == split}
    if set(labels) != {case["id"] for case in cases}:
        raise ValueError("Gold labels do not match the requested split")
    return manifest, cases, labels


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def benchmark_metrics(cases):
    matrix = {label: {prediction: 0 for prediction in PREDICTIONS} for label in LABELS}
    raw_matrix = {label: {prediction: 0 for prediction in PREDICTIONS} for label in LABELS}
    invalid_quotes = 0
    quote_count = 0
    cases_with_findings = 0
    cases_with_invalid_quotes = 0
    missing_evidence_cases = 0
    durations = []
    for case in cases:
        expected = case["expected_verdict"]
        judgment = case["judgment"]
        verdict = judgment["verdict"]
        matrix[expected][verdict] += 1
        scores = judgment["scores"]
        if verdict == "error":
            raw_verdict = "error"
        else:
            minimum = min(scores.values())
            raw_review = case.get("raw_needs_human_review", judgment["needs_human_review"]) or case.get("raw_human_review_reasons", [])
            raw_verdict = "fail" if minimum <= 1 else "review" if minimum == 2 or raw_review else "pass"
        raw_matrix[expected][raw_verdict] += 1
        findings = judgment["findings"]
        if findings:
            cases_with_findings += 1
        invalid = sum(finding.get("evidence_valid") is not True for finding in findings)
        quote_count += len(findings)
        invalid_quotes += invalid
        cases_with_invalid_quotes += invalid > 0
        missing_evidence_cases += any(reason.startswith("Missing cited findings for:") for reason in judgment["human_review_reasons"])
        if isinstance(case.get("elapsed_seconds"), (float, int)):
            durations.append(case["elapsed_seconds"])
    total = len(cases)
    matches = sum(matrix[label][label] for label in LABELS)
    errors = sum(matrix[label]["error"] for label in LABELS)
    class_metrics = {}
    for label in LABELS:
        true_positive = matrix[label][label]
        predicted = sum(matrix[expected][label] for expected in LABELS)
        actual = sum(matrix[label].values())
        class_metrics[label] = {"support": actual, "precision": _ratio(true_positive, predicted),
                                "recall": _ratio(true_positive, actual),
                                "f1": _ratio(2 * true_positive, actual + predicted)}
    unsafe = sum(sum(matrix[label].values()) for label in ("fail", "review"))
    false_passes = matrix["fail"]["pass"] + matrix["review"]["pass"]
    unsafe_flagged = sum(matrix[label][prediction] for label in ("fail", "review") for prediction in ("fail", "review"))
    return {"total": total, "matched": matches, "exact_agreement_all_cases": _ratio(matches, total),
            "successful_judgments": total - errors, "error_count": errors, "coverage": _ratio(total - errors, total),
            "macro_f1": statistics.mean(class_metrics[label]["f1"] or 0 for label in LABELS),
            "class_metrics": class_metrics, "confusion_matrix": matrix, "raw_score_confusion_matrix": raw_matrix,
            "false_passes": false_passes, "unsafe_cases": unsafe, "false_pass_rate": _ratio(false_passes, unsafe),
            "fail_to_pass": matrix["fail"]["pass"], "review_to_pass": matrix["review"]["pass"],
            "unsafe_flagged": unsafe_flagged, "unsafe_routing_rate": _ratio(unsafe_flagged, unsafe),
            "false_alarms_on_pass": matrix["pass"]["fail"] + matrix["pass"]["review"],
            "quotes": quote_count, "invalid_quotes": invalid_quotes,
            "quote_integrity_rate": _ratio(quote_count - invalid_quotes, quote_count),
            "cases_with_findings": cases_with_findings, "cases_with_invalid_quotes": cases_with_invalid_quotes,
            "cases_missing_criterion_evidence": missing_evidence_cases,
            "median_seconds_per_case": statistics.median(durations) if durations else None,
            "total_measured_seconds": sum(durations)}


def run_benchmark(directory, split, model, output, base_url="http://127.0.0.1:11434",
                  timeout=240, retry_errors=False):
    manifest, cases, labels = load_frozen(directory, split)
    judge = OllamaJudge(base_url=base_url, model=model, timeout=timeout)
    tags = judge._request("/api/tags").get("models", [])
    info = next((item for item in tags if item["name"] == model), None)
    if info is None:
        raise ValueError(f"Model {model} is not installed")
    digest = info.get("digest")
    if not digest:
        raise ValueError("Ollama did not provide a model digest")
    version = judge._request("/api/version").get("version", "unavailable")
    output = Path(output)
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    configuration = {"prompt_sha256": manifest["prompt_sha256"], "prompt_version": PROMPT_VERSION,
                     "core_sha256": manifest["core_sha256"], "ollama_version": version,
                     "model_digest": digest, "model_details": info.get("details", {}), "temperature": 0,
                     "seed": 42, "num_ctx": 8192, "num_predict": 1600, "base_url": base_url}
    run = {"schema_version": 1, "mode": "ollama", "model": model, "split": split,
           "generated_at": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
           "rubric": RUBRIC, "dataset_sha256": manifest["splits"][split]["sha256"],
           "label_provenance": manifest["label_provenance"], "human_approved": False,
           "cases": [], "requested_total": len(cases), "completed": False}
    interrupted = False
    for index, case in enumerate(cases, 1):
        # Gold labels are attached only after inference or cache lookup.
        key = fingerprint({"input": case, "configuration": configuration})
        cache_path = cache / f"{key}.json"
        saved = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else None
        if saved is not None and saved.get("cache_key") != key:
            raise ValueError("Cache binding mismatch")
        reuse = saved is not None and not (retry_errors and saved["case"]["judgment"]["verdict"] == "error")
        print(f"[{index}/{len(cases)}] {model} {split} {case['id']}{' (cached)' if reuse else ''}", flush=True)
        if reuse:
            result = saved["case"]
        else:
            result = dict(case)
            started = time.monotonic()
            try:
                result["judgment"] = judge.judge(result)
                raw = parse_judgment(result["raw_response"])
                result["raw_needs_human_review"] = raw["needs_human_review"]
                result["raw_human_review_reasons"] = raw["human_review_reasons"]
            except KeyboardInterrupt:
                result["judgment"] = error_judgment("Run interrupted during this case")
                interrupted = True
            except (ValueError, RuntimeError, KeyError) as exc:
                result["error"] = str(exc)
                result["judgment"] = error_judgment(str(exc))
            result["elapsed_seconds"] = time.monotonic() - started
            if not interrupted:
                cache_path.write_text(json.dumps({"cache_key": key, "case": result}, ensure_ascii=False, indent=2), encoding="utf-8")
        result["expected_verdict"] = labels[case["id"]]["expected_verdict"]
        result["category"] = labels[case["id"]]["category"]
        result["gold_rationale"] = labels[case["id"]]["independent_rationale"]
        run["cases"].append(result)
        print("  " + result["judgment"]["verdict"].upper(), flush=True)
        run["summary"] = summarize(run["cases"])
        run["benchmark_metrics"] = benchmark_metrics(run["cases"])
        (output / "checkpoint.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        if interrupted:
            break
    run["completed"] = not interrupted
    write_reports(run, output)
    (output / "metrics.json").write_text(json.dumps(run["benchmark_metrics"], indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run["benchmark_metrics"], indent=2), flush=True)
    return 130 if interrupted else 1 if run["benchmark_metrics"]["error_count"] else 0


def _percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"


def select_calibration(paths, output):
    """Record the candidate before holdout results are available."""
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if len(runs) < 2 or any(run.get("split") != "calibration" or not run.get("completed") for run in runs):
        raise ValueError("Selection requires at least two complete calibration runs")
    if len({run["dataset_sha256"] for run in runs}) != 1 or len({run["configuration"]["prompt_sha256"] for run in runs}) != 1:
        raise ValueError("Calibration runs must use the same inputs and prompt")
    if len({run["model"] for run in runs}) != len(runs):
        raise ValueError("Selection needs distinct models")
    candidates = []
    for run in runs:
        metrics = benchmark_metrics(run["cases"])
        candidates.append({"model": run["model"], "model_digest": run["configuration"]["model_digest"], "metrics": metrics})
    ranked = sorted(candidates, key=lambda item: (item["metrics"]["error_count"], item["metrics"]["false_passes"],
                                                 -item["metrics"]["macro_f1"], item["model"]))
    selection = {"selected_model": ranked[0]["model"], "created_at": datetime.now(timezone.utc).isoformat(),
                 "selection_rule": "Fewest errors, then fewest false passes, then highest macro F1; lexical model name breaks ties.",
                 "dataset_sha256": runs[0]["dataset_sha256"], "prompt_sha256": runs[0]["configuration"]["prompt_sha256"],
                 "candidates": candidates, "human_approved": False,
                 "status": "Exploratory selection from synthetic calibration data, pending holdout and human review."}
    output = Path(output)
    if output.exists():
        raise ValueError("Selection is already recorded; use a new benchmark version rather than selecting again after holdout")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    return selection


def compare_runs(paths, output):
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if any(not run.get("completed") for run in runs):
        raise ValueError("Comparison requires complete runs")
    prompt_ids = {run["configuration"]["prompt_sha256"] for run in runs}
    if len(prompt_ids) != 1:
        raise ValueError("Runs used different prompts")
    for split in {run["split"] for run in runs}:
        peers = [run for run in runs if run["split"] == split]
        if len({run["dataset_sha256"] for run in peers}) != 1:
            raise ValueError("Comparison runs used different split inputs")
    lines = ["# Local judge comparison", "", "Synthetic, agent-authored cases with independent agent label review. Human approval is pending.", "",
             "The prompt was frozen before testing. These results diagnose behavior on this authored set; they do not establish client readiness.", "",
             "| Split | Judge | Exact agreement (all cases) | Macro F1 | False passes / unsafe | Errors | Quotes verified | Median seconds |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    summaries = []
    for run in sorted(runs, key=lambda item: (item["split"], item["model"])):
        metrics = benchmark_metrics(run["cases"])
        summaries.append({"model": run["model"], "split": run["split"], "metrics": metrics,
                          "model_digest": run["configuration"]["model_digest"]})
        median = metrics["median_seconds_per_case"]
        lines.append(f"| {run['split']} | {run['model']} | {metrics['matched']}/{metrics['total']} ({_percent(metrics['exact_agreement_all_cases'])}) | {metrics['macro_f1']:.3f} | {metrics['false_passes']}/{metrics['unsafe_cases']} | {metrics['error_count']} | {_percent(metrics['quote_integrity_rate'])} | {median:.1f} |" if median is not None else "")
    lines += ["", "Quote verification checks that a substring exists in the cited reference. It does not check whether the quotation logically supports the finding.", "",
              "Errors remain in the agreement denominator. A review escalation on a genuinely bad answer can be useful routing while still disagreeing with its expected fail label.", "",
              "## Training decision", "", "Do not fine-tune from this small synthetic benchmark. Labels need human approval and broader representative examples. Keep the holdout separate from training. Fix systematic protocol or citation failures and validate on a new untouched set before claiming improvement.", "",
              "Both models are prespecified comparisons. Any model preference based on these holdout results is exploratory and needs confirmation on fresh data.", ""]
    for run in runs:
        metrics = benchmark_metrics(run["cases"])
        lines += [f"## {run['model']} — {run['split']}", "",
                  f"False passes from fail: {metrics['fail_to_pass']}; from review: {metrics['review_to_pass']}. False alarms on correct answers: {metrics['false_alarms_on_pass']}.", "",
                  "| Expected / predicted | Pass | Fail | Review | Error |", "| --- | ---: | ---: | ---: | ---: |"]
        for label in LABELS:
            lines.append(f"| {label} | " + " | ".join(str(metrics["confusion_matrix"][label][prediction]) for prediction in PREDICTIONS) + " |")
        lines += ["", "### Disagreements", ""]
        for case in run["cases"]:
            if case["judgment"]["verdict"] != case["expected_verdict"]:
                # Escape untrusted text for Markdown and HTML preview.
                safe = lambda value: html.escape(str(value)).replace("\n", " ").replace("|", "\\|")
                lines += [f"- **{safe(case['id'])}**: expected {case['expected_verdict']}, produced {case['judgment']['verdict']}. Label rationale: {safe(case['gold_rationale'])}"]
        lines.append("")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "comparison.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    # A simple self-contained HTML view with all counts and limitations.
    body = html.escape("\n".join(lines))
    (output / "comparison.html").write_text("<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>Judge comparison</title><style>body{max-width:1100px;margin:35px auto;padding:20px;background:#f5f7fa;color:#243348;font:15px/1.55 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;padding:25px;border:1px solid #d9e2ec;border-radius:10px}</style><pre>" + body + "</pre>", encoding="utf-8")
    return summaries
