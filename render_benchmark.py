"""Render the completed frozen benchmark as a self-contained comparison dashboard.

This command is read-only until every run and its calibration selection are
verified. It only writes comparison.html and decision.md in the comparison
output directory. It does not run inference, tune prompts, or train models.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

from evaluation.benchmarking import benchmark_metrics


ROOT = Path(__file__).resolve().parent
MODELS = ("qwen2.5-coder:7b", "qwen2.5:7b")
SPLITS = ("calibration", "holdout")
LABELS = ("pass", "fail", "review")
PREDICTIONS = LABELS + ("error",)
INPUT_KEYS = ("id", "question", "answer", "references", "requirements")
TRAINING_DECISION = (
    "Fine-tuning deferred; need human-approved representative labels and fresh validation."
)


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required benchmark artifact is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read benchmark artifact {path}: {exc}") from exc


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required frozen data is missing: {path}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read frozen data {path}: {exc}") from exc


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value, label):
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid timestamp for {label}") from exc
    require(parsed.tzinfo is not None, f"Timestamp for {label} must include its timezone")
    return parsed


def load_bundle(root=ROOT):
    """Verify frozen inputs, final results, and a selection made before holdout."""
    root = Path(root)
    dataset = root / "evaluation" / "benchmark" / "v1"
    output = root / "evaluation" / "output" / "benchmark"
    manifest = read_json(dataset / "manifest.json")
    selection = read_json(dataset / "selection.json")
    require(manifest.get("version") == 1, "Renderer expects frozen benchmark version 1")
    require(manifest.get("human_approved") is False,
            "This renderer describes agent-reviewed labels with human approval pending")
    require(file_sha256(root / "evaluation" / "core.py") == manifest.get("core_sha256"),
            "Evaluator source changed after the dataset was frozen")
    gold = read_jsonl(dataset / "gold.jsonl")
    require(fingerprint(gold) == manifest.get("gold_sha256"), "Frozen gold-label hash mismatch")
    gold_ids = [row["id"] for row in gold]
    require(len(gold_ids) == len(set(gold_ids)), "Frozen gold labels have duplicate IDs")
    require(set(manifest["splits"]) == set(SPLITS), "Unexpected benchmark splits")
    family_sets = [set(manifest["splits"][split]["families"]) for split in SPLITS]
    require(not family_sets[0] & family_sets[1], "Policy families overlap between splits")
    require(manifest["splits"]["calibration"]["count"] == 23
            and manifest["splits"]["holdout"]["count"] == 25,
            "This dashboard expects the frozen 23-case calibration and 25-case holdout")

    runs, paths, metrics, inputs = {}, {}, {}, {}
    for split in SPLITS:
        split_info = manifest["splits"][split]
        inputs[split] = read_jsonl(dataset / f"{split}.jsonl")
        require(fingerprint(inputs[split]) == split_info["sha256"],
                f"Frozen {split} input hash mismatch")
        require([case["id"] for case in inputs[split]] == split_info["ids"],
                f"Frozen {split} input IDs differ from manifest")
        require(len(inputs[split]) == split_info["count"], f"Frozen {split} count mismatch")
        labels = {row["id"]: row for row in gold if row["split"] == split}
        require(set(labels) == set(split_info["ids"]), f"Frozen {split} gold-label IDs differ")
        require([labels[case_id]["family_id"] for case_id in split_info["ids"]]
                == split_info["families"], f"Frozen {split} family metadata mismatch")
        require({label: sum(row["expected_verdict"] == label for row in labels.values())
                 for label in LABELS} == split_info["labels"], f"Frozen {split} label counts mismatch")
        require(all(set(case) == set(INPUT_KEYS) for case in inputs[split]),
                f"Frozen {split} inputs contain unexpected metadata")
        for model in MODELS:
            prefix = "coder" if model == MODELS[0] else "general"
            key = (split, model)
            path = output / f"{prefix}-{split}" / "results.json"
            run = read_json(path)
            label = f"{model} {split}"
            require(run.get("completed") is True, f"Run is incomplete: {label}")
            require(run.get("model") == model and run.get("split") == split
                    and run.get("mode") == "ollama", f"Run identity mismatch: {label}")
            require(run.get("requested_total") == split_info["count"]
                    and len(run.get("cases", [])) == split_info["count"],
                    f"Run case count mismatch: {label}")
            require(run.get("human_approved") is False, f"Unexpected label provenance: {label}")
            require(run.get("dataset_sha256") == split_info["sha256"],
                    f"Run input hash differs from frozen manifest: {label}")
            require([{field: case[field] for field in INPUT_KEYS} for case in run["cases"]]
                    == inputs[split], f"Run case inputs differ from frozen inputs: {label}")
            for case in run["cases"]:
                gold_case = labels[case["id"]]
                require(case.get("expected_verdict") == gold_case["expected_verdict"],
                        f"Run label mismatch: {label} {case['id']}")
                require(fingerprint({field: case[field] for field in INPUT_KEYS})
                        == gold_case["case_sha256"], f"Frozen case hash mismatch: {case['id']}")
                require(case["judgment"]["verdict"] in PREDICTIONS,
                        f"Unexpected predicted verdict: {label} {case['id']}")
            configuration = run["configuration"]
            for field in ("prompt_sha256", "prompt_version", "core_sha256"):
                require(configuration.get(field) == manifest.get(field),
                        f"Run {field} differs from frozen manifest: {label}")
            require(bool(configuration.get("model_digest")), f"Missing model digest: {label}")
            require(bool(configuration.get("ollama_version")), f"Missing Ollama version: {label}")
            timestamp(run.get("generated_at"), label)
            runs[key], paths[key] = run, path
            metrics[key] = benchmark_metrics(run["cases"])

    common_fields = ("prompt_sha256", "prompt_version", "core_sha256", "ollama_version",
                     "temperature", "seed", "num_ctx", "num_predict", "base_url")
    first_configuration = next(iter(runs.values()))["configuration"]
    for key, run in runs.items():
        require(all(field in run["configuration"] and run["configuration"][field]
                    == first_configuration[field] for field in common_fields),
                f"Runtime, prompt, or generation settings differ: {key}")
    for model in MODELS:
        require(runs[("calibration", model)]["configuration"]
                == runs[("holdout", model)]["configuration"],
                f"Model configuration changed between splits: {model}")

    require(selection.get("selected_model") in MODELS, "Selection names an unexpected model")
    require(selection.get("human_approved") is False, "Unexpected selection approval metadata")
    require(selection.get("dataset_sha256") == manifest["splits"]["calibration"]["sha256"]
            and selection.get("prompt_sha256") == manifest["prompt_sha256"],
            "Selection does not bind the frozen calibration inputs and prompt")
    expected_binding = {
        "dataset_version": manifest["version"], "gold_sha256": manifest["gold_sha256"],
        "splits": {split: manifest["splits"][split]["sha256"] for split in SPLITS},
        "configurations": {model: runs[("calibration", model)]["configuration"] for model in MODELS},
    }
    require(selection.get("workflow_binding") == expected_binding,
            "Selection workflow binding differs from final frozen runs")
    require(selection.get("calibration_file_sha256") == {
        model: file_sha256(paths[("calibration", model)]) for model in MODELS},
        "Calibration results changed after selection")
    candidates = selection.get("candidates", [])
    require(len(candidates) == len(MODELS) and {item["model"] for item in candidates} == set(MODELS),
            "Selection does not include exactly the two prespecified candidates")
    for candidate in candidates:
        require(candidate.get("metrics") == metrics[("calibration", candidate["model"])]
                and candidate.get("model_digest")
                == runs[("calibration", candidate["model"])]["configuration"]["model_digest"],
                "Selection candidate metrics or model digest differ from calibration results")
    ranked = sorted(MODELS, key=lambda model: (
        metrics[("calibration", model)]["error_count"],
        metrics[("calibration", model)]["false_passes"],
        -metrics[("calibration", model)]["macro_f1"], model))
    require(selection["selected_model"] == ranked[0], "Recorded selection disagrees with calibration rule")
    selected_at = timestamp(selection.get("created_at"), "calibration selection")
    for model in MODELS:
        require(timestamp(runs[("calibration", model)]["generated_at"], "calibration run")
                <= selected_at < timestamp(runs[("holdout", model)]["generated_at"], "holdout run"),
                f"Selection was not recorded between calibration and holdout starts: {model}")

    audit_path = output / "semantic_audit.json"
    audit = read_json(audit_path) if audit_path.is_file() else None
    if audit is not None:
        audit_bindings = audit.get("run_sha256", {})
        require(bool(audit_bindings) and set(audit_bindings) <= set(MODELS),
                "Semantic audit names unexpected or missing model bindings")
        for model, digest in audit_bindings.items():
            require(digest == file_sha256(paths[("calibration", model)]),
                    f"Semantic audit refers to a different calibration run: {model}")
        seen = set()
        for review in audit.get("reviews", []):
            key = (review["model"], review["id"])
            require(key not in seen, "Semantic audit has duplicate case reviews")
            seen.add(key)
            require(review["model"] in audit_bindings,
                    "Semantic audit review has no matching calibration file binding")
            calibration_cases = {case["id"]: case for case in runs[("calibration", review["model"])]["cases"]}
            require(review["id"] in calibration_cases, "Semantic audit names an unknown calibration case")
            case = calibration_cases[review["id"]]
            require(case["expected_verdict"] != case["judgment"]["verdict"],
                    "Semantic audit subset includes a case outside calibration disagreements")
            require(isinstance(review.get("verdict_supported"), bool)
                    and (review.get("findings_supported") is None
                         or isinstance(review.get("findings_supported"), bool))
                    and review.get("suggested_verdict") in LABELS
                    and isinstance(review.get("reasoning_errors"), list)
                    and all(isinstance(error, str) for error in review["reasoning_errors"]),
                    "Semantic audit review has invalid fields")
            require("review_routing_supported" not in review
                    or isinstance(review["review_routing_supported"], bool),
                    "Semantic audit review has invalid routing metadata")
    return {"manifest": manifest, "selection": selection, "runs": runs,
            "metrics": metrics, "paths": paths, "audit": audit}


def escape(value):
    return html.escape(str(value), quote=True)


def percent(value):
    return "N/A" if value is None else f"{value * 100:.1f}%"


def ratio_text(numerator, denominator):
    value = numerator / denominator if denominator else None
    return f"{numerator}/{denominator} ({percent(value)})"


def markdown_text(value):
    """Escape inline Markdown as well as raw HTML; never embed audit text here."""
    text = str(value).replace("\r", " ").replace("\n", " ")
    for character in "\\`*_{}[]()#+-.!|":
        text = text.replace(character, "\\" + character)
    return html.escape(text, quote=True)


def display_time(value):
    india_time = timestamp(value, "display time").astimezone(timezone(timedelta(hours=5, minutes=30)))
    return india_time.strftime("%d %b %Y, %I:%M %p IST")


CSS = """
:root{color-scheme:light;--ink:#17283e;--muted:#56667a;--line:#dde5ee;
--blue:#155be7;--wash:#f2f6fb;--green:#076746;--amber:#88570d}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);
color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--blue);text-underline-offset:3px}main{max-width:1250px;margin:auto;padding:36px 28px 60px}
header{padding:28px 0 16px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;
font-size:11px;font-weight:750;color:var(--blue)}h1{font-size:clamp(30px,5vw,48px);
line-height:1.13;letter-spacing:-.04em;margin:14px 0}h2{font-size:24px;letter-spacing:-.025em;
line-height:1.3;margin:0 0 8px}h3{font-size:17px;margin:0 0 10px}p{margin:0 0 14px}
.lede{max-width:830px;font-size:17px;color:var(--muted)}.muted,small{color:var(--muted)}
.nav{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;margin:22px 0}
.notice{background:#fff9e9;border:1px solid #ead7a6;border-radius:12px;padding:18px 22px;
color:#64430d;margin-bottom:26px}.notice p:last-child{margin:0}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:24px 0}
.card,.section,.matrix{background:white;border:1px solid var(--line);border-radius:14px}
.card{padding:23px}.card .label{font-size:12px;font-weight:650;color:var(--muted)}
.stat{font-size:38px;font-weight:750;letter-spacing:-.04em;line-height:1.3;margin:8px 0}
.model-stat{font-size:22px;letter-spacing:-.025em;overflow-wrap:anywhere;margin:12px 0;
font-weight:750;line-height:1.35}.card p{font-size:13px;margin:0;color:var(--muted)}
.section{padding:26px;margin-top:24px;scroll-margin-top:20px}.intro{max-width:900px;color:var(--muted)}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:9px;margin-top:20px}
table{border-collapse:collapse;width:100%;font-size:13px;text-align:left;min-width:1150px}
caption{text-align:left;padding:13px 15px;color:var(--muted);background:#f9fbfe;font-size:12px}
th{background:#f4f7fb;color:#43546b;font-size:11px;text-transform:uppercase;letter-spacing:.045em}
th,td{padding:15px 13px;border-bottom:1px solid var(--line);vertical-align:top}
tbody tr:last-child td,tbody tr:last-child th{border-bottom:0}.selected td{background:#f4f8ff}
td strong{display:block;font-size:14px;white-space:nowrap}td small{display:block;max-width:220px}
.pill{display:inline-block;padding:3px 8px;border-radius:999px;background:#eaf1ff;color:#2159b3;
font-size:10px;font-weight:700;letter-spacing:.02em;margin-top:6px}.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.notes{font-size:12px;color:var(--muted);margin:15px 0 0;max-width:1000px}
.matrix-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:20px}.matrix{padding:20px;
border-radius:10px}.matrix .table-wrap{margin-top:14px}.matrix table{min-width:350px;text-align:right}
.matrix th:first-child,.matrix td:first-child{text-align:left}.matrix th,.matrix td{padding:11px}
.diagonal{background:#eaf8f1;font-weight:750;color:var(--green)}.matrix p{font-size:12px;margin:8px 0 0}
.decision{border-left:4px solid var(--blue)}.decision ul{margin:14px 0;padding-left:22px}
.decision li{margin:7px 0}.audit-list{margin:16px 0;padding-left:22px}.audit-list li{margin:12px 0}
.audit-item{background:#f7f9fc;border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:12px 0}
details{margin:16px 0;border:1px solid var(--line);border-radius:9px;background:#fbfcfe}
summary{cursor:pointer;font-size:13px;font-weight:650;padding:14px 17px}details>div{padding:0 17px 17px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word;font:12px/1.6 ui-monospace,
Consolas,monospace;background:#f3f6fb;border-radius:7px;padding:15px;margin:5px 0}
dl{display:grid;grid-template-columns:180px 1fr;gap:8px 16px;font-size:13px}dt{color:var(--muted)}
dd{margin:0;overflow-wrap:anywhere}footer{font-size:12px;color:var(--muted);padding:25px 0}
@media(max-width:780px){main{padding:22px 16px 40px}.cards,.matrix-grid{grid-template-columns:1fr}
.card{padding:18px}.section{padding:20px 16px}.stat{font-size:32px}dl{grid-template-columns:1fr;gap:2px}
dd{margin-bottom:8px}.notice{padding:16px}.nav{gap:12px}}
"""


def metrics_table(bundle, split):
    selection = bundle["selection"]["selected_model"]
    rows = []
    for model in MODELS:
        m = bundle["metrics"][(split, model)]
        correct = m["class_metrics"]["pass"]["support"]
        verified = m["quotes"] - m["invalid_quotes"]
        median = m["median_seconds_per_case"]
        selected = model == selection
        badge = '<span class="pill">Selected on calibration</span>' if selected else ""
        runtime = "N/A" if median is None else f"{median:.1f} s"
        rows.append(
            f'<tr class="{"selected" if selected else ""}"><td><strong>{escape(model)}</strong>{badge}</td>'
            f'<td class="num">{ratio_text(m["matched"], m["total"])}</td>'
            f'<td class="num">{m["macro_f1"]:.3f}</td>'
            f'<td class="num">{ratio_text(m["false_passes"], m["unsafe_cases"])}</td>'
            f'<td class="num">{ratio_text(m["unsafe_flagged"], m["unsafe_cases"])}</td>'
            f'<td class="num">{ratio_text(m["false_alarms_on_pass"], correct)}</td>'
            f'<td class="num">{ratio_text(m["confusion_matrix"]["pass"]["pass"], correct)}</td>'
            f'<td class="num">{ratio_text(m["error_count"], m["total"])}</td>'
            f'<td class="num">{ratio_text(verified, m["quotes"])}</td>'
            f'<td class="num">{ratio_text(m["cases_missing_criterion_evidence"], m["total"])}</td>'
            f'<td class="num">{runtime}</td></tr>'
        )
    count = bundle["manifest"]["splits"][split]["count"]
    return (
        '<div class="table-wrap"><table>'
        f'<caption>Both judges received the same {count} frozen {escape(split)} cases.</caption>'
        '<thead><tr><th scope="col">Judge</th><th scope="col">Exact label agreement<br>All cases</th>'
        '<th scope="col">Macro F1</th><th scope="col">False passes<br>Fail + review labels</th>'
        '<th scope="col">Routed fail/review<br>Fail + review labels</th>'
        '<th scope="col">False alarms<br>Pass labels</th><th scope="col">Pass recall<br>Pass labels</th>'
        '<th scope="col">Errors<br>All cases</th>'
        '<th scope="col">Quote integrity<br>Exact source matches</th>'
        '<th scope="col">Evidence gaps<br>All cases</th>'
        '<th scope="col">Median<br>Time / case</th></tr></thead><tbody>'
        + "".join(rows) + '</tbody></table></div>'
    )


def confusion_matrix(bundle, model):
    m = bundle["metrics"][("holdout", model)]
    rows = []
    for label in LABELS:
        cells = "".join(
            f'<td class="{"diagonal" if label == prediction else "num"}">'
            f'{m["confusion_matrix"][label][prediction]}</td>' for prediction in PREDICTIONS)
        rows.append(f'<tr><th scope="row">{label.title()}</th>{cells}</tr>')
    badge = '<span class="pill">Selected before holdout</span>' if model == bundle["selection"]["selected_model"] else ""
    return (
        f'<article class="matrix"><h3>{escape(model)}</h3>{badge}<div class="table-wrap"><table>'
        '<caption>Rows: expected label · Columns: final predicted verdict</caption><thead><tr>'
        '<th scope="col">Expected</th>'
        + "".join(f'<th scope="col">{prediction.title()}</th>' for prediction in PREDICTIONS)
        + '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
        f'<p>Fail → pass: {m["fail_to_pass"]}; review → pass: {m["review_to_pass"]}. '
        f'Errors stay in the {m["total"]}-case denominator.</p></article>'
    )


def audit_html(bundle):
    audit = bundle["audit"]
    if audit is None:
        return ('<p class="intro">No final semantic-audit artifact was available when this dashboard '
                'was rendered. Findings and labels still require human review.</p>')
    reviews = audit.get("reviews", [])
    parts = [
        f'<p class="intro">A separate agent reviewed {len(reviews)} calibration disagreement cases. '
        'This audit covers a subset of rationales and findings; it does not assess all cases or provide '
        'human-approved labels. Suggested verdicts are qualitative review suggestions and do not change '
        'the frozen benchmark metrics. A review escalation may remain procedurally justified by invalid '
        'or missing evidence even when the finding’s reasoning is faulty.</p>',
    ]
    for model in MODELS:
        model_reviews = [review for review in reviews if review["model"] == model]
        if not model_reviews:
            continue
        total_disagreements = sum(case["expected_verdict"] != case["judgment"]["verdict"]
                                  for case in bundle["runs"][("calibration", model)]["cases"])
        with_errors = sum(bool(review["reasoning_errors"]) for review in model_reviews)
        parts.append(f'<p><strong>{escape(model)}</strong>: {len(model_reviews)} of '
                     f'{total_disagreements} calibration disagreement cases reviewed; '
                     f'{with_errors} reviewed cases have reported reasoning or citation errors. '
                     'These counts describe this review subset.</p>')
        examples = []
        for review in model_reviews:
            errors = review["reasoning_errors"]
            if not errors:
                continue
            routing = review.get("review_routing_supported")
            routing_note = (" Review routing is supported by the evidence protocol."
                            if routing is True else " Review routing is unsupported in this agent review."
                            if routing is False else "")
            examples.append(
                f'<article class="audit-item"><h3>{escape(review["id"])}</h3><ul>'
                + "".join(f'<li>{escape(error)}</li>' for error in errors)
                + f'</ul><p class="notes">Agent-suggested document verdict: '
                f'{escape(review["suggested_verdict"])}.{routing_note} Human review required.</p></article>')
        if examples:
            parts.append(f'<details><summary>{escape(model)} · reported reasoning and citation-error examples</summary>'
                         '<div>' + "".join(examples) + '</div></details>')
    parts.append('<details><summary>Complete bound semantic-audit data</summary><div><pre>'
                 + escape(json.dumps(audit, ensure_ascii=False, indent=2)) + '</pre></div></details>')
    return "\n".join(parts)


def render_html(bundle):
    manifest, selection = bundle["manifest"], bundle["selection"]
    selected = selection["selected_model"]
    calibration = bundle["metrics"][("calibration", selected)]
    holdout = bundle["metrics"][("holdout", selected)]
    distribution = lambda split: ", ".join(
        f'{manifest["splits"][split]["labels"][label]} {label}' for label in LABELS)
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Local judge benchmark · comparison</title><style>' + CSS + '</style></head><body><main>',
        '<header><div class="eyebrow">Local judge benchmark · Frozen version 1</div>'
        '<h1>Two local judges, one frozen test.</h1>'
        '<p class="lede">A comparison of two Qwen 7B models on 48 fictional, document-grounded cases. '
        'The candidate was selected from calibration results before either model saw the holdout.</p>'
        '<nav class="nav" aria-label="Report sections"><a href="#holdout">Holdout results</a>'
        '<a href="#calibration">Calibration selection</a><a href="#matrices">Confusion matrices</a>'
        '<a href="#reasoning">Reasoning review</a><a href="#metadata">Reproducibility</a></nav></header>',
        '<aside class="notice"><p><strong>Human approval pending.</strong> Cases were authored by agents '
        'and independently reviewed by another agent. Labels are synthetic and have not been approved '
        'by a human. These observations describe this small authored sample; they do not establish '
        'performance on real client data.</p></aside>',
        '<div class="cards"><article class="card"><div class="label">Candidate selected on calibration</div>'
        f'<div class="model-stat">{escape(selected)}</div><p>Chosen before holdout using errors, '
        'false passes, then macro F1.</p></article><article class="card">'
        '<div class="label">Selected judge · holdout label matches</div>'
        f'<div class="stat">{holdout["matched"]} / {holdout["total"]}</div>'
        f'<p>{percent(holdout["exact_agreement_all_cases"])} exact agreement across all 25 cases, '
        'including errors.</p></article><article class="card">'
        '<div class="label">Training decision</div><div class="model-stat">Fine-tuning deferred</div>'
        '<p>Obtain human-approved representative labels and fresh validation first.</p></article></div>',
        '<section class="section" id="holdout"><h2>Holdout observations</h2>'
        f'<p class="intro">25 unseen policy families: {distribution("holdout")}. '
        'Both models are shown as prespecified comparisons. The selected candidate remains the one '
        'chosen from calibration; any new preference based on this table would need fresh validation.</p>',
        metrics_table(bundle, "holdout"),
        '<p class="notes">Exact agreement uses the final verdict after citation validation. '
        'False passes are pass verdicts on expected fail or review cases; the denominator is all '
        '16 such holdout cases. False alarms are fail or review verdicts on the 9 expected pass cases. '
        'Pass recall counts how many expected pass cases the judge actually accepts. Zero false '
        'passes alone can result from accepting no answers and does not establish useful judging. '
        'Errors are reported separately and remain in the agreement denominator. Macro F1 averages '
        'pass, fail, and review F1; an error receives no label-match credit. Routed fail/review counts '
        'expected fail/review cases sent to either fail or review. Evidence gaps count cases lacking '
        'a cited finding for one or more low-scored criteria.</p></section>',
        '<section class="section" id="calibration"><h2>Selection made before holdout</h2>'
        f'<p class="intro">23 calibration cases: {distribution("calibration")}. '
        f'<strong>{escape(selected)}</strong> was recorded on {escape(display_time(selection["created_at"]))}. '
        'Selection used only these calibration results.</p>',
        metrics_table(bundle, "calibration"),
        '<p class="notes">Prespecified rule: fewest errors, then fewest false passes, then highest '
        'macro F1; lexical model name breaks ties. The renderer verifies the selection’s file hashes, '
        'runtime binding, and timestamp before producing this report. Compare pass recall and false '
        'alarms alongside false passes: a judge that accepts no correct answers can still make '
        'zero false passes.</p></section>',
        '<section class="section" id="matrices"><h2>Where holdout labels differed</h2>'
        '<p class="intro">Each matrix contains all 25 cases. Green cells are exact label matches. '
        'A review escalation can route a faulty answer to a person while still differing from its '
        'expected fail label.</p><div class="matrix-grid">',
        *(confusion_matrix(bundle, model) for model in MODELS),
        '</div></section>',
        '<section class="section" id="reasoning"><h2>Quote matching is only an integrity check</h2>'
        '<p class="intro">Quote integrity counts findings whose quoted text exists as an exact '
        'substring in the cited reference. It does not establish that the quote logically supports '
        'the finding. Passing this check cannot certify sound reasoning.</p>',
        audit_html(bundle),
        '</section>',
        '<section class="section decision"><h2>What these results support</h2>'
        f'<p><strong>{escape(TRAINING_DECISION)}</strong></p>'
        f'<p>The calibration-selected judge matched {holdout["matched"]} of the 25 heldout labels. '
        f'It produced {holdout["false_passes"]} false passes among {holdout["unsafe_cases"]} expected '
        f'fail/review cases, {holdout["false_alarms_on_pass"]} false alarms among 9 expected pass cases, '
        f'and {holdout["error_count"]} errors. These are observed sample counts.</p>'
        '<ul><li>Have a human approve representative labels and verify findings against their source passages.</li>'
        '<li>Diagnose repeated reasoning errors and repair citation/evidence scaffolding after this '
        'frozen benchmark, using a new dataset version to evaluate changes.</li>'
        '<li>Use a fresh untouched validation set after any prompt, model, or training changes. '
        'Keep this frozen holdout separate from training.</li></ul>'
        '<p class="muted">No fine-tuning has occurred in this benchmark. Neither software checks '
        'nor these synthetic label matches establish readiness for unattended client audits.</p></section>',
        '<section class="section" id="metadata"><h2>Reproducibility details</h2>'
        '<p class="intro">All four complete runs use the same frozen prompt, evaluator source, '
        'Ollama version, and generation settings. Model digests are stable across splits. '
        'Policy families are disjoint between calibration and holdout.</p>',
        '<details><summary>Frozen dataset, prompt, and selection</summary><div><pre>',
        escape(json.dumps({"version": manifest["version"], "label_provenance": manifest["label_provenance"],
                           "human_approved": manifest["human_approved"], "prompt_version": manifest["prompt_version"],
                           "prompt_sha256": manifest["prompt_sha256"], "core_sha256": manifest["core_sha256"],
                           "gold_sha256": manifest["gold_sha256"], "splits": manifest["splits"],
                           "selected_model": selected, "selection_created_at": selection["created_at"],
                           "selection_rule": selection["selection_rule"]}, ensure_ascii=False, indent=2)),
        '</pre></div></details>',
    ]
    for model in MODELS:
        run = bundle["runs"][("calibration", model)]
        runtime = {split: {
            "generated_at": bundle["runs"][(split, model)]["generated_at"],
            "total_measured_seconds": bundle["metrics"][(split, model)]["total_measured_seconds"],
            "median_seconds_per_case": bundle["metrics"][(split, model)]["median_seconds_per_case"],
            "cases_with_invalid_quotes": bundle["metrics"][(split, model)]["cases_with_invalid_quotes"],
            "cases_missing_criterion_evidence": bundle["metrics"][(split, model)]["cases_missing_criterion_evidence"],
        } for split in SPLITS}
        parts.append(f'<details><summary>{escape(model)} · model digest and runtime</summary><div><pre>'
                     + escape(json.dumps({"configuration": run["configuration"], "runs": runtime},
                                         ensure_ascii=False, indent=2)) + '</pre></div></details>')
    parts += ['</section><footer>Generated from verified completed results. All data remains local. '
              'Synthetic benchmark · human review required.</footer></main></body></html>']
    return "\n".join(parts)


def render_decision(bundle):
    selected = bundle["selection"]["selected_model"]
    calibration = bundle["metrics"][("calibration", selected)]
    holdout = bundle["metrics"][("holdout", selected)]
    lines = [
        "# Judge selection and training decision", "",
        f"**{TRAINING_DECISION}**", "",
        "The 48 cases are fictional, machine-authored examples with independent agent label review. "
        "They are not human-approved labels or representative client data. Calibration contains 23 "
        "cases; holdout contains 25. Underlying policy families are disjoint.", "",
        f"The calibration rule selected **{markdown_text(selected)}** before either holdout run started: "
        "fewest errors, then fewest false passes, then highest macro F1; lexical model name breaks ties. "
        f"On calibration it matched {calibration['matched']}/23 labels, made "
        f"{calibration['false_passes']}/{calibration['unsafe_cases']} false passes on expected fail/review "
        f"cases, and returned {calibration['error_count']} errors.", "",
        f"On holdout the selected judge matched **{holdout['matched']}/25** labels "
        f"({percent(holdout['exact_agreement_all_cases'])}), with macro F1 {holdout['macro_f1']:.3f}. "
        f"It made {holdout['false_passes']}/{holdout['unsafe_cases']} false passes, "
        f"{holdout['false_alarms_on_pass']}/9 false alarms on expected pass cases, and "
        f"{holdout['error_count']}/25 errors. These are observed counts in this small synthetic sample, "
        "not estimates of reliability on client work.", "",
        "| Split | Judge | Exact label agreement, all cases | Macro F1 | False passes / fail + review | "
        "False alarms / pass | Pass recall | Errors / all cases | Exact quote matches / findings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for split in SPLITS:
        for model in MODELS:
            m = bundle["metrics"][(split, model)]
            pass_count = m["class_metrics"]["pass"]["support"]
            lines.append(
                f"| {split} | {markdown_text(model)} | {ratio_text(m['matched'], m['total'])} | "
                f"{m['macro_f1']:.3f} | {ratio_text(m['false_passes'], m['unsafe_cases'])} | "
                f"{ratio_text(m['false_alarms_on_pass'], pass_count)} | "
                f"{ratio_text(m['confusion_matrix']['pass']['pass'], pass_count)} | "
                f"{ratio_text(m['error_count'], m['total'])} | "
                f"{ratio_text(m['quotes'] - m['invalid_quotes'], m['quotes'])} |")
    lines += [
        "", "Errors remain in the label-agreement denominator. False passes mean predicted pass "
        "on expected fail or review. False alarms mean predicted fail or review on expected pass; "
        "runtime errors are reported separately. Final verdicts incorporate citation validation. "
        "Useful review routing can still differ from an expected fail label. Pass recall measures "
        "acceptance of expected pass cases; zero false passes alone can result from accepting no "
        "correct answers and does not establish useful judging.", "",
        "Exact quote matches check only that a substring exists in its cited reference. They do not "
        "establish that the finding is logically supported. Human review must check both the evidence "
        "and its interpretation.", "",
    ]
    for split in SPLITS:
        for model in MODELS:
            m = bundle["metrics"][(split, model)]
            lines += [f"{markdown_text(model)} on {split}: {m['unsafe_flagged']}/{m['unsafe_cases']} "
                      "expected fail/review cases were routed to fail or review; "
                      f"{m['cases_missing_criterion_evidence']}/{m['total']} cases lacked cited findings "
                      "for one or more low-scored criteria.", ""]
    if bundle["audit"] is not None:
        lines += ["A separate agent reasoning audit is included in comparison.html. It provides "
                  "machine-reviewed examples, not human approval; review those examples and conclusions "
                  "before relying on the findings.", ""]
    lines += [
        "Fine-tuning is premature because these labels lack human approval and the dataset is a small "
        "synthetic diagnostic set. No fine-tuning has been performed. First obtain human-approved "
        "representative examples, investigate repeated reasoning and protocol errors, and compare any "
        "changes on fresh untouched validation data. Keep this frozen holdout separate from training.", "",
        "Repair citation/evidence scaffolding after this frozen run using a new dataset version. "
        "A judge can identify an answer’s real fault yet receive a final review verdict because an "
        "additional low score lacks a valid cited finding; a calibration disagreement alone does not "
        "establish that the judge missed the answer’s fault.", "",
        "Both prespecified models are reported. The candidate remains the one selected from "
        "calibration. Any preference formed after inspecting holdout results needs confirmation on "
        "fresh data. These results do not establish readiness for unattended client audits.", "",
        "The renderer verified four completed runs, case inputs and labels, frozen hashes, common "
        "prompt/evaluator/runtime settings, stable model digests across splits, disjoint policy "
        "families, and the calibration selection’s hashes and timing.", "",
    ]
    return "\n".join(lines)


def render(root=ROOT):
    bundle = load_bundle(root)
    dashboard = render_html(bundle)
    decision = render_decision(bundle)
    output = Path(root) / "evaluation" / "output" / "benchmark-comparison"
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.html").write_text(dashboard, encoding="utf-8")
    (output / "decision.md").write_text(decision, encoding="utf-8")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        output = render()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Comparison rendering failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dashboard: {output / 'comparison.html'}")
    print(f"Decision: {output / 'decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
