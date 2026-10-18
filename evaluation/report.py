"""Self-contained, evidence-first reports for local chatbot evaluations."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from pathlib import Path


_VERDICTS = ("pass", "fail", "review", "error")
_SCORES = (
    ("groundedness", "Groundedness"),
    ("instruction_following", "Instruction following"),
    ("completeness", "Completeness"),
)


def _text(value: object) -> str:
    return "" if value is None else str(value)


def _html(value: object) -> str:
    return html.escape(_text(value), quote=True)


def _md(value: object) -> str:
    """Keep supplied content from becoming Markdown or embedded HTML."""
    text = html.escape(_text(value), quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|~\-])", r"\\\1", text)


def _dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _objects(value: object) -> list[dict]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _verdict(case: dict) -> str:
    value = _dict(case.get("judgment")).get("verdict", "error")
    return value if value in _VERDICTS else "error"


def _review(case: dict) -> bool:
    judgment = _dict(case.get("judgment"))
    return (
        bool(judgment.get("needs_human_review"))
        or _verdict(case) in ("review", "error")
        or any(finding.get("evidence_valid") is False for finding in _objects(judgment.get("findings")))
    )


def _reasons(case: dict) -> list[str]:
    value = _dict(case.get("judgment")).get("human_review_reasons", [])
    reasons = [_text(item) for item in value] if isinstance(value, list) else [_text(value)]
    reasons = [item for item in reasons if item]
    if _review(case) and not reasons:
        reasons = ["Evaluation failed; inspect the error and assess this case manually." if _verdict(case) == "error" else "A reviewer should check the answer and its supporting evidence."]
    return reasons


def _requirements(case: dict) -> list[str]:
    value = case.get("requirements", [])
    if isinstance(value, list):
        return [_text(item) for item in value if _text(item)]
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False, indent=2)]
    return [_text(value)] if value else []


def _agreement(run: dict) -> str:
    value = _dict(run.get("summary")).get("agreement")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or value > 100:
        return "Unavailable"
    percent = value * 100 if value <= 1 else value
    return f"{percent:.1f}%"


def _labelled(run: dict, cases: list[dict]) -> int:
    value = _dict(run.get("summary")).get("labelled")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return sum(case.get("expected_verdict") in ("pass", "fail", "review") and _verdict(case) != "error" for case in cases)


def _incomplete(run: dict, total: int) -> str:
    if run.get("completed") is not False:
        return ""
    requested = run.get("requested_total")
    progress = f"{total} of {requested} requested cases are included." if isinstance(requested, int) and not isinstance(requested, bool) and requested >= total else f"{total} cases are included."
    return f"This run did not complete. {progress} These are partial results."


def _counts(cases: list[dict]) -> dict[str, int]:
    return {verdict: sum(_verdict(case) == verdict for case in cases) for verdict in _VERDICTS}


def _notice(run: dict) -> tuple[str, str]:
    if run.get("mode") == "demo":
        return (
            "Illustrative saved judgments — no Qwen run occurred.",
            "This demonstration shows the report format using saved judgments. It does not measure a live model's performance.",
        )
    return (
        "Automated judgments require human review.",
        "Check the answers, reference excerpts and findings before using these judgments for client decisions. Rubric scores are ratings, not confidence probabilities.",
    )


def _html_case(case: dict, index: int) -> str:
    judgment = _dict(case.get("judgment"))
    scores = _dict(judgment.get("scores"))
    verdict = _verdict(case)
    review = _review(case)
    expected = case.get("expected_verdict")
    expected_html = f'<span class="expected">Expected label: {_html(expected)}</span>' if expected is not None else ""
    review_html = '<span class="review-tag">Human review needed</span>' if review else ""
    score_html = "".join(
        f'<div><dt>{label}</dt><dd>{_html(scores.get(key)) if scores.get(key) is not None else "Unavailable"}</dd></div>'
        for key, label in _SCORES
    )
    requirements = _requirements(case)
    requirements_html = (
        '<div class="requirements"><h4>Answer requirements</h4><ul>'
        + "".join(f"<li>{_html(item)}</li>" for item in requirements)
        + "</ul></div>"
    ) if requirements else ""
    reasons = _reasons(case)
    review_block = (
        '<div class="review-callout"><strong>Reviewer attention</strong><ul>'
        + "".join(f"<li>{_html(item)}</li>" for item in reasons)
        + "</ul></div>"
    ) if review else ""
    error_block = f'<div class="error-callout"><strong>Evaluation error</strong><p class="preserve">{_html(case.get("error"))}</p></div>' if case.get("error") else ""
    findings_html = ""
    for finding in _objects(judgment.get("findings")):
        quote = finding.get("quote")
        quote_html = f'<blockquote class="preserve">{_html(quote)}</blockquote>' if quote else '<p class="muted">No supporting quote recorded.</p>'
        if finding.get("evidence_valid") is False:
            evidence_status = '<div class="evidence-warning"><strong>Unverified evidence: quote or reference did not match.</strong> This finding is untrusted. Check the supplied references before accepting its claim or explanation.</div>'
        elif finding.get("evidence_valid") is True:
            evidence_status = '<p class="evidence-check">Quote found in the supplied reference. This checks quote existence only; it does not validate that the quote supports the claim.</p>'
        else:
            evidence_status = '<p class="evidence-check">Evidence matching status unavailable. Review the source text before accepting this finding.</p>'
        findings_html += (
            '<div class="finding">'
            f'<div class="finding-label">{_html(finding.get("category") or "Finding")}<span>Reference: {_html(finding.get("reference_id") or "Unspecified")}</span></div>'
            f'<p class="preserve"><strong>Claim:</strong> {_html(finding.get("claim"))}</p>'
            f'{evidence_status}{quote_html}<p class="preserve">{_html(finding.get("explanation"))}</p></div>'
        )
    if not findings_html:
        findings_html = '<p class="muted">No claim-level findings were recorded. Review the rationale and reference excerpts below.</p>'
    references = _objects(case.get("references"))
    references_html = "".join(
        f'<div class="reference"><h4>{_html(ref.get("id") or "Reference")}</h4><p class="preserve">{_html(ref.get("text"))}</p></div>'
        for ref in references
    ) or '<p class="muted">No reference excerpts were supplied.</p>'
    return f"""
<article class="case" id="case-{index}">
  <div class="case-header"><div><span class="eyebrow">Case {index + 1}</span><h3>{_html(case.get('id') or f'Case {index + 1}')}</h3></div><div class="case-tags"><span class="badge {verdict}">{verdict.title()}</span>{review_html}{expected_html}</div></div>
  {review_block}{error_block}
  <div class="answer-grid"><section><h4>Question</h4><p class="preserve">{_html(case.get('question'))}</p>{requirements_html}</section><section><h4>Chatbot answer</h4><p class="preserve">{_html(case.get('answer') or 'No answer recorded.')}</p></section></div>
  <div class="judgment"><h4>Judgment rationale</h4><p class="preserve">{_html(judgment.get('rationale') or 'No rationale recorded.')}</p><dl class="scores">{score_html}</dl></div>
  <section class="evidence"><h4>Claim-level evidence</h4>{findings_html}</section>
  <details class="references"><summary>Reference excerpts ({len(references)})</summary>{references_html}</details>
</article>"""


def _render_html(run: dict, cases: list[dict]) -> str:
    counts = _counts(cases)
    flagged = sum(_review(case) for case in cases)
    labelled = _labelled(run, cases)
    notice, detail = _notice(run)
    incomplete = _incomplete(run, len(cases))
    incomplete_html = f'<aside class="notice incomplete"><strong>Incomplete evaluation run</strong><p>{_html(incomplete)}</p></aside>' if incomplete else ""
    metrics = [("Total cases", len(cases), "total")]
    metrics.extend((name.title(), counts[name], name) for name in _VERDICTS)
    metrics.append(("Flagged for review", flagged, "flagged"))
    metrics_html = "".join(f'<div class="metric {kind}"><span>{label}</span><strong>{value}</strong></div>' for label, value, kind in metrics)
    review_links = "".join(
        f'<a href="#case-{index}"><span>{_html(case.get("id") or f"Case {index + 1}")}</span><span class="badge {_verdict(case)}">{_verdict(case).title()}</span></a>'
        for index, case in enumerate(cases) if _review(case)
    )
    queue = '<div class="review-links">' + review_links + '</div>' if review_links else '<p class="muted">No cases were flagged for additional review. Automated judgments still require a reviewer’s sign-off.</p>'
    rubric = json.dumps(run.get("rubric", {}), ensure_ascii=False, indent=2)
    cases_html = "".join(_html_case(case, index) for index, case in enumerate(cases)) or '<div class="panel"><p>No evaluation cases were recorded.</p></div>'
    mode = "Demonstration" if run.get("mode") == "demo" else "Local model evaluation"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Chatbot evaluation report</title>
<style>
.evidence-warning{{background:#fcebed;color:#9f2638;border:1px solid #efbdc4;border-radius:6px;padding:10px 12px;margin:9px 0}}.evidence-warning strong{{display:block;margin-bottom:3px}}.evidence-check{{font-size:11px;color:#66738a;margin:8px 0}}.notice.incomplete{{background:#fcebed;border-color:#e8a5b0;border-left-color:#ae3447}}.notice.incomplete strong,.notice.incomplete p{{color:#943244}}
:root{{color-scheme:light;--ink:#202d3d;--muted:#66738a;--line:#dee5ed;--paper:#fff;--bg:#f3f6fa;--accent:#245bbd}}
*{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1180px;margin:0 auto;padding:42px 28px 50px}}h1,h2,h3,h4,p{{margin:0}}h1{{font-size:34px;letter-spacing:-1.2px;line-height:1.15;margin:9px 0 14px}}h2{{font-size:20px;letter-spacing:-.35px}}h3{{font-size:19px;overflow-wrap:anywhere}}h4{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-bottom:9px}}p+ p{{margin-top:8px}}a{{color:var(--accent)}}.eyebrow{{font-size:11px;font-weight:750;letter-spacing:1.5px;text-transform:uppercase;color:var(--accent)}}.topline{{display:flex;justify-content:space-between;gap:20px;align-items:center}}.mode{{font-size:12px;color:var(--muted);border:1px solid var(--line);border-radius:20px;padding:5px 11px;background:white}}.meta{{display:flex;flex-wrap:wrap;gap:8px 26px;color:var(--muted);font-size:13px;overflow-wrap:anywhere}}.meta strong{{color:var(--ink);font-weight:600}}.notice{{background:#fff7e9;border:1px solid #e9c77c;border-left:4px solid #c38c20;border-radius:10px;padding:16px 18px;margin:26px 0 18px}}.notice strong{{display:block;font-size:15px;color:#715018;margin-bottom:4px}}.notice p{{font-size:13px;color:#715d3e}}.metrics{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;margin-bottom:12px}}.metric{{border:1px solid var(--line);border-radius:10px;background:white;padding:15px 16px}}.metric span{{font-size:11px;color:var(--muted);display:block;font-weight:600}}.metric strong{{display:block;font-size:28px;line-height:1.3;margin-top:3px;font-variant-numeric:tabular-nums}}.metric.pass strong{{color:#227248}}.metric.fail strong{{color:#ae3447}}.metric.review strong,.metric.flagged strong{{color:#9b650b}}.metric.error strong{{color:#7051a0}}.agreement{{font-size:12px;color:var(--muted);padding:4px 0 20px}}.agreement strong{{color:var(--ink)}}.panel,.case{{border:1px solid var(--line);border-radius:12px;background:white;box-shadow:0 2px 4px #243e5e04}}.panel{{padding:20px 22px;margin-bottom:24px}}.panel p{{font-size:13px;margin-top:8px}}.review-links{{display:flex;flex-wrap:wrap;gap:8px;margin-top:13px}}.review-links a{{display:flex;align-items:center;gap:10px;padding:6px 10px;border:1px solid var(--line);border-radius:7px;text-decoration:none;font-size:13px;overflow-wrap:anywhere}}.review-links a:hover{{background:#f4f7fd}}.section-heading{{display:flex;align-items:center;justify-content:space-between;margin:30px 0 14px}}.section-heading a{{font-size:12px;text-decoration:none}}.case{{margin-bottom:18px;padding:22px;scroll-margin-top:16px}}.case-header{{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:18px}}.case-header .eyebrow{{color:var(--muted);font-size:10px}}.case-header h3{{margin-top:2px}}.case-tags{{display:flex;flex-wrap:wrap;justify-content:flex-end;align-items:center;gap:8px}}.badge{{display:inline-block;border-radius:5px;font-size:11px;line-height:1.3;font-weight:700;padding:5px 9px}}.badge.pass{{background:#e8f5ee;color:#227248}}.badge.fail{{background:#fcebed;color:#ae3447}}.badge.review{{background:#fff2d8;color:#94600b}}.badge.error{{background:#eee9f8;color:#7051a0}}.review-tag{{font-size:11px;color:#8a600e;font-weight:650}}.expected{{font-size:11px;color:var(--muted)}}.answer-grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.35fr);gap:20px;background:#f7f9fc;border:1px solid #e9edf3;border-radius:8px;padding:17px}}.answer-grid section+section{{border-left:1px solid var(--line);padding-left:20px}}.preserve{{white-space:pre-wrap;overflow-wrap:anywhere}}.requirements{{margin-top:17px}}ul{{margin:7px 0 0;padding-left:20px}}li+li{{margin-top:4px}}.requirements li{{font-size:13px}}.judgment{{padding:19px 0}}.scores{{display:flex;flex-wrap:wrap;gap:12px 30px;padding:12px 15px;margin:14px 0 0;border:1px solid var(--line);border-radius:7px}}.scores div{{display:flex;gap:8px;align-items:center}}.scores dt{{font-size:12px;color:var(--muted)}}.scores dd{{margin:0;font-size:13px;font-weight:700}}.evidence{{border-top:1px solid var(--line);padding-top:17px}}.finding{{border:1px solid var(--line);border-radius:7px;padding:14px 15px;margin-top:10px;font-size:13px}}.finding-label{{font-size:11px;font-weight:700;color:var(--accent);display:flex;justify-content:space-between;gap:15px;margin-bottom:9px;overflow-wrap:anywhere}}.finding-label span{{font-weight:500;color:var(--muted)}}blockquote{{margin:11px 0;padding:9px 13px;background:#f3f6fc;border-left:3px solid #91afd9;color:#3f5068}}.muted{{color:var(--muted);font-size:13px}}details{{font-size:13px}}summary{{cursor:pointer;font-weight:600}}.references{{border-top:1px solid var(--line);margin-top:17px;padding-top:13px}}.reference{{background:#f7f9fc;border-radius:6px;padding:14px;margin-top:10px}}.reference h4{{color:var(--ink);text-transform:none;letter-spacing:0}}.review-callout,.error-callout{{border:1px solid #eed8a7;background:#fff9ed;border-radius:7px;padding:12px 15px;margin:0 0 16px;font-size:13px;color:#795719}}.error-callout{{background:#f5f0fc;border-color:#dfd1f3;color:#694697}}.error-callout p{{margin-top:6px}}.rubric pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,"Cascadia Code",Consolas,monospace;background:#f7f9fc;padding:16px;border-radius:7px}}footer{{color:var(--muted);font-size:11px;display:flex;justify-content:space-between;gap:20px;margin-top:27px}}footer a{{text-decoration:none}}
@media(max-width:850px){{.metrics{{grid-template-columns:repeat(3,minmax(0,1fr))}}.case-header{{flex-direction:column;gap:10px}}.case-tags{{justify-content:flex-start}}}}
@media(max-width:600px){{main{{padding:25px 16px}}h1{{font-size:28px}}.topline{{align-items:flex-start}}.mode{{max-width:140px;text-align:center}}.metric{{padding:12px}}.metric strong{{font-size:25px}}.case{{padding:16px}}.answer-grid{{grid-template-columns:1fr;padding:14px;gap:15px}}.answer-grid section+section{{padding:15px 0 0;border-left:0;border-top:1px solid var(--line)}}.finding-label{{flex-direction:column;gap:2px}}footer{{flex-direction:column;gap:5px}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}.panel,.case{{box-shadow:none}}.case-header,.finding,.review-callout,.scores{{break-inside:avoid}}.notice{{margin-top:18px}}.review-links a{{color:var(--ink)}}.section-heading a,footer a{{display:none}}details.reference{{display:block}}}}
</style></head><body><main id="top">
<header><div class="topline"><span class="eyebrow">Evaluation audit</span><span class="mode">{mode}</span></div><h1>Chatbot evaluation report</h1><div class="meta"><span>Model: <strong>{_html(run.get('model') or 'Unspecified')}</strong></span><span>Generated: <strong>{_html(run.get('generated_at') or 'Unavailable')}</strong></span><span>Schema: <strong>{_html(run.get('schema_version', 1))}</strong></span></div></header>
<aside class="notice"><strong>{_html(notice)}</strong><p>{_html(detail)}</p></aside>
{incomplete_html}
<section class="metrics" aria-label="Verdict counts">{metrics_html}</section>
<p class="agreement"><strong>Expected-label agreement: {_html(_agreement(run))}</strong> · {labelled} successfully judged labelled case{'s' if labelled != 1 else ''}. Agreement compares automated verdicts with supplied expected labels; it is not confidence. Scores use the supplied rubric.</p>
<section class="panel"><h2>Flagged review queue <span class="muted">({flagged})</span></h2>{queue}</section>
<div class="section-heading"><h2>Case findings</h2><a href="#rubric">View scoring rubric ↓</a></div>{cases_html}
<section class="panel rubric" id="rubric"><h2>Scoring rubric</h2><p class="muted">Use this rubric alongside the reference excerpts when reviewing a verdict. Rubric ratings do not express statistical confidence.</p><details><summary>Show rubric</summary><pre>{_html(rubric)}</pre></details></section>
<footer><span>Local evaluation report · Includes questions, answers, source excerpts and automated judgments.</span><a href="#top">Back to top ↑</a></footer>
</main></body></html>"""


def _render_markdown(run: dict, cases: list[dict]) -> str:
    counts = _counts(cases)
    flagged = sum(_review(case) for case in cases)
    labelled = _labelled(run, cases)
    notice, detail = _notice(run)
    lines = [
        "# Chatbot evaluation report", "",
        f"**{_md(notice)}**", "", _md(detail), "",
        f"Model: **{_md(run.get('model') or 'Unspecified')}**  ",
        f"Generated: {_md(run.get('generated_at') or 'Unavailable')}  ",
        f"Mode: {_md(run.get('mode'))}", "",
        "## Verdict summary", "",
        "| Total | Pass | Fail | Review | Error | Flagged for human review |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {len(cases)} | {counts['pass']} | {counts['fail']} | {counts['review']} | {counts['error']} | {flagged} |", "",
        f"Expected-label agreement: **{_md(_agreement(run))}** across {labelled} successfully judged labelled cases. This compares verdicts with supplied expected labels; it is not confidence.", "",
        "## Flagged review queue", "",
    ]
    incomplete = _incomplete(run, len(cases))
    if incomplete:
        lines[6:6] = ["**Incomplete evaluation run**", "", _md(incomplete), ""]
    flagged_cases = [case for case in cases if _review(case)]
    if flagged_cases:
        lines.extend(f"- **{_md(case.get('id') or 'Unnamed case')}** ({_verdict(case)}): {_md(' '.join(_reasons(case)))}" for case in flagged_cases)
    else:
        lines.append("No cases were flagged for additional review. Automated judgments still require a reviewer’s sign-off.")
    for index, case in enumerate(cases):
        judgment = _dict(case.get("judgment"))
        scores = _dict(judgment.get("scores"))
        lines.extend(["", f"## Case {index + 1}: {_md(case.get('id') or 'Unnamed case')}", "", f"**Automated verdict: {_verdict(case).upper()}**"])
        if case.get("expected_verdict") is not None:
            lines.extend(["", f"Expected label: {_md(case.get('expected_verdict'))}"])
        if _review(case):
            lines.extend(["", "**Human review needed**", ""])
            lines.extend(f"- {_md(reason)}" for reason in _reasons(case))
        if case.get("error"):
            lines.extend(["", "### Evaluation error", "", _md(case.get("error"))])
        lines.extend(["", "### Question", "", _md(case.get("question"))])
        if _requirements(case):
            lines.extend(["", "### Answer requirements", ""])
            lines.extend(f"- {_md(item)}" for item in _requirements(case))
        lines.extend(["", "### Chatbot answer", "", _md(case.get("answer") or "No answer recorded."), "", "### Judgment rationale", "", _md(judgment.get("rationale") or "No rationale recorded."), "", "| Rubric dimension | Rating |", "| --- | --- |"])
        for key, label in _SCORES:
            lines.append(f"| {label} | {_md(scores.get(key)) if scores.get(key) is not None else 'Unavailable'} |")
        lines.extend(["", "Scores are rubric ratings, not confidence probabilities.", "", "### Claim-level evidence", ""])
        findings = _objects(judgment.get("findings"))
        if not findings:
            lines.append("No claim-level findings were recorded. Review the rationale and reference excerpts below.")
        for finding in findings:
            if finding.get("evidence_valid") is False:
                evidence_status = "**Unverified evidence: quote or reference did not match.** This finding is untrusted. Check the supplied references before accepting its claim or explanation."
            elif finding.get("evidence_valid") is True:
                evidence_status = "Quote found in the supplied reference. This checks quote existence only; it does not validate that the quote supports the claim."
            else:
                evidence_status = "Evidence matching status unavailable. Review the source text before accepting this finding."
            lines.extend([
                f"**{_md(finding.get('category') or 'Finding')}**", "",
                evidence_status, "",
                f"Claim: {_md(finding.get('claim'))}", "",
                f"Reference: {_md(finding.get('reference_id') or 'Unspecified')}", "",
                f"Source quote: {_md(finding.get('quote') or 'No supporting quote recorded.')}", "",
                _md(finding.get("explanation")), "",
            ])
        lines.extend(["", "### Reference excerpts", ""])
        refs = _objects(case.get("references"))
        if not refs:
            lines.append("No reference excerpts were supplied.")
        for ref in refs:
            lines.extend([f"**{_md(ref.get('id') or 'Reference')}**", "", _md(ref.get("text")), ""])
    lines.extend(["", "## Scoring rubric", "", "Rubric ratings do not express statistical confidence.", ""])
    rubric = json.dumps(run.get("rubric", {}), ensure_ascii=False, indent=2)
    lines.extend(_md(line.lstrip()) + "  " for line in rubric.splitlines())
    lines.extend(["", "The accompanying results.json contains the complete run data. reviewer.csv provides a blank reviewer verdict and notes field for each case.", ""])
    return "\n".join(lines)


def _csv_safe(value: object) -> str:
    text = _text(value)
    return "'" + text if text.startswith(("=", "+", "-", "@", "\t", "\r", "\n")) else text


def write_reports(run: dict, out_dir: Path) -> list[Path]:
    """Write HTML, Markdown, full JSON and a blank reviewer worksheet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _objects(run.get("cases"))
    html_path = out_dir / "report.html"
    markdown_path = out_dir / "report.md"
    json_path = out_dir / "results.json"
    csv_path = out_dir / "reviewer.csv"
    html_path.write_text(_render_html(run, cases), encoding="utf-8")
    markdown_path.write_text(_render_markdown(run, cases), encoding="utf-8")
    json_path.write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "automated_verdict", "reviewer_verdict", "notes"])
        for case in cases:
            writer.writerow([_csv_safe(case.get("id")), _verdict(case), "", ""])
    return [html_path, markdown_path, json_path, csv_path]
