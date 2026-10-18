"""Local human-review queue for chatbot evaluation cases.

Run:
    python review_app.py --cases evaluation/benchmark/v1/blind_review.jsonl

The server binds to localhost only and writes one JSON object per reviewed case
to the selected output JSONL file. Human labels are never written back into the
benchmark inputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


LABELS = ("pass", "fail", "review")
CONFIDENCE = ("low", "medium", "high")


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}, line {line_number}: invalid JSON") from exc
        if not isinstance(case, dict):
            raise ValueError(f"{path}, line {line_number}: case must be an object")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{path}, line {line_number}: id must be nonempty")
        if case_id in seen:
            raise ValueError(f"{path}, line {line_number}: duplicate id {case_id!r}")
        if not isinstance(case.get("question"), str) or not isinstance(case.get("answer"), str):
            raise ValueError(f"{path}, line {line_number}: question and answer are required")
        references = case.get("references")
        if not isinstance(references, list) or not references:
            raise ValueError(f"{path}, line {line_number}: references must be a nonempty list")
        for reference in references:
            if not isinstance(reference, dict) or not isinstance(reference.get("id"), str) or not isinstance(reference.get("text"), str):
                raise ValueError(f"{path}, line {line_number}: invalid reference")
        requirements = case.get("requirements", [])
        if not isinstance(requirements, list) or not all(isinstance(item, str) for item in requirements):
            raise ValueError(f"{path}, line {line_number}: requirements must be a string list")
        clean = {
            "id": case_id,
            "question": case["question"],
            "answer": case["answer"],
            "references": [{"id": ref["id"], "text": ref["text"]} for ref in references],
            "requirements": requirements,
        }
        for key in ("expected_verdict", "split", "family_id", "category"):
            if key in case:
                clean[key] = case[key]
        cases.append(clean)
        seen.add(case_id)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def load_reviews(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    reviews = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            review = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}, line {line_number}: invalid JSON") from exc
        if not isinstance(review, dict) or not isinstance(review.get("id"), str):
            raise ValueError(f"{path}, line {line_number}: review id is required")
        reviews[review["id"]] = review
    return reviews


def save_review(path: Path, review: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_reviews(path)
    existing[review["id"]] = review
    ordered = [existing[key] for key in sorted(existing)]
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered),
        encoding="utf-8",
    )


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _case_html(case: dict, review: dict | None) -> str:
    review = review or {}
    selected = review.get("human_verdict", "")
    confidence = review.get("confidence", "")
    options = "".join(
        f'<option value="{label}" {"selected" if selected == label else ""}>{label.title()}</option>'
        for label in LABELS
    )
    confidence_options = "".join(
        f'<option value="{level}" {"selected" if confidence == level else ""}>{level.title()}</option>'
        for level in CONFIDENCE
    )
    requirements = "".join(f"<li>{_escape(item)}</li>" for item in case.get("requirements", []))
    references = "".join(
        f'<article class="reference"><h3>{_escape(ref["id"])}</h3><p>{_escape(ref["text"])}</p></article>'
        for ref in case["references"]
    )
    expected = (
        f'<span class="expected">Bootstrap label: {_escape(case["expected_verdict"])}</span>'
        if case.get("expected_verdict")
        else ""
    )
    return f"""
<main>
  <header>
    <p class="eyebrow">Human annotation queue</p>
    <h1>{_escape(case["id"])}</h1>
    <p class="meta">{expected} {_escape(case.get("category", ""))}</p>
  </header>
  <section class="grid">
    <article class="panel"><h2>Question</h2><p class="preserve">{_escape(case["question"])}</p>
      {"<h3>Requirements</h3><ul>" + requirements + "</ul>" if requirements else ""}
    </article>
    <article class="panel"><h2>Chatbot answer</h2><p class="preserve">{_escape(case["answer"])}</p></article>
  </section>
  <section class="panel"><h2>References</h2>{references}</section>
  <form class="panel" method="post" action="/review">
    <input type="hidden" name="id" value="{_escape(case["id"])}">
    <label>Human verdict
      <select name="human_verdict" required><option value="">Choose...</option>{options}</select>
    </label>
    <label>Confidence
      <select name="confidence" required><option value="">Choose...</option>{confidence_options}</select>
    </label>
    <label>Rationale
      <textarea name="rationale" required placeholder="Explain the decisive evidence and any uncertainty.">{_escape(review.get("rationale", ""))}</textarea>
    </label>
    <button type="submit">Save label and continue</button>
  </form>
</main>
"""


def render_page(cases: list[dict], reviews: dict[str, dict], index: int, message: str = "") -> str:
    index = max(0, min(index, len(cases) - 1))
    case = cases[index]
    reviewed = sum(case["id"] in reviews for case in cases)
    notice = f'<div class="notice">{_escape(message)}</div>' if message else ""
    previous = f'<a href="/?index={index - 1}">Previous</a>' if index else "<span>Previous</span>"
    next_link = f'<a href="/?index={index + 1}">Next</a>' if index < len(cases) - 1 else "<span>Next</span>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Human annotation queue</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;color:#172033;background:#f4f6f9;margin:0}}
main{{max-width:1050px;margin:auto;padding:28px 18px 48px}}header{{margin-bottom:18px}}
h1{{margin:3px 0;font-size:30px}}h2{{font-size:17px;margin:0 0 10px}}h3{{font-size:14px;margin:17px 0 6px}}
.eyebrow{{text-transform:uppercase;letter-spacing:1px;color:#285bb5;font-size:11px;font-weight:700}}
.meta{{color:#657189;font-size:13px}}.expected{{margin-right:14px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.panel{{background:white;border:1px solid #dce3ec;border-radius:10px;padding:18px;margin-bottom:14px}}
.preserve{{white-space:pre-wrap;overflow-wrap:anywhere}}ul{{padding-left:21px}}label{{display:block;font-weight:650;margin:14px 0}}
select,textarea{{display:block;width:100%;margin-top:6px;padding:9px;border:1px solid #bbc6d5;border-radius:6px;font:inherit;box-sizing:border-box}}
textarea{{min-height:125px;resize:vertical}}button{{background:#245bbd;color:white;border:0;border-radius:6px;padding:10px 15px;font-weight:700;cursor:pointer}}
.reference{{background:#f6f8fb;border-radius:6px;padding:12px;margin-top:9px}}.reference h3{{margin:0 0 4px}}
.notice{{background:#e9f5ed;border:1px solid #a8d5b5;padding:10px;border-radius:6px;margin-bottom:14px}}
nav{{display:flex;justify-content:space-between;margin-top:15px}}nav a,nav span{{padding:8px 12px;border:1px solid #ccd5e2;border-radius:6px;text-decoration:none;color:#24509c}}
@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>{notice}{_case_html(case, reviews.get(case["id"]))}
<nav><span>{previous}</span><strong>{index + 1} / {len(cases)} reviewed: {reviewed}</strong><span>{next_link}</span></nav>
</body></html>"""


class ReviewHandler(BaseHTTPRequestHandler):
    cases: list[dict] = []
    reviews: dict[str, dict] = {}
    output: Path

    def _respond(self, body: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        try:
            index = int(query.get("index", ["0"])[0])
        except ValueError:
            index = 0
        self._respond(render_page(self.cases, self.reviews, index))

    def do_POST(self) -> None:
        if self.path != "/review":
            self._respond("Not found", 404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        values = parse_qs(self.rfile.read(length).decode("utf-8"))
        case_id = values.get("id", [""])[0]
        verdict = values.get("human_verdict", [""])[0]
        confidence = values.get("confidence", [""])[0]
        rationale = values.get("rationale", [""])[0].strip()
        case_index = next((i for i, case in enumerate(self.cases) if case["id"] == case_id), 0)
        if verdict not in LABELS or confidence not in CONFIDENCE or not rationale:
            self._respond(render_page(self.cases, self.reviews, case_index, "Choose a verdict, confidence, and rationale."))
            return
        review = {
            "id": case_id,
            "human_verdict": verdict,
            "confidence": confidence,
            "rationale": rationale,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_review(self.output, review)
        self.reviews[case_id] = review
        next_index = min(case_index + 1, len(self.cases) - 1)
        self._respond(render_page(self.cases, self.reviews, next_index, f"Saved {case_id}."))

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Review chatbot evaluation cases locally.")
    parser.add_argument("--cases", type=Path, required=True, help="Input JSONL case file")
    parser.add_argument("--out", type=Path, default=Path("evaluation/output/human_reviews.jsonl"))
    parser.add_argument("--host", default="127.0.0.1", help="Use localhost unless explicitly protected")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--host must be localhost for the local review tool")
    try:
        cases = load_cases(args.cases)
        reviews = load_reviews(args.out)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    ReviewHandler.cases = cases
    ReviewHandler.reviews = reviews
    ReviewHandler.output = args.out
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Review {len(cases)} cases at http://{args.host}:{args.port}/")
    print(f"Labels are saved to {args.out.resolve()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
