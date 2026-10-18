"""Run a local Qwen-assisted chatbot audit: python evaluate.py --help."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from evaluation.core import OllamaJudge, PROMPT_VERSION, RUBRIC, case_digest, error_judgment, load_cases, summarize, validate_judgment
from evaluation.report import write_reports


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit exported chatbot answers against supplied documents using local Ollama.")
    parser.add_argument("--cases", type=Path, default=ROOT / "evaluation/samples/cases.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation/output")
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=180, help="Per-case Ollama timeout in seconds")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=1600)
    parser.add_argument("--limit", type=int, help="Evaluate only the first N cases")
    parser.add_argument("--demo", action="store_true", help="Use illustrative saved sample judgments, without calling a model")
    parser.add_argument("--doctor", action="store_true", help="Check local Ollama and selected model, then exit")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        judge = OllamaJudge(args.base_url, args.model, args.timeout, args.num_ctx, args.num_predict)
        if args.doctor:
            status = judge.doctor()
            print(json.dumps(status, indent=2))
            return 0 if status["available"] else 1
        cases = load_cases(args.cases)
        if args.limit:
            cases = cases[:args.limit]
        fixtures = None
        if args.demo:
            # Stored hashes bind ratings to their original inputs, even when a
            # user edits the default sample file in place. Never compare a file to itself.
            bundle = json.loads((ROOT / "evaluation/samples/demo_judgments.json").read_text(encoding="utf-8"))
            hashes = bundle.get("case_sha256", {})
            if not isinstance(hashes, dict) or any(hashes.get(case["id"]) != case_digest(case) for case in cases):
                raise ValueError("--demo works only with unchanged supplied sample cases. Use Ollama for custom or modified cases.")
            fixtures = bundle["judgments"]
        args.out.mkdir(parents=True, exist_ok=True)
        run = {"schema_version": 1, "mode": "demo" if args.demo else "ollama",
               "model": "illustrative saved judgments" if args.demo else args.model,
               "generated_at": datetime.now(timezone.utc).isoformat(), "rubric": RUBRIC, "cases": []}
        run["configuration"] = {"base_url": args.base_url, "num_ctx": args.num_ctx,
                                "num_predict": args.num_predict, "temperature": 0, "seed": 42,
                                "prompt_version": PROMPT_VERSION}
        interrupted = False
        for index, case in enumerate(cases, 1):
            print(f"[{index}/{len(cases)}] Evaluating {case['id']}...", flush=True)
            try:
                case["judgment"] = validate_judgment(fixtures[case["id"]], case) if args.demo else judge.judge(case)
            except KeyboardInterrupt:
                case["judgment"] = error_judgment("Run interrupted during this case.")
                interrupted = True
            except (RuntimeError, ValueError, KeyError) as exc:
                case["error"] = str(exc)
                case["judgment"] = error_judgment(str(exc))
            run["cases"].append(case)
            run["summary"] = summarize(run["cases"])
            # Save a recoverable checkpoint after each completed case.
            checkpoint = args.out / "checkpoint.json"
            checkpoint.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  {case['judgment']['verdict'].upper()}", flush=True)
            if interrupted:
                break
        run["requested_total"] = len(cases)
        run["completed"] = not interrupted
        paths = write_reports(run, args.out)
        print(json.dumps(run["summary"], indent=2))
        for path in paths:
            print(path.resolve())
        return 130 if interrupted else 1 if run["summary"]["error"] else 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
