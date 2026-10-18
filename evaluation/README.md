# Local chatbot evaluation

A first portfolio project for offering chatbot quality audits. It uses the same
Ollama server and `qwen2.5-coder:7b` model as the coding agent in this folder.
The existing coding agent is separate; this evaluator never executes model
instructions or client code.

Give it a question, the chatbot's answer, reference documents, and optional
requirements. It evaluates groundedness, instruction following, and completeness,
checks that cited quotations exist in the references, and exports an HTML report,
a Markdown report, JSON results, and a CSV worksheet for a human reviewer.

Python 3.10+ is sufficient. There are no new Python dependencies.

## Run on Windows

From PowerShell:

```powershell
Set-Location 'D:\LocalAI\agent'
python evaluate.py --doctor
python evaluate.py --out evaluation/output/qwen
```

The doctor command checks the server and model. If Ollama is stopped, start it
with `ollama serve`. If the model is missing, `ollama pull qwen2.5-coder:7b`
downloads it. The model is already available on the machine used to build this
project.

Open `evaluation/output/qwen/report.html` to inspect the results. Use the CSV
worksheet to record independent reviewer verdicts and notes. Filling in the CSV
does not yet update the report automatically.

An offline illustration is also included:

```powershell
python evaluate.py --demo --out evaluation/output/demo
```

The demo uses manually authored saved judgments for the unchanged sample cases.
It makes no model requests. Its agreement with the sample labels is illustrative,
not evidence of Qwen's evaluation ability. The real run reports its actual model
and timestamp, including disagreements with the sample expectations.

## Your own audit

Save one JSON object per line in a UTF-8 `.jsonl` file. Example:

```json
{"id":"returns-01","question":"How long do I have to return an item?","answer":"You have 60 days.","references":[{"id":"returns-policy","text":"Unused items can be returned within 14 days of delivery."}],"requirements":["State the return window accurately."]}
```

Then run:

```powershell
python evaluate.py --cases client_cases.jsonl --out evaluation/output/client-audit
```

## Human annotation workflow

Automated verdicts are not training labels until a human has reviewed them.
Use the local annotation queue for the frozen bootstrap cases:

```powershell
python review_app.py `
  --cases evaluation/benchmark/v1/blind_review.jsonl `
  --out evaluation/benchmark/v1/human_reviews.jsonl
```

Open the printed localhost URL. Each case requires a human verdict, confidence,
and rationale. The app stores labels separately from the benchmark inputs and
can also review any future JSONL file in the same case format. Do not mark the
bootstrap labels as human-approved until the queue has been completed and the
result has been checked for missing, duplicate, or low-confidence labels.

After review, export only high-confidence labels for training:

```powershell
python prepare_training.py `
  --cases evaluation/benchmark/v1/blind_review.jsonl `
  --reviews evaluation/benchmark/v1/human_reviews.jsonl `
  --out evaluation/output/training-data
```

The exporter requires at least 30 completed high-confidence labels and creates
separate `train.jsonl`, `validation.jsonl`, and `test.jsonl` files plus a
manifest. The test split must remain untouched while prompts, adapters, or
training settings are changed. The current synthetic benchmark is a bootstrap
queue, not evidence that a trained model is production-ready.

This version audits exported answers; it does not connect to or crawl a client's
chatbot. The references are inline plain text, so copy relevant passages from the
client's approved documentation. Use short, focused excerpts. The evaluator
rejects oversized inputs instead of silently shortening them.

Optional `expected_verdict` values (`pass`, `fail`, `review`) let you compare
judgments with independent human labels. Labels are excluded from judge prompts.
The report gives agreement only over successfully judged labelled cases;
generation errors are counted separately. This small sample is a functional
check, not a validated benchmark or a forecast of performance on client data.

## Ratings and review

Each criterion uses 0-4: 0 fundamentally wrong, 1 major error, 2 partial or
uncertain, 3 adequate with a minor weakness, 4 fully satisfies the requirement.
The tool derives verdicts from these scores rather than trusting a model's
requested verdict:

- Any 0 or 1: **fail**, provided the issue has checked citations.
- Otherwise any 2 or a request for human review: **review**.
- Otherwise: **pass**.
- Invalid citations, missing evidence for a low score, or contradictory findings:
  **review**, overriding the automated score.
- Invalid JSON, invalid schema, unavailable server, timeout, or truncated output:
  **error**, requiring review; never pass.

Quote checks establish only that a quotation exists. They do not prove that the
quote logically supports the finding. A model can still miss errors or misread
policy. Review the evidence before presenting findings to a client. Prompt
instructions reduce exposure to malicious text in documents, but this is not a
tested prompt-injection defense.

All outputs are written locally. The default Ollama endpoint is local. If you
set `--base-url` to another server, the question, answer, and documents are sent
to that server. Outputs include client text and raw judge responses; use a
sanitized copy for your public portfolio.

## Useful controls

```powershell
python evaluate.py --limit 1 --out evaluation/output/smoke
python evaluate.py --model qwen2.5-coder:7b --timeout 240
python evaluate.py --num-ctx 16384 --num-predict 2400
python -m unittest evaluation.test_core evaluation.test_transport -v
```

`--timeout` applies to each case. Runs continue after an individual case error
and return exit code 1 if any error occurred. Failed answers are audit results,
so they do not cause a nonzero exit. `checkpoint.json` is saved after each case.
An interrupted run exports its completed portion with `completed: false` and
`requested_total` in JSON. Reusing an output directory replaces that report;
choose a separate directory for each audit.

The transport uses Ollama's JSON schema format and checks completion metadata.
It is separate from the existing `OllamaClient` so the evaluation tool needs no
additional SDK and can inspect truncated generations. API references:
[Generate a response](https://docs.ollama.com/api/generate),
[Structured outputs](https://docs.ollama.com/capabilities/structured-outputs).

## A concrete first client deliverable

Offer a small audit using the client's documentation and a jointly agreed set of
questions. Deliver a reviewed list of incorrect or unsupported answers, the
supporting policy excerpts, and suggested corrections. This project supplies the
reporting workflow; the reviewer supplies the final judgment.

Before fine-tuning, collect a larger independently labelled set, keep a separate
unseen test set, and compare model judgments with those labels. Fine-tuning is
not part of this first version.

## What the first live tests showed

All 35 software checks passed. The local Qwen run produced reports for six
development cases and four newly authored cases. On the four fresh cases the
final prompt matched three expected verdicts; the fourth was escalated to review
because the model omitted a required citation. This tiny check is not an
estimate of general accuracy. Some matching verdicts still had weak reasoning.

The development examples exposed false positive findings, an overlooked
omission, and rewritten quotations. Prompt changes improved some examples but
did not make the judge consistently correct. Review every automated judgment,
including passes, before using it in a paid audit.

See [VALIDATION.md](VALIDATION.md) for the observed results and file locations.
The final prompt was not changed after looking at the four fresh-case results.

## Larger comparison completed

A subsequent frozen benchmark used 48 new fictional cases with independent
agent label review, split into 23 calibration and 25 holdout cases. Both local
7B models completed all cases on Ollama 0.40.4 with the same fixed prompt and
generation settings; all 59 software checks passed.

Calibration selected `qwen2.5-coder:7b` before holdout. On holdout it matched
12/25 expected labels, compared with 9/25 for `qwen2.5:7b`. It falsely flagged
6/9 expected-pass answers; the general model flagged 7/9. Both routed all 16
expected fail/review cases to fail or review. There were no runtime errors.
These are observed counts on agent-reviewed synthetic cases, not estimates of
reliability on client data.

The comparison is complete. Fine-tuning is deferred: clarify the scoring and
evidence protocol, obtain human-approved representative examples, and test any
changes on a new untouched set. Neither judge should approve client answers
without review. The frozen holdout remains separate from training.

See [the comparison dashboard](output/benchmark-comparison/comparison.html),
[training decision](output/benchmark-comparison/decision.md), and
[benchmark protocol](benchmark/README.md).
