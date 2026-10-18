# First local evaluation results

Run date: 13 October 2026. Model: local `qwen2.5-coder:7b` through Ollama.
Temperature 0, seed 42, context 8192, output limit 1600 tokens. No fine-tuning.

The software pipeline passed **59 checks**, covering strict schema validation,
quote matching, missing evidence, held-out reference labels, stale demo ratings,
network and truncation failures, report escaping, and CSV formula protection.
Those checks test software behavior; they do not establish model judgment quality.

| Run | Cases | Pass / fail / review / error | Exact label agreement |
| --- | ---: | --- | --- |
| Initial live prompt | 6 development | 4 / 0 / 2 / 0 | 2 of 6 |
| Revised live prompt (version 2) | Same 6 development | 0 / 3 / 3 / 0 | 3 of 6 |
| Frozen version 2 on fresh cases | 4 newly authored | 1 / 1 / 2 / 0 | 3 of 4 |

The saved offline demo is separate. Its six judgments were authored together
with the labels, so its 6-of-6 agreement is illustrative and measures no model.

The revised prompt explicitly checks each requested detail, places evidence
assessment before scores, and asks for source conflicts to be reviewed. It
caught a missing refund timeline that the initial prompt missed. It also falsely
flagged a correct admission of an undocumented fact and invented an omission in
another correct answer. The validator detected a rewritten source quotation
and routed that case to review.

For the four fresh software-policy examples, the final prompt matched the
supported answer, a missing support-ticket field, and the source-conflict label.
It identified the incorrect export-retention number, but failed to supply a
groundedness citation, so the tool escalated the result to review rather than
accepting its failure rating. Even the matching source-conflict verdict had a
weak explanation that incorrectly demanded a resolution from conflicting
documents. Matching a verdict is not enough to trust the reasoning.

This is a working audit prototype with an inconsistent model judge. A quotation
existing in a source does not prove that it supports the finding. Human review
is necessary for every result, including automated passes and failures. A larger
independently labelled set is needed before making reliability claims or offering
unattended evaluation to clients.

## Inspect the saved outputs

- `output/qwen/`: initial live run.
- `output/qwen-v2/`: revised live development run.
- `output/holdout/`: final run on the four newly authored cases.
- `output/demo/`: clearly labelled offline illustrative judgments.

Each directory contains `report.html`, `report.md`, `results.json`, and
`reviewer.csv`. JSON preserves the original answer, references, raw model response,
completion metadata, scores, and evidence-check status. The reviewer worksheet
starts blank for independent human assessment.

The four new cases were not used to change the final prompt. The small authored
sets are functional checks rather than a standardized or representative benchmark.

## Subsequent larger comparison

The next benchmark used 48 new agent-authored, independently agent-reviewed
cases, with 23 calibration and 25 heldout cases. Both local 7B models were run
on Ollama 0.40.4 using the same frozen prompt and settings. All 59 software checks
passed, and all 96 case evaluations finished without runtime errors.

| Judge | Calibration label matches | Holdout label matches | Holdout false alarms on expected-pass cases |
| --- | ---: | ---: | ---: |
| qwen2.5-coder:7b | 14/23 | 12/25 | 6/9 |
| qwen2.5:7b | 10/23 | 9/25 | 7/9 |

The coding model was selected using calibration results before holdout. Both
models routed all 16 expected fail/review holdout cases to fail or review. That
does not establish reliability: they also flagged most expected-pass answers,
and independent agent review found faulty reasoning and citation problems.

Fine-tuning remains deferred until labels are human-approved. The repository
now includes a local annotation queue (`review_app.py`), a high-confidence
training-data exporter (`prepare_training.py`), and a Hugging Face/Colab
groundedness-specialist pipeline under `training/`. Public hallucination data
must not be treated as complete project verdict labels. Any changes need a new
untouched validation set; this frozen holdout is not training data.

See [the completed comparison](output/benchmark-comparison/comparison.html)
and [training decision](output/benchmark-comparison/decision.md) for exact counts,
evidence metrics, source bindings, and reasoning-audit examples.
