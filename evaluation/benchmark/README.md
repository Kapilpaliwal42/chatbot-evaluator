# Frozen local judge benchmark

This version contains 48 fictional, document-grounded cases. Two agents authored
the examples and a separate agent reviewed all labels without seeing the author
labels. One disagreement was adjudicated before any benchmark inference: a
faithful answer admitting an unspecified product variant satisfies the stated
requirement and therefore passes. Its inputs did not change.

Final labels: 17 pass, 20 fail, 11 review. These are **agent-reviewed synthetic
labels**, not human-approved truth or representative client data.

## Splits and bindings

- Calibration: 23 cases (8 pass, 10 fail, 5 review).
- Holdout: 25 cases (9 pass, 10 fail, 6 review).
- Underlying policy families are disjoint between the splits.
- Judge inputs contain no expected verdict or label rationale. Labels live in
  `v1/gold.jsonl` and are attached to results only after inference.
- `v1/manifest.json` binds the input data, labels, prompt, schema, rubric, and
  evaluator source with SHA-256 hashes. Changes invalidate the run.
- `v1/preparation.json` binds the original blinded review inputs. Freezing also
  checks that authored inputs still match what the independent reviewer saw.
- `v1/human_review.csv` has empty human-label and notes columns. A human has not
  approved these labels.

The prompt is version 2, frozen before benchmark results. No tuning or training
uses the holdout. The two prespecified judges are `qwen2.5-coder:7b` and
`qwen2.5:7b`; both run locally through Ollama, one at a time.

## Complete the comparison

From `D:\LocalAI\agent`:

```powershell
python complete_benchmark.py --check
python complete_benchmark.py --status
python complete_benchmark.py
```

The controller resumes completed work, waits for the general model to be
installed, evaluates both calibration splits, records the preferred model using
only calibration results, then evaluates both holdout splits and creates the
comparison. A file lock prevents duplicate controllers. The same prompt,
model digest, Ollama version, generation settings, and input hashes are checked
before cached work is reused. The controller does not install or train models.

The download command is `ollama pull qwen2.5:7b`. The user chose to retain this
same-size comparison despite a slow download. The official model is about
4.7 GB: [Ollama model page](https://ollama.com/library/qwen2.5:7b).

Progress is saved in `../output/benchmark/workflow-state.json` and
`../output/benchmark/status.html`. Stage reports are in
`../output/benchmark/{coder-calibration,general-calibration,general-holdout,coder-holdout}`.
Final comparison files are in `../output/benchmark-comparison`.

## Interpret results

The four-run comparison has completed on Ollama 0.40.4. Calibration selected
the coding model before holdout. Its holdout label agreement was 12/25, versus
9/25 for the general model. The coding model flagged 6/9 expected-pass cases;
the general model flagged 7/9. Both routed all 16 expected fail/review cases to
fail or review, with zero runtime errors. These small synthetic observations
do not establish readiness for unattended client work.

Final artifacts: [comparison dashboard](../output/benchmark-comparison/comparison.html)
and [training decision](../output/benchmark-comparison/decision.md).

The previous coding-model calibration on Ollama 0.40.3 was archived at
`../output/benchmark/history/coder-calibration-ollama-0.40.3/results.json`.
The final comparison reran it on 0.40.4 for matching runtime versions.

Exact agreement includes errors in the denominator. The comparison also reports
per-label performance, macro F1, false passes from fail/review cases, false alarms
on good answers, evidence integrity, and runtime. It distinguishes raw scoring
from the verdict after validation: missing or invalid citations can turn a
failure rating into a review escalation.

An exact quote check only establishes that a substring appears in the source.
It does not establish that the finding logically follows. Independent semantic
review of the initial coding-model findings is saved separately.

The initial coding-model calibration matched 14 of 23 labels, with no runtime
errors. It falsely failed 5 of 8 correct answers. Four of 25 quotations failed
the exact source check. No unsafe calibration case was accepted after the
validation rules, but this small synthetic result does not establish a low
real-world false-pass rate.

Fine-tuning is deferred. First obtain human-approved, representative cases and
validate the chosen judge on fresh data. These authored examples provide a
diagnostic baseline; they are not a sufficient training corpus or evidence for
unattended client audits.

Software checks:

```powershell
python -m unittest evaluation.test_core evaluation.test_transport evaluation.test_benchmark evaluation.test_workflow -v
```
