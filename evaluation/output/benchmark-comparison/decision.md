# Judge selection and training decision

**Fine-tuning deferred; need human-approved representative labels and fresh validation.**

The 48 cases are fictional, machine-authored examples with independent agent label review. They are not human-approved labels or representative client data. Calibration contains 23 cases; holdout contains 25. Underlying policy families are disjoint.

The calibration rule selected **qwen2\.5\-coder:7b** before either holdout run started: fewest errors, then fewest false passes, then highest macro F1; lexical model name breaks ties. On calibration it matched 14/23 labels, made 0/15 false passes on expected fail/review cases, and returned 0 errors.

On holdout the selected judge matched **12/25** labels (48.0%), with macro F1 0.482. It made 0/16 false passes, 6/9 false alarms on expected pass cases, and 0/25 errors. These are observed counts in this small synthetic sample, not estimates of reliability on client work.

| Split | Judge | Exact label agreement, all cases | Macro F1 | False passes / fail + review | False alarms / pass | Pass recall | Errors / all cases | Exact quote matches / findings |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibration | qwen2\.5\-coder:7b | 14/23 (60.9%) | 0.610 | 0/15 (0.0%) | 5/8 (62.5%) | 3/8 (37.5%) | 0/23 (0.0%) | 21/25 (84.0%) |
| calibration | qwen2\.5:7b | 10/23 (43.5%) | 0.360 | 0/15 (0.0%) | 8/8 (100.0%) | 0/8 (0.0%) | 0/23 (0.0%) | 27/34 (79.4%) |
| holdout | qwen2\.5\-coder:7b | 12/25 (48.0%) | 0.482 | 0/16 (0.0%) | 6/9 (66.7%) | 3/9 (33.3%) | 0/25 (0.0%) | 30/33 (90.9%) |
| holdout | qwen2\.5:7b | 9/25 (36.0%) | 0.369 | 0/16 (0.0%) | 7/9 (77.8%) | 2/9 (22.2%) | 0/25 (0.0%) | 38/48 (79.2%) |

Errors remain in the label-agreement denominator. False passes mean predicted pass on expected fail or review. False alarms mean predicted fail or review on expected pass; runtime errors are reported separately. Final verdicts incorporate citation validation. Useful review routing can still differ from an expected fail label. Pass recall measures acceptance of expected pass cases; zero false passes alone can result from accepting no correct answers and does not establish useful judging.

Exact quote matches check only that a substring exists in its cited reference. They do not establish that the finding is logically supported. Human review must check both the evidence and its interpretation.

qwen2\.5\-coder:7b on calibration: 15/15 expected fail/review cases were routed to fail or review; 8/23 cases lacked cited findings for one or more low-scored criteria.

qwen2\.5:7b on calibration: 15/15 expected fail/review cases were routed to fail or review; 8/23 cases lacked cited findings for one or more low-scored criteria.

qwen2\.5\-coder:7b on holdout: 16/16 expected fail/review cases were routed to fail or review; 8/25 cases lacked cited findings for one or more low-scored criteria.

qwen2\.5:7b on holdout: 16/16 expected fail/review cases were routed to fail or review; 11/25 cases lacked cited findings for one or more low-scored criteria.

A separate agent reasoning audit is included in comparison.html. It provides machine-reviewed examples, not human approval; review those examples and conclusions before relying on the findings.

Fine-tuning is premature because these labels lack human approval and the dataset is a small synthetic diagnostic set. No fine-tuning has been performed. First obtain human-approved representative examples, investigate repeated reasoning and protocol errors, and compare any changes on fresh untouched validation data. Keep this frozen holdout separate from training.

Repair citation/evidence scaffolding after this frozen run using a new dataset version. A judge can identify an answer’s real fault yet receive a final review verdict because an additional low score lacks a valid cited finding; a calibration disagreement alone does not establish that the judge missed the answer’s fault.

Both prespecified models are reported. The candidate remains the one selected from calibration. Any preference formed after inspecting holdout results needs confirmation on fresh data. These results do not establish readiness for unattended client audits.

The renderer verified four completed runs, case inputs and labels, frozen hashes, common prompt/evaluator/runtime settings, stable model digests across splits, disjoint policy families, and the calibration selection’s hashes and timing.
