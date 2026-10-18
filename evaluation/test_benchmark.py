"""Software checks for frozen benchmarks, independent labels, and honest metrics.

All fixtures here are artificial test data, not benchmark evaluation results or
human-verified ground truth. No live model or network connection is used.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation import benchmarking as benchmark
from evaluation.core import CRITERIA, build_prompt, error_judgment


def measured_case(expected, predicted, findings=None, reasons=None):
    if predicted == "error":
        judgment = error_judgment("Artificial transport error")
    else:
        score = {"pass": 4, "fail": 1, "review": 2}[predicted]
        judgment = {
            "verdict": predicted,
            "scores": {criterion: score for criterion in CRITERIA},
            "rationale": "Artificial metric fixture.",
            "findings": findings or [],
            "needs_human_review": predicted == "review",
            "human_review_reasons": reasons or [],
        }
    return {"expected_verdict": expected, "judgment": judgment}


def authored_fixtures():
    """Six distinct policy families; two examples for each possible gold label."""
    rows = []
    for index, (label, limit) in enumerate((
        ("pass", 2), ("pass", 3), ("fail", 4), ("fail", 5),
        ("review", 6), ("review", 8),
    )):
        case_id = f"fixture-{index}"
        refs = [{"id": case_id, "text": f"Room {index} admits up to {limit} guests."}]
        if label == "review":
            refs.append({"id": f"alternate-{index}",
                         "text": f"Room {index} admits up to {limit + 1} guests. Neither source has priority."})
        rows.append({
            "id": case_id,
            "question": f"What is the guest limit for room {index}?",
            "answer": f"The limit is {9 if label == 'fail' else limit} guests.",
            "references": refs,
            "requirements": ["State the documented guest limit."],
            "expected_verdict": label,
            "gold_rationale": f"Author rationale for artificial {label} fixture {index}.",
            "category": "artificial_guest_limit",
            "family_id": f"distinct-room-policy-{index}",
        })
    return rows


# A separate fixed review oracle for these software fixtures. Review creation
# reads the blinded inputs, never the authored-label file or author rationales.
FIXTURE_REVIEW_LABELS = {
    "fixture-0": "pass", "fixture-1": "pass", "fixture-2": "fail",
    "fixture-3": "fail", "fixture-4": "review", "fixture-5": "review",
}


def prepare_review(directory, rows=None, seed=41):
    root = Path(directory)
    rows = authored_fixtures() if rows is None else rows
    sources = [root / "source-one.jsonl", root / "source-two.jsonl"]
    benchmark.write_jsonl(sources[0], rows[::2])
    benchmark.write_jsonl(sources[1], rows[1::2])
    dataset = root / "frozen"
    benchmark.make_blind_review(sources, dataset, seed=seed)
    reviews = [{
        "id": row["id"],
        "independent_verdict": FIXTURE_REVIEW_LABELS[row["references"][0]["id"]],
        "rationale": "Independent software-fixture check of the supplied room limit.",
    } for row in benchmark.read_jsonl(dataset / "blind_review.jsonl")]
    review_path = root / "independent-review.jsonl"
    benchmark.write_jsonl(review_path, reviews)
    return dataset, review_path


def prepare_frozen(directory):
    dataset, review_path = prepare_review(directory)
    benchmark.freeze_dataset(dataset, review_path, seed=41)
    return dataset


class BenchmarkMetricTests(unittest.TestCase):
    def test_errors_remain_in_agreement_support_and_confusion_denominators(self):
        cases = [measured_case("pass", "pass"), measured_case("pass", "error"),
                 measured_case("fail", "fail"), measured_case("fail", "pass"),
                 measured_case("review", "review"), measured_case("review", "error")]
        metrics = benchmark.benchmark_metrics(cases)
        self.assertEqual(metrics["total"], 6)
        self.assertEqual(metrics["matched"], 3)
        self.assertEqual(metrics["exact_agreement_all_cases"], 0.5)
        self.assertEqual(metrics["error_count"], 2)
        self.assertEqual(metrics["successful_judgments"], 4)
        self.assertAlmostEqual(metrics["coverage"], 4 / 6)
        self.assertEqual(metrics["confusion_matrix"], {
            "pass": {"pass": 1, "fail": 0, "review": 0, "error": 1},
            "fail": {"pass": 1, "fail": 1, "review": 0, "error": 0},
            "review": {"pass": 0, "fail": 0, "review": 1, "error": 1},
        })
        for label in benchmark.LABELS:
            self.assertEqual(metrics["class_metrics"][label]["support"], 2)
            self.assertEqual(metrics["class_metrics"][label]["recall"], 0.5)

    def test_fail_to_pass_and_review_to_pass_are_counted_separately(self):
        metrics = benchmark.benchmark_metrics([
            measured_case("fail", "pass"), measured_case("review", "pass"),
            measured_case("review", "pass"), measured_case("fail", "error"),
            measured_case("pass", "review"),
        ])
        self.assertEqual(metrics["fail_to_pass"], 1)
        self.assertEqual(metrics["review_to_pass"], 2)
        self.assertEqual(metrics["false_passes"], 3)
        self.assertEqual(metrics["unsafe_cases"], 4)
        self.assertEqual(metrics["false_pass_rate"], 0.75)
        self.assertEqual(metrics["false_alarms_on_pass"], 1)
        self.assertEqual(metrics["unsafe_routing_rate"], 0)

    def test_quote_integrity_is_unknown_when_there_are_no_findings(self):
        metrics = benchmark.benchmark_metrics([measured_case("pass", "pass")])
        self.assertEqual(metrics["quotes"], 0)
        self.assertEqual(metrics["invalid_quotes"], 0)
        self.assertIsNone(metrics["quote_integrity_rate"])
        self.assertEqual(metrics["cases_with_findings"], 0)

    def test_quote_integrity_counts_unverified_and_missing_validity(self):
        metrics = benchmark.benchmark_metrics([
            measured_case("fail", "review", findings=[
                {"evidence_valid": True}, {"evidence_valid": False}, {}],
                reasons=["Missing cited findings for: completeness."]),
            measured_case("pass", "pass"),
        ])
        self.assertEqual(metrics["quotes"], 3)
        self.assertEqual(metrics["invalid_quotes"], 2)
        self.assertAlmostEqual(metrics["quote_integrity_rate"], 1 / 3)
        self.assertEqual(metrics["cases_with_findings"], 1)
        self.assertEqual(metrics["cases_with_invalid_quotes"], 1)
        self.assertEqual(metrics["cases_missing_criterion_evidence"], 1)

    def test_raw_scores_and_evidence_overridden_verdict_are_distinguished(self):
        case = measured_case("fail", "review")
        case["judgment"]["scores"]["groundedness"] = 1
        case["raw_needs_human_review"] = False
        metrics = benchmark.benchmark_metrics([case])
        self.assertEqual(metrics["confusion_matrix"]["fail"]["review"], 1)
        self.assertEqual(metrics["raw_score_confusion_matrix"]["fail"]["fail"], 1)


class FrozenDatasetTests(unittest.TestCase):
    def test_blinded_inputs_have_randomized_ids_and_no_author_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, _ = prepare_review(directory)
            blind = benchmark.read_jsonl(dataset / "blind_review.jsonl")
            self.assertEqual(len(blind), 6)
            self.assertEqual({row["id"] for row in blind},
                             {f"bench-{index:03d}" for index in range(1, 7)})
            self.assertNotEqual([row["references"][0]["id"] for row in blind],
                                [row["id"] for row in authored_fixtures()])
            for row in blind:
                self.assertEqual(set(row), set(benchmark.INPUT_KEYS))
                prompt = build_prompt(row)
                for secret in ("expected_verdict", "gold_rationale", "family_id", "original_id"):
                    self.assertNotIn(secret, prompt)

    def test_split_is_stratified_family_disjoint_and_keeps_gold_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = prepare_frozen(directory)
            manifest, calibration, calibration_gold = benchmark.load_frozen(dataset, "calibration")
            _, holdout, holdout_gold = benchmark.load_frozen(dataset, "holdout")
            for split in ("calibration", "holdout"):
                self.assertEqual(manifest["splits"][split]["labels"],
                                 {"pass": 1, "fail": 1, "review": 1})
                self.assertEqual(manifest["splits"][split]["count"], 3)
            self.assertFalse(set(manifest["splits"]["calibration"]["families"]) &
                             set(manifest["splits"]["holdout"]["families"]))
            self.assertFalse(set(calibration_gold) & set(holdout_gold))
            self.assertEqual(len(calibration_gold) + len(holdout_gold), 6)
            self.assertFalse(manifest["human_approved"])
            for row in calibration + holdout:
                self.assertEqual(set(row), set(benchmark.INPUT_KEYS))
                self.assertNotIn("expected_verdict", row)
            gold = benchmark.read_jsonl(dataset / "gold.jsonl")
            self.assertEqual(benchmark.fingerprint(gold), manifest["gold_sha256"])
            for row in calibration + holdout:
                label = (calibration_gold | holdout_gold)[row["id"]]
                self.assertEqual(label["case_sha256"], benchmark.fingerprint(row))

    def test_same_seed_reproduces_split_inputs_and_families(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            left, right = prepare_frozen(first), prepare_frozen(second)
            left_manifest, _, _ = benchmark.load_frozen(left, "holdout")
            right_manifest, _, _ = benchmark.load_frozen(right, "holdout")
            self.assertEqual(left_manifest["splits"], right_manifest["splits"])
            self.assertEqual(left_manifest["gold_sha256"], right_manifest["gold_sha256"])

    def test_freeze_rejects_disagreement_before_writing_manifest_or_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, review_path = prepare_review(directory)
            reviews = benchmark.read_jsonl(review_path)
            reviews[0]["independent_verdict"] = (
                "fail" if reviews[0]["independent_verdict"] != "fail" else "pass")
            benchmark.write_jsonl(review_path, reviews)
            with self.assertRaisesRegex(ValueError, "disagreements"):
                benchmark.freeze_dataset(dataset, review_path)
            self.assertFalse((dataset / "manifest.json").exists())
            self.assertFalse((dataset / "gold.jsonl").exists())
            self.assertTrue((dataset / "label_disagreements.json").exists())

    def test_freeze_rejects_authored_input_edits_after_blind_review(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, review_path = prepare_review(directory)
            authored = benchmark.read_jsonl(dataset / "authored_labels.jsonl")
            authored[0]["answer"] = "This input was changed after independent review."
            benchmark.write_jsonl(dataset / "authored_labels.jsonl", authored)
            with self.assertRaisesRegex(ValueError, "Authored inputs changed after independent review"):
                benchmark.freeze_dataset(dataset, review_path)
            self.assertFalse((dataset / "manifest.json").exists())
            self.assertFalse((dataset / "gold.jsonl").exists())

    def test_freeze_rejects_blind_input_edits_against_preparation_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset, review_path = prepare_review(directory)
            preparation = json.loads((dataset / "preparation.json").read_text(encoding="utf-8"))
            blind = benchmark.read_jsonl(dataset / "blind_review.jsonl")
            self.assertEqual(preparation["blind_inputs_sha256"], benchmark.fingerprint(blind))
            blind[0]["references"][0]["text"] = "This reference was changed after preparation."
            benchmark.write_jsonl(dataset / "blind_review.jsonl", blind)
            with self.assertRaisesRegex(ValueError, "Blind review inputs changed after preparation"):
                benchmark.freeze_dataset(dataset, review_path)
            self.assertFalse((dataset / "manifest.json").exists())
            self.assertFalse((dataset / "gold.jsonl").exists())

    def test_frozen_input_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = prepare_frozen(directory)
            inputs = benchmark.read_jsonl(dataset / "holdout.jsonl")
            inputs[0]["answer"] = "Changed after the holdout was frozen."
            benchmark.write_jsonl(dataset / "holdout.jsonl", inputs)
            with self.assertRaisesRegex(ValueError, "Frozen inputs changed"):
                benchmark.load_frozen(dataset, "holdout")

    def test_gold_label_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = prepare_frozen(directory)
            gold = benchmark.read_jsonl(dataset / "gold.jsonl")
            gold[0]["expected_verdict"] = "pass" if gold[0]["expected_verdict"] != "pass" else "fail"
            benchmark.write_jsonl(dataset / "gold.jsonl", gold)
            with self.assertRaisesRegex(ValueError, "Frozen gold labels changed"):
                benchmark.load_frozen(dataset, "holdout")

    def test_prompt_edit_is_rejected_before_loading_frozen_data(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = prepare_frozen(directory)
            with patch.object(benchmark, "SYSTEM_PROMPT", benchmark.SYSTEM_PROMPT + " Changed."):
                with self.assertRaisesRegex(ValueError, "Judge prompt changed"):
                    benchmark.load_frozen(dataset, "holdout")

    def test_label_in_frozen_model_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = prepare_frozen(directory)
            inputs = benchmark.read_jsonl(dataset / "holdout.jsonl")
            inputs[0]["expected_verdict"] = "pass"
            benchmark.write_jsonl(dataset / "holdout.jsonl", inputs)
            with self.assertRaisesRegex(ValueError, "must not include labels"):
                benchmark.load_frozen(dataset, "holdout")

    def test_author_metadata_is_required_and_duplicate_families_rejected(self):
        for field in ("family_id", "category", "gold_rationale"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                rows = authored_fixtures()
                del rows[0][field]
                with self.assertRaisesRegex(ValueError, f"missing {field}"):
                    prepare_review(directory, rows)
        with tempfile.TemporaryDirectory() as directory:
            rows = authored_fixtures()
            rows[1]["family_id"] = rows[0]["family_id"]
            with self.assertRaisesRegex(ValueError, "policy families"):
                prepare_review(directory, rows)


class BenchmarkCacheTests(unittest.TestCase):
    def test_cache_is_reused_only_for_same_model_digest_prompt_and_input(self):
        calls = []
        digest = ["digest-one"]

        class FakeJudge:
            def __init__(self, **kwargs):
                self.model = kwargs["model"]

            def _request(self, route):
                if route == "/api/version":
                    return {"version": "test-ollama"}
                if route != "/api/tags":
                    raise AssertionError("Unexpected fake transport request")
                return {"models": [{"name": self.model, "digest": digest[0], "details": {}}]}

            def judge(self, case):
                calls.append(copy.deepcopy(case))
                if set(case) != set(benchmark.INPUT_KEYS):
                    raise AssertionError("Labels or metadata reached model inference")
                raw = {
                    "scores": {criterion: 4 for criterion in CRITERIA},
                    "rationale": "Artificial model response for cache test.",
                    "findings": [], "needs_human_review": False,
                    "human_review_reasons": [],
                }
                case["raw_response"] = json.dumps(raw)
                return {**raw, "verdict": "pass"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = prepare_frozen(root / "initial")
            output = root / "output"
            with patch.object(benchmark, "OllamaJudge", FakeJudge), \
                    patch.object(benchmark, "write_reports"), patch("builtins.print"):
                def run(source):
                    return benchmark.run_benchmark(source, "holdout", "fake:tiny", output)

                self.assertEqual(run(dataset), 0)
                self.assertEqual(len(calls), 3)
                self.assertEqual(run(dataset), 0)
                self.assertEqual(len(calls), 3, "Unchanged inputs should reuse cached judgments")
                digest[0] = "digest-two"
                self.assertEqual(run(dataset), 0)
                self.assertEqual(len(calls), 6, "Model digest changes must invalidate cache")

                # A deliberately new dataset version permits a new prompt, while
                # sharing the output directory to exercise cache invalidation.
                with patch.object(benchmark, "SYSTEM_PROMPT", benchmark.SYSTEM_PROMPT + " New version."):
                    changed_prompt_dataset = prepare_frozen(root / "new-prompt")
                    self.assertEqual(run(changed_prompt_dataset), 0)
                    self.assertEqual(len(calls), 9, "Prompt changes must invalidate cache")

                rows = authored_fixtures()
                for row in rows:
                    row["requirements"].append("Use a brief answer in this new dataset version.")
                changed_inputs, review = prepare_review(root / "new-input", rows)
                benchmark.freeze_dataset(changed_inputs, review, seed=41)
                self.assertEqual(run(changed_inputs), 0)
                self.assertEqual(len(calls), 12, "Input changes must invalidate cache")
            self.assertEqual(len(list((output / "cache").glob("*.json"))), 12)
            self.assertTrue(all("expected_verdict" not in case for case in calls))


if __name__ == "__main__":
    unittest.main()
