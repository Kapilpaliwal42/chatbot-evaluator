"""Regression checks for independent, evidence-grounded local evaluation."""

import csv
import html
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate

from evaluation.core import (
    build_prompt,
    case_digest,
    load_cases,
    parse_judgment,
    summarize,
    validate_judgment,
)
from evaluation.report import write_reports


def sample_case():
    return {
        "id": "refund-001",
        "question": "How long do customers have to request a refund?",
        "answer": "Customers have 30 days to request a refund.",
        "references": [{"id": "policy", "text": "Refunds may be requested within 30 days of purchase."}],
        "requirements": ["State the refund window."],
    }


def sample_judgment(score=4, findings=None):
    return {
        "scores": {"groundedness": score, "instruction_following": 4, "completeness": 4},
        "rationale": "The answer states the refund window supported by the policy.",
        "findings": [] if findings is None else findings,
        "needs_human_review": False,
        "human_review_reasons": [],
    }


def sample_finding(**overrides):
    finding = {
        "category": "groundedness",
        "claim": "Customers have 90 days to request a refund.",
        "reference_id": "policy",
        "quote": "within 30 days of purchase",
        "explanation": "The claim differs from the policy's 30-day refund window.",
    }
    finding.update(overrides)
    return finding


class JudgmentTests(unittest.TestCase):
    def test_supported_pass_is_accepted(self):
        result = validate_judgment(sample_judgment(), sample_case())
        self.assertEqual(result["verdict"], "pass")
        self.assertFalse(result["needs_human_review"])

    def test_verdict_is_derived_from_scores_not_model_label(self):
        judgment = sample_judgment(score=1, findings=[sample_finding()])
        result = validate_judgment(judgment, sample_case())
        self.assertEqual(result["verdict"], "fail")
        self.assertTrue(result["findings"][0]["evidence_valid"])

    def test_model_cannot_inject_a_passing_verdict(self):
        judgment = sample_judgment(score=1, findings=[sample_finding()])
        judgment["verdict"] = "pass"
        with self.assertRaises(ValueError):
            validate_judgment(judgment, sample_case())

    def test_marginal_score_requires_review(self):
        result = validate_judgment(sample_judgment(score=2, findings=[sample_finding()]), sample_case())
        self.assertEqual(result["verdict"], "review")

    def test_fabricated_citation_forces_review_even_with_failing_score(self):
        for finding in (
            sample_finding(reference_id="invented-policy"),
            sample_finding(quote="within 90 days of purchase"),
        ):
            with self.subTest(finding=finding):
                result = validate_judgment(sample_judgment(score=1, findings=[finding]), sample_case())
                self.assertEqual(result["verdict"], "review")
                self.assertTrue(result["needs_human_review"])
                self.assertFalse(result["findings"][0]["evidence_valid"])
                self.assertTrue(result["human_review_reasons"])

    def test_quote_matching_is_exact(self):
        finding = sample_finding(quote="within 30 DAYS of purchase")
        result = validate_judgment(sample_judgment(score=1, findings=[finding]), sample_case())
        self.assertEqual(result["verdict"], "review")
        self.assertFalse(result["findings"][0]["evidence_valid"])

    def test_unexplained_failure_requires_human_review(self):
        result = validate_judgment(sample_judgment(score=0), sample_case())
        self.assertEqual(result["verdict"], "review")
        self.assertTrue(result["needs_human_review"])
        self.assertTrue(result["human_review_reasons"])

    def test_invalid_scores_are_rejected(self):
        for value in (True, False, -1, 5, 4.0, "4", None):
            with self.subTest(value=value):
                judgment = sample_judgment()
                judgment["scores"]["groundedness"] = value
                with self.assertRaises(ValueError):
                    validate_judgment(judgment, sample_case())

    def test_incomplete_score_dimensions_are_rejected(self):
        judgment = sample_judgment()
        del judgment["scores"]["completeness"]
        with self.assertRaises(ValueError):
            validate_judgment(judgment, sample_case())

    def test_malformed_schema_cannot_become_a_pass(self):
        mutations = (
            ("rationale", ""),
            ("rationale", 42),
            ("findings", "none"),
            ("needs_human_review", "false"),
            ("human_review_reasons", "none"),
            ("human_review_reasons", [42]),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                judgment = sample_judgment()
                judgment[key] = value
                with self.assertRaises(ValueError):
                    validate_judgment(judgment, sample_case())

    def test_missing_required_judgment_fields_are_rejected(self):
        for key in sample_judgment():
            with self.subTest(key=key):
                judgment = sample_judgment()
                del judgment[key]
                with self.assertRaises(ValueError):
                    validate_judgment(judgment, sample_case())

    def test_unknown_finding_category_is_rejected(self):
        judgment = sample_judgment(findings=[sample_finding(category="style")])
        with self.assertRaises(ValueError):
            validate_judgment(judgment, sample_case())

    def test_empty_quote_is_rejected_as_malformed(self):
        judgment = sample_judgment(score=1, findings=[sample_finding(quote="")])
        with self.assertRaises(ValueError):
            validate_judgment(judgment, sample_case())

    def test_each_problem_criterion_requires_its_own_evidence(self):
        judgment = sample_judgment(score=1, findings=[sample_finding()])
        judgment["scores"]["completeness"] = 1
        result = validate_judgment(judgment, sample_case())
        self.assertEqual(result["verdict"], "review")
        self.assertTrue(result["needs_human_review"])
        self.assertTrue(any("completeness" in reason for reason in result["human_review_reasons"]))

    def test_positive_scores_with_negative_findings_require_review(self):
        result = validate_judgment(sample_judgment(findings=[sample_finding()]), sample_case())
        self.assertEqual(result["verdict"], "review")
        self.assertTrue(result["needs_human_review"])


class PromptAndParsingTests(unittest.TestCase):
    def test_expected_label_is_held_out_of_judge_prompt(self):
        case = sample_case()
        case["expected_verdict"] = "pass"
        pass_prompt = build_prompt(case)
        case["expected_verdict"] = "fail"
        self.assertEqual(pass_prompt, build_prompt(case))
        self.assertNotIn("expected_verdict", pass_prompt)
        self.assertIn(case["question"], pass_prompt)
        self.assertIn(case["answer"], pass_prompt)
        self.assertIn(case["references"][0]["text"], pass_prompt)

    def test_plain_json_and_single_json_fence_are_supported(self):
        judgment = sample_judgment()
        serialized = json.dumps(judgment)
        self.assertEqual(parse_judgment(serialized), judgment)
        self.assertEqual(parse_judgment("```json\n" + serialized + "\n```"), judgment)

    def test_prose_and_multiple_json_objects_are_not_greedily_extracted(self):
        serialized = json.dumps(sample_judgment())
        for text in (
            "Here is my evaluation: " + serialized,
            serialized + "\n" + serialized,
            "```json\n" + serialized + "\n```\nExtra explanation.",
            "```json\n" + serialized + "\n```\n```json\n" + serialized + "\n```",
            "[]",
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_judgment(text)


class CaseLoadingTests(unittest.TestCase):
    def write_cases(self, directory, cases):
        path = Path(directory) / "cases.jsonl"
        path.write_text("\n".join(json.dumps(case) for case in cases) + "\n", encoding="utf-8")
        return path

    def test_load_valid_case(self):
        with tempfile.TemporaryDirectory() as directory:
            case = sample_case()
            loaded = load_cases(self.write_cases(directory, [case]))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["id"], case["id"])

    def test_duplicate_case_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
                load_cases(self.write_cases(directory, [sample_case(), sample_case()]))

    def test_duplicate_reference_ids_are_rejected(self):
        case = sample_case()
        case["references"].append({"id": "policy", "text": "A contradictory policy."})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "(?i)duplicate"):
                load_cases(self.write_cases(directory, [case]))

    def test_missing_case_field_reports_input_line(self):
        bad_case = sample_case()
        bad_case["id"] = "refund-002"
        del bad_case["answer"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError) as error:
                load_cases(self.write_cases(directory, [sample_case(), bad_case]))
        self.assertIn("2", str(error.exception))

    def test_invalid_expected_label_is_rejected(self):
        case = sample_case()
        case["expected_verdict"] = "excellent"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_cases(self.write_cases(directory, [case]))

    def test_requirements_are_optional(self):
        case = sample_case()
        del case["requirements"]
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_cases(self.write_cases(directory, [case]))
        self.assertEqual(len(loaded), 1)


class SummaryTests(unittest.TestCase):
    def test_agreement_excludes_unlabelled_and_failed_judgments(self):
        cases = [
            {"id": "a", "expected_verdict": "pass", "judgment": {"verdict": "pass"}},
            {"id": "b", "expected_verdict": "fail", "judgment": {"verdict": "review"}},
            {"id": "c", "judgment": {"verdict": "pass"}},
            {"id": "d", "expected_verdict": "fail", "judgment": {"verdict": "error"}, "error": "Invalid judge JSON"},
        ]
        result = summarize(cases)
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["pass"], 2)
        self.assertEqual(result["fail"], 0)
        self.assertEqual(result["review"], 1)
        self.assertEqual(result["error"], 1)
        self.assertEqual(result["labelled"], 2)
        self.assertEqual(result["agreement"], 0.5)

    def test_agreement_is_absent_without_successful_labels(self):
        result = summarize([{"id": "a", "judgment": {"verdict": "pass"}}])
        self.assertEqual(result["labelled"], 0)
        self.assertIsNone(result["agreement"])


class ReportTests(unittest.TestCase):
    def test_html_escapes_untrusted_case_and_judgment_content(self):
        def hostile(name):
            return '<script data-field="' + name + '">window.' + name + '=true</script>'

        case = sample_case()
        case["id"] = hostile("caseid")
        case["question"] = hostile("question")
        case["answer"] = hostile("answer")
        case["requirements"] = [hostile("requirement")]
        case["references"] = [{"id": hostile("referenceid"), "text": hostile("reference")}]
        case["judgment"] = sample_judgment(score=1, findings=[sample_finding(
            claim=hostile("claim"), reference_id=hostile("referenceid"),
            quote=hostile("quote"), explanation=hostile("explanation"),
        )])
        case["judgment"]["verdict"] = "review"
        case["judgment"]["rationale"] = hostile("rationale")
        case["judgment"]["needs_human_review"] = True
        case["judgment"]["human_review_reasons"] = [hostile("reviewreason")]
        run = {"mode": "demo", "cases": [case], "summary": {"agreement": None}}
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(run, Path(directory))
            report = next(path for path in paths if path.suffix == ".html").read_text(encoding="utf-8")
        for field in ("caseid", "question", "answer", "requirement", "referenceid", "reference",
                      "claim", "quote", "explanation", "rationale", "reviewreason"):
            with self.subTest(field=field):
                payload = hostile(field)
                self.assertNotIn(payload, report)
                self.assertIn(html.escape(payload, quote=True), report)

    def test_reviewer_csv_prevents_case_id_formula_execution(self):
        case = sample_case()
        case["id"] = '=HYPERLINK("https://example.invalid", "Click")'
        case["judgment"] = {"verdict": "pass"}
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports({"cases": [case]}, Path(directory))
            csv_path = next(path for path in paths if path.suffix == ".csv")
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[1][0], "'" + case["id"])
        self.assertEqual(rows[1][2:], ["", ""])


class DemoBindingTests(unittest.TestCase):
    def write_bound_sample(self, directory):
        root = Path(directory)
        samples = root / "evaluation" / "samples"
        samples.mkdir(parents=True)
        case = sample_case()
        source = samples / "cases.jsonl"
        source.write_text(json.dumps(case) + "\n", encoding="utf-8")
        fixtures = {
            "case_sha256": {case["id"]: case_digest(case)},
            "judgments": {case["id"]: sample_judgment()},
        }
        (samples / "demo_judgments.json").write_text(json.dumps(fixtures), encoding="utf-8")
        return root, source, case

    def test_edited_default_sample_cannot_inherit_saved_passing_judgment(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, case = self.write_bound_sample(directory)
            case["answer"] = "Customers have 90 days to request a refund."
            source.write_text(json.dumps(case) + "\n", encoding="utf-8")
            output = root / "output"
            with patch("evaluate.ROOT", root), patch("evaluation.core.OllamaJudge._request") as request:
                code = evaluate.main(["--demo", "--cases", str(source), "--out", str(output)])
            self.assertEqual(code, 1)
            self.assertFalse((output / "report.html").exists())
            self.assertFalse((output / "results.json").exists())
            request.assert_not_called()

    def test_original_bound_sample_runs_demo_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root, source, _ = self.write_bound_sample(directory)
            output = root / "output"
            with patch("evaluate.ROOT", root), patch("evaluation.core.OllamaJudge._request") as request:
                code = evaluate.main(["--demo", "--cases", str(source), "--out", str(output)])
            self.assertEqual(code, 0)
            results = json.loads((output / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(results["summary"]["pass"], 1)
            self.assertEqual(results["mode"], "demo")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
