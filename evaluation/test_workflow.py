"""Controller tests use fake frozen data and never invoke model inference."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import complete_benchmark as workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.controller = workflow.Controller(self.root, wait_hours=1, poll_seconds=5)
        self.manifest = {"version": 1, "prompt_sha256": "prompt", "core_sha256": "core",
                         "gold_sha256": "gold", "splits": {
                             "calibration": {"sha256": "calibration-input"},
                             "holdout": {"sha256": "holdout-input"}}}
        self.cases = {"calibration": [{"id": "cal-1"}], "holdout": [{"id": "hold-1"}]}
        self.models = {model: {"name": model, "digest": model + "-digest", "details": {"size": "7B"}}
                       for model in workflow.MODELS}
        self.events = []
        for name, replacement in (("load_frozen", self.fake_frozen),
                                  ("local_models", lambda: (self.models, "test-version")),
                                  ("run_benchmark", self.fake_run),
                                  ("select_calibration", self.fake_select),
                                  ("compare_runs", self.fake_compare)):
            mocked = patch.object(workflow, name, side_effect=replacement).start()
            setattr(self, name + "_mock", mocked)
            self.addCleanup(patch.stopall)

    def fake_frozen(self, directory, split):
        return self.manifest, self.cases[split], {}

    def fake_run(self, directory, split, model, output):
        if split == "holdout":
            selection = workflow.read_json(self.controller.selection)
            self.assertIn("workflow_binding", selection)
            self.assertIn("calibration_file_sha256", selection)
        self.events.append(("run", split, model))
        result = {"schema_version": 1, "mode": "ollama", "completed": True,
                  "model": model, "split": split, "requested_total": 1,
                  "configuration": self.controller.expected_configuration(self.manifest, self.models[model], "test-version"),
                  "dataset_sha256": self.manifest["splits"][split]["sha256"],
                  "cases": self.cases[split]}
        workflow.atomic_write(Path(output) / "results.json", json.dumps(result))
        return 0

    def fake_select(self, paths, output):
        self.events.append(("select",))
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(workflow.read_json(path)["split"] == "calibration" for path in paths))
        value = {"selected_model": workflow.GENERAL}
        workflow.atomic_write(output, json.dumps(value))
        return value

    def fake_compare(self, paths, output):
        self.events.append(("compare",))
        self.assertEqual(len(paths), 4)
        for name in ("comparison.json", "comparison.md", "comparison.html"):
            workflow.atomic_write(Path(output) / name, "{}" if name.endswith(".json") else "Comparison")

    def test_selection_precedes_both_holdout_runs_and_gpu_jobs_are_sequential(self):
        self.assertEqual(self.controller.execute(), 0)
        self.assertEqual(self.events, [
            ("run", "calibration", workflow.CODER),
            ("run", "calibration", workflow.GENERAL), ("select",),
            ("run", "holdout", workflow.GENERAL),
            ("run", "holdout", workflow.CODER), ("compare",)])
        self.assertEqual(workflow.read_json(self.controller.output / "workflow-state.json")["status"], "completed")

    def test_matching_completed_work_is_not_repeated(self):
        self.assertEqual(self.controller.execute(), 0)
        self.events.clear()
        self.assertEqual(self.controller.execute(), 0)
        self.assertEqual(self.events, [])

    def test_model_change_after_selection_fails_without_rerunning(self):
        self.assertEqual(self.controller.execute(), 0)
        self.events.clear()
        self.models[workflow.CODER]["digest"] = "replacement-digest"
        self.assertEqual(self.controller.execute(), 1)
        self.assertEqual(self.events, [])
        state = workflow.read_json(self.controller.output / "workflow-state.json")
        self.assertEqual(state["status"], "failed")
        self.assertIn("changed after calibration selection", state["detail"])

    def test_model_wait_deadline_reports_failure_and_never_opens_holdout(self):
        self.models.pop(workflow.GENERAL)
        now = [0]
        self.controller.clock = lambda: now[0]
        self.controller.sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
        self.controller.wait_hours = 5 / 3600
        self.assertEqual(self.controller.execute(), 1)
        state = workflow.read_json(self.controller.output / "workflow-state.json")
        self.assertEqual(state["status"], "failed")
        self.assertIn("not available", state["detail"])
        self.assertEqual(self.events, [("run", "calibration", workflow.CODER)])
        self.assertFalse(self.controller.selection.exists())

    def test_existing_holdout_without_selection_refuses_late_selection(self):
        workflow.atomic_write(self.controller.run_path("holdout", workflow.GENERAL), "{}")
        self.assertEqual(self.controller.execute(), 1)
        self.assertEqual(self.events, [])
        self.assertFalse(self.controller.selection.exists())
        state = workflow.read_json(self.controller.output / "workflow-state.json")
        self.assertIn("without a prior recorded selection", state["detail"])

    def test_lock_blocks_second_controller_without_overwriting_state(self):
        self.controller.state("running", "owner", "Original worker")
        original = (self.controller.output / "workflow-state.json").read_bytes()
        with workflow.workflow_lock(self.controller.output / "workflow.lock"):
            self.assertEqual(self.controller.execute(), 2)
        self.assertEqual((self.controller.output / "workflow-state.json").read_bytes(), original)
        self.assertEqual(self.events, [])

    def test_preflight_is_read_only_and_status_html_escapes_detail(self):
        result = self.controller.check()
        self.assertTrue(result["frozen_verified"])
        self.assertFalse(self.controller.output.exists())
        self.controller.state("waiting", "download", "<script>alert('x')</script>")
        page = (self.controller.output / "status.html").read_text(encoding="utf-8")
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn('http-equiv="refresh" content="60"', page)


if __name__ == "__main__":
    unittest.main()
