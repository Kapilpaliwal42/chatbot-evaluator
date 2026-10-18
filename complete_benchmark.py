"""Resume the frozen two-7B-model comparison, with one GPU job at a time."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import time
import uuid

from evaluation.benchmarking import (OllamaJudge, PROMPT_VERSION, compare_runs,
                                    load_frozen, run_benchmark,
                                    select_calibration)


ROOT = Path(__file__).resolve().parent
CODER = "qwen2.5-coder:7b"
GENERAL = "qwen2.5:7b"
MODELS = (CODER, GENERAL)
BASE_URL = "http://127.0.0.1:11434"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def workflow_lock(path):
    """The operating system releases this lock if the controller exits."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0, 2)
    if not handle.tell():
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("A benchmark controller already holds the workflow lock") from exc
    try:
        yield
    finally:
        handle.seek(0)
        if os.name == "nt":
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def local_models():
    """Read the local service; this controller never downloads a model."""
    judge = OllamaJudge(base_url=BASE_URL, model=CODER, timeout=20)
    tags = judge._request("/api/tags").get("models", [])
    version = judge._request("/api/version").get("version")
    if not version:
        raise ValueError("Ollama did not provide its runtime version")
    found = {item.get("name"): item for item in tags}
    for model in MODELS:
        if model in found and not found[model].get("digest"):
            raise ValueError(f"Ollama did not provide the digest for {model}")
    return found, version


class Controller:
    def __init__(self, root=ROOT, wait_hours=12, poll_seconds=60,
                 clock=time.monotonic, sleep=time.sleep):
        if (not math.isfinite(wait_hours) or not math.isfinite(poll_seconds)
                or wait_hours < 0 or poll_seconds < 5):
            raise ValueError("wait-hours must be nonnegative and poll-seconds at least 5")
        self.root = Path(root)
        self.dataset = self.root / "evaluation" / "benchmark" / "v1"
        self.output = self.root / "evaluation" / "output" / "benchmark"
        self.comparison = self.root / "evaluation" / "output" / "benchmark-comparison"
        self.selection = self.dataset / "selection.json"
        self.wait_hours, self.poll_seconds = wait_hours, poll_seconds
        self.clock, self.sleep = clock, sleep
        self.phase = "starting"

    def state(self, status, phase, detail):
        self.phase = phase
        value = {"pid": os.getpid(), "status": status, "phase": phase,
                 "updated_at": datetime.now(timezone.utc).isoformat(),
                 "detail": str(detail),
                 "comparison_path": str(self.comparison / "comparison.html")}
        atomic_write(self.output / "workflow-state.json", json.dumps(value, indent=2) + "\n")
        safe = lambda item: html.escape(str(item), quote=True)
        refresh = '<meta http-equiv="refresh" content="60">' if status in ("waiting", "running") else ""
        body = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                + refresh + "<title>Judge comparison progress</title><style>"
                "body{max-width:760px;margin:50px auto;padding:24px;background:#f4f7fb;"
                "color:#243348;font:16px/1.6 system-ui}main{background:white;padding:28px;"
                "border:1px solid #d9e2ec;border-radius:12px}p{overflow-wrap:anywhere}"
                "small{color:#607086}</style></head><body><main><h1>Judge comparison</h1>"
                f"<p><strong>{safe(status.title())}</strong> · {safe(phase)}</p>"
                f"<p>{safe(detail)}</p><p><small>Updated {safe(value['updated_at'])}</small></p>")
        if status == "completed":
            relative = os.path.relpath(self.comparison / "comparison.html", self.output).replace(os.sep, "/")
            body += f'<p><a href="{safe(relative)}">Open the completed comparison</a></p>'
        body += "<p><small>Synthetic benchmark; human review of labels and findings is pending.</small></p></main></body></html>"
        atomic_write(self.output / "status.html", body)
        print(f"{status}: {phase}: {detail}", flush=True)

    def frozen(self):
        manifest, calibration, _ = load_frozen(self.dataset, "calibration")
        holdout_manifest, holdout, _ = load_frozen(self.dataset, "holdout")
        if manifest != holdout_manifest:
            raise ValueError("Frozen manifest changed while checking the splits")
        return manifest, {"calibration": calibration, "holdout": holdout}

    def check(self):
        manifest, cases = self.frozen()
        models, version = local_models()
        return {"dataset_version": manifest["version"], "frozen_verified": True,
                "splits": {key: len(value) for key, value in cases.items()},
                "ollama_version": version,
                "models": {model: {"available": model in models,
                                    "digest": models.get(model, {}).get("digest")}
                           for model in MODELS}}

    def run_path(self, split, model):
        name = "coder" if model == CODER else "general"
        return self.output / f"{name}-{split}" / "results.json"

    def expected_configuration(self, manifest, info, version):
        return {"prompt_sha256": manifest["prompt_sha256"], "prompt_version": PROMPT_VERSION,
                "core_sha256": manifest["core_sha256"], "ollama_version": version,
                "model_digest": info["digest"], "model_details": info.get("details", {}),
                "temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 1600,
                "base_url": BASE_URL}

    def complete_run(self, split, model, manifest, cases, models, version):
        path = self.run_path(split, model)
        if not path.exists():
            return False
        try:
            run = read_json(path)
            expected_ids = [case["id"] for case in cases[split]]
            return (run.get("completed") is True and run.get("model") == model
                    and run.get("split") == split and run.get("mode") == "ollama"
                    and run.get("schema_version") == 1
                    and run.get("configuration") == self.expected_configuration(manifest, models[model], version)
                    and run.get("dataset_sha256") == manifest["splits"][split]["sha256"]
                    and run.get("requested_total") == len(expected_ids)
                    and [case.get("id") for case in run.get("cases", [])] == expected_ids)
        except (ValueError, KeyError, TypeError):
            return False

    def ensure_run(self, split, model, manifest, cases, models, version):
        if self.complete_run(split, model, manifest, cases, models, version):
            self.state("running", f"{model} {split}", "Reusing a complete run with matching frozen inputs and model configuration.")
            return
        self.state("running", f"{model} {split}", "Evaluating cases; matching per-case cache entries resume prior work.")
        code = run_benchmark(self.dataset, split, model, self.run_path(split, model).parent)
        if code == 130:
            raise RuntimeError("Benchmark was interrupted; resume to use its saved cache")
        if not self.complete_run(split, model, manifest, cases, models, version):
            raise RuntimeError(f"{model} {split} did not produce a complete, correctly bound result")
        # Code 1 represents completed judgments with error cases, retained in the comparison.
        if code not in (0, 1):
            raise RuntimeError(f"Benchmark returned unexpected status {code}")

    def wait_general(self):
        deadline = self.clock() + self.wait_hours * 3600
        while True:
            try:
                models, version = local_models()
                if GENERAL in models:
                    if CODER not in models:
                        raise ValueError(f"Required installed model is missing: {CODER}")
                    return models, version
                detail = f"Waiting for the already requested {GENERAL} download; no additional model is being installed."
            except RuntimeError as exc:
                detail = f"Waiting for the local Ollama service: {exc}"
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(f"{GENERAL} was not available within {self.wait_hours:g} hours")
            self.state("waiting", "general model availability", detail)
            self.sleep(min(self.poll_seconds, remaining))

    def binding(self, manifest, models, version):
        return {"dataset_version": manifest["version"], "gold_sha256": manifest["gold_sha256"],
                "splits": {name: data["sha256"] for name, data in manifest["splits"].items()},
                "configurations": {model: self.expected_configuration(manifest, models[model], version)
                                   for model in MODELS}}

    def validate_selection(self, binding):
        selection = read_json(self.selection)
        if selection.get("workflow_binding") != binding:
            raise ValueError("Configuration changed after calibration selection; use a new benchmark version")
        if selection.get("selected_model") not in MODELS:
            raise ValueError("Recorded selection names an unexpected model")
        expected_files = {model: file_sha256(self.run_path("calibration", model)) for model in MODELS}
        if selection.get("calibration_file_sha256") != expected_files:
            raise ValueError("Calibration results changed after selection; use a new benchmark version")
        return selection

    def run(self):
        self.state("running", "frozen verification", "Verifying the prompt, evaluation code, input splits, and gold-label hashes.")
        manifest, cases = self.frozen()
        models, version = local_models()
        if CODER not in models:
            raise ValueError(f"Required installed model is missing: {CODER}")
        # A selected run may never silently change configuration or recalibrate.
        if self.selection.exists():
            models, version = self.wait_general()
            binding = self.binding(manifest, models, version)
            self.validate_selection(binding)
        else:
            if any(self.run_path("holdout", model).exists()
                   or (self.run_path("holdout", model).parent / "checkpoint.json").exists()
                   for model in MODELS):
                raise ValueError("Holdout results exist without a prior recorded selection; use a new benchmark version")
            self.ensure_run("calibration", CODER, manifest, cases, models, version)
            models, version = self.wait_general()
            self.ensure_run("calibration", CODER, manifest, cases, models, version)
            self.ensure_run("calibration", GENERAL, manifest, cases, models, version)
            binding = self.binding(manifest, models, version)
            self.state("running", "calibration selection", "Recording the candidate using calibration results before any holdout evaluation.")
            # The API writes its own file; publish the fully bound selection atomically.
            # A crash before publication leaves no selection and no holdout work.
            pending = self.selection.with_name("selection." + uuid.uuid4().hex + ".pending.json")
            selection = select_calibration([self.run_path("calibration", model) for model in MODELS], pending)
            selection["workflow_binding"] = binding
            selection["calibration_file_sha256"] = {model: file_sha256(self.run_path("calibration", model)) for model in MODELS}
            atomic_write(self.selection, json.dumps(selection, indent=2) + "\n")
            pending.unlink()
        self.validate_selection(binding)
        for model in (GENERAL, CODER):
            self.ensure_run("holdout", model, manifest, cases, models, version)
        paths = [self.run_path(split, model) for split in ("calibration", "holdout") for model in MODELS]
        comparison_binding = {"workflow_binding": binding, "selection_sha256": file_sha256(self.selection),
                              "run_sha256": {str(path.relative_to(self.root)): file_sha256(path) for path in paths}}
        receipt = self.comparison / "workflow-binding.json"
        complete = all((self.comparison / name).is_file() for name in ("comparison.json", "comparison.md", "comparison.html"))
        if not (complete and receipt.exists() and read_json(receipt) == comparison_binding):
            self.state("running", "comparison", "Writing the four-run comparison, including errors and evidence limitations.")
            compare_runs(paths, self.comparison)
            atomic_write(receipt, json.dumps(comparison_binding, indent=2) + "\n")
        self.state("completed", "comparison", "Both calibration and untouched holdout splits are complete for both 7B models. Human review is pending.")
        return 0

    def execute(self):
        try:
            with workflow_lock(self.output / "workflow.lock"):
                try:
                    return self.run()
                except (Exception, KeyboardInterrupt) as exc:
                    detail = str(exc) or "Controller interrupted; saved results can be resumed."
                    self.state("failed", self.phase, detail)
                    return 1
        except RuntimeError as exc:
            # Do not overwrite the state of the controller that owns the lock.
            print(str(exc), flush=True)
            return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Read saved state without starting work or taking the lock")
    parser.add_argument("--check", action="store_true", help="Read-only frozen-data and local-model preflight")
    parser.add_argument("--wait-hours", type=float, default=12)
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.status and args.check:
        parser.error("Choose only one of --status and --check")
    try:
        controller = Controller(wait_hours=args.wait_hours, poll_seconds=args.poll_seconds)
        if args.status:
            path = controller.output / "workflow-state.json"
            print(json.dumps(read_json(path) if path.exists() else {"status": "not_started"}, indent=2))
            return 0
        if args.check:
            print(json.dumps(controller.check(), indent=2))
            return 0
        return controller.execute()
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Preflight failed: {exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
