import argparse
import csv
import json
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import yaml


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def parse_args():
    parser = argparse.ArgumentParser(description="Launch local dashboard for experiment monitoring.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-root", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return parser.parse_args()


def read_json_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except (ValueError, json.JSONDecodeError):
        return None


def read_train_log(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def read_yaml_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def get_nested_value(payload, dotted_key):
    current = payload
    for key in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def discover_experiment_configs(project_root):
    exp_dir = os.path.join(project_root, "configs", "experiments")
    registry = {}
    if not os.path.isdir(exp_dir):
        return registry
    for file_name in sorted(os.listdir(exp_dir)):
        if not file_name.endswith(".yaml"):
            continue
        file_path = os.path.join(exp_dir, file_name)
        payload = read_yaml_if_exists(file_path) or {}
        experiment_name = payload.get("experiment_name")
        if experiment_name:
            registry[experiment_name] = {"path": file_path, "config": payload}
    return registry


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def find_resume_candidate(project_root, experiment_name):
    runs_root = os.path.join(project_root, "outputs", "runs")
    if not os.path.isdir(runs_root):
        return None
    candidates = []
    for run_name in os.listdir(runs_root):
        run_dir = os.path.join(runs_root, run_name)
        if not os.path.isdir(run_dir):
            continue
        status = read_json_if_exists(os.path.join(run_dir, "status.json")) or {}
        if status.get("experiment_name") != experiment_name:
            continue
        resume_checkpoint_path = status.get("resume_checkpoint_path") or os.path.join(run_dir, "resume_checkpoint.pt")
        if status.get("status") == "stopped" and os.path.exists(resume_checkpoint_path):
            candidates.append((run_name, run_dir, resume_checkpoint_path, status))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, run_dir, resume_checkpoint_path, status = candidates[0]
    return {
        "run_dir": run_dir,
        "resume_checkpoint_path": resume_checkpoint_path,
        "epoch": status.get("epoch", 0),
    }


def move_run(run_name, source_root, target_root):
    src = os.path.join(source_root, run_name)
    dst = os.path.join(target_root, run_name)
    if not os.path.isdir(src):
        raise FileNotFoundError("Run directory not found: {}".format(src))
    ensure_dir(target_root)
    if os.path.exists(dst):
        raise FileExistsError("Target already exists: {}".format(dst))
    shutil.move(src, dst)


def build_catalog(plan, experiment_registry):
    catalog = []
    seen = set()
    for stage_cfg in plan.get("stages", []):
        for exp_cfg in stage_cfg.get("experiments", []):
            experiment_name = exp_cfg.get("experiment_name")
            info = experiment_registry.get(experiment_name, {})
            config = info.get("config", {})
            catalog.append({
                "experiment_name": experiment_name,
                "stage": stage_cfg.get("stage"),
                "dimension": stage_cfg.get("dimension"),
                "value": exp_cfg.get("value"),
                "description": stage_cfg.get("description"),
                "config_path": info.get("path"),
                "config": config,
            })
            seen.add(experiment_name)
    for experiment_name, info in sorted(experiment_registry.items()):
        if experiment_name in seen:
            continue
        config = info.get("config", {})
        metadata = config.get("metadata", {})
        catalog.append({
            "experiment_name": experiment_name,
            "stage": metadata.get("stage", "unplanned"),
            "dimension": metadata.get("dimension"),
            "value": metadata.get("value"),
            "description": "Unplanned experiment",
            "config_path": info.get("path"),
            "config": config,
        })
    return catalog


class ScheduleManager(object):
    def __init__(self, project_root):
        self.project_root = project_root
        self.plan = read_yaml_if_exists(os.path.join(project_root, "configs", "experiment_plan.yaml")) or {"stages": []}
        self.registry = discover_experiment_configs(project_root)
        self.catalog = build_catalog(self.plan, self.registry)
        self.queue_path = os.path.join(project_root, "outputs", "schedule", "queue.json")
        self.log_dir = os.path.join(project_root, "outputs", "logs")
        ensure_dir(os.path.dirname(self.queue_path))
        ensure_dir(self.log_dir)
        self.lock = threading.Lock()
        self.process = None
        self.worker = None
        self.stop_requested = False
        self.state = self._load_state()

    def _default_state(self):
        return {
            "items": [],
            "runner_status": "idle",
            "current_item_id": None,
            "updated_at": time.time(),
        }

    def _load_state(self):
        state = read_json_if_exists(self.queue_path) or self._default_state()
        for item in state.get("items", []):
            if item.get("status") == "running":
                item["status"] = "pending"
        state["runner_status"] = "idle"
        state["current_item_id"] = None
        self._save_state(state)
        return state

    def _save_state(self, state=None):
        payload = state if state is not None else self.state
        payload["updated_at"] = time.time()
        with open(self.queue_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

    def _find_catalog_entry(self, experiment_name):
        for item in self.catalog:
            if item["experiment_name"] == experiment_name:
                return item
        return None

    def get_state(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def get_catalog(self):
        return self.catalog

    def add_item(self, experiment_name):
        entry = self._find_catalog_entry(experiment_name)
        if not entry or not entry.get("config_path"):
            raise ValueError("Unknown experiment: {}".format(experiment_name))
        resume_candidate = find_resume_candidate(self.project_root, experiment_name)
        with self.lock:
            item_id = "{}-{}".format(int(time.time() * 1000), len(self.state["items"]))
            self.state["items"].append({
                "id": item_id,
                "experiment_name": experiment_name,
                "stage": entry.get("stage"),
                "dimension": entry.get("dimension"),
                "value": entry.get("value"),
                "config_path": entry.get("config_path"),
                "status": "pending",
                "created_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "return_code": None,
                "log_path": None,
                "run_dir": resume_candidate.get("run_dir") if resume_candidate else None,
                "resume_checkpoint_path": resume_candidate.get("resume_checkpoint_path") if resume_candidate else None,
                "resume_epoch": resume_candidate.get("epoch") if resume_candidate else 0,
            })
            self._save_state()

    def remove_item(self, item_id):
        with self.lock:
            self.state["items"] = [item for item in self.state["items"] if item["id"] != item_id]
            self._save_state()

    def clear_items(self):
        with self.lock:
            if self.state.get("runner_status") == "running":
                raise RuntimeError("Cannot clear queue while runner is active.")
            self.state["items"] = []
            self._save_state()

    def move_item(self, item_id, direction):
        with self.lock:
            items = self.state["items"]
            idx = next((i for i, item in enumerate(items) if item["id"] == item_id), None)
            if idx is None:
                raise ValueError("Queue item not found: {}".format(item_id))
            target = idx - 1 if direction == "up" else idx + 1
            if target < 0 or target >= len(items):
                return
            items[idx], items[target] = items[target], items[idx]
            self._save_state()

    def start(self):
        with self.lock:
            if self.worker and self.worker.is_alive():
                return
            self.stop_requested = False
            self.state["runner_status"] = "running"
            self._save_state()
            self.worker = threading.Thread(target=self._worker_loop, name="schedule-runner", daemon=True)
            self.worker.start()

    def stop(self):
        with self.lock:
            self.stop_requested = True
            proc = self.process
            self.state["runner_status"] = "stopping"
            self._save_state()
        if proc is not None:
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def _worker_loop(self):
        while True:
            with self.lock:
                if self.stop_requested:
                    self.state["runner_status"] = "stopped"
                    self.state["current_item_id"] = None
                    self.process = None
                    self._save_state()
                    return

                next_item = None
                for item in self.state["items"]:
                    if item.get("status") == "pending":
                        next_item = item
                        break

                if next_item is None:
                    self.state["runner_status"] = "idle"
                    self.state["current_item_id"] = None
                    self.process = None
                    self._save_state()
                    return

                next_item["status"] = "running"
                next_item["started_at"] = time.time()
                self.state["current_item_id"] = next_item["id"]
                log_path = os.path.join(
                    self.log_dir,
                    "queue_{}_{}.log".format(
                        int(next_item["started_at"]),
                        next_item["experiment_name"],
                    ),
                )
                next_item["log_path"] = log_path
                self._save_state()

            command = [
                sys.executable,
                os.path.join(self.project_root, "scripts", "run_experiment.py"),
                "--config",
                next_item["config_path"],
            ]
            if next_item.get("run_dir") and next_item.get("resume_checkpoint_path"):
                command.extend(["--resume-run-dir", next_item["run_dir"]])
                command.extend(["--resume-checkpoint", next_item["resume_checkpoint_path"]])
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write("Starting {}\n".format(" ".join(command)))
                log_file.flush()
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                proc = subprocess.Popen(
                    command,
                    cwd=self.project_root,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ.copy(), TRAFFIC_SIGNS_FORCE_NUM_WORKERS_ZERO="1"),
                    creationflags=creationflags,
                )
                with self.lock:
                    self.process = proc
                    self._save_state()
                return_code = proc.wait()

            with self.lock:
                next_item["finished_at"] = time.time()
                next_item["return_code"] = return_code
                if self.stop_requested:
                    next_item["status"] = "stopped"
                    self.state["runner_status"] = "stopped"
                    self.state["current_item_id"] = None
                    self.process = None
                    self._save_state()
                    return
                next_item["status"] = "completed" if return_code == 0 else "failed"
                if next_item["status"] == "completed":
                    next_item["resume_checkpoint_path"] = None
                self.state["current_item_id"] = None
                self.process = None
                self._save_state()


def build_run_state(project_root, runs_root, schedule_manager=None):
    plan = read_yaml_if_exists(os.path.join(project_root, "configs", "experiment_plan.yaml")) or {"stages": []}
    experiment_registry = discover_experiment_configs(project_root)
    archived_root = os.path.join(project_root, "outputs", "archived_runs")
    experiment_index = {}
    for stage_cfg in plan.get("stages", []):
        for exp_cfg in stage_cfg.get("experiments", []):
            config_info = experiment_registry.get(exp_cfg.get("experiment_name"), {})
            experiment_index[exp_cfg.get("experiment_name")] = {
                "stage": stage_cfg.get("stage"),
                "dimension": stage_cfg.get("dimension"),
                "value": exp_cfg.get("value"),
                "config_path": config_info.get("path"),
                "config": config_info.get("config"),
            }
    active_run_dirs = sorted(
        [os.path.join(runs_root, name) for name in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, name))],
        reverse=True,
    ) if os.path.isdir(runs_root) else []
    archived_run_dirs = sorted(
        [os.path.join(archived_root, name) for name in os.listdir(archived_root) if os.path.isdir(os.path.join(archived_root, name))],
        reverse=True,
    ) if os.path.isdir(archived_root) else []
    run_dirs = active_run_dirs + archived_run_dirs

    runs = []
    for run_dir in run_dirs:
        is_archived = os.path.normpath(run_dir).startswith(os.path.normpath(archived_root))
        status = read_json_if_exists(os.path.join(run_dir, "status.json")) or {}
        metrics = read_json_if_exists(os.path.join(run_dir, "metrics.json")) or {}
        resolved_config = read_yaml_if_exists(os.path.join(run_dir, "resolved_config.yaml")) or {}
        log_rows = read_train_log(os.path.join(run_dir, "train_log.csv"))
        latest = log_rows[-1] if log_rows else None
        metadata = resolved_config.get("metadata", {})
        experiment_name = status.get("experiment_name") or metrics.get("experiment_name") or os.path.basename(run_dir)
        fallback = experiment_index.get(experiment_name, {})
        if not metadata and fallback:
            metadata = {
                "stage": fallback.get("stage"),
                "dimension": fallback.get("dimension"),
                "value": fallback.get("value"),
            }
        if not resolved_config or not resolved_config.get("model"):
            resolved_config = fallback.get("config") or resolved_config
        runs.append({
            "run_dir": run_dir,
            "run_name": os.path.basename(run_dir),
            "status": "archived" if is_archived else status.get("status", "unknown"),
            "archived": is_archived,
            "experiment_name": experiment_name,
            "metadata": metadata,
            "config": resolved_config,
            "epoch": status.get("epoch", 0),
            "total_epochs": status.get("total_epochs", 0),
            "best_val_acc": status.get("best_val_acc", metrics.get("best_val_acc")),
            "best_epoch": status.get("best_epoch", metrics.get("best_epoch")),
            "latest": latest,
            "metrics": metrics,
            "log_rows": log_rows,
            "curve_path": "/runs/{}/curves.png".format(os.path.basename(run_dir)),
            "lr_curve_path": "/runs/{}/lr_curve.png".format(os.path.basename(run_dir)),
            "confusion_matrix_path": "/runs/{}/confusion_matrix.png".format(os.path.basename(run_dir)),
        })

    comparisons = []
    for stage_cfg in plan.get("stages", []):
        stage_name = stage_cfg.get("stage")
        stage_runs = [run for run in runs if run.get("metadata", {}).get("stage") == stage_name]
        rows = []
        for run in stage_runs:
            value = run.get("metadata", {}).get("value")
            if value is None:
                value = get_nested_value(run.get("config", {}), stage_cfg.get("dimension", ""))
            rows.append({
                "experiment_name": run["experiment_name"],
                "run_name": run["run_name"],
                "value": value,
                "best_val_acc": safe_float(run.get("best_val_acc")),
                "test_acc": safe_float((run.get("metrics") or {}).get("test_acc")),
                "best_val_macro_f1": safe_float((run.get("metrics") or {}).get("best_val_macro_f1")),
                "test_macro_f1": safe_float((run.get("metrics") or {}).get("test_macro_f1")),
                "runtime_sec": safe_float((run.get("metrics") or {}).get("total_runtime_sec")),
                "trainable_params": (run.get("metrics") or {}).get("trainable_params"),
                "status": run.get("status"),
                "archived": run.get("archived"),
                "run_dir": run.get("run_dir"),
            })
        comparisons.append({
            "stage": stage_name,
            "dimension": stage_cfg.get("dimension"),
            "description": stage_cfg.get("description"),
            "rows": rows,
        })

    return {
        "version": "dashboard-2026-04-17-v2",
        "plan": plan,
        "runs": runs,
        "comparisons": comparisons,
        "catalog": schedule_manager.get_catalog() if schedule_manager else [],
        "schedule": schedule_manager.get_state() if schedule_manager else {"items": [], "runner_status": "idle", "current_item_id": None},
    }


def make_handler(project_root):
    dashboard_root = os.path.join(project_root, "web")
    runs_root = os.path.join(project_root, "outputs", "runs")
    archived_root = os.path.join(project_root, "outputs", "archived_runs")
    schedule_manager = ScheduleManager(project_root)

    class Handler(BaseHTTPRequestHandler):
        def _send_bytes(self, content, content_type="text/plain; charset=utf-8", status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path in ("/", "/index.html"):
                with open(os.path.join(dashboard_root, "index.html"), "rb") as file_obj:
                    return self._send_bytes(file_obj.read(), "text/html; charset=utf-8")

            if path == "/api/state":
                payload = json.dumps(build_run_state(project_root, runs_root, schedule_manager), ensure_ascii=False).encode("utf-8")
                return self._send_bytes(payload, "application/json; charset=utf-8")

            if path.startswith("/runs/"):
                rel_path = path[len("/runs/"):].replace("/", os.sep)
                for root in [runs_root, archived_root]:
                    abs_path = os.path.normpath(os.path.join(root, rel_path))
                    if abs_path.startswith(os.path.normpath(root)) and os.path.isfile(abs_path):
                        content_type = "image/png" if abs_path.lower().endswith(".png") else "text/plain; charset=utf-8"
                        with open(abs_path, "rb") as file_obj:
                            return self._send_bytes(file_obj.read(), content_type)

            return self._send_bytes(b"Not Found", status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path not in (
                "/api/archive",
                "/api/restore",
                "/api/schedule/add",
                "/api/schedule/remove",
                "/api/schedule/move",
                "/api/schedule/clear",
                "/api/schedule/start",
                "/api/schedule/stop",
            ):
                return self._send_bytes(b"Not Found", status=404)

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            payload = json.loads(raw_body.decode("utf-8"))

            try:
                if parsed.path == "/api/archive":
                    run_name = payload.get("run_name")
                    if not run_name:
                        raise ValueError("run_name is required")
                    move_run(run_name, runs_root, archived_root)
                elif parsed.path == "/api/restore":
                    run_name = payload.get("run_name")
                    if not run_name:
                        raise ValueError("run_name is required")
                    move_run(run_name, archived_root, runs_root)
                elif parsed.path == "/api/schedule/add":
                    schedule_manager.add_item(payload.get("experiment_name"))
                elif parsed.path == "/api/schedule/remove":
                    schedule_manager.remove_item(payload.get("item_id"))
                elif parsed.path == "/api/schedule/move":
                    schedule_manager.move_item(payload.get("item_id"), payload.get("direction"))
                elif parsed.path == "/api/schedule/clear":
                    schedule_manager.clear_items()
                elif parsed.path == "/api/schedule/start":
                    schedule_manager.start()
                else:
                    schedule_manager.stop()
            except Exception as exc:
                return self._send_bytes(json.dumps({"error": str(exc)}).encode("utf-8"), "application/json", 400)

            response = json.dumps({"ok": True}).encode("utf-8")
            return self._send_bytes(response, "application/json; charset=utf-8")

        def log_message(self, format, *args):
            return

    return Handler


def main():
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(os.path.abspath(args.project_root)))
    print("Dashboard running at http://{}:{}/".format(args.host, args.port))
    server.serve_forever()


if __name__ == "__main__":
    main()
