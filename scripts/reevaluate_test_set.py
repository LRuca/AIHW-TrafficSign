import argparse
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import yaml

from traffic_signs.analysis import save_classwise_metrics, save_confusion_matrix_figure, save_json
from traffic_signs.data import build_dataloaders
from traffic_signs.models import build_model
from traffic_signs.train import evaluate, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Re-evaluate existing runs on the current test set.")
    parser.add_argument("--runs-root", default="outputs/runs", help="Directory containing run folders.")
    parser.add_argument("--run-dir", action="append", default=None, help="Specific run directory to re-evaluate.")
    return parser.parse_args()


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def iter_run_dirs(runs_root, explicit_run_dirs):
    if explicit_run_dirs:
        for run_dir in explicit_run_dirs:
            if os.path.isdir(run_dir):
                yield os.path.abspath(run_dir)
        return

    runs_root = os.path.abspath(runs_root)
    for name in sorted(os.listdir(runs_root)):
        run_dir = os.path.join(runs_root, name)
        if os.path.isdir(run_dir):
            yield run_dir


def reevaluate_run(run_dir):
    config_path = os.path.join(run_dir, "resolved_config.yaml")
    best_path = os.path.join(run_dir, "best.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")
    status_path = os.path.join(run_dir, "status.json")

    if not os.path.exists(config_path) or not os.path.exists(best_path) or not os.path.exists(metrics_path):
        return None

    config = load_yaml(config_path)
    config["model"]["pretrained"] = False
    config["data"]["num_workers"] = 0
    set_seed(config["train"]["seed"])
    device = resolve_device(config["train"]["device"])

    bundle = build_dataloaders(config)
    if bundle.get("test_loader") is None:
        return None

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
    )
    test_metrics = evaluate(model, bundle["test_loader"], criterion, device, config)

    metrics_payload = load_json(metrics_path)
    metrics_payload["test_acc"] = test_metrics["acc"]
    metrics_payload["test_macro_f1"] = test_metrics["macro_f1"]
    metrics_payload["test_size"] = len(bundle["test_samples"])
    save_json(metrics_path, metrics_payload)

    if os.path.exists(status_path):
        status_payload = load_json(status_path)
        status_payload.setdefault("metrics", {})
        status_payload["metrics"]["test_acc"] = test_metrics["acc"]
        status_payload["metrics"]["test_macro_f1"] = test_metrics["macro_f1"]
        status_payload["metrics"]["test_size"] = len(bundle["test_samples"])
        save_json(status_path, status_payload)

    test_predictions_df = pd.DataFrame(
        {
            "image_path": test_metrics["paths"],
            "y_true": test_metrics["targets"],
            "y_pred": test_metrics["preds"],
            "confidence": test_metrics["confidences"],
        }
    )
    test_predictions_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False, encoding="utf-8-sig")
    save_classwise_metrics(
        test_predictions_df,
        bundle["class_names"],
        bundle["train_samples"],
        os.path.join(run_dir, "test_classwise_metrics.csv"),
    )
    save_confusion_matrix_figure(
        test_metrics["targets"],
        test_metrics["preds"],
        bundle["class_names"],
        os.path.join(run_dir, "test_confusion_matrix.png"),
        "{} test confusion matrix".format(config["experiment_name"]),
    )

    return {
        "run_dir": run_dir,
        "experiment_name": config["experiment_name"],
        "test_acc": test_metrics["acc"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_size": len(bundle["test_samples"]),
    }


def main():
    args = parse_args()
    results = []
    for run_dir in iter_run_dirs(args.runs_root, args.run_dir):
        result = reevaluate_run(run_dir)
        if result is not None:
            results.append(result)
            print(
                "{experiment_name},{test_size},{test_acc:.6f},{test_macro_f1:.6f},{run_dir}".format(
                    **result
                )
            )

    print("re-evaluated_runs={}".format(len(results)))


if __name__ == "__main__":
    main()
