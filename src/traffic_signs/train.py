import math
import os
import random
import signal
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from traffic_signs.analysis import save_classwise_metrics, save_confusion_matrix_figure, save_curves, save_json, save_lr_curve
from traffic_signs.data import build_dataloaders, compute_class_weights
from traffic_signs.models import build_model, set_finetune_mode


class TrainingInterrupted(Exception):
    pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_accuracy(targets, preds):
    if not targets:
        return 0.0
    correct = 0
    for target, pred in zip(targets, preds):
        if int(target) == int(pred):
            correct += 1
    return float(correct) / float(len(targets))


def compute_macro_f1(targets, preds, num_classes):
    eps = 1e-12
    f1_scores = []
    for class_idx in range(num_classes):
        tp = 0
        fp = 0
        fn = 0
        for target, pred in zip(targets, preds):
            target = int(target)
            pred = int(pred)
            if pred == class_idx and target == class_idx:
                tp += 1
            elif pred == class_idx and target != class_idx:
                fp += 1
            elif pred != class_idx and target == class_idx:
                fn += 1
        precision = tp / float(tp + fp + eps)
        recall = tp / float(tp + fn + eps)
        f1_scores.append(2.0 * precision * recall / float(precision + recall + eps))
    return float(sum(f1_scores)) / float(max(1, num_classes))


def make_output_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "{}_seed{}".format(config["experiment_name"], config["train"]["seed"])
    run_dir = os.path.join(config["output_root"], "{}_{}".format(timestamp, run_name))
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_live_status(run_dir, payload):
    save_json(os.path.join(run_dir, "status.json"), payload)


def write_resume_checkpoint(
    run_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_acc,
    best_epoch,
    config,
):
    checkpoint_payload = {
        "epoch": epoch,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "experiment_name": config["experiment_name"],
        "train_seed": config["train"]["seed"],
    }
    torch.save(checkpoint_payload, os.path.join(run_dir, "resume_checkpoint.pt"))


def flush_epoch_logs(logs, run_dir, experiment_name):
    log_df = pd.DataFrame(logs)
    log_df.to_csv(os.path.join(run_dir, "train_log.csv"), index=False, encoding="utf-8-sig")
    if not log_df.empty:
        save_curves(log_df, os.path.join(run_dir, "curves.png"), experiment_name)
        save_lr_curve(log_df, os.path.join(run_dir, "lr_curve.png"), "Learning Rate")


def save_config_snapshot(config, run_dir):
    import yaml

    with open(os.path.join(run_dir, "resolved_config.yaml"), "w", encoding="utf-8") as file_obj:
        yaml.safe_dump(config, file_obj, sort_keys=False, allow_unicode=True)


def resolve_device(device_name):
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_optimizer(model, config):
    opt_cfg = config["optimizer"]
    params = [p for p in model.parameters() if p.requires_grad]
    name = opt_cfg["name"].lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    if name == "momentum":
        return torch.optim.SGD(params, lr=opt_cfg["lr"], momentum=opt_cfg["momentum"], weight_decay=opt_cfg["weight_decay"])
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=opt_cfg["lr"], momentum=opt_cfg["momentum"], weight_decay=opt_cfg["weight_decay"])
    if name == "adam":
        return torch.optim.Adam(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    raise ValueError("Unsupported optimizer: {}".format(opt_cfg["name"]))


def create_scheduler(optimizer, config):
    sched_cfg = config["scheduler"]
    epochs = config["train"]["epochs"]
    name = sched_cfg["name"].lower()
    min_lr = sched_cfg.get("min_lr", 1e-6)

    if name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=sched_cfg.get("gamma", 0.97))
    if name == "polynomial":
        power = sched_cfg.get("power", 2.0)
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: max((1.0 - float(epoch) / float(max(1, epochs))) ** power, min_lr / max(optimizer.defaults["lr"], 1e-12)),
        )
    raise ValueError("Unsupported scheduler: {}".format(sched_cfg["name"]))


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, config):
    model.train()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    amp_enabled = bool(config["train"].get("amp", True)) and device.type == "cuda"
    grad_clip = float(config["train"].get("grad_clip_norm", 0.0))

    max_train_batches = config["train"].get("max_train_batches")
    for step_idx, (images, targets, _) in enumerate(loader):
        if max_train_batches is not None and step_idx >= int(max_train_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    seen_samples = max(1, len(all_targets))
    epoch_loss = total_loss / seen_samples
    epoch_acc = compute_accuracy(all_targets, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_preds = []
    all_probs = []
    all_paths = []

    max_val_batches = config["train"].get("max_val_batches")
    for step_idx, (images, targets, paths) in enumerate(loader):
        if max_val_batches is not None and step_idx >= int(max_val_batches):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * images.size(0)
        all_targets.extend(targets.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())
        all_probs.extend(probs.max(dim=1)[0].detach().cpu().tolist())
        all_paths.extend(list(paths))

    seen_samples = max(1, len(all_targets))
    metrics = {
        "loss": total_loss / seen_samples,
        "acc": compute_accuracy(all_targets, all_preds),
        "macro_f1": compute_macro_f1(all_targets, all_preds, num_classes=config["data"]["num_classes"]),
        "targets": all_targets,
        "preds": all_preds,
        "confidences": all_probs,
        "paths": all_paths,
    }
    return metrics


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_resume_checkpoint(config):
    checkpoint_path = config.get("model", {}).get("checkpoint_path")
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None
    return torch.load(checkpoint_path, map_location="cpu")


def _signal_to_name(sig):
    if sig == getattr(signal, "SIGTERM", None):
        return "SIGTERM"
    if sig == getattr(signal, "SIGINT", None):
        return "SIGINT"
    if sig == getattr(signal, "SIGBREAK", None):
        return "SIGBREAK"
    return str(sig)


def run_experiment(config, project_root):
    set_seed(config["train"]["seed"])
    device = resolve_device(config["train"]["device"])
    runtime_cfg = config.setdefault("runtime", {})
    run_dir = runtime_cfg.get("resume_run_dir") or make_output_dir(config)
    os.makedirs(run_dir, exist_ok=True)
    save_config_snapshot(config, run_dir)
    write_live_status(
        run_dir,
        {
            "experiment_name": config["experiment_name"],
            "status": "initializing",
            "epoch": 0,
            "total_epochs": config["train"]["epochs"],
            "best_val_acc": None,
            "run_dir": run_dir,
        },
    )

    bundle = build_dataloaders(config)
    model = build_model(config).to(device)
    set_finetune_mode(model, config, epoch=0)

    class_weights = None
    if config["train"].get("class_weighting", False):
        class_weights = compute_class_weights(bundle["train_samples"], config["data"]["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(config["train"].get("label_smoothing", 0.0)),
    )

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    scaler = torch.cuda.amp.GradScaler(enabled=(config["train"].get("amp", True) and device.type == "cuda"))
    optimizer_rebuilt = False

    log_path = os.path.join(run_dir, "train_log.csv")
    if os.path.exists(log_path):
        logs = pd.read_csv(log_path, encoding="utf-8-sig").to_dict("records")
    else:
        logs = []
    best_val_acc = -1.0
    best_epoch = -1
    best_metrics = None
    best_state_path = os.path.join(run_dir, "best.pt")
    start_time = time.time()
    start_epoch = 0
    resume_payload = load_resume_checkpoint(config)
    interrupted = {"requested": False, "signal_name": None}

    def handle_interrupt(sig, frame):
        interrupted["requested"] = True
        interrupted["signal_name"] = _signal_to_name(sig)
        raise TrainingInterrupted()

    previous_handlers = {}
    signal_map = [sig for sig in [getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None), getattr(signal, "SIGBREAK", None)] if sig is not None]
    for sig in signal_map:
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_interrupt)

    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        scaler_state_dict = resume_payload.get("scaler_state_dict") or {}
        if scaler_state_dict:
            scaler.load_state_dict(scaler_state_dict)
        start_epoch = int(resume_payload.get("epoch", 0))
        best_val_acc = float(resume_payload.get("best_val_acc", -1.0))
        best_epoch = int(resume_payload.get("best_epoch", -1))
        if config["finetune"]["mode"] == "freeze_then_full" and start_epoch > config["finetune"].get("freeze_epochs", 0):
            optimizer_rebuilt = True
        write_live_status(
            run_dir,
            {
                "experiment_name": config["experiment_name"],
                "status": "resumed",
                "epoch": start_epoch,
                "total_epochs": config["train"]["epochs"],
                "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
                "best_epoch": best_epoch if best_epoch >= 0 else None,
                "run_dir": run_dir,
                "resumed_from_checkpoint": config["model"].get("checkpoint_path"),
            },
        )

    try:
        for epoch in range(start_epoch, config["train"]["epochs"]):
            if config["finetune"]["mode"] == "freeze_then_full":
                set_finetune_mode(model, config, epoch=epoch)
                if epoch == config["finetune"].get("freeze_epochs", 0) and not optimizer_rebuilt:
                    optimizer = create_optimizer(model, config)
                    scheduler = create_scheduler(optimizer, config)
                    optimizer_rebuilt = True

            epoch_start = time.time()
            train_loss, train_acc = train_one_epoch(model, bundle["train_loader"], criterion, optimizer, device, scaler, config)
            val_metrics = evaluate(model, bundle["val_loader"], criterion, device, config)
            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            log_row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": current_lr,
                "epoch_time_sec": time.time() - epoch_start,
            }
            logs.append(log_row)
            print(log_row)
            flush_epoch_logs(logs, run_dir, config["experiment_name"])

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                best_epoch = epoch + 1
                best_metrics = val_metrics
                torch.save(model.state_dict(), best_state_path)

            write_resume_checkpoint(
                run_dir,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch + 1,
                best_val_acc,
                best_epoch,
                config,
            )
            if config["train"].get("save_last", True):
                torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

            write_live_status(
                run_dir,
                {
                    "experiment_name": config["experiment_name"],
                    "status": "running",
                    "epoch": epoch + 1,
                    "total_epochs": config["train"]["epochs"],
                    "latest": log_row,
                    "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
                    "best_epoch": best_epoch if best_epoch >= 0 else None,
                    "run_dir": run_dir,
                    "resume_checkpoint_path": os.path.join(run_dir, "resume_checkpoint.pt"),
                },
            )

        if config["train"].get("save_last", True):
            torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))

        log_df = pd.DataFrame(logs)
        flush_epoch_logs(logs, run_dir, config["experiment_name"])

        metrics_payload = {
            "experiment_name": config["experiment_name"],
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_metrics["macro_f1"] if best_metrics else None,
            "trainable_params": count_trainable_params(model),
            "total_runtime_sec": time.time() - start_time,
            "device": str(device),
            "train_size": len(bundle["train_samples"]),
            "val_size": len(bundle["val_samples"]),
            "test_size": len(bundle["test_samples"]),
        }

        if best_metrics is None and os.path.exists(best_state_path):
            model.load_state_dict(torch.load(best_state_path, map_location=device))
            best_metrics = evaluate(model, bundle["val_loader"], criterion, device, config)
            metrics_payload["best_val_macro_f1"] = best_metrics["macro_f1"]

        test_metrics = None
        if bundle.get("test_loader") is not None and os.path.exists(best_state_path):
            model.load_state_dict(torch.load(best_state_path, map_location=device))
            test_metrics = evaluate(model, bundle["test_loader"], criterion, device, config)
            metrics_payload["test_acc"] = test_metrics["acc"]
            metrics_payload["test_macro_f1"] = test_metrics["macro_f1"]

        save_json(os.path.join(run_dir, "metrics.json"), metrics_payload)

        if config["runtime"].get("save_curves", True):
            save_curves(log_df, os.path.join(run_dir, "curves.png"), config["experiment_name"])
            save_lr_curve(log_df, os.path.join(run_dir, "lr_curve.png"), "Learning Rate")

        predictions_df = pd.DataFrame({
            "image_path": best_metrics["paths"],
            "y_true": best_metrics["targets"],
            "y_pred": best_metrics["preds"],
            "confidence": best_metrics["confidences"],
        })
        if config["runtime"].get("save_predictions", True):
            predictions_df.to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False, encoding="utf-8-sig")

        if config["runtime"].get("save_confusion_matrix", True):
            save_confusion_matrix_figure(
                best_metrics["targets"],
                best_metrics["preds"],
                bundle["class_names"],
                os.path.join(run_dir, "confusion_matrix.png"),
                "{} confusion matrix".format(config["experiment_name"]),
            )

        if config["runtime"].get("save_classwise_metrics", True):
            save_classwise_metrics(
                predictions_df,
                bundle["class_names"],
                bundle["train_samples"],
                os.path.join(run_dir, "classwise_metrics.csv"),
            )

        if test_metrics is not None:
            test_predictions_df = pd.DataFrame({
                "image_path": test_metrics["paths"],
                "y_true": test_metrics["targets"],
                "y_pred": test_metrics["preds"],
                "confidence": test_metrics["confidences"],
            })
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

        resume_checkpoint_path = os.path.join(run_dir, "resume_checkpoint.pt")
        if os.path.exists(resume_checkpoint_path):
            os.remove(resume_checkpoint_path)

        write_live_status(
            run_dir,
            {
                "experiment_name": config["experiment_name"],
                "status": "completed",
                "epoch": config["train"]["epochs"],
                "total_epochs": config["train"]["epochs"],
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "run_dir": run_dir,
                "metrics": metrics_payload,
            },
        )

        print("Run artifacts saved to {}".format(run_dir))
    except TrainingInterrupted:
        interrupted_epoch = logs[-1]["epoch"] if logs else start_epoch
        if config["train"].get("save_last", True):
            torch.save(model.state_dict(), os.path.join(run_dir, "last.pt"))
        write_resume_checkpoint(
            run_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            interrupted_epoch,
            best_val_acc,
            best_epoch,
            config,
        )
        write_live_status(
            run_dir,
            {
                "experiment_name": config["experiment_name"],
                "status": "stopped",
                "epoch": interrupted_epoch,
                "total_epochs": config["train"]["epochs"],
                "best_val_acc": best_val_acc if best_val_acc >= 0 else None,
                "best_epoch": best_epoch if best_epoch >= 0 else None,
                "run_dir": run_dir,
                "resume_checkpoint_path": os.path.join(run_dir, "resume_checkpoint.pt"),
                "stop_reason": interrupted["signal_name"] or "external_stop",
            },
        )
        print("Training interrupted. Resume checkpoint saved to {}".format(os.path.join(run_dir, "resume_checkpoint.pt")))
        raise
    finally:
        for sig, previous_handler in previous_handlers.items():
            signal.signal(sig, previous_handler)
