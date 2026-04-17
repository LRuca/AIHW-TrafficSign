import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2, ensure_ascii=False)


def save_curves(log_df, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], label="train_loss")
    axes[0].plot(log_df["epoch"], log_df["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(log_df["epoch"], log_df["train_acc"], label="train_acc")
    axes[1].plot(log_df["epoch"], log_df["val_acc"], label="val_acc")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_lr_curve(log_df, out_path, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(log_df["epoch"], log_df["lr"], label="lr")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_figure(y_true, y_pred, class_names, out_path, title):
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, pred in zip(y_true, y_pred):
        cm[int(truth), int(pred)] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return cm


def save_classwise_metrics(predictions_df, class_names, train_samples, out_csv_path):
    train_counts = {}
    for _, label in train_samples:
        train_counts[label] = train_counts.get(label, 0) + 1

    rows = []
    for class_idx, class_name in enumerate(class_names):
        class_df = predictions_df[predictions_df["y_true"] == class_idx]
        total = len(class_df)
        correct = int((class_df["y_true"] == class_df["y_pred"]).sum())
        acc = float(correct) / float(total) if total else 0.0
        rows.append({
            "class_idx": class_idx,
            "class_name": class_name,
            "train_count": train_counts.get(class_idx, 0),
            "val_count": total,
            "class_accuracy": acc,
        })
    pd.DataFrame(rows).to_csv(out_csv_path, index=False, encoding="utf-8-sig")
