import timm
import torch.nn as nn
from torchvision import models


def replace_mobilenet_classifier(model, num_classes, dropout, activation):
    in_features = model.classifier[-1].in_features
    if activation == "gelu":
        act_layer = nn.GELU()
    else:
        act_layer = nn.ReLU(inplace=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 256),
        act_layer,
        nn.Dropout(p=dropout),
        nn.Linear(256, num_classes),
    )
    return model


def build_model(config):
    model_cfg = config["model"]
    model_name = model_cfg["name"].lower()
    num_classes = config["data"]["num_classes"]
    dropout = model_cfg.get("dropout", 0.2)
    activation = model_cfg.get("activation", "relu").lower()

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=model_cfg.get("pretrained", True))
        model = replace_mobilenet_classifier(model, num_classes, dropout, activation)
        return model

    if "deit" in model_name or "vit" in model_name:
        model = timm.create_model(
            model_cfg["name"],
            pretrained=model_cfg.get("pretrained", True),
            num_classes=num_classes,
            img_size=config["data"]["image_size"],
        )
        return model

    raise ValueError("Unsupported model name: {}".format(model_cfg["name"]))


def set_finetune_mode(model, config, epoch=0):
    mode = config["finetune"]["mode"]
    freeze_epochs = config["finetune"].get("freeze_epochs", 0)

    for param in model.parameters():
        param.requires_grad = True

    if mode == "full":
        return

    if mode == "head_only":
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "head"):
            for param in model.head.parameters():
                param.requires_grad = True
        return

    if mode == "last_stage":
        for param in model.parameters():
            param.requires_grad = False
        if hasattr(model, "features"):
            for param in model.features[-1].parameters():
                param.requires_grad = True
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, "blocks"):
            for param in model.blocks[-1].parameters():
                param.requires_grad = True
            if hasattr(model, "norm"):
                for param in model.norm.parameters():
                    param.requires_grad = True
            for param in model.head.parameters():
                param.requires_grad = True
        return

    if mode == "freeze_then_full":
        if epoch < freeze_epochs:
            return set_finetune_mode(model, {"finetune": {"mode": "head_only"}}, epoch=epoch)
        return

    raise ValueError("Unsupported finetune mode: {}".format(mode))
