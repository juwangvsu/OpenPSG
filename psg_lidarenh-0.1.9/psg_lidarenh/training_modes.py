from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


FINETUNE = "finetune"
FREEZE_IMAGE_BACKBONE = "freeze_image_backbone"
FREEZE_PSGTR = "freeze_psgtr"
REINITIALIZE_PSGTR_FREEZE_BACKBONE = "reinitialize_psgtr_freeze_backbone"

TRAINING_MODES = (
    FINETUNE,
    FREEZE_IMAGE_BACKBONE,
    FREEZE_PSGTR,
    REINITIALIZE_PSGTR_FREEZE_BACKBONE,
)


def image_backbone_module(model: nn.Module) -> nn.Module:
    """Return the convolutional image backbone, excluding position embeddings.

    Hugging Face DETR exposes ``model.backbone`` as a wrapper containing
    ``conv_encoder`` and ``position_embedding``. The fallback supports older
    compatible implementations where the wrapper itself is the convolutional
    backbone.
    """

    detr_model = getattr(model, "model", None)
    if detr_model is None or not hasattr(detr_model, "backbone"):
        raise AttributeError("Model does not expose model.backbone")
    backbone = detr_model.backbone
    return getattr(backbone, "conv_encoder", backbone)


def is_lidar_parameter(name: str) -> bool:
    return name.startswith("lidar_encoder.") or name.startswith(
        "lidar_query_fusion."
    )


@dataclass(frozen=True)
class TrainabilityReport:
    mode: str
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_tensors: int
    frozen_tensors: int
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "frozen_parameters": self.frozen_parameters,
            "trainable_tensors": self.trainable_tensors,
            "frozen_tensors": self.frozen_tensors,
            "trainable_names": list(self.trainable_names),
            "frozen_names": list(self.frozen_names),
        }


def configure_trainability(model: nn.Module, mode: str) -> TrainabilityReport:
    if mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode: {mode}")

    # Reset first so loading a checkpoint does not preserve stale flags from a
    # previous Python process or a manually modified model object.
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if mode in {
        FREEZE_IMAGE_BACKBONE,
        REINITIALIZE_PSGTR_FREEZE_BACKBONE,
    }:
        for parameter in image_backbone_module(model).parameters():
            parameter.requires_grad_(False)
    elif mode == FREEZE_PSGTR:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(is_lidar_parameter(name))

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_parameters = 0
    frozen_parameters = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_parameters += count
        else:
            frozen_names.append(name)
            frozen_parameters += count

    if trainable_parameters == 0:
        raise RuntimeError(f"Training mode {mode!r} leaves no trainable parameters")
    return TrainabilityReport(
        mode=mode,
        total_parameters=trainable_parameters + frozen_parameters,
        trainable_parameters=trainable_parameters,
        frozen_parameters=frozen_parameters,
        trainable_tensors=len(trainable_names),
        frozen_tensors=len(frozen_names),
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
    )


def enforce_frozen_module_modes(model: nn.Module, mode: str) -> None:
    """Keep frozen modules in evaluation mode after ``model.train()``.

    Freezing parameters alone does not stop BatchNorm running-stat updates or
    dropout. This function makes the selected freeze mode behavior literal.
    """

    if mode in {
        FREEZE_IMAGE_BACKBONE,
        REINITIALIZE_PSGTR_FREEZE_BACKBONE,
    }:
        image_backbone_module(model).eval()
    elif mode == FREEZE_PSGTR:
        model.eval()
        model.lidar_encoder.train()
        model.lidar_query_fusion.train()


def optimizer_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: float,
    backbone_learning_rate: float,
) -> list[dict[str, object]]:
    backbone_ids = {
        id(parameter)
        for parameter in image_backbone_module(model).parameters()
        if parameter.requires_grad
    }
    backbone: list[torch.Tensor] = []
    other: list[torch.Tensor] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_ids:
            backbone.append(parameter)
        else:
            other.append(parameter)

    groups: list[dict[str, object]] = []
    if other:
        groups.append(
            {
                "name": "non_backbone",
                "params": other,
                "lr": learning_rate,
            }
        )
    if backbone:
        groups.append(
            {
                "name": "image_backbone",
                "params": backbone,
                "lr": backbone_learning_rate,
            }
        )
    if not groups:
        raise RuntimeError("No trainable parameters were selected")
    return groups
