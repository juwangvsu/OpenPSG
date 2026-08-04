from __future__ import annotations

import torch
from torch import nn

from psg_lidarenh.training_modes import (
    FINETUNE,
    FREEZE_IMAGE_BACKBONE,
    FREEZE_PSGTR,
    REINITIALIZE_PSGTR_FREEZE_BACKBONE,
    configure_trainability,
    enforce_frozen_module_modes,
    optimizer_parameter_groups,
)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Module()
        self.model.backbone.conv_encoder = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
        )
        self.model.backbone.position_embedding = nn.Linear(4, 4)
        self.transformer = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)
        self.lidar_encoder = nn.Linear(4, 4)
        self.lidar_query_fusion = nn.Sequential(
            nn.Linear(4, 4),
            nn.Dropout(0.2),
        )


def names_with_grad(model: nn.Module) -> set[str]:
    return {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def test_full_finetune_keeps_everything_trainable() -> None:
    model = FakeModel()
    report = configure_trainability(model, FINETUNE)
    assert report.frozen_parameters == 0
    assert len(names_with_grad(model)) == len(list(model.named_parameters()))


def test_freeze_image_backbone_only_freezes_conv_encoder() -> None:
    model = FakeModel()
    configure_trainability(model, FREEZE_IMAGE_BACKBONE)
    trainable = names_with_grad(model)
    assert not any(name.startswith("model.backbone.conv_encoder") for name in trainable)
    assert "model.backbone.position_embedding.weight" in trainable
    assert "transformer.weight" in trainable
    assert "lidar_encoder.weight" in trainable

    model.train()
    enforce_frozen_module_modes(model, FREEZE_IMAGE_BACKBONE)
    assert not model.model.backbone.conv_encoder.training
    assert model.model.backbone.position_embedding.training
    assert model.lidar_encoder.training


def test_freeze_psgtr_trains_only_lidar_modules() -> None:
    model = FakeModel()
    configure_trainability(model, FREEZE_PSGTR)
    trainable = names_with_grad(model)
    assert trainable
    assert all(
        name.startswith("lidar_encoder.")
        or name.startswith("lidar_query_fusion.")
        for name in trainable
    )

    model.train()
    enforce_frozen_module_modes(model, FREEZE_PSGTR)
    assert not model.model.training
    assert not model.transformer.training
    assert not model.head.training
    assert model.lidar_encoder.training
    assert model.lidar_query_fusion.training


def test_reinitialized_mode_uses_same_freeze_policy_as_backbone_mode() -> None:
    model = FakeModel()
    configure_trainability(model, REINITIALIZE_PSGTR_FREEZE_BACKBONE)
    trainable = names_with_grad(model)
    assert not any(name.startswith("model.backbone.conv_encoder") for name in trainable)
    assert "transformer.weight" in trainable
    assert "head.weight" in trainable
    assert "lidar_encoder.weight" in trainable


def test_optimizer_groups_skip_frozen_backbone() -> None:
    model = FakeModel()
    configure_trainability(model, FREEZE_IMAGE_BACKBONE)
    groups = optimizer_parameter_groups(
        model,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
    )
    assert [group["name"] for group in groups] == ["non_backbone"]
    assert groups[0]["lr"] == 1e-4


def test_optimizer_groups_separate_trainable_backbone() -> None:
    model = FakeModel()
    configure_trainability(model, FINETUNE)
    groups = optimizer_parameter_groups(
        model,
        learning_rate=1e-4,
        backbone_learning_rate=1e-5,
    )
    assert [group["name"] for group in groups] == [
        "non_backbone",
        "image_backbone",
    ]
    assert groups[1]["lr"] == 1e-5
