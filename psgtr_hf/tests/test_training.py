from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from psgtr_hf.training import evaluate, make_optimizer, train_one_epoch


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.backbone = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)

    def forward(self, pixel_values, pixel_mask, labels):
        del pixel_mask, labels
        value = pixel_values.mean(dim=(1, 2, 3), keepdim=True)
        prediction = self.head(self.model.backbone(value.flatten(1)))
        loss = prediction.square().mean()
        return SimpleNamespace(loss=loss, loss_dict={"loss_fake": loss})


def batches() -> list[dict]:
    return [
        {
            "pixel_values": torch.full((1, 3, 2, 2), float(index + 1)),
            "pixel_mask": torch.ones((1, 2, 2), dtype=torch.long),
            "labels": [
                {
                    "class_labels": torch.tensor([0]),
                    "boxes": torch.tensor([[0.5, 0.5, 1.0, 1.0]]),
                    "masks": torch.ones((1, 2, 2)),
                    "relations": torch.empty((0, 3), dtype=torch.long),
                }
            ],
        }
        for index in range(3)
    ]


def test_training_and_evaluation_loop() -> None:
    model = FakeModel()
    optimizer = make_optimizer(model, lr=1e-3, backbone_lr=1e-4, weight_decay=0)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[1]["lr"] == 1e-4
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    metrics, global_step = train_one_epoch(
        model,
        batches(),
        optimizer,
        scaler,
        torch.device("cpu"),
        epoch=1,
        global_step=0,
        accumulation_steps=2,
        max_grad_norm=0.1,
        amp=False,
        log_every=10,
    )
    assert global_step == 2
    assert metrics["loss"] >= 0
    validation = evaluate(model, batches(), torch.device("cpu"), amp=False)
    assert validation["loss"] >= 0
