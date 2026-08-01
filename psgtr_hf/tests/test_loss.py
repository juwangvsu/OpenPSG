from types import SimpleNamespace

import torch

from psgtr_hf.loss import PsgtrLoss


def config() -> SimpleNamespace:
    return SimpleNamespace(
        num_object_labels=4,
        num_relation_labels=3,
        object_class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        relation_class_cost=2.0,
        subject_class_loss_coefficient=1.0,
        object_class_loss_coefficient=1.0,
        relation_class_loss_coefficient=2.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        dice_loss_coefficient=1.0,
        object_eos_coefficient=0.02,
        no_relation_coefficient=0.02,
        train_object_background=False,
    )


def outputs() -> dict[str, torch.Tensor]:
    return {
        "subject_logits": torch.randn(1, 5, 5, requires_grad=True),
        "object_logits": torch.randn(1, 5, 5, requires_grad=True),
        "relation_logits": torch.randn(1, 5, 4, requires_grad=True),
        "subject_boxes": torch.rand(1, 5, 4, requires_grad=True),
        "object_boxes": torch.rand(1, 5, 4, requires_grad=True),
        "subject_masks": torch.randn(1, 5, 8, 8, requires_grad=True),
        "object_masks": torch.randn(1, 5, 8, 8, requires_grad=True),
    }


def test_loss_backward() -> None:
    predictions = outputs()
    labels = [
        {
            "class_labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]
            ),
            "masks": torch.stack(
                [torch.eye(8), torch.flip(torch.eye(8), [1])]
            ),
            "relations": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
    ]

    loss, loss_dict = PsgtrLoss(config())(predictions, labels)
    assert torch.isfinite(loss)
    assert "loss_relation_ce" in loss_dict
    loss.backward()
    assert all(value.grad is not None for value in predictions.values())


def test_empty_relation_image() -> None:
    predictions = outputs()
    labels = [
        {
            "class_labels": torch.empty(0, dtype=torch.long),
            "boxes": torch.empty(0, 4),
            "masks": torch.empty(0, 8, 8),
            "relations": torch.empty(0, 3, dtype=torch.long),
        }
    ]

    loss, _ = PsgtrLoss(config())(predictions, labels)
    assert torch.isfinite(loss)
    loss.backward()
