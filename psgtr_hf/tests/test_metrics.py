from __future__ import annotations

import torch

from psgtr_hf.metrics import PsgEvaluationAccumulator
from psgtr_hf.training import select_random_evaluation_indices


def test_perfect_panoptic_and_predicate_metrics() -> None:
    masks = torch.zeros((2, 8, 8), dtype=torch.bool)
    masks[0, 1:5, 1:4] = True
    masks[1, 2:7, 5:7] = True
    prediction = {
        "scores": torch.tensor([0.9]),
        "subject_scores": torch.tensor([0.95]),
        "object_scores": torch.tensor([0.94]),
        "relation_scores": torch.tensor([0.92]),
        "subject_labels": torch.tensor([0]),
        "object_labels": torch.tensor([1]),
        "relation_labels": torch.tensor([0]),
        "subject_masks": masks[:1].float(),
        "object_masks": masks[1:].float(),
    }
    target = {
        "class_labels": torch.tensor([0, 1]),
        "masks": masks.float(),
        "relations": torch.tensor([[0, 1, 0]]),
        "size": torch.tensor([8, 8]),
    }
    accumulator = PsgEvaluationAccumulator(
        num_object_classes=2,
        num_thing_classes=2,
        num_predicate_classes=1,
        recall_ks=(1, 20),
    )
    accumulator.update(
        prediction,
        target,
        entity_score_threshold=0.25,
        mask_threshold=0.5,
        iou_threshold=0.5,
        thing_nms_threshold=0.8,
    )
    result = accumulator.compute(
        object_classes=("person", "bicycle"),
        predicate_classes=("riding",),
    )
    metrics = result["metrics"]
    assert metrics["evaluated_images"] == 1
    assert metrics["pq"] == 1.0
    assert metrics["sq"] == 1.0
    assert metrics["rq"] == 1.0
    assert metrics["predicate_recall_at_1"] == 1.0
    assert metrics["predicate_mean_recall_at_20"] == 1.0


def test_random_evaluation_indices_are_fixed_and_unique() -> None:
    first = select_random_evaluation_indices(1000, 200, 17)
    second = select_random_evaluation_indices(1000, 200, 17)
    assert first == second
    assert len(first) == 200
    assert len(set(first)) == 200
    assert select_random_evaluation_indices(3, 200, 17) == [0, 1, 2]
