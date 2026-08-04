from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import torch


class Metadata:
    thing_classes = ("person", "car")
    stuff_classes = ()
    object_classes = thing_classes
    predicate_classes = ("beside",)


class InferenceOnlyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_labels = "unset"

    def forward(self, pixel_values, pixel_mask, lidar_points, labels=None):
        del pixel_values, pixel_mask, lidar_points
        self.seen_labels = labels
        if labels is not None:
            raise AssertionError("evaluation must not run the training criterion")
        return SimpleNamespace(loss=None, loss_dict=None)

    def post_process_triplets(self, outputs, target_sizes, score_threshold, top_k, mask_threshold):
        del outputs, target_sizes, score_threshold, top_k, mask_threshold
        return [{}]


def test_evaluation_does_not_compute_training_loss(monkeypatch) -> None:
    metrics_module = types.ModuleType("psgtr_hf.metrics")
    psgtr_module = types.ModuleType("psgtr_hf")

    class Accumulator:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.image_count = torch.zeros(1, dtype=torch.float64)

        def distributed_reduce(self, device):
            del device

        def compute(self, **kwargs):
            del kwargs
            return {"metrics": {"evaluated_images": int(self.image_count.item())}}

    metrics_module.PsgEvaluationAccumulator = Accumulator
    psgtr_module.metrics = metrics_module
    monkeypatch.setitem(sys.modules, "psgtr_hf", psgtr_module)
    monkeypatch.setitem(sys.modules, "psgtr_hf.metrics", metrics_module)
    sys.modules.pop("psg_lidarenh.evaluation", None)
    evaluation = importlib.import_module("psg_lidarenh.evaluation")

    def update(accumulator, prediction, target, **kwargs):
        del prediction, target, kwargs
        accumulator.image_count += 1

    monkeypatch.setattr(evaluation, "_fast_accumulator_update", update)
    target_masks = torch.zeros((2, 8, 8), dtype=torch.float32)
    loader = [{
        "pixel_values": torch.zeros((1, 3, 8, 8)),
        "pixel_mask": torch.ones((1, 8, 8), dtype=torch.long),
        "lidar_points": [torch.zeros((1, 4))],
        "labels": [{
            "class_labels": torch.tensor([0, 1]),
            "boxes": torch.zeros((2, 4)),
            "masks": target_masks,
            "relations": torch.tensor([[0, 1, 0]]),
            "segment_ids": torch.tensor([1, 2]),
            "size": torch.tensor([8, 8]),
            "original_size": torch.tensor([8, 8]),
            "image_id": torch.tensor(1),
        }],
    }]
    model = InferenceOnlyModel()
    result = evaluation.evaluate_model(
        model,
        loader,
        torch.device("cpu"),
        Metadata(),
        amp=False,
        reduce_across_processes=False,
    )
    assert model.seen_labels is None
    assert result["evaluation_implementation"]["training_loss_computed"] is False
    assert "loss" not in result["metrics"]
