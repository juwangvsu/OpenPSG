from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from psgtr_hf.inference import (
    combine_panels,
    render_ground_truth,
    render_predictions,
    select_dataset_indices,
)


def test_select_dataset_indices() -> None:
    dataset = SimpleNamespace(
        samples=[{"image_id": 10}, {"image_id": 20}, {"image_id": 30}],
        split="train",
        __len__=lambda self: 3,
    )

    class Dataset:
        samples = dataset.samples
        split = "train"

        def __len__(self) -> int:
            return 3

    actual = Dataset()
    assert select_dataset_indices(
        actual,
        random_count=None,
        indices=["2,0", "2"],
        image_ids=None,
        seed=1,
    ) == [2, 0]
    assert select_dataset_indices(
        actual,
        random_count=None,
        indices=None,
        image_ids=["30", "10"],
        seed=1,
    ) == [2, 0]
    assert len(
        select_dataset_indices(
            actual,
            random_count=2,
            indices=None,
            image_ids=None,
            seed=1,
        )
    ) == 2


def test_render_prediction_and_ground_truth(tmp_path: Path) -> None:
    image = Image.new("RGB", (32, 24), "white")
    masks = torch.zeros((2, 24, 32), dtype=torch.bool)
    masks[0, 2:14, 2:14] = True
    masks[1, 8:22, 18:30] = True
    raw = {
        "image": image,
        "image_id": 99,
        "class_labels": torch.tensor([0, 1]),
        "masks": masks,
        "relations": torch.tensor([[0, 1, 0]]),
        "segment_ids": torch.tensor([1, 2]),
    }
    prediction = {
        "scores": torch.tensor([0.8]),
        "subject_scores": torch.tensor([0.9]),
        "object_scores": torch.tensor([0.8]),
        "relation_scores": torch.tensor([0.7]),
        "subject_labels": torch.tensor([0]),
        "object_labels": torch.tensor([1]),
        "relation_labels": torch.tensor([0]),
        "subject_boxes": torch.tensor([[2.0, 2.0, 14.0, 14.0]]),
        "object_boxes": torch.tensor([[18.0, 8.0, 30.0, 22.0]]),
        "subject_masks": masks[:1],
        "object_masks": masks[1:],
        "query_indices": torch.tensor([4]),
    }
    gt_image, gt_lines, gt_json = render_ground_truth(
        raw,
        ["person", "cup"],
        ["holding"],
        10,
    )
    pred_image, pred_lines, pred_json = render_predictions(
        image,
        prediction,
        ["person", "cup"],
        ["holding"],
    )
    combined = combine_panels(gt_image, pred_image)
    output = tmp_path / "comparison.png"
    combined.save(output)
    assert output.is_file()
    assert "holding" in gt_lines[0]
    assert "score=0.800" in pred_lines[0]
    assert gt_json["relations"][0]["subject_label"] == "person"
    assert pred_json[0]["query_index"] == 4
