from __future__ import annotations

from types import SimpleNamespace

import torch
from PIL import Image

from psg_lidarenh.inference import (
    _ground_truth_payload,
    _prediction_payload,
    render_comparison,
    select_indices,
)


class Dataset:
    def __init__(self) -> None:
        self.base_dataset = SimpleNamespace(
            samples=[
                {"image_id": 10},
                {"image_id": 20},
                {"image_id": 30},
                {"image_id": 40},
            ]
        )

    def __len__(self) -> int:
        return 4


def target() -> dict[str, torch.Tensor]:
    masks = torch.zeros((2, 16, 24), dtype=torch.bool)
    masks[0, 2:10, 2:9] = True
    masks[1, 5:14, 13:21] = True
    return {
        "class_labels": torch.tensor([0, 1]),
        "masks": masks,
        "relations": torch.tensor([[0, 1, 0]]),
        "image_id": torch.tensor(10),
        "size": torch.tensor([16, 24]),
    }


def prediction() -> dict[str, torch.Tensor]:
    masks = target()["masks"].float()
    return {
        "scores": torch.tensor([0.9]),
        "subject_scores": torch.tensor([0.95]),
        "object_scores": torch.tensor([0.94]),
        "relation_scores": torch.tensor([0.92]),
        "subject_labels": torch.tensor([0]),
        "object_labels": torch.tensor([1]),
        "relation_labels": torch.tensor([0]),
        "subject_masks": masks[:1],
        "object_masks": masks[1:],
    }


def test_select_indices_by_index_image_id_and_random() -> None:
    dataset = Dataset()
    assert select_indices(
        dataset,
        random_count=None,
        indices=[3, 1],
        image_ids=None,
        seed=1,
    ) == [3, 1]
    assert select_indices(
        dataset,
        random_count=None,
        indices=None,
        image_ids=[30, 10],
        seed=1,
    ) == [2, 0]
    first = select_indices(
        dataset,
        random_count=3,
        indices=None,
        image_ids=None,
        seed=7,
    )
    second = select_indices(
        dataset,
        random_count=3,
        indices=None,
        image_ids=None,
        seed=7,
    )
    assert first == second
    assert len(first) == 3


def test_visualization_and_json_payloads(tmp_path) -> None:
    object_classes = ("person", "bicycle")
    predicate_classes = ("riding",)
    image = Image.new("RGB", (24, 16), "gray")
    comparison = render_comparison(
        image,
        target(),
        prediction(),
        object_classes,
        predicate_classes,
        max_relations=10,
    )
    output = tmp_path / "comparison.png"
    comparison.save(output)
    assert output.is_file()
    assert comparison.width == 48
    assert comparison.height > 16

    ground_truth = _ground_truth_payload(
        target(), object_classes, predicate_classes
    )
    predicted = _prediction_payload(
        prediction(), object_classes, predicate_classes
    )
    assert ground_truth["relations"][0]["predicate"] == "riding"
    assert predicted["relations"][0]["subject"]["category"] == "person"
    assert predicted["relations"][0]["object"]["category"] == "bicycle"
