from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from psgtr_hf.dataset import (
    OpenPsgDataset,
    PsgCollator,
    PsgImageTransforms,
    build_openpsg_dataloaders,
    rgb_to_segment_id,
)


def encode_id_map(ids: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            ids % 256,
            (ids // 256) % 256,
            (ids // (256 * 256)) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def make_dataset(root: Path) -> Path:
    image_root = root / "coco" / "train2017"
    panoptic_root = root / "coco" / "panoptic_train2017"
    image_root.mkdir(parents=True)
    panoptic_root.mkdir(parents=True)
    data = []
    for image_id in (1, 2):
        filename = f"{image_id:012d}.jpg"
        panoptic_filename = f"{image_id:012d}.png"
        image = np.zeros((8, 10, 3), dtype=np.uint8)
        image[:, :5, 0] = 255
        image[:, 5:, 1] = 255
        Image.fromarray(image).save(image_root / filename)
        ids = np.ones((8, 10), dtype=np.int64)
        ids[:, 5:] = 2
        Image.fromarray(encode_id_map(ids)).save(panoptic_root / panoptic_filename)
        data.append(
            {
                "file_name": filename,
                "pan_seg_file_name": panoptic_filename,
                "height": 8,
                "width": 10,
                "image_id": image_id,
                "segments_info": [
                    {"id": 1, "category_id": 0, "iscrowd": 0, "isthing": True},
                    {"id": 2, "category_id": 1, "iscrowd": 0, "isthing": True},
                ],
                "annotations": [
                    {"bbox": [0, 0, 5, 8], "category_id": 0},
                    {"bbox": [5, 0, 10, 8], "category_id": 1},
                ],
                "relations": [[0, 1, 0], [0, 1, 1], [0, 1, 1]],
            }
        )
    annotation = root / "psg" / "psg_train_val.json"
    annotation.parent.mkdir(parents=True)
    annotation.write_text(
        json.dumps(
            {
                "thing_classes": ["left", "right"],
                "stuff_classes": ["background"],
                "predicate_classes": ["beside", "looking at"],
                "test_image_ids": [2],
                "data": data,
            }
        )
    )
    return annotation


def fixed_transform(training: bool) -> PsgImageTransforms:
    return PsgImageTransforms(
        training=training,
        min_size_choices=(8,),
        max_size=10,
        flip_probability=0,
        crop_probability=0,
    )


def test_rgb_to_segment_id() -> None:
    ids = np.array([[1, 256, 65536]], dtype=np.int64)
    decoded = rgb_to_segment_id(Image.fromarray(encode_id_map(ids)))
    assert torch.equal(decoded, torch.from_numpy(ids))


def test_openpsg_split_targets_and_collation(tmp_path: Path) -> None:
    annotation = make_dataset(tmp_path)
    train = OpenPsgDataset(
        annotation,
        tmp_path,
        split="train",
        transforms=fixed_transform(True),
    )
    validation = OpenPsgDataset(
        annotation,
        tmp_path,
        split="validation",
        transforms=fixed_transform(False),
    )
    assert len(train) == 1
    assert len(validation) == 1
    assert train.metadata.object_classes == ("left", "right", "background")

    train_sample = train[0]
    target = train_sample["target"]
    assert target["relations"].shape == (1, 3)
    assert target["relations"][0, :2].tolist() == [0, 1]
    assert target["relations"][0, 2].item() in {0, 1}
    assert torch.allclose(
        target["boxes"],
        torch.tensor([[0.25, 0.5, 0.5, 1.0], [0.75, 0.5, 0.5, 1.0]]),
    )

    validation_target = validation[0]["target"]
    assert validation_target["relations"].tolist() == [[0, 1, 0], [0, 1, 1]]
    batch = PsgCollator()([train_sample, validation[0]])
    assert batch["pixel_values"].shape == (2, 3, 8, 10)
    assert batch["pixel_mask"].sum().item() == 160
    assert batch["labels"][0]["masks"].shape == (2, 8, 10)


def test_build_dataloaders(tmp_path: Path) -> None:
    annotation = make_dataset(tmp_path)
    train, validation = build_openpsg_dataloaders(
        annotation,
        tmp_path,
        batch_size=1,
        num_workers=0,
        train_min_sizes=(8,),
        validation_min_size=8,
        max_size=10,
        crop_probability=0,
        flip_probability=0,
    )
    assert next(iter(train))["pixel_values"].shape == (1, 3, 8, 10)
    assert next(iter(validation))["labels"][0]["relations"].shape == (2, 3)
