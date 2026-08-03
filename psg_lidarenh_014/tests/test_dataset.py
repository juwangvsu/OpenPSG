import json

import numpy as np
import torch
from torch.utils.data import Dataset

from psg_lidarenh.dataset import (
    LidarManifestDataset,
    resolve_openpsg_data_root,
)


class BaseDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return {
            "image_id": 7,
            "pixel_values": torch.zeros(3, 4, 4),
            "target": {"image_id": torch.tensor(7)},
        }


def test_manifest_loads_raw_scan_without_3d_annotations(tmp_path) -> None:
    points = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    points.tofile(tmp_path / "scan.bin")
    manifest = {
        "point_feature_count": 4,
        "samples": {"7": {"lidar_file": "scan.bin"}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    sample = LidarManifestDataset(BaseDataset(), path)[0]
    assert sample["lidar_points"].shape == (1, 4)
    assert "boxes_3d" not in sample["target"]


def test_data_root_accepts_conversion_root_or_coco_child(tmp_path) -> None:
    conversion_root = tmp_path / "converted"
    coco_root = conversion_root / "coco"
    (coco_root / "train2017").mkdir(parents=True)
    (coco_root / "panoptic_train2017").mkdir()

    assert resolve_openpsg_data_root(conversion_root) == conversion_root.resolve()
    assert resolve_openpsg_data_root(coco_root) == conversion_root.resolve()


def test_data_root_accepts_original_coco_directory(tmp_path) -> None:
    coco_root = tmp_path / "coco"
    (coco_root / "train2017").mkdir(parents=True)
    assert resolve_openpsg_data_root(coco_root) == tmp_path.resolve()
