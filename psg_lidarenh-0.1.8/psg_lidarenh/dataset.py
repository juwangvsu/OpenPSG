from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def _image_id(value: Any) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def resolve_openpsg_data_root(data_root: str | Path) -> Path:
    """Return the root expected by :class:`OpenPsgDataset`.

    ``OpenPsgDataset`` resolves images below ``<root>/coco/train2017`` and
    related directories. The KITTI-360 converter writes exactly that layout,
    but users naturally pass either the conversion output root or its ``coco``
    child. Accept both forms. This also keeps the common original-COCO form
    ``/datasets/coco`` working by returning ``/datasets``.
    """

    root = Path(data_root).expanduser().resolve()
    nested_markers = (
        root / "coco" / "train2017",
        root / "coco" / "val2017",
        root / "coco" / "panoptic_train2017",
        root / "coco" / "panoptic_val2017",
    )
    if any(path.exists() for path in nested_markers):
        return root

    direct_markers = (
        root / "train2017",
        root / "val2017",
        root / "panoptic_train2017",
        root / "panoptic_val2017",
    )
    if root.name == "coco" or any(path.exists() for path in direct_markers):
        return root.parent

    return root


def load_lidar_points(path: Path, feature_count: int) -> torch.Tensor:
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "points" if "points" in archive else archive.files[0]
        array = archive[key]
    elif path.suffix == ".bin":
        array = np.fromfile(path, dtype=np.float32)
        if array.size % feature_count:
            raise ValueError(
                f"{path} has {array.size} floats, not divisible by {feature_count}"
            )
        array = array.reshape(-1, feature_count)
    else:
        raise ValueError(f"Unsupported LiDAR file format: {path.suffix}")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != feature_count:
        raise ValueError(
            f"Expected LiDAR points [P, {feature_count}], got {array.shape}"
        )
    return torch.from_numpy(array.copy())


class LidarManifestDataset(Dataset):
    """Attach one raw LiDAR scan to each OpenPSG image sample.

    Manifest format::

        {
          "point_feature_count": 4,
          "samples": {
            "9001234": {"lidar_file": "/absolute/or/relative/frame.bin"}
          }
        }
    """

    def __init__(
        self,
        base_dataset: Dataset,
        manifest_file: str | Path,
        *,
        require_lidar: bool = True,
    ) -> None:
        self.base_dataset = base_dataset
        self.manifest_file = Path(manifest_file)
        document = json.loads(self.manifest_file.read_text(encoding="utf-8"))
        self.feature_count = int(document.get("point_feature_count", 4))
        samples = document.get("samples")
        if isinstance(samples, list):
            samples = {str(record["image_id"]): record for record in samples}
        if not isinstance(samples, Mapping):
            raise ValueError("Manifest 'samples' must be an object or list")
        self.samples = {str(key): value for key, value in samples.items()}
        self.root = self.manifest_file.parent
        self.require_lidar = bool(require_lidar)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = deepcopy(self.base_dataset[index])
        target_key = "target" if "target" in item else "labels"
        target = item[target_key]
        image_id = _image_id(item.get("image_id", target["image_id"]))
        record = self.samples.get(str(image_id))
        if record is None:
            if self.require_lidar:
                raise KeyError(f"No LiDAR manifest record for image_id={image_id}")
            points = torch.empty((0, self.feature_count), dtype=torch.float32)
        else:
            path = Path(record["lidar_file"])
            if not path.is_absolute():
                path = self.root / path
            points = load_lidar_points(path, self.feature_count)
        item[target_key] = target
        item["lidar_points"] = points
        return item


class LidarSceneGraphCollator:
    def __init__(self, base_collator: Any | None = None) -> None:
        if base_collator is None:
            from psgtr_hf.dataset import PsgCollator

            base_collator = PsgCollator()
        self.base_collator = base_collator

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        lidar_points = [sample["lidar_points"] for sample in samples]
        camera_samples = [
            {key: value for key, value in sample.items() if key != "lidar_points"}
            for sample in samples
        ]
        batch = self.base_collator(camera_samples)
        batch["lidar_points"] = lidar_points
        return batch
