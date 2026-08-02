from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

Split = Literal["train", "validation", "all"]


@dataclass(frozen=True)
class OpenPsgMetadata:
    thing_classes: tuple[str, ...]
    stuff_classes: tuple[str, ...]
    predicate_classes: tuple[str, ...]

    @property
    def object_classes(self) -> tuple[str, ...]:
        return self.thing_classes + self.stuff_classes

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "thing_classes": list(self.thing_classes),
            "stuff_classes": list(self.stuff_classes),
            "object_classes": list(self.object_classes),
            "predicate_classes": list(self.predicate_classes),
        }


def rgb_to_segment_id(image: Image.Image) -> torch.Tensor:
    """Decode a COCO panoptic RGB PNG into integer segment IDs."""

    array = np.asarray(image)
    if array.ndim == 2:
        return torch.from_numpy(np.array(array, dtype=np.int64, copy=True))
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"Unsupported panoptic image shape: {array.shape}")
    array = np.array(array[..., :3], dtype=np.int64, copy=True)
    ids = array[..., 0] + 256 * array[..., 1] + 256 * 256 * array[..., 2]
    return torch.from_numpy(ids)


def _resize_size(height: int, width: int, short_side: int, max_size: int) -> tuple[int, int]:
    scale = short_side / min(height, width)
    if max(height, width) * scale > max_size:
        scale = max_size / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize(
    image: Image.Image,
    masks: torch.Tensor,
    short_side: int,
    max_size: int,
) -> tuple[Image.Image, torch.Tensor]:
    new_height, new_width = _resize_size(image.height, image.width, short_side, max_size)
    if (new_height, new_width) == (image.height, image.width):
        return image, masks
    image = image.resize((new_width, new_height), resample=Image.Resampling.BILINEAR)
    masks = F.interpolate(
        masks[:, None].float(),
        size=(new_height, new_width),
        mode="nearest",
    )[:, 0].to(torch.bool)
    return image, masks


def _filter_empty_entities(
    class_labels: torch.Tensor,
    masks: torch.Tensor,
    relations: torch.Tensor,
    segment_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    keep = masks.flatten(1).any(1)
    if keep.all():
        return class_labels, masks, relations, segment_ids

    remap = torch.full((keep.numel(),), -1, dtype=torch.long)
    remap[keep] = torch.arange(int(keep.sum()), dtype=torch.long)
    if relations.numel():
        relation_keep = keep[relations[:, 0]] & keep[relations[:, 1]]
        relations = relations[relation_keep].clone()
        if relations.numel():
            relations[:, :2] = remap[relations[:, :2]]
        else:
            relations = torch.empty((0, 3), dtype=torch.long)
    return class_labels[keep], masks[keep], relations, segment_ids[keep]


def _boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() == 0:
        return torch.empty((0, 4), dtype=torch.float32)
    height, width = masks.shape[-2:]
    boxes = []
    for mask in masks:
        y, x = torch.where(mask)
        if x.numel() == 0:
            raise ValueError("Cannot derive a box from an empty mask")
        x0 = x.min().float()
        y0 = y.min().float()
        x1 = x.max().float() + 1
        y1 = y.max().float() + 1
        boxes.append(
            torch.stack(
                (
                    (x0 + x1) / (2 * width),
                    (y0 + y1) / (2 * height),
                    (x1 - x0) / width,
                    (y1 - y0) / height,
                )
            )
        )
    return torch.stack(boxes)


class PsgImageTransforms:
    """Joint image/mask transforms modeled after OpenPSG's PSGTR pipeline."""

    def __init__(
        self,
        *,
        training: bool,
        min_size_choices: Sequence[int] = tuple(range(480, 801, 32)),
        max_size: int = 1333,
        flip_probability: float = 0.5,
        crop_probability: float = 0.5,
        crop_resize_choices: Sequence[int] = (400, 500, 600),
        crop_min_size: int = 384,
        crop_max_size: int = 600,
        crop_attempts: int = 10,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        if not min_size_choices:
            raise ValueError("min_size_choices cannot be empty")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        self.training = training
        self.min_size_choices = tuple(int(value) for value in min_size_choices)
        self.max_size = int(max_size)
        self.flip_probability = float(flip_probability if training else 0.0)
        self.crop_probability = float(crop_probability if training else 0.0)
        self.crop_resize_choices = tuple(int(value) for value in crop_resize_choices)
        self.crop_min_size = int(crop_min_size)
        self.crop_max_size = int(crop_max_size)
        self.crop_attempts = int(crop_attempts)
        self.image_mean = torch.tensor(image_mean, dtype=torch.float32)[:, None, None]
        self.image_std = torch.tensor(image_std, dtype=torch.float32)[:, None, None]

    @staticmethod
    def _crop(
        image: Image.Image,
        masks: torch.Tensor,
        class_labels: torch.Tensor,
        relations: torch.Tensor,
        segment_ids: torch.Tensor,
        top: int,
        left: int,
        height: int,
        width: int,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image = image.crop((left, top, left + width, top + height))
        masks = masks[:, top : top + height, left : left + width]
        return (image, *_filter_empty_entities(class_labels, masks, relations, segment_ids))

    def _random_relation_preserving_crop(
        self,
        image: Image.Image,
        masks: torch.Tensor,
        class_labels: torch.Tensor,
        relations: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[Image.Image, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.crop_resize_choices or relations.numel() == 0:
            return image, class_labels, masks, relations, segment_ids

        image, masks = _resize(
            image,
            masks,
            random.choice(self.crop_resize_choices),
            self.max_size,
        )
        original = (image, class_labels, masks, relations, segment_ids)
        min_height = min(self.crop_min_size, image.height)
        max_height = min(self.crop_max_size, image.height)
        min_width = min(self.crop_min_size, image.width)
        max_width = min(self.crop_max_size, image.width)
        if min_height <= 0 or min_width <= 0:
            return original

        for _ in range(self.crop_attempts):
            crop_height = random.randint(min_height, max_height)
            crop_width = random.randint(min_width, max_width)
            top = random.randint(0, image.height - crop_height)
            left = random.randint(0, image.width - crop_width)
            candidate = self._crop(
                image,
                masks,
                class_labels,
                relations,
                segment_ids,
                top,
                left,
                crop_height,
                crop_width,
            )
            if candidate[3].numel():
                return candidate
        return original

    def __call__(
        self,
        image: Image.Image,
        class_labels: torch.Tensor,
        masks: torch.Tensor,
        relations: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.training and random.random() < self.flip_probability:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            masks = masks.flip(-1)

        if self.training and random.random() < self.crop_probability:
            image, class_labels, masks, relations, segment_ids = (
                self._random_relation_preserving_crop(
                    image,
                    masks,
                    class_labels,
                    relations,
                    segment_ids,
                )
            )

        short_side = (
            random.choice(self.min_size_choices)
            if self.training
            else self.min_size_choices[-1]
        )
        image, masks = _resize(image, masks, short_side, self.max_size)
        class_labels, masks, relations, segment_ids = _filter_empty_entities(
            class_labels,
            masks,
            relations,
            segment_ids,
        )
        boxes = _boxes_from_masks(masks)

        image_array = np.array(image, dtype=np.float32, copy=True) / 255.0
        pixel_values = torch.from_numpy(image_array).permute(2, 0, 1)
        pixel_values = (pixel_values - self.image_mean) / self.image_std
        target = {
            "class_labels": class_labels,
            "boxes": boxes,
            "masks": masks.float(),
            "relations": relations,
            "segment_ids": segment_ids,
            "size": torch.tensor([image.height, image.width], dtype=torch.long),
        }
        return pixel_values, target


class OpenPsgDataset(Dataset[dict[str, Any]]):
    """OpenPSG JSON + COCO panoptic images without MMDetection dependencies."""

    def __init__(
        self,
        annotation_file: str | Path,
        data_root: str | Path = "data",
        *,
        split: Split = "train",
        transforms: PsgImageTransforms | None = None,
        filter_empty_relations: bool = True,
        deduplicate_relations: bool = True,
        randomize_duplicate_relations: bool | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.annotation_file = Path(annotation_file)
        self.data_root = Path(data_root)
        if split not in {"train", "validation", "all"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.transforms = transforms or PsgImageTransforms(
            training=split == "train",
            min_size_choices=(800,),
        )
        self.deduplicate_relations = deduplicate_relations
        self.randomize_duplicate_relations = (
            split == "train"
            if randomize_duplicate_relations is None
            else bool(randomize_duplicate_relations)
        )

        with self.annotation_file.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        required = {"data", "thing_classes", "stuff_classes", "predicate_classes"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Annotation JSON is missing keys: {sorted(missing)}")

        self.metadata = OpenPsgMetadata(
            thing_classes=tuple(payload["thing_classes"]),
            stuff_classes=tuple(payload["stuff_classes"]),
            predicate_classes=tuple(payload["predicate_classes"]),
        )
        validation_ids = {int(value) for value in payload.get("test_image_ids", [])}
        if split == "validation" and not validation_ids:
            raise ValueError(
                "validation split requires test_image_ids in the annotation JSON"
            )

        samples = []
        for sample in payload["data"]:
            image_id = int(sample["image_id"])
            if split == "train" and image_id in validation_ids:
                continue
            if split == "validation" and image_id not in validation_ids:
                continue
            if filter_empty_relations and not sample.get("relations"):
                continue
            samples.append(sample)
        if max_samples is not None:
            samples = samples[:max_samples]
        if not samples:
            raise ValueError(f"No samples remain for split={split!r}")
        self.samples = samples

        coco_root = self.data_root / "coco"
        self.image_roots = (
            coco_root / "train2017",
            coco_root / "val2017",
            coco_root,
            self.data_root,
        )
        self.panoptic_roots = (
            coco_root / "panoptic_train2017",
            coco_root / "panoptic_val2017",
            coco_root,
            self.data_root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _resolve_file(filename: str, roots: Sequence[Path]) -> Path:
        path = Path(filename)
        if path.is_absolute() and path.is_file():
            return path
        candidates: list[Path] = []
        for root in roots:
            candidates.extend((root / path, root / path.name))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        checked = "\n  ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"Could not find {filename!r}; checked:\n  {checked}")

    def _relations(self, sample: dict[str, Any]) -> torch.Tensor:
        relations = [tuple(int(value) for value in relation) for relation in sample["relations"]]
        if self.deduplicate_relations and self.randomize_duplicate_relations:
            grouped: dict[tuple[int, int], list[int]] = {}
            for subject, object_, predicate in relations:
                grouped.setdefault((subject, object_), []).append(predicate)
            relations = [
                (subject, object_, random.choice(predicates))
                for (subject, object_), predicates in grouped.items()
            ]
        elif self.deduplicate_relations:
            relations = list(dict.fromkeys(relations))
        if not relations:
            return torch.empty((0, 3), dtype=torch.long)
        return torch.tensor(relations, dtype=torch.long)

    def load_raw_sample(self, index: int) -> dict[str, Any]:
        """Load one image and its original-resolution panoptic annotations.

        This public method is useful for deterministic inference and
        visualization. It performs no resizing, cropping, flipping, or image
        normalization.
        """

        sample = self.samples[index]
        image_path = self._resolve_file(sample["file_name"], self.image_roots)
        panoptic_path = self._resolve_file(
            sample["pan_seg_file_name"],
            self.panoptic_roots,
        )
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(panoptic_path) as source:
            id_map = rgb_to_segment_id(source)
        if id_map.shape != (image.height, image.width):
            raise ValueError(
                f"Image and panoptic PNG sizes differ for image_id={sample['image_id']}: "
                f"image={(image.height, image.width)}, panoptic={tuple(id_map.shape)}"
            )

        segments = sample["segments_info"]
        if not segments:
            raise ValueError(f"image_id={sample['image_id']} has no segments_info")
        segment_ids = torch.tensor([int(segment["id"]) for segment in segments])
        class_labels = torch.tensor(
            [int(segment["category_id"]) for segment in segments],
            dtype=torch.long,
        )
        if class_labels.min() < 0 or class_labels.max() >= len(self.metadata.object_classes):
            raise ValueError(f"Invalid category_id in image_id={sample['image_id']}")
        masks = id_map.unsqueeze(0) == segment_ids[:, None, None]
        relations = self._relations(sample)
        if relations.numel() and (
            relations[:, :2].min() < 0
            or relations[:, :2].max() >= len(segments)
            or relations[:, 2].min() < 0
            or relations[:, 2].max() >= len(self.metadata.predicate_classes)
        ):
            raise ValueError(f"Invalid relation in image_id={sample['image_id']}")

        return {
            "image": image,
            "image_id": int(sample["image_id"]),
            "image_path": image_path,
            "panoptic_path": panoptic_path,
            "class_labels": class_labels,
            "masks": masks,
            "relations": relations,
            "segment_ids": segment_ids,
            "original_size": torch.tensor(
                [image.height, image.width],
                dtype=torch.long,
            ),
        }

    def prepare_raw_sample(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Apply this dataset's transforms to a result of ``load_raw_sample``."""

        pixel_values, target = self.transforms(
            raw["image"],
            raw["class_labels"],
            raw["masks"],
            raw["relations"],
            raw["segment_ids"],
        )
        if self.split == "train" and target["relations"].numel() == 0:
            raise RuntimeError(
                f"Training transform removed every relation for image_id={raw['image_id']}"
            )
        target["image_id"] = torch.tensor(raw["image_id"], dtype=torch.long)
        target["original_size"] = raw["original_size"]
        return {
            "pixel_values": pixel_values,
            "target": target,
            "image_id": raw["image_id"],
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.prepare_raw_sample(self.load_raw_sample(index))


class PsgCollator:
    """Pad images and target masks while preserving variable-length relations."""

    def __init__(self, size_divisor: int = 1) -> None:
        if size_divisor <= 0:
            raise ValueError("size_divisor must be positive")
        self.size_divisor = size_divisor

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        max_height = max(sample["pixel_values"].shape[-2] for sample in samples)
        max_width = max(sample["pixel_values"].shape[-1] for sample in samples)
        max_height = math.ceil(max_height / self.size_divisor) * self.size_divisor
        max_width = math.ceil(max_width / self.size_divisor) * self.size_divisor

        batch_size = len(samples)
        channels = samples[0]["pixel_values"].shape[0]
        dtype = samples[0]["pixel_values"].dtype
        pixel_values = torch.zeros(
            (batch_size, channels, max_height, max_width),
            dtype=dtype,
        )
        pixel_mask = torch.zeros(
            (batch_size, max_height, max_width),
            dtype=torch.long,
        )
        labels: list[dict[str, torch.Tensor]] = []
        for batch_index, sample in enumerate(samples):
            values = sample["pixel_values"]
            height, width = values.shape[-2:]
            pixel_values[batch_index, :, :height, :width] = values
            pixel_mask[batch_index, :height, :width] = 1
            target = dict(sample["target"])
            masks = target["masks"]
            target["masks"] = F.pad(
                masks,
                (0, max_width - width, 0, max_height - height),
            )
            labels.append(target)

        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
            "image_ids": torch.tensor(
                [sample["image_id"] for sample in samples],
                dtype=torch.long,
            ),
        }


def seed_worker(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def build_openpsg_dataloaders(
    annotation_file: str | Path,
    data_root: str | Path = "data",
    *,
    batch_size: int = 1,
    num_workers: int = 2,
    seed: int = 42,
    train_min_sizes: Sequence[int] = tuple(range(480, 801, 32)),
    validation_min_size: int = 800,
    max_size: int = 1333,
    crop_probability: float = 0.5,
    flip_probability: float = 0.5,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = OpenPsgDataset(
        annotation_file,
        data_root,
        split="train",
        transforms=PsgImageTransforms(
            training=True,
            min_size_choices=train_min_sizes,
            max_size=max_size,
            crop_probability=crop_probability,
            flip_probability=flip_probability,
        ),
        max_samples=max_train_samples,
    )
    validation_dataset = OpenPsgDataset(
        annotation_file,
        data_root,
        split="validation",
        transforms=PsgImageTransforms(
            training=False,
            min_size_choices=(validation_min_size,),
            max_size=max_size,
        ),
        max_samples=max_validation_samples,
    )
    if distributed and world_size <= 1:
        raise ValueError("world_size must be greater than one in distributed mode")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}): {rank}")

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    validation_sampler = (
        DistributedSampler(
            validation_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
            drop_last=False,
        )
        if distributed
        else None
    )
    generator = torch.Generator().manual_seed(seed + rank)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "collate_fn": PsgCollator(),
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            **common,
        ),
        DataLoader(
            validation_dataset,
            shuffle=False,
            sampler=validation_sampler,
            **common,
        ),
    )
