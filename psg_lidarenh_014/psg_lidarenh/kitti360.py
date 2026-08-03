from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class KittiClassMapping:
    semantic_id: int
    kitti_name: str
    target_aliases: tuple[str, ...]
    compatibility: Literal["exact", "compatible", "approximate"]


# These IDs follow the official KITTI-360 / Cityscapes semantic definitions.
# Default conversion uses only exact or defensible compatible mappings so the
# pretrained OpenPSG classifier dimensions and row meanings remain unchanged.
KITTI_CLASS_MAPPINGS: tuple[KittiClassMapping, ...] = (
    KittiClassMapping(7, "road", ("road",), "exact"),
    KittiClassMapping(
        8,
        "sidewalk",
        ("pavement-merged", "pavement", "sidewalk"),
        "compatible",
    ),
    KittiClassMapping(
        11,
        "building",
        ("building-other-merged", "building", "house"),
        "compatible",
    ),
    KittiClassMapping(
        12,
        "wall",
        ("wall-other-merged", "wall", "wall-brick", "wall-stone"),
        "compatible",
    ),
    KittiClassMapping(
        13,
        "fence",
        ("fence-merged", "fence"),
        "compatible",
    ),
    KittiClassMapping(17, "pole", ("light",), "approximate"),
    KittiClassMapping(19, "traffic light", ("traffic light",), "exact"),
    KittiClassMapping(20, "traffic sign", ("stop sign",), "approximate"),
    KittiClassMapping(
        21,
        "vegetation",
        ("tree-merged", "potted plant"),
        "approximate",
    ),
    KittiClassMapping(
        22,
        "terrain",
        ("grass-merged", "dirt-merged"),
        "approximate",
    ),
    KittiClassMapping(
        23,
        "sky",
        ("sky-other-merged", "sky"),
        "compatible",
    ),
    KittiClassMapping(24, "person", ("person",), "exact"),
    KittiClassMapping(25, "rider", ("person",), "compatible"),
    KittiClassMapping(26, "car", ("car",), "exact"),
    KittiClassMapping(27, "truck", ("truck",), "exact"),
    KittiClassMapping(28, "bus", ("bus",), "exact"),
    KittiClassMapping(31, "train", ("train",), "exact"),
    KittiClassMapping(32, "motorcycle", ("motorcycle",), "exact"),
    KittiClassMapping(33, "bicycle", ("bicycle",), "exact"),
)


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def id_to_rgb(ids: np.ndarray) -> np.ndarray:
    values = ids.astype(np.int64, copy=False)
    if values.min(initial=0) < 0 or values.max(initial=0) >= 256**3:
        raise ValueError("Panoptic segment IDs must fit in 24 bits")
    return np.stack(
        (
            values % 256,
            (values // 256) % 256,
            (values // 65536) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    y, x = np.nonzero(mask)
    if not len(x):
        raise ValueError("Cannot compute a box from an empty mask")
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def parse_frame_reference(line: str) -> tuple[str, int] | None:
    sequence_match = re.search(r"2013_05_28_drive_(\d{4})_sync", line)
    frame_matches = re.findall(r"(?<!\d)(\d{10})(?!\d)", line)
    if sequence_match and frame_matches:
        sequence = f"2013_05_28_drive_{sequence_match.group(1)}_sync"
        return sequence, int(frame_matches[-1])
    columns = line.split()
    if len(columns) >= 2 and columns[-1].isdigit():
        sequence_match = re.search(r"(\d{4})", columns[0])
        if sequence_match:
            sequence = f"2013_05_28_drive_{sequence_match.group(1)}_sync"
            return sequence, int(columns[-1])
    return None


def load_split(path: Path) -> list[tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"KITTI-360 split file is missing: {path}")
    frames: list[tuple[str, int]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reference = parse_frame_reference(line)
        if reference is None:
            raise ValueError(f"Cannot parse {path}:{line_number}: {line!r}")
        frames.append(reference)
    return frames


def sequence_name(sequence_id: int) -> str:
    if not 0 <= sequence_id <= 9999:
        raise ValueError("KITTI-360 sequence ID must be between 0 and 9999")
    return f"2013_05_28_drive_{sequence_id:04d}_sync"


def create_link(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if os.path.lexists(destination):
        if not overwrite:
            return
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unsupported link mode: {mode}")


def resolve_category_mapping(
    thing_classes: list[str],
    stuff_classes: list[str],
    *,
    include_approximate: bool,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    all_classes = thing_classes + stuff_classes
    by_name = {normalize_name(name): index for index, name in enumerate(all_classes)}
    resolved: dict[int, dict[str, Any]] = {}
    report: list[dict[str, Any]] = []
    for entry in KITTI_CLASS_MAPPINGS:
        allowed = entry.compatibility != "approximate" or include_approximate
        category_id = None
        target_name = None
        if allowed:
            for alias in entry.target_aliases:
                candidate = by_name.get(normalize_name(alias))
                if candidate is not None:
                    category_id = candidate
                    target_name = all_classes[candidate]
                    break
        record = {
            "kitti_semantic_id": entry.semantic_id,
            "kitti_name": entry.kitti_name,
            "compatibility": entry.compatibility,
            "target_category_id": category_id,
            "target_name": target_name,
            "included": category_id is not None,
        }
        report.append(record)
        if category_id is not None:
            resolved[entry.semantic_id] = {
                **record,
                "isthing": category_id < len(thing_classes),
            }
    return resolved, report


def _dominant_semantic(semantic_map: np.ndarray, mask: np.ndarray) -> int:
    values = semantic_map[mask].astype(np.int64)
    if not values.size:
        return -1
    counts = np.bincount(values)
    return int(counts.argmax())


def _predicate_id(predicates: list[str], name: str) -> int | None:
    normalized = normalize_name(name)
    for index, predicate in enumerate(predicates):
        if normalize_name(predicate) == normalized:
            return index
    return None


def generate_spatial_2d_relations(
    segments: list[dict[str, Any]],
    masks: list[np.ndarray],
    predicate_classes: list[str],
    *,
    max_relations: int,
) -> list[list[int]]:
    """Generate conservative OpenPSG-vocabulary pseudo-relations.

    The official OpenPSG spatial predicates include ``over``, ``beside``,
    ``on``, and ``in``. No depth ordering or activity predicates are fabricated.
    """

    predicate = {
        name: _predicate_id(predicate_classes, name)
        for name in ("over", "beside", "on", "in")
    }
    boxes = [segment["bbox"] for segment in segments]
    relations: list[tuple[float, list[int]]] = []
    for subject in range(len(segments)):
        sx0, sy0, sx1, sy1 = boxes[subject]
        sw = max(1, sx1 - sx0)
        sh = max(1, sy1 - sy0)
        scx = 0.5 * (sx0 + sx1)
        scy = 0.5 * (sy0 + sy1)
        for object_ in range(len(segments)):
            if subject == object_:
                continue
            ox0, oy0, ox1, oy1 = boxes[object_]
            ow = max(1, ox1 - ox0)
            oh = max(1, oy1 - oy0)
            ocx = 0.5 * (ox0 + ox1)
            ocy = 0.5 * (oy0 + oy1)
            horizontal_gap = max(0.0, max(ox0 - sx1, sx0 - ox1))
            vertical_overlap = max(0.0, min(sy1, oy1) - max(sy0, oy0))
            vertical_overlap_ratio = vertical_overlap / max(1.0, min(sh, oh))

            if (
                predicate["beside"] is not None
                and vertical_overlap_ratio >= 0.4
                and horizontal_gap <= 0.75 * max(sw, ow)
            ):
                distance = abs(scx - ocx) / max(1.0, sw + ow)
                relations.append(
                    (distance, [subject, object_, int(predicate["beside"])])
                )

            if (
                predicate["over"] is not None
                and scy < ocy
                and abs(scx - ocx) <= 0.75 * max(sw, ow)
            ):
                distance = (ocy - scy) / max(1.0, sh + oh)
                relations.append(
                    (distance + 0.1, [subject, object_, int(predicate["over"])])
                )

            if predicate["in"] is not None:
                intersection = np.logical_and(masks[subject], masks[object_]).sum()
                containment = intersection / max(1, int(masks[subject].sum()))
                if containment >= 0.9:
                    relations.append(
                        (0.0, [subject, object_, int(predicate["in"])])
                    )

            if predicate["on"] is not None and segments[object_]["isthing"] is False:
                lower_band = np.zeros_like(masks[subject])
                band_start = max(sy0, sy1 - max(2, sh // 8))
                lower_band[band_start:sy1, sx0:sx1] = masks[subject][
                    band_start:sy1,
                    sx0:sx1,
                ]
                dilated_object = np.asarray(
                    Image.fromarray(masks[object_].astype(np.uint8) * 255).filter(
                        ImageFilter.MaxFilter(5)
                    )
                ) > 0
                contact = np.logical_and(lower_band, dilated_object).sum()
                if contact >= max(1, int(lower_band.sum() * 0.05)):
                    relations.append(
                        (0.0, [subject, object_, int(predicate["on"])])
                    )

    unique: dict[tuple[int, int, int], float] = {}
    for score, relation in relations:
        key = tuple(relation)
        unique[key] = min(score, unique.get(key, float("inf")))
    ranked = sorted(unique.items(), key=lambda item: (item[1], item[0]))
    return [list(key) for key, _ in ranked[:max_relations]]


class Kitti360Converter:
    def __init__(
        self,
        kitti360_root: str | Path,
        source_openpsg_json: str | Path,
        output_root: str | Path,
        *,
        camera: str = "image_00",
        link_mode: str = "symlink",
        relation_mode: str = "spatial2d",
        include_approximate_mappings: bool = False,
        min_segment_area: int = 64,
        max_relations_per_image: int = 128,
        require_relations: bool = True,
        strict_lidar: bool = True,
        overwrite: bool = False,
    ) -> None:
        self.kitti360_root = Path(kitti360_root).resolve()
        self.source_openpsg_json = Path(source_openpsg_json).resolve()
        self.output_root = Path(output_root).resolve()
        self.camera = camera
        self.link_mode = link_mode
        self.relation_mode = relation_mode
        self.include_approximate_mappings = include_approximate_mappings
        self.min_segment_area = int(min_segment_area)
        self.max_relations_per_image = int(max_relations_per_image)
        self.require_relations = bool(require_relations)
        self.strict_lidar = bool(strict_lidar)
        self.overwrite = bool(overwrite)
        source = json.loads(self.source_openpsg_json.read_text(encoding="utf-8"))
        self.thing_classes = list(source["thing_classes"])
        self.stuff_classes = list(source["stuff_classes"])
        self.predicate_classes = list(source["predicate_classes"])
        self.category_mapping, self.mapping_report = resolve_category_mapping(
            self.thing_classes,
            self.stuff_classes,
            include_approximate=self.include_approximate_mappings,
        )

    @staticmethod
    def image_id(sequence: str, frame: int) -> int:
        match = re.search(r"drive_(\d{4})_sync", sequence)
        if match is None:
            raise ValueError(f"Invalid KITTI-360 sequence name: {sequence}")
        return int(match.group(1)) * 1_000_000 + int(frame)

    def _paths(
        self,
        sequence: str,
        frame: int,
    ) -> tuple[Path, Path, Path, Path]:
        name = f"{frame:010d}.png"
        rgb = (
            self.kitti360_root
            / "data_2d_raw"
            / sequence
            / self.camera
            / "data_rect"
            / name
        )
        semantic = (
            self.kitti360_root
            / "data_2d_semantics"
            / "train"
            / sequence
            / self.camera
            / "semantic"
            / name
        )
        instance = (
            self.kitti360_root
            / "data_2d_semantics"
            / "train"
            / sequence
            / self.camera
            / "instance"
            / name
        )
        lidar = (
            self.kitti360_root
            / "data_3d_raw"
            / sequence
            / "velodyne_points"
            / "data"
            / f"{frame:010d}.bin"
        )
        return rgb, semantic, instance, lidar

    def _convert_frame(
        self,
        split: str,
        sequence: str,
        frame: int,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        rgb_path, semantic_path, instance_path, lidar_path = self._paths(
            sequence,
            frame,
        )
        required = (rgb_path, semantic_path, instance_path)
        if not all(path.is_file() for path in required):
            return None
        if self.strict_lidar and not lidar_path.is_file():
            return None
        semantic_map = np.asarray(Image.open(semantic_path), dtype=np.int64)
        instance_map = np.asarray(Image.open(instance_path), dtype=np.int64)
        if semantic_map.shape != instance_map.shape:
            raise ValueError(f"Semantic/instance size mismatch for {sequence}/{frame}")
        height, width = semantic_map.shape

        thing_masks: list[tuple[int, int, np.ndarray]] = []
        stuff_masks: dict[int, np.ndarray] = {}
        for raw_segment_id in np.unique(instance_map):
            raw_segment_id = int(raw_segment_id)
            if raw_segment_id <= 0:
                continue
            raw_mask = instance_map == raw_segment_id
            semantic_id = _dominant_semantic(semantic_map, raw_mask)
            mapping = self.category_mapping.get(semantic_id)
            if mapping is None:
                continue
            category_id = int(mapping["target_category_id"])
            if bool(mapping["isthing"]):
                if int(raw_mask.sum()) >= self.min_segment_area:
                    thing_masks.append((raw_segment_id, category_id, raw_mask))
            else:
                if category_id not in stuff_masks:
                    stuff_masks[category_id] = raw_mask.copy()
                else:
                    stuff_masks[category_id] |= raw_mask

        panoptic_ids = np.zeros((height, width), dtype=np.int64)
        segments: list[dict[str, Any]] = []
        masks: list[np.ndarray] = []
        used_ids: set[int] = set()
        for raw_segment_id, category_id, mask in sorted(thing_masks):
            segment_id = raw_segment_id
            while segment_id in used_ids or segment_id == 0:
                segment_id += 1
            used_ids.add(segment_id)
            panoptic_ids[mask] = segment_id
            bbox = bbox_from_mask(mask)
            segments.append(
                {
                    "id": segment_id,
                    "category_id": category_id,
                    "isthing": True,
                    "iscrowd": False,
                    "area": int(mask.sum()),
                    "bbox": bbox,
                }
            )
            masks.append(mask)
        for category_id, mask in sorted(stuff_masks.items()):
            if int(mask.sum()) < self.min_segment_area:
                continue
            segment_id = 100_000 + category_id
            while segment_id in used_ids:
                segment_id += 1_000
            used_ids.add(segment_id)
            panoptic_ids[mask] = segment_id
            bbox = bbox_from_mask(mask)
            segments.append(
                {
                    "id": segment_id,
                    "category_id": category_id,
                    "isthing": False,
                    "iscrowd": False,
                    "area": int(mask.sum()),
                    "bbox": bbox,
                }
            )
            masks.append(mask)

        if len(segments) < 2:
            return None
        if self.relation_mode == "spatial2d":
            relations = generate_spatial_2d_relations(
                segments,
                masks,
                self.predicate_classes,
                max_relations=self.max_relations_per_image,
            )
        elif self.relation_mode == "none":
            relations = []
        else:
            raise ValueError(f"Unsupported relation mode: {self.relation_mode}")
        if self.require_relations and not relations:
            return None

        sequence_id = int(re.search(r"drive_(\d{4})_sync", sequence).group(1))
        output_name = f"{sequence_id:04d}_{frame:010d}.png"
        image_id = self.image_id(sequence, frame)
        image_directory = "train2017" if split == "train" else "val2017"
        panoptic_directory = (
            "panoptic_train2017" if split == "train" else "panoptic_val2017"
        )
        output_rgb = self.output_root / "coco" / image_directory / output_name
        output_panoptic = (
            self.output_root / "coco" / panoptic_directory / output_name
        )
        create_link(rgb_path, output_rgb, self.link_mode, self.overwrite)
        output_panoptic.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(id_to_rgb(panoptic_ids)).save(output_panoptic)

        segment_info = [
            {key: value for key, value in segment.items() if key != "bbox"}
            for segment in segments
        ]
        annotations = [
            {
                "bbox": segment["bbox"],
                "category_id": segment["category_id"],
            }
            for segment in segments
        ]
        record = {
            "file_name": output_name,
            "pan_seg_file_name": output_name,
            "image_id": image_id,
            "height": height,
            "width": width,
            "segments_info": segment_info,
            "annotations": annotations,
            "relations": relations,
        }
        manifest = {
            "lidar_file": str(lidar_path.resolve()),
            "sequence": sequence,
            "frame": frame,
        }
        return record, manifest

    def convert(
        self,
        *,
        max_frames: int | None = None,
        sequence_id: int = 0,
    ) -> dict[str, Any]:
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        selected_sequence = sequence_name(sequence_id)
        semantics_root = self.kitti360_root / "data_2d_semantics" / "train"
        split_files = {
            "train": semantics_root / "2013_05_28_drive_train_frames.txt",
            "validation": semantics_root / "2013_05_28_drive_val_frames.txt",
        }
        split_limits = {
            "train": max_frames,
            "validation": (
                max(1, math.ceil(max_frames * 0.1))
                if max_frames is not None
                else None
            ),
        }
        records: list[dict[str, Any]] = []
        validation_ids: list[int] = []
        manifest_samples: dict[str, dict[str, Any]] = {}
        statistics: dict[str, Any] = {}
        for split, split_file in split_files.items():
            all_frames = load_split(split_file)
            selected = [
                (sequence, frame)
                for sequence, frame in all_frames
                if sequence == selected_sequence
            ]
            target = split_limits[split]
            kept = 0
            skipped = 0
            attempted = 0
            for sequence, frame in selected:
                if target is not None and kept >= target:
                    break
                attempted += 1
                converted = self._convert_frame(split, sequence, frame)
                if converted is None:
                    skipped += 1
                    continue
                record, manifest = converted
                records.append(record)
                manifest_samples[str(record["image_id"])] = manifest
                if split == "validation":
                    validation_ids.append(record["image_id"])
                kept += 1
            statistics[split] = {
                "listed_all_sequences": len(all_frames),
                "listed_selected_sequence": len(selected),
                "requested": target,
                "attempted": attempted,
                "kept": kept,
                "skipped": skipped,
                "exhausted_before_requested": (
                    target is not None and kept < target
                ),
            }

        annotations = {
            "thing_classes": self.thing_classes,
            "stuff_classes": self.stuff_classes,
            "predicate_classes": self.predicate_classes,
            "test_image_ids": validation_ids,
            "data": records,
        }
        annotation_path = self.output_root / "annotations" / "psg_train_val.json"
        manifest_path = self.output_root / "lidar" / "manifest.json"
        mapping_path = self.output_root / "annotations" / "category_mapping.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps(annotations, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "point_feature_count": 4,
                    "samples": manifest_samples,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mapping_path.write_text(
            json.dumps(
                {
                    "source_openpsg_json": str(self.source_openpsg_json),
                    "include_approximate_mappings": self.include_approximate_mappings,
                    "mappings": self.mapping_report,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary = {
            "output_root": str(self.output_root),
            "annotation_file": str(annotation_path),
            "lidar_manifest": str(manifest_path),
            "category_mapping": str(mapping_path),
            "sequence_id": sequence_id,
            "sequence": selected_sequence,
            "max_train_frames": max_frames,
            "max_validation_frames": split_limits["validation"],
            "images": len(records),
            "training_images": len(records) - len(validation_ids),
            "validation_images": len(validation_ids),
            "statistics": statistics,
        }
        (self.output_root / "conversion_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert KITTI-360 into OpenPSG + raw LiDAR manifest layout"
    )
    parser.add_argument(
        "--kitti360-root",
        type=Path,
        default=Path("/data/jwang/datasets/kitti360"),
    )
    parser.add_argument("--source-openpsg-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--camera", choices=("image_00", "image_01"), default="image_00")
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
    )
    parser.add_argument(
        "--relation-mode",
        choices=("spatial2d", "none"),
        default="spatial2d",
    )
    parser.add_argument("--include-approximate-mappings", action="store_true")
    parser.add_argument("--min-segment-area", type=int, default=64)
    parser.add_argument("--max-relations-per-image", type=int, default=128)
    parser.add_argument(
        "--require-relations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-lidar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        help=(
            "Maximum successfully converted training frames. When provided, "
            "the converter additionally keeps ceil(0.1 * N) validation frames."
        ),
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=0,
        help=(
            "KITTI-360 sequence number. Default 0 selects "
            "2013_05_28_drive_0000_sync."
        ),
    )
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if not 0 <= args.seq <= 9999:
        parser.error("--seq must be between 0 and 9999")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    converter = Kitti360Converter(
        args.kitti360_root,
        args.source_openpsg_json,
        args.output_root,
        camera=args.camera,
        link_mode=args.link_mode,
        relation_mode=args.relation_mode,
        include_approximate_mappings=args.include_approximate_mappings,
        min_segment_area=args.min_segment_area,
        max_relations_per_image=args.max_relations_per_image,
        require_relations=args.require_relations,
        strict_lidar=args.strict_lidar,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            converter.convert(
                max_frames=args.max_frames,
                sequence_id=args.seq,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
