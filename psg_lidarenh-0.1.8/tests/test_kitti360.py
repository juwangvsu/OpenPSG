import json
from pathlib import Path

import numpy as np
from PIL import Image

from psg_lidarenh.kitti360 import (
    Kitti360Converter,
    id_to_rgb,
    parse_frame_reference,
    resolve_category_mapping,
)


THING = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
]
STUFF = [
    "road",
    "fence-merged",
    "sky-other-merged",
    "pavement-merged",
    "building-other-merged",
    "wall-other-merged",
    "tree-merged",
    "grass-merged",
]
PREDICATES = ["over", "in front of", "beside", "on", "in", "attached to"]


def test_frame_reference_parser() -> None:
    result = parse_frame_reference(
        "2013_05_28_drive_0009_sync/image_00/data_rect/0000001234.png"
    )
    assert result == ("2013_05_28_drive_0009_sync", 1234)


def test_mapping_keeps_approximate_classes_disabled() -> None:
    mapping, report = resolve_category_mapping(
        THING,
        STUFF,
        include_approximate=False,
    )
    assert mapping[26]["target_category_id"] == THING.index("car")
    assert mapping[7]["target_category_id"] == len(THING) + STUFF.index("road")
    assert 21 not in mapping
    vegetation = next(item for item in report if item["kitti_semantic_id"] == 21)
    assert vegetation["included"] is False


def _write_frame(root: Path, sequence: str, frame: int) -> None:
    name = f"{frame:010d}.png"
    rgb_path = root / "data_2d_raw" / sequence / "image_00" / "data_rect" / name
    semantic_path = (
        root
        / "data_2d_semantics"
        / "train"
        / sequence
        / "image_00"
        / "semantic"
        / name
    )
    instance_path = (
        root
        / "data_2d_semantics"
        / "train"
        / sequence
        / "image_00"
        / "instance"
        / name
    )
    lidar_path = (
        root
        / "data_3d_raw"
        / sequence
        / "velodyne_points"
        / "data"
        / f"{frame:010d}.bin"
    )
    for path in (rgb_path, semantic_path, instance_path, lidar_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(np.zeros((12, 16, 3), dtype=np.uint8)).save(rgb_path)
    semantic = np.zeros((12, 16), dtype=np.uint8)
    instance = np.zeros((12, 16), dtype=np.uint16)
    semantic[6:, :] = 7
    instance[6:, :] = 7000
    semantic[3:8, 5:10] = 26
    instance[3:8, 5:10] = 26001
    Image.fromarray(semantic).save(semantic_path)
    Image.fromarray(instance).save(instance_path)
    np.array([[1.0, 2.0, 0.0, 0.5]], dtype=np.float32).tofile(lidar_path)


def test_end_to_end_conversion(tmp_path) -> None:
    kitti = tmp_path / "kitti360"
    sequence = "2013_05_28_drive_0000_sync"
    _write_frame(kitti, sequence, 1)
    _write_frame(kitti, sequence, 2)
    split_root = kitti / "data_2d_semantics" / "train"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "2013_05_28_drive_train_frames.txt").write_text(
        f"{sequence}/image_00/data_rect/0000000001.png\n"
    )
    (split_root / "2013_05_28_drive_val_frames.txt").write_text(
        f"{sequence}/image_00/data_rect/0000000002.png\n"
    )
    source = tmp_path / "source_psg.json"
    source.write_text(
        json.dumps(
            {
                "thing_classes": THING,
                "stuff_classes": STUFF,
                "predicate_classes": PREDICATES,
                "test_image_ids": [],
                "data": [],
            }
        )
    )
    output = tmp_path / "converted"
    summary = Kitti360Converter(
        kitti,
        source,
        output,
        link_mode="symlink",
        min_segment_area=2,
    ).convert()
    assert summary["images"] == 2
    annotations = json.loads(
        (output / "annotations" / "psg_train_val.json").read_text()
    )
    assert annotations["thing_classes"] == THING
    assert annotations["stuff_classes"] == STUFF
    assert annotations["predicate_classes"] == PREDICATES
    assert len(annotations["test_image_ids"]) == 1
    first = annotations["data"][0]
    categories = {entry["category_id"] for entry in first["segments_info"]}
    assert THING.index("car") in categories
    assert len(THING) + STUFF.index("road") in categories
    assert first["relations"]
    manifest = json.loads((output / "lidar" / "manifest.json").read_text())
    sample = next(iter(manifest["samples"].values()))
    assert "boxes_3d" not in sample
    assert Path(sample["lidar_file"]).is_absolute()
    assert (output / "coco" / "train2017" / "0000_0000000001.png").is_symlink()


def test_max_frames_adds_ten_percent_validation_and_filters_sequence(tmp_path) -> None:
    kitti = tmp_path / "kitti360"
    sequence_0 = "2013_05_28_drive_0000_sync"
    sequence_1 = "2013_05_28_drive_0001_sync"
    for frame in range(1, 13):
        _write_frame(kitti, sequence_0, frame)
    for frame in range(101, 105):
        _write_frame(kitti, sequence_1, frame)

    split_root = kitti / "data_2d_semantics" / "train"
    split_root.mkdir(parents=True, exist_ok=True)
    train_lines = [
        *(f"{sequence_0}/image_00/data_rect/{frame:010d}.png\n" for frame in range(1, 11)),
        *(f"{sequence_1}/image_00/data_rect/{frame:010d}.png\n" for frame in range(101, 103)),
    ]
    validation_lines = [
        *(f"{sequence_0}/image_00/data_rect/{frame:010d}.png\n" for frame in range(11, 13)),
        *(f"{sequence_1}/image_00/data_rect/{frame:010d}.png\n" for frame in range(103, 105)),
    ]
    (split_root / "2013_05_28_drive_train_frames.txt").write_text(
        "".join(train_lines)
    )
    (split_root / "2013_05_28_drive_val_frames.txt").write_text(
        "".join(validation_lines)
    )

    source = tmp_path / "source_psg.json"
    source.write_text(
        json.dumps(
            {
                "thing_classes": THING,
                "stuff_classes": STUFF,
                "predicate_classes": PREDICATES,
                "test_image_ids": [],
                "data": [],
            }
        )
    )
    output = tmp_path / "converted"
    summary = Kitti360Converter(
        kitti,
        source,
        output,
        min_segment_area=2,
    ).convert(max_frames=10, sequence_id=0)

    assert summary["sequence"] == sequence_0
    assert summary["training_images"] == 10
    assert summary["validation_images"] == 1
    assert summary["images"] == 11
    assert summary["max_train_frames"] == 10
    assert summary["max_validation_frames"] == 1
    assert summary["statistics"]["train"]["listed_selected_sequence"] == 10
    assert summary["statistics"]["validation"]["listed_selected_sequence"] == 2

    annotations = json.loads(
        (output / "annotations" / "psg_train_val.json").read_text()
    )
    assert len(annotations["data"]) == 11
    assert len(annotations["test_image_ids"]) == 1
    assert all(
        record["file_name"].startswith("0000_")
        for record in annotations["data"]
    )


def test_sequence_one_selects_only_sequence_one(tmp_path) -> None:
    kitti = tmp_path / "kitti360"
    sequence_0 = "2013_05_28_drive_0000_sync"
    sequence_1 = "2013_05_28_drive_0001_sync"
    _write_frame(kitti, sequence_0, 1)
    _write_frame(kitti, sequence_0, 2)
    _write_frame(kitti, sequence_1, 101)
    _write_frame(kitti, sequence_1, 102)
    split_root = kitti / "data_2d_semantics" / "train"
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "2013_05_28_drive_train_frames.txt").write_text(
        f"{sequence_0}/image_00/data_rect/0000000001.png\n"
        f"{sequence_1}/image_00/data_rect/0000000101.png\n"
    )
    (split_root / "2013_05_28_drive_val_frames.txt").write_text(
        f"{sequence_0}/image_00/data_rect/0000000002.png\n"
        f"{sequence_1}/image_00/data_rect/0000000102.png\n"
    )
    source = tmp_path / "source_psg.json"
    source.write_text(
        json.dumps(
            {
                "thing_classes": THING,
                "stuff_classes": STUFF,
                "predicate_classes": PREDICATES,
                "test_image_ids": [],
                "data": [],
            }
        )
    )
    output = tmp_path / "converted"
    summary = Kitti360Converter(
        kitti,
        source,
        output,
        min_segment_area=2,
    ).convert(sequence_id=1)
    assert summary["sequence"] == sequence_1
    assert summary["training_images"] == 1
    assert summary["validation_images"] == 1
    annotations = json.loads(
        (output / "annotations" / "psg_train_val.json").read_text()
    )
    assert all(
        record["file_name"].startswith("0001_")
        for record in annotations["data"]
    )
