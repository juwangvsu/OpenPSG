from __future__ import annotations

import argparse
import colorsys
import json
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image, ImageDraw, ImageFont


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LiDAR-enhanced PSGTR inference on selected training samples "
            "and save ground-truth/prediction visualizations"
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing coco/, or the coco directory itself; "
            "both forms are accepted"
        ),
    )
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--lidar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation"),
        default="train",
        help="Dataset split to sample; defaults to training data",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--random-count",
        type=int,
        help="Randomly select this many samples; default: 8",
    )
    selection.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Exact dataset indices to run",
    )
    selection.add_argument(
        "--image-ids",
        type=int,
        nargs="+",
        help="Exact OpenPSG image IDs to run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-visual-relations",
        type=int,
        default=15,
        help="Maximum number of GT and predicted relation lines drawn per image",
    )
    args = parser.parse_args(argv)
    if args.random_count is None and args.indices is None and args.image_ids is None:
        args.random_count = 8
    if args.random_count is not None and args.random_count <= 0:
        parser.error("--random-count must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_visual_relations <= 0:
        parser.error("--max-visual-relations must be positive")
    return args


def resolve_checkpoint(path: Path) -> Path:
    marker = path / "last_checkpoint"
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    return path


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _extract_record_image_id(record: Any) -> int | None:
    if isinstance(record, dict) and "image_id" in record:
        return _as_int(record["image_id"])
    image_id = getattr(record, "image_id", None)
    if image_id is not None:
        return _as_int(image_id)
    return None


def dataset_image_ids(dataset: Any) -> list[int]:
    """Return image IDs in the exact filtered dataset order.

    Current ``OpenPsgDataset`` versions retain filtered records in one of the
    common attributes below. A slow item-loading fallback is used only when an
    unfamiliar PSGTR-HF version does not expose those records.
    """

    base = getattr(dataset, "base_dataset", dataset)
    for attribute in ("samples", "data", "records", "annotations"):
        records = getattr(base, attribute, None)
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        if len(records) != len(dataset):
            continue
        values = [_extract_record_image_id(record) for record in records]
        if all(value is not None for value in values):
            return [int(value) for value in values if value is not None]

    print(
        "dataset records are not exposed; resolving --image-ids by loading samples",
        flush=True,
    )
    result: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = sample.get("target", sample.get("labels", {}))
        result.append(_as_int(sample.get("image_id", target["image_id"])))
    return result


def select_indices(
    dataset: Any,
    *,
    random_count: int | None,
    indices: Sequence[int] | None,
    image_ids: Sequence[int] | None,
    seed: int,
) -> list[int]:
    if indices is not None:
        selected = [int(index) for index in indices]
        invalid = [index for index in selected if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"Dataset indices out of range for length {len(dataset)}: {invalid}"
            )
        return selected

    if image_ids is not None:
        ordered_ids = dataset_image_ids(dataset)
        index_by_id = {image_id: index for index, image_id in enumerate(ordered_ids)}
        missing = [int(image_id) for image_id in image_ids if int(image_id) not in index_by_id]
        if missing:
            raise KeyError(f"Image IDs are not present in the selected split: {missing}")
        return [index_by_id[int(image_id)] for image_id in image_ids]

    count = min(int(random_count or 8), len(dataset))
    generator = random.Random(seed)
    return generator.sample(range(len(dataset)), count)


def _object_classes(metadata: Any) -> tuple[str, ...]:
    classes = getattr(metadata, "object_classes", None)
    if classes is not None:
        return tuple(classes)
    return tuple(metadata.thing_classes) + tuple(metadata.stuff_classes)


def _tensor_image(pixel_values: torch.Tensor, height: int, width: int) -> Image.Image:
    values = pixel_values.detach().float().cpu()[:, :height, :width]
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < -0.05 or maximum > 1.05:
        mean = values.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = values.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        values = values * std + mean
    values = values.clamp(0, 1)
    array = (values.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _color(index: int) -> tuple[int, int, int]:
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return int(red * 255), int(green * 255), int(blue * 255)


def _boxes_from_masks(masks: torch.Tensor) -> list[list[float]]:
    masks = masks.detach().cpu().to(torch.bool)
    boxes: list[list[float]] = []
    for mask in masks:
        rows, columns = torch.where(mask)
        if rows.numel() == 0:
            boxes.append([0.0, 0.0, 0.0, 0.0])
        else:
            boxes.append(
                [
                    float(columns.min()),
                    float(rows.min()),
                    float(columns.max() + 1),
                    float(rows.max() + 1),
                ]
            )
    return boxes


def _overlay_mask(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
    alpha: int = 72,
) -> None:
    binary = (mask.detach().cpu() > 0).to(torch.uint8).numpy() * alpha
    overlay = Image.new("RGBA", image.size, (*color, 0))
    overlay.putalpha(Image.fromarray(binary, mode="L"))
    image.alpha_composite(overlay)


def _draw_box(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    x0, y0, x1, y1 = (float(value) for value in box)
    if x1 <= x0 or y1 <= y0:
        return
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    text_box = draw.textbbox((x0, y0), label)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    draw.rectangle(
        (x0, max(0, y0 - text_height - 4), x0 + text_width + 6, y0),
        fill=(*color, 220),
    )
    draw.text((x0 + 3, max(0, y0 - text_height - 2)), label, fill=(0, 0, 0, 255))


def _append_caption(
    image: Image.Image,
    title: str,
    lines: Sequence[str],
) -> Image.Image:
    font = ImageFont.load_default()
    line_height = 15
    caption_height = 30 + line_height * max(1, len(lines))
    result = Image.new("RGB", (image.width, image.height + caption_height), "white")
    result.paste(image.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(result)
    draw.text((8, image.height + 7), title, fill="black", font=font)
    for line_index, line in enumerate(lines):
        draw.text(
            (8, image.height + 25 + line_index * line_height),
            line,
            fill="black",
            font=font,
        )
    return result


def _ground_truth_payload(
    target: dict[str, Any],
    object_classes: Sequence[str],
    predicate_classes: Sequence[str],
) -> dict[str, Any]:
    labels = target["class_labels"].detach().cpu().to(torch.long)
    masks = target["masks"].detach().cpu()
    boxes = _boxes_from_masks(masks)
    entities = [
        {
            "entity_index": index,
            "category_id": int(category_id),
            "category": object_classes[int(category_id)],
            "box_xyxy": boxes[index],
        }
        for index, category_id in enumerate(labels.tolist())
    ]
    relations = []
    for subject, object_, predicate in target["relations"].detach().cpu().tolist():
        relations.append(
            {
                "subject_index": int(subject),
                "subject_category": object_classes[int(labels[subject])],
                "predicate_id": int(predicate),
                "predicate": predicate_classes[int(predicate)],
                "object_index": int(object_),
                "object_category": object_classes[int(labels[object_])],
            }
        )
    return {"entities": entities, "relations": relations}


def _prediction_payload(
    prediction: dict[str, torch.Tensor],
    object_classes: Sequence[str],
    predicate_classes: Sequence[str],
) -> dict[str, Any]:
    subject_masks = prediction["subject_masks"].detach().cpu()
    object_masks = prediction["object_masks"].detach().cpu()
    subject_boxes = _boxes_from_masks(subject_masks)
    object_boxes = _boxes_from_masks(object_masks)
    count = int(prediction["scores"].shape[0])
    relations = []
    for index in range(count):
        subject_id = int(prediction["subject_labels"][index])
        object_id = int(prediction["object_labels"][index])
        predicate_id = int(prediction["relation_labels"][index])
        relations.append(
            {
                "rank": index + 1,
                "score": float(prediction["scores"][index]),
                "subject": {
                    "category_id": subject_id,
                    "category": object_classes[subject_id],
                    "score": float(prediction["subject_scores"][index]),
                    "box_xyxy": subject_boxes[index],
                },
                "predicate": {
                    "predicate_id": predicate_id,
                    "predicate": predicate_classes[predicate_id],
                    "score": float(prediction["relation_scores"][index]),
                },
                "object": {
                    "category_id": object_id,
                    "category": object_classes[object_id],
                    "score": float(prediction["object_scores"][index]),
                    "box_xyxy": object_boxes[index],
                },
            }
        )
    return {"relations": relations}


def render_comparison(
    image: Image.Image,
    target: dict[str, Any],
    prediction: dict[str, torch.Tensor],
    object_classes: Sequence[str],
    predicate_classes: Sequence[str],
    *,
    max_relations: int,
) -> Image.Image:
    gt = image.convert("RGBA")
    gt_masks = target["masks"].detach().cpu()
    gt_labels = target["class_labels"].detach().cpu().to(torch.long)
    gt_boxes = _boxes_from_masks(gt_masks)
    for entity_index, (mask, category_id, box) in enumerate(
        zip(gt_masks, gt_labels.tolist(), gt_boxes)
    ):
        color = _color(entity_index)
        _overlay_mask(gt, mask, color)
        _draw_box(
            ImageDraw.Draw(gt),
            box,
            color,
            f"{entity_index}:{object_classes[int(category_id)]}",
        )
    gt_lines = []
    relations = target["relations"].detach().cpu().tolist()
    for subject, object_, predicate in relations[:max_relations]:
        gt_lines.append(
            f"{subject}:{object_classes[int(gt_labels[subject])]} --"
            f"{predicate_classes[int(predicate)]}--> "
            f"{object_}:{object_classes[int(gt_labels[object_])]}"
        )
    if len(relations) > max_relations:
        gt_lines.append(f"... {len(relations) - max_relations} more relations")
    if not gt_lines:
        gt_lines.append("No ground-truth relations")
    gt_panel = _append_caption(gt, "Ground truth", gt_lines)

    predicted = image.convert("RGBA")
    subject_masks = prediction["subject_masks"].detach().cpu()
    object_masks = prediction["object_masks"].detach().cpu()
    subject_boxes = _boxes_from_masks(subject_masks)
    object_boxes = _boxes_from_masks(object_masks)
    prediction_lines: list[str] = []
    count = min(max_relations, int(prediction["scores"].shape[0]))
    for index in range(count):
        subject_id = int(prediction["subject_labels"][index])
        object_id = int(prediction["object_labels"][index])
        predicate_id = int(prediction["relation_labels"][index])
        subject_color = _color(index * 2)
        object_color = _color(index * 2 + 1)
        _overlay_mask(predicted, subject_masks[index], subject_color, alpha=54)
        _overlay_mask(predicted, object_masks[index], object_color, alpha=54)
        draw = ImageDraw.Draw(predicted)
        _draw_box(
            draw,
            subject_boxes[index],
            subject_color,
            f"S{index + 1}:{object_classes[subject_id]}",
        )
        _draw_box(
            draw,
            object_boxes[index],
            object_color,
            f"O{index + 1}:{object_classes[object_id]}",
        )
        prediction_lines.append(
            f"#{index + 1} {object_classes[subject_id]} --"
            f"{predicate_classes[predicate_id]}--> {object_classes[object_id]} "
            f"score={float(prediction['scores'][index]):.3f}"
        )
    if int(prediction["scores"].shape[0]) > max_relations:
        prediction_lines.append(
            f"... {int(prediction['scores'].shape[0]) - max_relations} more predictions"
        )
    if not prediction_lines:
        prediction_lines.append("No predictions above threshold")
    prediction_panel = _append_caption(predicted, "Prediction", prediction_lines)

    height = max(gt_panel.height, prediction_panel.height)
    comparison = Image.new("RGB", (gt_panel.width + prediction_panel.width, height), "white")
    comparison.paste(gt_panel, (0, 0))
    comparison.paste(prediction_panel, (gt_panel.width, 0))
    return comparison


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> list[dict[str, Any]]:
    from .dataset import LidarSceneGraphCollator
    from .modeling_psg_lidarenh import (
        PsgLidarEnhForPanopticSceneGraphGeneration,
    )
    from .training import build_dataset

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint = resolve_checkpoint(args.checkpoint)
    dataset = build_dataset(args, args.split, training=False)
    selected = select_indices(
        dataset,
        random_count=args.random_count,
        indices=args.indices,
        image_ids=args.image_ids,
        seed=args.seed,
    )
    metadata = dataset.base_dataset.metadata
    object_classes = _object_classes(metadata)
    predicate_classes = tuple(metadata.predicate_classes)
    model = PsgLidarEnhForPanopticSceneGraphGeneration.from_pretrained(
        checkpoint,
        use_safetensors=True,
    ).to(device)
    model.eval()
    collator = LidarSceneGraphCollator()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    for position, dataset_index in enumerate(selected, start=1):
        batch = collator([dataset[dataset_index]])
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        lidar_points = [points.to(device) for points in batch["lidar_points"]]
        target = batch["labels"][0]
        target_size = target["size"].detach().cpu().to(torch.long)
        height, width = (int(value) for value in target_size.tolist())
        image_id = _as_int(target["image_id"])

        print(
            f"infer {position}/{len(selected)} index={dataset_index} image_id={image_id}",
            flush=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=args.amp and device.type == "cuda",
        ):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                lidar_points=lidar_points,
                labels=None,
            )
        predictions = model.post_process_triplets(
            outputs,
            target_sizes=target_size.unsqueeze(0).to(device),
            score_threshold=args.score_threshold,
            top_k=args.top_k,
            mask_threshold=args.mask_threshold,
        )
        prediction = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in predictions[0].items()
        }
        target_cpu = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        image = _tensor_image(batch["pixel_values"][0], height, width)
        comparison = render_comparison(
            image,
            target_cpu,
            prediction,
            object_classes,
            predicate_classes,
            max_relations=args.max_visual_relations,
        )
        stem = f"index-{dataset_index:06d}_image-{image_id}"
        image_path = args.output_dir / f"{stem}.png"
        json_path = args.output_dir / f"{stem}.json"
        comparison.save(image_path)
        payload = {
            "checkpoint": str(checkpoint.resolve()),
            "split": args.split,
            "dataset_index": int(dataset_index),
            "image_id": image_id,
            "lidar_point_count": int(batch["lidar_points"][0].shape[0]),
            "image_size": [height, width],
            "ground_truth": _ground_truth_payload(
                target_cpu,
                object_classes,
                predicate_classes,
            ),
            "prediction": _prediction_payload(
                prediction,
                object_classes,
                predicate_classes,
            ),
            "visualization": image_path.name,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest.append(
            {
                "dataset_index": int(dataset_index),
                "image_id": image_id,
                "visualization": image_path.name,
                "result": json_path.name,
                "prediction_count": len(payload["prediction"]["relations"]),
                "ground_truth_relation_count": len(payload["ground_truth"]["relations"]),
            }
        )

    (args.output_dir / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint.resolve()),
                "split": args.split,
                "selection": {
                    "random_count": args.random_count,
                    "indices": args.indices,
                    "image_ids": args.image_ids,
                    "seed": args.seed,
                },
                "score_threshold": args.score_threshold,
                "top_k": args.top_k,
                "mask_threshold": args.mask_threshold,
                "samples": manifest,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    run_inference(parse_args(argv))


if __name__ == "__main__":
    main()
