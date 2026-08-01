from __future__ import annotations

import argparse
import colorsys
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .dataset import OpenPsgDataset, PsgCollator, PsgImageTransforms


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PSGTR checkpoint on selected OpenPSG samples and save "
            "prediction-versus-ground-truth visualizations."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "A checkpoint directory, or a training output directory containing "
            "last_checkpoint."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="train",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--random-count",
        type=int,
        default=None,
        help="Select this many random samples. Default: 8.",
    )
    selection.add_argument(
        "--indices",
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Dataset indices, separated by spaces or commas.",
    )
    selection.add_argument(
        "--image-ids",
        nargs="+",
        default=None,
        metavar="IMAGE_ID",
        help="COCO/OpenPSG image IDs, separated by spaces or commas.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--max-ground-truth-relations", type=int, default=20)
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if args.random_count is None and args.indices is None and args.image_ids is None:
        args.random_count = 8
    if args.random_count is not None and args.random_count <= 0:
        parser.error("--random-count must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.min_size <= 0 or args.max_size <= 0:
        parser.error("--min-size and --max-size must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be in [0, 1]")
    if not 0.0 <= args.mask_threshold <= 1.0:
        parser.error("--mask-threshold must be in [0, 1]")
    if args.max_ground_truth_relations <= 0:
        parser.error("--max-ground-truth-relations must be positive")
    if args.panel_width < 320:
        parser.error("--panel-width must be at least 320")
    return args


def _parse_integer_tokens(values: Sequence[str], option: str) -> list[int]:
    result: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                result.append(int(token))
            except ValueError as error:
                raise ValueError(f"Invalid integer for {option}: {token!r}") from error
    if not result:
        raise ValueError(f"{option} did not contain any integers")
    return result


def select_dataset_indices(
    dataset: OpenPsgDataset,
    *,
    random_count: int | None,
    indices: Sequence[str] | None,
    image_ids: Sequence[str] | None,
    seed: int,
) -> list[int]:
    if indices is not None:
        selected = _parse_integer_tokens(indices, "--indices")
        invalid = [index for index in selected if index < 0 or index >= len(dataset)]
        if invalid:
            raise IndexError(
                f"Dataset indices out of range [0, {len(dataset)}): {invalid}"
            )
    elif image_ids is not None:
        requested = _parse_integer_tokens(image_ids, "--image-ids")
        index_by_image_id = {
            int(sample["image_id"]): index
            for index, sample in enumerate(dataset.samples)
        }
        missing = [image_id for image_id in requested if image_id not in index_by_image_id]
        if missing:
            raise KeyError(
                f"Image IDs are not present in split={dataset.split!r}: {missing}"
            )
        selected = [index_by_image_id[image_id] for image_id in requested]
    else:
        count = 8 if random_count is None else random_count
        if count > len(dataset):
            raise ValueError(
                f"--random-count={count} exceeds dataset size {len(dataset)}"
            )
        selected = random.Random(seed).sample(range(len(dataset)), count)

    # Preserve caller order while preventing accidental duplicate output files.
    return list(dict.fromkeys(selected))


def resolve_checkpoint(path: Path) -> Path:
    path = path.expanduser().resolve()
    marker = path / "last_checkpoint"
    if marker.is_file():
        target = Path(marker.read_text(encoding="utf-8").strip()).expanduser()
        if not target.is_absolute():
            target = path / target
        path = target.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Checkpoint has no config.json: {path}")
    if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"Checkpoint has no model weights: {path}")
    return path


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _chunks(values: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _color(index: int, *, saturation: float = 0.70, value: float = 0.95) -> tuple[int, int, int]:
    hue = (index * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return round(red * 255), round(green * 255), round(blue * 255)


def _absolute_boxes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    boxes = torch.zeros((len(masks), 4), dtype=torch.float32)
    for index, mask in enumerate(masks):
        y, x = torch.nonzero(mask, as_tuple=True)
        if x.numel():
            boxes[index] = torch.tensor(
                [x.min(), y.min(), x.max() + 1, y.max() + 1],
                dtype=torch.float32,
            )
    return boxes


def _overlay_mask(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
    alpha: int,
) -> Image.Image:
    mask_array = mask.detach().to(device="cpu", dtype=torch.bool).numpy()
    if mask_array.shape != (image.height, image.width):
        raise ValueError(
            f"Mask shape {mask_array.shape} does not match image size "
            f"{(image.height, image.width)}"
        )
    overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
    overlay[mask_array, :3] = color
    overlay[mask_array, 3] = alpha
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay, "RGBA"))


def _draw_box_label(
    draw: ImageDraw.ImageDraw,
    box: Sequence[float],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    width: int = 3,
) -> None:
    x0, y0, x1, y1 = (float(value) for value in box)
    draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
    text_box = draw.textbbox((x0, y0), label, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    top = max(0, y0 - text_height - 5)
    draw.rectangle((x0, top, x0 + text_width + 6, top + text_height + 5), fill=(*color, 230))
    draw.text((x0 + 3, top + 2), label, fill=(0, 0, 0, 255), font=font)


def _resize_for_panel(image: Image.Image, panel_width: int) -> tuple[Image.Image, float]:
    if image.width <= panel_width:
        return image, 1.0
    scale = panel_width / image.width
    height = max(1, round(image.height * scale))
    return image.resize((panel_width, height), Image.Resampling.LANCZOS), scale


def _wrap_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[str],
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        words = line.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if draw.textlength(candidate, font=font) <= max_width:
                current = candidate
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def _panel(
    title: str,
    image: Image.Image,
    lines: Sequence[str],
    panel_width: int,
) -> Image.Image:
    title_font = _font(24)
    text_font = _font(17)
    padding = 14
    image, _ = _resize_for_panel(image, panel_width - 2 * padding)
    measure = Image.new("RGB", (panel_width, 1), "white")
    measure_draw = ImageDraw.Draw(measure)
    wrapped = _wrap_lines(measure_draw, lines, text_font, panel_width - 2 * padding)
    line_height = max(20, measure_draw.textbbox((0, 0), "Ag", font=text_font)[3] + 5)
    title_height = max(32, measure_draw.textbbox((0, 0), title, font=title_font)[3] + 12)
    height = padding + title_height + image.height + padding + line_height * max(1, len(wrapped)) + padding
    panel = Image.new("RGB", (panel_width, height), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((padding, padding), title, fill="black", font=title_font)
    image_top = padding + title_height
    panel.paste(image.convert("RGB"), (padding, image_top))
    y = image_top + image.height + padding
    if not wrapped:
        wrapped = ["(none)"]
    for line in wrapped:
        draw.text((padding, y), line, fill="black", font=text_font)
        y += line_height
    return panel


def render_ground_truth(
    raw: dict[str, Any],
    object_classes: Sequence[str],
    predicate_classes: Sequence[str],
    max_relations: int,
) -> tuple[Image.Image, list[str], dict[str, Any]]:
    image = raw["image"].convert("RGBA")
    masks = raw["masks"].to(torch.bool)
    boxes = _absolute_boxes_from_masks(masks)
    class_labels = raw["class_labels"]
    all_relations = raw["relations"]
    relations = all_relations[:max_relations]
    involved = sorted(set(relations[:, :2].flatten().tolist())) if relations.numel() else []

    for entity_index in involved:
        image = _overlay_mask(image, masks[entity_index], _color(entity_index), 62)
    draw = ImageDraw.Draw(image)
    label_font = _font(15)
    for entity_index in involved:
        class_id = int(class_labels[entity_index])
        _draw_box_label(
            draw,
            boxes[entity_index].tolist(),
            f"#{entity_index} {object_classes[class_id]}",
            _color(entity_index),
            label_font,
        )

    lines: list[str] = []
    relation_records: list[dict[str, Any]] = []
    for relation_index, (subject, object_, predicate) in enumerate(
        all_relations.tolist(),
        start=1,
    ):
        subject_class = int(class_labels[subject])
        object_class = int(class_labels[object_])
        relation_records.append(
            {
                "subject_entity_index": subject,
                "subject_label_id": subject_class,
                "subject_label": object_classes[subject_class],
                "predicate_id": predicate,
                "predicate": predicate_classes[predicate],
                "object_entity_index": object_,
                "object_label_id": object_class,
                "object_label": object_classes[object_class],
            }
        )
        if relation_index <= max_relations:
            lines.append(
                f"{relation_index:02d}. #{subject} {object_classes[subject_class]} "
                f"--{predicate_classes[predicate]}--> "
                f"#{object_} {object_classes[object_class]}"
            )
    if len(all_relations) > max_relations:
        lines.append(
            f"... {len(all_relations) - max_relations} more ground-truth relations omitted"
        )

    entities = [
        {
            "entity_index": index,
            "segment_id": int(raw["segment_ids"][index]),
            "label_id": int(class_id),
            "label": object_classes[int(class_id)],
            "box_xyxy": boxes[index].tolist(),
        }
        for index, class_id in enumerate(class_labels.tolist())
    ]
    return image.convert("RGB"), lines, {"entities": entities, "relations": relation_records}


def render_predictions(
    original: Image.Image,
    prediction: dict[str, torch.Tensor],
    object_classes: Sequence[str],
    predicate_classes: Sequence[str],
) -> tuple[Image.Image, list[str], list[dict[str, Any]]]:
    image = original.convert("RGBA")
    count = len(prediction["scores"])
    for index in range(count):
        image = _overlay_mask(image, prediction["subject_masks"][index], _color(2 * index), 48)
        image = _overlay_mask(image, prediction["object_masks"][index], _color(2 * index + 1), 48)

    draw = ImageDraw.Draw(image)
    label_font = _font(15)
    lines: list[str] = []
    records: list[dict[str, Any]] = []
    for index in range(count):
        subject_label_id = int(prediction["subject_labels"][index])
        object_label_id = int(prediction["object_labels"][index])
        predicate_id = int(prediction["relation_labels"][index])
        score = float(prediction["scores"][index])
        subject_box = prediction["subject_boxes"][index].tolist()
        object_box = prediction["object_boxes"][index].tolist()
        _draw_box_label(
            draw,
            subject_box,
            f"S{index + 1} {object_classes[subject_label_id]}",
            _color(2 * index),
            label_font,
        )
        _draw_box_label(
            draw,
            object_box,
            f"O{index + 1} {object_classes[object_label_id]}",
            _color(2 * index + 1),
            label_font,
        )
        lines.append(
            f"{index + 1:02d}. {object_classes[subject_label_id]} "
            f"--{predicate_classes[predicate_id]}--> "
            f"{object_classes[object_label_id]}  score={score:.3f}"
        )
        records.append(
            {
                "query_index": int(prediction["query_indices"][index]),
                "score": score,
                "subject_score": float(prediction["subject_scores"][index]),
                "object_score": float(prediction["object_scores"][index]),
                "predicate_score": float(prediction["relation_scores"][index]),
                "subject_label_id": subject_label_id,
                "subject_label": object_classes[subject_label_id],
                "predicate_id": predicate_id,
                "predicate": predicate_classes[predicate_id],
                "object_label_id": object_label_id,
                "object_label": object_classes[object_label_id],
                "subject_box_xyxy": subject_box,
                "object_box_xyxy": object_box,
            }
        )
    return image.convert("RGB"), lines, records


def combine_panels(left: Image.Image, right: Image.Image, gap: int = 18) -> Image.Image:
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + gap + right.width, height), (238, 238, 238))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def _cpu_prediction(prediction: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in prediction.items()}


def run(args: argparse.Namespace) -> Path:
    from .modeling_psgtr import PsgtrForPanopticSceneGraphGeneration

    checkpoint = resolve_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    dataset = OpenPsgDataset(
        args.annotation_file,
        args.data_root,
        split=args.split,
        transforms=PsgImageTransforms(
            training=False,
            min_size_choices=(args.min_size,),
            max_size=args.max_size,
        ),
        filter_empty_relations=True,
        deduplicate_relations=False,
    )
    selected = select_dataset_indices(
        dataset,
        random_count=args.random_count,
        indices=args.indices,
        image_ids=args.image_ids,
        seed=args.seed,
    )

    model = PsgtrForPanopticSceneGraphGeneration.from_pretrained(
        checkpoint,
        use_safetensors=True,
    ).to(device)
    model.eval()
    if model.config.num_object_labels != len(dataset.metadata.object_classes):
        raise ValueError("Checkpoint object vocabulary does not match the dataset")
    if model.config.num_relation_labels != len(dataset.metadata.predicate_classes):
        raise ValueError("Checkpoint predicate vocabulary does not match the dataset")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    collator = PsgCollator()
    amp_enabled = args.amp and device.type == "cuda"
    results: list[dict[str, Any]] = []
    object_classes = dataset.metadata.object_classes
    predicate_classes = dataset.metadata.predicate_classes

    for selected_batch in _chunks(selected, args.batch_size):
        raw_samples = [dataset.load_raw_sample(index) for index in selected_batch]
        prepared_samples = [
            dataset.prepare_raw_sample(raw)
            for raw in raw_samples
        ]
        batch = collator(prepared_samples)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        target_sizes = torch.stack(
            [raw["original_size"] for raw in raw_samples]
        ).to(device)
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with torch.inference_mode(), amp_context:
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            predictions = model.post_process_triplets(
                outputs,
                target_sizes=target_sizes,
                score_threshold=args.score_threshold,
                top_k=args.top_k,
                mask_threshold=args.mask_threshold,
            )

        for dataset_index, raw, prediction in zip(
            selected_batch,
            raw_samples,
            predictions,
        ):
            prediction = _cpu_prediction(prediction)
            gt_image, gt_lines, ground_truth = render_ground_truth(
                raw,
                object_classes,
                predicate_classes,
                args.max_ground_truth_relations,
            )
            pred_image, pred_lines, prediction_records = render_predictions(
                raw["image"],
                prediction,
                object_classes,
                predicate_classes,
            )
            gt_panel = _panel(
                f"Ground truth | image_id={raw['image_id']} | index={dataset_index}",
                gt_image,
                gt_lines,
                args.panel_width,
            )
            pred_panel = _panel(
                f"Predictions | threshold={args.score_threshold:g} | top_k={args.top_k}",
                pred_image,
                pred_lines,
                args.panel_width,
            )
            comparison = combine_panels(gt_panel, pred_panel)
            stem = f"{len(results):04d}_image-{raw['image_id']}_index-{dataset_index}"
            visualization_path = args.output_dir / f"{stem}.png"
            comparison.save(visualization_path)
            record = {
                "dataset_index": int(dataset_index),
                "image_id": int(raw["image_id"]),
                "file_name": str(raw["image_path"]),
                "panoptic_file_name": str(raw["panoptic_path"]),
                "original_size": raw["original_size"].tolist(),
                "visualization": visualization_path.name,
                "ground_truth": ground_truth,
                "predictions": prediction_records,
            }
            (args.output_dir / f"{stem}.json").write_text(
                json.dumps(record, indent=2) + "\n",
                encoding="utf-8",
            )
            results.append(record)
            print(
                f"saved={visualization_path} predictions={len(prediction_records)} ",
                f"ground_truth_relations={len(ground_truth['relations'])}",
                flush=True,
            )

    manifest = {
        "checkpoint": str(checkpoint),
        "annotation_file": str(args.annotation_file.resolve()),
        "data_root": str(args.data_root.resolve()),
        "split": args.split,
        "seed": args.seed,
        "score_threshold": args.score_threshold,
        "top_k": args.top_k,
        "mask_threshold": args.mask_threshold,
        "selected_indices": selected,
        "selected_image_ids": [record["image_id"] for record in results],
        "results": results,
    }
    manifest_path = args.output_dir / "results.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"results={manifest_path}", flush=True)
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))
