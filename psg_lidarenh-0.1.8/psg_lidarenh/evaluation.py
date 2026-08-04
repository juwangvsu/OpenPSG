from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

from psgtr_hf.metrics import PsgEvaluationAccumulator


def move_labels(
    labels: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    return [
        {
            key: value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in target.items()
        }
        for target in labels
    ]


def metadata_object_classes(metadata: Any) -> tuple[str, ...]:
    classes = getattr(metadata, "object_classes", None)
    if classes is not None:
        return tuple(classes)
    return tuple(metadata.thing_classes) + tuple(metadata.stuff_classes)


def _reduce_loss_terms(
    loss_terms: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Reduce possibly different loss dictionaries in a fixed global order."""

    if not (dist.is_available() and dist.is_initialized()):
        return loss_terms
    world_size = dist.get_world_size()
    keys_by_rank: list[list[str] | None] = [None] * world_size
    dist.all_gather_object(keys_by_rank, sorted(loss_terms))
    keys = sorted({key for rank_keys in keys_by_rank for key in (rank_keys or [])})
    reduced: dict[str, torch.Tensor] = {}
    for key in keys:
        value = loss_terms.get(
            key,
            torch.zeros(1, dtype=torch.float64, device=device),
        )
        dist.all_reduce(value)
        reduced[key] = value
    return reduced


def _mask_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Return exact xyxy boxes for boolean masks without Python pixel loops."""

    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W]")
    count, height, width = masks.shape
    if count == 0:
        return torch.empty((0, 4), dtype=torch.float32, device=masks.device)
    masks = masks.to(torch.bool)
    any_x = masks.any(dim=1)
    any_y = masks.any(dim=2)
    x_coordinates = torch.arange(width, device=masks.device)
    y_coordinates = torch.arange(height, device=masks.device)
    x_min = torch.where(any_x, x_coordinates, width).amin(dim=1)
    x_max = torch.where(any_x, x_coordinates, -1).amax(dim=1) + 1
    y_min = torch.where(any_y, y_coordinates, height).amin(dim=1)
    y_max = torch.where(any_y, y_coordinates, -1).amax(dim=1) + 1
    boxes = torch.stack((x_min, y_min, x_max, y_max), dim=-1).to(torch.float32)
    nonempty = masks.flatten(1).any(dim=1)
    return torch.where(nonempty[:, None], boxes, torch.zeros_like(boxes))


def _box_iou_matrix(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 or second.numel() == 0:
        return torch.zeros(
            (first.shape[0], second.shape[0]),
            dtype=torch.float32,
            device=first.device,
        )
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp_min(0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp_min(0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp_min(1e-6)


def _class_aware_box_nms(
    labels: torch.Tensor,
    scores: torch.Tensor,
    boxes: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Greedy class-aware NMS using one small box-IoU matrix."""

    if labels.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    order = scores.argsort(descending=True)
    pairwise = _box_iou_matrix(boxes, boxes).detach().cpu()
    labels_cpu = labels.detach().cpu()
    order_cpu = order.detach().cpu().tolist()
    kept: list[int] = []
    for index in order_cpu:
        duplicate = any(
            int(labels_cpu[index]) == int(labels_cpu[previous])
            and float(pairwise[index, previous]) > threshold
            for previous in kept
        )
        if not duplicate:
            kept.append(index)
    return torch.tensor(kept, dtype=torch.long, device=labels.device)


def _assemble_panoptic_fast(
    prediction: dict[str, torch.Tensor],
    *,
    num_thing_classes: int,
    entity_score_threshold: float,
    mask_threshold: float,
    thing_nms_threshold: float,
    max_endpoints: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse relation endpoints into one panoptic map in O(HW).

    The previous implementation repeatedly computed full-resolution mask IoU
    inside Python NMS loops. With 200 endpoint masks at KITTI-360 resolution,
    the first evaluation sample could take many minutes. This implementation
    uses class-aware box NMS and chunked pixel assignment instead.
    """

    labels = torch.cat(
        (prediction["subject_labels"], prediction["object_labels"]),
        dim=0,
    ).to(device=device, dtype=torch.long)
    scores = torch.cat(
        (prediction["subject_scores"], prediction["object_scores"]),
        dim=0,
    ).to(device=device, dtype=torch.float32)
    probabilities = torch.cat(
        (prediction["subject_masks"], prediction["object_masks"]),
        dim=0,
    ).to(device=device, dtype=torch.float32)
    if probabilities.ndim != 3:
        raise ValueError("Predicted endpoint masks must have shape [N, H, W]")
    height, width = probabilities.shape[-2:]

    keep = scores >= entity_score_threshold
    labels = labels[keep]
    scores = scores[keep]
    probabilities = probabilities[keep]
    if labels.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros((height, width), dtype=torch.long, device=device),
        )

    binary = probabilities >= mask_threshold
    nonempty = binary.flatten(1).any(dim=1)
    labels = labels[nonempty]
    scores = scores[nonempty]
    probabilities = probabilities[nonempty]
    binary = binary[nonempty]
    if labels.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros((height, width), dtype=torch.long, device=device),
        )

    if labels.numel() > max_endpoints:
        selected = scores.topk(max_endpoints).indices
        labels = labels[selected]
        scores = scores[selected]
        probabilities = probabilities[selected]
        binary = binary[selected]

    boxes = _mask_boxes(binary)
    thing_indices = torch.nonzero(
        labels < num_thing_classes,
        as_tuple=False,
    ).flatten()
    kept_things = _class_aware_box_nms(
        labels[thing_indices],
        scores[thing_indices],
        boxes[thing_indices],
        thing_nms_threshold,
    )
    thing_indices = thing_indices[kept_things]

    candidate_labels: list[torch.Tensor] = []
    candidate_scores: list[torch.Tensor] = []
    candidate_probabilities: list[torch.Tensor] = []
    for index in thing_indices:
        candidate_labels.append(labels[index])
        candidate_scores.append(scores[index])
        candidate_probabilities.append(probabilities[index])

    stuff_labels = labels[labels >= num_thing_classes].unique(sorted=True)
    for label in stuff_labels:
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        candidate_labels.append(label)
        candidate_scores.append(scores[indices].max())
        candidate_probabilities.append(probabilities[indices].max(dim=0).values)

    if not candidate_labels:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.zeros((height, width), dtype=torch.long, device=device),
        )

    candidate_labels_tensor = torch.stack(candidate_labels).to(torch.long)
    candidate_scores_tensor = torch.stack(candidate_scores).to(torch.float32)
    candidate_probabilities_tensor = torch.stack(candidate_probabilities).to(torch.float32)

    best_score = torch.full(
        (height, width),
        -1.0,
        dtype=torch.float32,
        device=device,
    )
    best_index = torch.full(
        (height, width),
        -1,
        dtype=torch.long,
        device=device,
    )
    chunk_size = 16
    for start in range(0, len(candidate_labels), chunk_size):
        end = min(start + chunk_size, len(candidate_labels))
        chunk = candidate_probabilities_tensor[start:end]
        weighted = chunk * candidate_scores_tensor[start:end, None, None]
        weighted = weighted.masked_fill(chunk < mask_threshold, -1.0)
        chunk_score, chunk_index = weighted.max(dim=0)
        update = chunk_score > best_score
        best_score = torch.where(update, chunk_score, best_score)
        best_index = torch.where(update, chunk_index + start, best_index)

    assigned = best_score >= 0
    output_labels: list[torch.Tensor] = []
    output_map = torch.zeros((height, width), dtype=torch.long, device=device)
    output_index = 1
    for candidate_index, label in enumerate(candidate_labels_tensor):
        mask = assigned & (best_index == candidate_index)
        if mask.any():
            output_labels.append(label)
            output_map[mask] = output_index
            output_index += 1

    if not output_labels:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            output_map,
        )
    return torch.stack(output_labels), output_map


def _target_panoptic_map(target_masks: torch.Tensor) -> torch.Tensor:
    if target_masks.ndim != 3:
        raise ValueError("Target masks must have shape [N, H, W]")
    height, width = target_masks.shape[-2:]
    target_map = torch.zeros(
        (height, width),
        dtype=torch.long,
        device=target_masks.device,
    )
    for index in range(target_masks.shape[0]):
        target_map[target_masks[index]] = index + 1
    return target_map


def _panoptic_counts_from_maps(
    predicted_labels: torch.Tensor,
    predicted_map: torch.Tensor,
    target_labels: torch.Tensor,
    target_map: torch.Tensor,
    *,
    num_classes: int,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute exact PQ counts from two non-overlapping segment maps."""

    predicted_count = int(predicted_labels.numel())
    target_count = int(target_labels.numel())
    dtype = torch.float64
    iou_sum = torch.zeros(num_classes, dtype=dtype, device=predicted_map.device)
    true_positive = torch.zeros_like(iou_sum)
    false_positive = torch.bincount(
        predicted_labels,
        minlength=num_classes,
    ).to(dtype)
    false_negative = torch.bincount(
        target_labels,
        minlength=num_classes,
    ).to(dtype)
    if predicted_count == 0 or target_count == 0:
        return iou_sum.cpu(), true_positive.cpu(), false_positive.cpu(), false_negative.cpu()

    predicted_area = torch.bincount(
        predicted_map.flatten(),
        minlength=predicted_count + 1,
    )[1:].to(torch.float64)
    target_area = torch.bincount(
        target_map.flatten(),
        minlength=target_count + 1,
    )[1:].to(torch.float64)
    joint_index = predicted_map * (target_count + 1) + target_map
    intersections = torch.bincount(
        joint_index.flatten(),
        minlength=(predicted_count + 1) * (target_count + 1),
    ).reshape(predicted_count + 1, target_count + 1)[1:, 1:].to(torch.float64)
    union = predicted_area[:, None] + target_area[None, :] - intersections
    iou = intersections / union.clamp_min(1.0)
    same_class = predicted_labels[:, None] == target_labels[None, :]
    matches = torch.nonzero(same_class & (iou > iou_threshold), as_tuple=False)
    if matches.numel() == 0:
        return iou_sum.cpu(), true_positive.cpu(), false_positive.cpu(), false_negative.cpu()

    # For non-overlapping panoptic maps, IoU > 0.5 guarantees a unique match.
    predicted_indices = matches[:, 0]
    target_indices = matches[:, 1]
    matched_labels = predicted_labels[predicted_indices]
    matched_iou = iou[predicted_indices, target_indices]
    true_positive += torch.bincount(matched_labels, minlength=num_classes).to(dtype)
    false_positive -= torch.bincount(matched_labels, minlength=num_classes).to(dtype)
    false_negative -= torch.bincount(
        target_labels[target_indices],
        minlength=num_classes,
    ).to(dtype)
    iou_sum.scatter_add_(0, matched_labels, matched_iou)
    return iou_sum.cpu(), true_positive.cpu(), false_positive.cpu(), false_negative.cpu()


def _predicate_recall_bbox_counts(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    num_predicates: int,
    ks: Sequence[int],
    mask_threshold: float,
    iou_threshold: float,
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Compute class-grounded predicate recall using endpoint box IoU.

    Bounding-box grounding is the standard inexpensive SGG recall criterion.
    It avoids the old implementation's repeated full-resolution mask-IoU loops.
    """

    relations = target["relations"].to(device=device, dtype=torch.long)
    totals = torch.zeros(num_predicates, dtype=torch.float64)
    matched = {
        int(k): torch.zeros(num_predicates, dtype=torch.float64)
        for k in ks
    }
    if relations.numel() == 0:
        return totals, matched

    target_labels = target["class_labels"].to(device=device, dtype=torch.long)
    target_masks = target["masks"].to(device=device) >= mask_threshold
    subject_masks = prediction["subject_masks"].to(device=device) >= mask_threshold
    object_masks = prediction["object_masks"].to(device=device) >= mask_threshold
    target_boxes = _mask_boxes(target_masks)
    subject_boxes = _mask_boxes(subject_masks)
    object_boxes = _mask_boxes(object_masks)
    subject_iou = _box_iou_matrix(subject_boxes, target_boxes)
    object_iou = _box_iou_matrix(object_boxes, target_boxes)

    scores = prediction["scores"].to(device=device)
    order = scores.argsort(descending=True)
    subject_labels = prediction["subject_labels"].to(device=device, dtype=torch.long)
    object_labels = prediction["object_labels"].to(device=device, dtype=torch.long)
    relation_labels = prediction["relation_labels"].to(device=device, dtype=torch.long)

    subject_indices = relations[:, 0]
    object_indices = relations[:, 1]
    predicate_indices = relations[:, 2]
    totals += torch.bincount(
        predicate_indices.cpu(),
        minlength=num_predicates,
    ).to(torch.float64)

    target_subject_labels = target_labels[subject_indices]
    target_object_labels = target_labels[object_indices]
    for k in ks:
        top = order[: min(int(k), order.numel())]
        if top.numel() == 0:
            continue
        correct = (
            (relation_labels[top, None] == predicate_indices[None, :])
            & (subject_labels[top, None] == target_subject_labels[None, :])
            & (object_labels[top, None] == target_object_labels[None, :])
            & (subject_iou[top[:, None], subject_indices[None, :]] >= iou_threshold)
            & (object_iou[top[:, None], object_indices[None, :]] >= iou_threshold)
        )
        recalled = correct.any(dim=0)
        if recalled.any():
            matched[int(k)] += torch.bincount(
                predicate_indices[recalled].cpu(),
                minlength=num_predicates,
            ).to(torch.float64)
    return totals, matched


def _fast_accumulator_update(
    accumulator: PsgEvaluationAccumulator,
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    entity_score_threshold: float,
    mask_threshold: float,
    iou_threshold: float,
    thing_nms_threshold: float,
    max_panoptic_endpoints: int,
    device: torch.device,
) -> None:
    height, width = (int(value) for value in target["size"].tolist())
    target_masks = target["masks"][:, :height, :width].to(device=device) >= mask_threshold
    target_labels = target["class_labels"].to(device=device, dtype=torch.long)
    predicted_labels, predicted_map = _assemble_panoptic_fast(
        prediction,
        num_thing_classes=accumulator.num_thing_classes,
        entity_score_threshold=entity_score_threshold,
        mask_threshold=mask_threshold,
        thing_nms_threshold=thing_nms_threshold,
        max_endpoints=max_panoptic_endpoints,
        device=device,
    )
    target_map = _target_panoptic_map(target_masks)
    pq_counts = _panoptic_counts_from_maps(
        predicted_labels,
        predicted_map,
        target_labels,
        target_map,
        num_classes=accumulator.num_object_classes,
        iou_threshold=iou_threshold,
    )
    accumulator.pq_iou_sum += pq_counts[0]
    accumulator.pq_tp += pq_counts[1]
    accumulator.pq_fp += pq_counts[2]
    accumulator.pq_fn += pq_counts[3]

    predicate_total, predicate_matched = _predicate_recall_bbox_counts(
        prediction,
        {**target, "masks": target_masks},
        num_predicates=accumulator.num_predicate_classes,
        ks=accumulator.recall_ks,
        mask_threshold=mask_threshold,
        iou_threshold=iou_threshold,
        device=device,
    )
    accumulator.predicate_total += predicate_total
    for k in accumulator.recall_ks:
        accumulator.predicate_matched[int(k)] += predicate_matched[int(k)]
    accumulator.image_count += 1


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    metadata: Any,
    *,
    amp: bool,
    recall_ks: Sequence[int] = (20, 50, 100),
    entity_score_threshold: float = 0.25,
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    thing_nms_threshold: float = 0.8,
    max_panoptic_endpoints: int = 100,
    reduce_across_processes: bool = True,
    progress_label: str | None = None,
    progress_every: int = 10,
) -> dict[str, Any]:
    unwrapped = model.module if hasattr(model, "module") else model
    unwrapped.eval()
    accumulator = PsgEvaluationAccumulator(
        num_object_classes=len(metadata_object_classes(metadata)),
        num_thing_classes=len(metadata.thing_classes),
        num_predicate_classes=len(metadata.predicate_classes),
        recall_ks=tuple(int(value) for value in recall_ks),
    )
    image_count = torch.zeros(1, dtype=torch.float64, device=device)
    processed = 0
    total = len(loader.dataset) if hasattr(loader, "dataset") else None

    for batch_number, batch in enumerate(loader, start=1):
        if progress_label and batch_number == 1:
            print(f"{progress_label} batch=1 stage=forward", flush=True)
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        lidar_points = [
            points.to(device, non_blocking=True)
            for points in batch["lidar_points"]
        ]
        labels = move_labels(batch["labels"], device)
        # Metrics use the ground-truth labels below, but inference must not
        # invoke the training criterion. Passing labels here runs triplet-level
        # Hungarian matching against every pseudo-relation in the frame. KITTI
        # frames can contain hundreds or thousands of generated relations,
        # making the first evaluation forward appear to hang for many minutes.
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            outputs = unwrapped(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                lidar_points=lidar_points,
                labels=None,
            )
        if progress_label and batch_number == 1:
            print(f"{progress_label} batch=1 stage=postprocess", flush=True)
        image_count += len(labels)

        target_sizes = torch.stack([target["size"] for target in labels])
        predictions = unwrapped.post_process_triplets(
            outputs,
            target_sizes=target_sizes,
            score_threshold=0.0,
            top_k=max(max(recall_ks), 100),
            mask_threshold=mask_threshold,
        )
        if progress_label and batch_number == 1:
            print(f"{progress_label} batch=1 stage=metrics", flush=True)
        for prediction, target in zip(predictions, labels):
            _fast_accumulator_update(
                accumulator,
                prediction,
                target,
                entity_score_threshold=entity_score_threshold,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
                thing_nms_threshold=thing_nms_threshold,
                max_panoptic_endpoints=max_panoptic_endpoints,
                device=device,
            )
        processed += len(labels)
        if progress_label and (
            processed == 1
            or processed == total
            or processed % max(1, progress_every) == 0
        ):
            suffix = f"/{total}" if total is not None else ""
            print(f"{progress_label} processed={processed}{suffix}", flush=True)

    if reduce_across_processes and dist.is_available() and dist.is_initialized():
        dist.all_reduce(image_count)
        accumulator.distributed_reduce(device)

    result = accumulator.compute(
        object_classes=metadata_object_classes(metadata),
        predicate_classes=tuple(metadata.predicate_classes),
    )
    result["evaluation_implementation"] = {
        "training_loss_computed": False,
        "panoptic_quality": "exact segment-map intersection after class-aware box NMS",
        "predicate_recall_basis": "subject/object bbox IoU",
        "max_panoptic_endpoints": int(max_panoptic_endpoints),
    }
    return result
