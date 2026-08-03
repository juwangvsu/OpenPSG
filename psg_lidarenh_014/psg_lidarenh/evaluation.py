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


@torch.no_grad()
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
) -> dict[str, Any]:
    unwrapped = model.module if hasattr(model, "module") else model
    model.eval()
    accumulator = PsgEvaluationAccumulator(
        num_object_classes=len(metadata_object_classes(metadata)),
        num_thing_classes=len(metadata.thing_classes),
        num_predicate_classes=len(metadata.predicate_classes),
        recall_ks=tuple(int(value) for value in recall_ks),
    )
    loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
    image_count = torch.zeros(1, dtype=torch.float64, device=device)
    loss_terms: dict[str, torch.Tensor] = {}

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        lidar_points = [
            points.to(device, non_blocking=True)
            for points in batch["lidar_points"]
        ]
        labels = move_labels(batch["labels"], device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            # Evaluation datasets are partitioned without padding, so ranks
            # can have different numbers of batches. Calling the DDP wrapper
            # here would run its per-forward buffer broadcast and deadlock when
            # a shorter rank exits before a longer rank. Gradients are disabled
            # and metrics are reduced explicitly below, so evaluate the local
            # underlying module directly.
            outputs = unwrapped(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                lidar_points=lidar_points,
                labels=labels,
            )
        if outputs.loss is not None:
            loss_sum += outputs.loss.detach().to(torch.float64) * len(labels)
        image_count += len(labels)
        if outputs.loss_dict:
            for name, value in outputs.loss_dict.items():
                loss_terms.setdefault(
                    name,
                    torch.zeros(1, dtype=torch.float64, device=device),
                )
                loss_terms[name] += value.detach().to(torch.float64) * len(labels)

        target_sizes = torch.stack([target["size"] for target in labels])
        predictions = unwrapped.post_process_triplets(
            outputs,
            target_sizes=target_sizes,
            score_threshold=0.0,
            top_k=max(max(recall_ks), 100),
            mask_threshold=mask_threshold,
        )
        for prediction, target in zip(predictions, labels):
            accumulator.update(
                prediction,
                target,
                entity_score_threshold=entity_score_threshold,
                mask_threshold=mask_threshold,
                iou_threshold=iou_threshold,
                thing_nms_threshold=thing_nms_threshold,
            )

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum)
        dist.all_reduce(image_count)
        for value in loss_terms.values():
            dist.all_reduce(value)
    accumulator.distributed_reduce(device)
    result = accumulator.compute(
        object_classes=metadata_object_classes(metadata),
        predicate_classes=tuple(metadata.predicate_classes),
    )
    denominator = image_count.clamp_min(1.0)
    result["metrics"]["loss"] = float((loss_sum / denominator).item())
    for name, value in sorted(loss_terms.items()):
        result["metrics"][name] = float((value / denominator).item())
    return result
