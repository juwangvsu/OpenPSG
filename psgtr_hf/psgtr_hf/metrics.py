from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.distributed as dist


def _mask_iou_matrix(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for two boolean mask sets."""

    if first.ndim != 3 or second.ndim != 3:
        raise ValueError("Masks must have shape [N, H, W]")
    if first.shape[-2:] != second.shape[-2:]:
        raise ValueError("Mask sets must have the same spatial size")
    if first.shape[0] == 0 or second.shape[0] == 0:
        return torch.zeros(
            (first.shape[0], second.shape[0]),
            dtype=torch.float64,
            device=first.device,
        )
    first_flat = first.flatten(1).to(torch.float64)
    second_flat = second.flatten(1).to(torch.float64)
    intersection = first_flat @ second_flat.transpose(0, 1)
    union = (
        first_flat.sum(1)[:, None]
        + second_flat.sum(1)[None, :]
        - intersection
    )
    return intersection / union.clamp_min(1.0)


def _mask_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    intersection = torch.logical_and(first, second).sum().item()
    union = torch.logical_or(first, second).sum().item()
    return float(intersection / union) if union else 0.0


def assemble_panoptic_prediction(
    prediction: dict[str, torch.Tensor],
    *,
    num_thing_classes: int,
    entity_score_threshold: float = 0.25,
    mask_threshold: float = 0.5,
    thing_nms_threshold: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collapse triplet endpoints into one non-overlapping panoptic prediction.

    PSGTR predicts a subject and object mask for every relation query, so the
    same physical entity can appear many times. This function collects all
    endpoints, merges same-class stuff candidates, suppresses duplicate thing
    candidates, and assigns each pixel to the highest scoring remaining mask.
    """

    labels = torch.cat(
        (prediction["subject_labels"], prediction["object_labels"]),
        dim=0,
    ).to(torch.long)
    # Endpoint candidates inherit the complete triplet confidence. This keeps
    # no-relation queries from flooding the panoptic reconstruction with
    # otherwise high object-class scores.
    scores = torch.cat(
        (prediction["scores"], prediction["scores"]),
        dim=0,
    ).to(torch.float32)
    probabilities = torch.cat(
        (prediction["subject_masks"], prediction["object_masks"]),
        dim=0,
    ).to(torch.float32)
    if probabilities.ndim != 3:
        raise ValueError("Predicted endpoint masks must have shape [N, H, W]")
    height, width = probabilities.shape[-2:]
    if labels.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, height, width), dtype=torch.bool),
        )

    keep = scores >= entity_score_threshold
    labels = labels[keep]
    scores = scores[keep]
    probabilities = probabilities[keep]
    if labels.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, height, width), dtype=torch.bool),
        )

    binary = probabilities >= mask_threshold
    nonempty = binary.flatten(1).any(1)
    labels = labels[nonempty]
    scores = scores[nonempty]
    probabilities = probabilities[nonempty]
    binary = binary[nonempty]
    if labels.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, height, width), dtype=torch.bool),
        )

    candidate_labels: list[int] = []
    candidate_scores: list[torch.Tensor] = []
    candidate_probabilities: list[torch.Tensor] = []

    # Things remain instance candidates and are deduplicated class-wise.
    thing_indices = torch.nonzero(labels < num_thing_classes, as_tuple=False).flatten()
    thing_indices = thing_indices[scores[thing_indices].argsort(descending=True)]
    kept_thing_indices: list[int] = []
    for index_tensor in thing_indices:
        index = int(index_tensor.item())
        duplicate = False
        for previous in kept_thing_indices:
            if int(labels[previous].item()) != int(labels[index].item()):
                continue
            if _mask_iou(binary[previous], binary[index]) > thing_nms_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept_thing_indices.append(index)
        candidate_labels.append(int(labels[index].item()))
        candidate_scores.append(scores[index])
        candidate_probabilities.append(probabilities[index])

    # Panoptic stuff has one segment per semantic class, so merge all endpoint
    # proposals of the same stuff class before resolving overlaps.
    stuff_labels = labels[labels >= num_thing_classes].unique(sorted=True)
    for label_tensor in stuff_labels:
        label = int(label_tensor.item())
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        candidate_labels.append(label)
        candidate_scores.append(scores[indices].max())
        candidate_probabilities.append(probabilities[indices].max(dim=0).values)

    if not candidate_labels:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, height, width), dtype=torch.bool),
        )

    candidate_scores_tensor = torch.stack(candidate_scores)
    candidate_probabilities_tensor = torch.stack(candidate_probabilities)
    weighted = candidate_probabilities_tensor * candidate_scores_tensor[:, None, None]
    weighted = weighted.masked_fill(
        candidate_probabilities_tensor < mask_threshold,
        -1.0,
    )
    winning_score, winning_index = weighted.max(dim=0)
    assigned = winning_score >= 0

    output_labels: list[int] = []
    output_masks: list[torch.Tensor] = []
    for index, label in enumerate(candidate_labels):
        mask = assigned & (winning_index == index)
        if mask.any():
            output_labels.append(label)
            output_masks.append(mask.cpu())

    if not output_labels:
        return (
            torch.empty((0,), dtype=torch.long),
            torch.empty((0, height, width), dtype=torch.bool),
        )
    return torch.tensor(output_labels, dtype=torch.long), torch.stack(output_masks)


def panoptic_quality_counts(
    predicted_labels: torch.Tensor,
    predicted_masks: torch.Tensor,
    target_labels: torch.Tensor,
    target_masks: torch.Tensor,
    *,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return class-wise IoU sum, TP, FP, and FN for standard PQ."""

    iou_sum = torch.zeros(num_classes, dtype=torch.float64)
    true_positive = torch.zeros(num_classes, dtype=torch.float64)
    false_positive = torch.zeros(num_classes, dtype=torch.float64)
    false_negative = torch.zeros(num_classes, dtype=torch.float64)
    predicted_labels = predicted_labels.cpu()
    predicted_masks = predicted_masks.cpu().to(torch.bool)
    target_labels = target_labels.cpu()
    target_masks = target_masks.cpu().to(torch.bool)

    for class_id in range(num_classes):
        predicted_indices = torch.nonzero(
            predicted_labels == class_id,
            as_tuple=False,
        ).flatten()
        target_indices = torch.nonzero(
            target_labels == class_id,
            as_tuple=False,
        ).flatten()
        if predicted_indices.numel() == 0:
            false_negative[class_id] += target_indices.numel()
            continue
        if target_indices.numel() == 0:
            false_positive[class_id] += predicted_indices.numel()
            continue

        pairwise = _mask_iou_matrix(
            predicted_masks[predicted_indices],
            target_masks[target_indices],
        )
        candidates = torch.nonzero(pairwise > iou_threshold, as_tuple=False)
        if candidates.numel():
            candidate_scores = pairwise[candidates[:, 0], candidates[:, 1]]
            order = candidate_scores.argsort(descending=True)
        else:
            order = torch.empty((0,), dtype=torch.long)
        used_predicted: set[int] = set()
        used_target: set[int] = set()
        for candidate_index in order:
            predicted_local = int(candidates[candidate_index, 0].item())
            target_local = int(candidates[candidate_index, 1].item())
            if predicted_local in used_predicted or target_local in used_target:
                continue
            used_predicted.add(predicted_local)
            used_target.add(target_local)
            true_positive[class_id] += 1
            iou_sum[class_id] += pairwise[predicted_local, target_local]
        false_positive[class_id] += predicted_indices.numel() - len(used_predicted)
        false_negative[class_id] += target_indices.numel() - len(used_target)

    return iou_sum, true_positive, false_positive, false_negative


def predicate_recall_counts(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    num_predicates: int,
    ks: Sequence[int] = (20, 50, 100),
    mask_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Count standard mask-grounded scene-graph predicate recall at K."""

    relations = target["relations"].cpu().to(torch.long)
    totals = torch.zeros(num_predicates, dtype=torch.float64)
    matched = {
        int(k): torch.zeros(num_predicates, dtype=torch.float64)
        for k in ks
    }
    if relations.numel() == 0:
        return totals, matched

    target_labels = target["class_labels"].cpu().to(torch.long)
    target_masks = target["masks"].cpu() >= mask_threshold
    predicted_scores = prediction["scores"].detach().cpu()
    order = predicted_scores.argsort(descending=True)
    predicted_subject_labels = prediction["subject_labels"].detach().cpu()
    predicted_object_labels = prediction["object_labels"].detach().cpu()
    predicted_relation_labels = prediction["relation_labels"].detach().cpu()
    predicted_subject_masks = prediction["subject_masks"].detach().cpu() >= mask_threshold
    predicted_object_masks = prediction["object_masks"].detach().cpu() >= mask_threshold

    totals += torch.bincount(
        relations[:, 2],
        minlength=num_predicates,
    ).to(torch.float64)

    for k in ks:
        top = order[: min(int(k), order.numel())]
        for subject_index, object_index, predicate_index in relations.tolist():
            subject_label = int(target_labels[subject_index].item())
            object_label = int(target_labels[object_index].item())
            predicate_index = int(predicate_index)
            candidate_indices = top[
                (predicted_relation_labels[top] == predicate_index)
                & (predicted_subject_labels[top] == subject_label)
                & (predicted_object_labels[top] == object_label)
            ]
            recalled = False
            for prediction_index_tensor in candidate_indices:
                prediction_index = int(prediction_index_tensor.item())
                subject_iou = _mask_iou(
                    predicted_subject_masks[prediction_index],
                    target_masks[subject_index],
                )
                if subject_iou < iou_threshold:
                    continue
                object_iou = _mask_iou(
                    predicted_object_masks[prediction_index],
                    target_masks[object_index],
                )
                if object_iou >= iou_threshold:
                    recalled = True
                    break
            if recalled:
                matched[int(k)][predicate_index] += 1

    return totals, matched


@dataclass
class PsgEvaluationAccumulator:
    num_object_classes: int
    num_thing_classes: int
    num_predicate_classes: int
    recall_ks: tuple[int, ...] = (20, 50, 100)

    def __post_init__(self) -> None:
        self.pq_iou_sum = torch.zeros(self.num_object_classes, dtype=torch.float64)
        self.pq_tp = torch.zeros(self.num_object_classes, dtype=torch.float64)
        self.pq_fp = torch.zeros(self.num_object_classes, dtype=torch.float64)
        self.pq_fn = torch.zeros(self.num_object_classes, dtype=torch.float64)
        self.predicate_total = torch.zeros(
            self.num_predicate_classes,
            dtype=torch.float64,
        )
        self.predicate_matched = {
            int(k): torch.zeros(self.num_predicate_classes, dtype=torch.float64)
            for k in self.recall_ks
        }
        self.image_count = torch.zeros(1, dtype=torch.float64)

    def update(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        *,
        entity_score_threshold: float,
        mask_threshold: float,
        iou_threshold: float,
        thing_nms_threshold: float,
    ) -> None:
        height, width = (int(value) for value in target["size"].tolist())
        target_masks = target["masks"][:, :height, :width].cpu() >= mask_threshold
        predicted_labels, predicted_masks = assemble_panoptic_prediction(
            prediction,
            num_thing_classes=self.num_thing_classes,
            entity_score_threshold=entity_score_threshold,
            mask_threshold=mask_threshold,
            thing_nms_threshold=thing_nms_threshold,
        )
        counts = panoptic_quality_counts(
            predicted_labels,
            predicted_masks,
            target["class_labels"],
            target_masks,
            num_classes=self.num_object_classes,
            iou_threshold=iou_threshold,
        )
        self.pq_iou_sum += counts[0]
        self.pq_tp += counts[1]
        self.pq_fp += counts[2]
        self.pq_fn += counts[3]

        predicate_total, predicate_matched = predicate_recall_counts(
            prediction,
            {
                **target,
                "masks": target_masks,
            },
            num_predicates=self.num_predicate_classes,
            ks=self.recall_ks,
            mask_threshold=mask_threshold,
            iou_threshold=iou_threshold,
        )
        self.predicate_total += predicate_total
        for k in self.recall_ks:
            self.predicate_matched[int(k)] += predicate_matched[int(k)]
        self.image_count += 1

    def distributed_reduce(self, device: torch.device) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        tensors = [
            self.pq_iou_sum,
            self.pq_tp,
            self.pq_fp,
            self.pq_fn,
            self.predicate_total,
            *(self.predicate_matched[int(k)] for k in self.recall_ks),
            self.image_count,
        ]
        for tensor in tensors:
            reduced = tensor.to(device)
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            tensor.copy_(reduced.cpu())

    @staticmethod
    def _pq_average(
        iou_sum: torch.Tensor,
        true_positive: torch.Tensor,
        false_positive: torch.Tensor,
        false_negative: torch.Tensor,
        class_mask: torch.Tensor,
    ) -> tuple[float, float, float, int]:
        denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
        valid = class_mask & (denominator > 0)
        if not valid.any():
            return 0.0, 0.0, 0.0, 0
        pq = iou_sum / denominator.clamp_min(1e-12)
        sq = torch.where(
            true_positive > 0,
            iou_sum / true_positive.clamp_min(1e-12),
            torch.zeros_like(iou_sum),
        )
        rq = true_positive / denominator.clamp_min(1e-12)
        return (
            float(pq[valid].mean().item()),
            float(sq[valid].mean().item()),
            float(rq[valid].mean().item()),
            int(valid.sum().item()),
        )

    def compute(
        self,
        *,
        object_classes: Sequence[str],
        predicate_classes: Sequence[str],
    ) -> dict[str, Any]:
        all_classes = torch.ones(self.num_object_classes, dtype=torch.bool)
        thing_classes = torch.arange(self.num_object_classes) < self.num_thing_classes
        stuff_classes = ~thing_classes
        pq, sq, rq, pq_classes = self._pq_average(
            self.pq_iou_sum,
            self.pq_tp,
            self.pq_fp,
            self.pq_fn,
            all_classes,
        )
        pq_th, sq_th, rq_th, thing_count = self._pq_average(
            self.pq_iou_sum,
            self.pq_tp,
            self.pq_fp,
            self.pq_fn,
            thing_classes,
        )
        pq_st, sq_st, rq_st, stuff_count = self._pq_average(
            self.pq_iou_sum,
            self.pq_tp,
            self.pq_fp,
            self.pq_fn,
            stuff_classes,
        )
        metrics: dict[str, float | int] = {
            "evaluated_images": int(self.image_count.item()),
            "pq": pq,
            "sq": sq,
            "rq": rq,
            "pq_th": pq_th,
            "sq_th": sq_th,
            "rq_th": rq_th,
            "pq_st": pq_st,
            "sq_st": sq_st,
            "rq_st": rq_st,
            "pq_classes": pq_classes,
            "thing_pq_classes": thing_count,
            "stuff_pq_classes": stuff_count,
        }
        predicate_details: dict[str, list[dict[str, Any]]] = {}
        total_relations = float(self.predicate_total.sum().item())
        valid_predicates = self.predicate_total > 0
        for k in self.recall_ks:
            matched = self.predicate_matched[int(k)]
            micro = float(matched.sum().item() / total_relations) if total_relations else 0.0
            class_recall = torch.where(
                valid_predicates,
                matched / self.predicate_total.clamp_min(1.0),
                torch.zeros_like(matched),
            )
            mean = (
                float(class_recall[valid_predicates].mean().item())
                if valid_predicates.any()
                else 0.0
            )
            metrics[f"predicate_recall_at_{k}"] = micro
            metrics[f"predicate_mean_recall_at_{k}"] = mean
            predicate_details[str(k)] = [
                {
                    "predicate_id": index,
                    "predicate": predicate_classes[index],
                    "ground_truth": int(self.predicate_total[index].item()),
                    "matched": int(matched[index].item()),
                    "recall": (
                        float(class_recall[index].item())
                        if self.predicate_total[index] > 0
                        else None
                    ),
                }
                for index in range(self.num_predicate_classes)
            ]

        denominator = self.pq_tp + 0.5 * self.pq_fp + 0.5 * self.pq_fn
        class_pq = torch.where(
            denominator > 0,
            self.pq_iou_sum / denominator.clamp_min(1e-12),
            torch.zeros_like(denominator),
        )
        panoptic_details = [
            {
                "category_id": index,
                "category": object_classes[index],
                "isthing": index < self.num_thing_classes,
                "pq": float(class_pq[index].item()) if denominator[index] > 0 else None,
                "iou_sum": float(self.pq_iou_sum[index].item()),
                "true_positive": int(self.pq_tp[index].item()),
                "false_positive": int(self.pq_fp[index].item()),
                "false_negative": int(self.pq_fn[index].item()),
            }
            for index in range(self.num_object_classes)
        ]
        return {
            "metrics": metrics,
            "predicate_recall_by_class": predicate_details,
            "panoptic_quality_by_class": panoptic_details,
        }
