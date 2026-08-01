from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


class PsgtrLossConfig(Protocol):
    num_object_labels: int
    num_relation_labels: int
    object_class_cost: float
    bbox_cost: float
    giou_cost: float
    relation_class_cost: float
    subject_class_loss_coefficient: float
    object_class_loss_coefficient: float
    relation_class_loss_coefficient: float
    bbox_loss_coefficient: float
    giou_loss_coefficient: float
    dice_loss_coefficient: float
    object_eos_coefficient: float
    no_relation_coefficient: float
    train_object_background: bool


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def box_iou(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-6), union


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou, union = box_iou(boxes1, boxes2)
    top_left = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    enclosing_size = (bottom_right - top_left).clamp(min=0)
    enclosing_area = enclosing_size[:, :, 0] * enclosing_size[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    normalizer: float,
) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (probabilities * targets).sum(1) + 1
    denominator = probabilities.sum(1) + targets.sum(1) + 1
    return (1 - numerator / denominator).sum() / normalizer


class PsgtrHungarianMatcher(nn.Module):
    """Match complete subject-predicate-object triplets to decoder queries."""

    def __init__(self, config: PsgtrLossConfig) -> None:
        super().__init__()
        self.num_object_labels = config.num_object_labels
        self.num_relation_labels = config.num_relation_labels
        self.object_class_cost = config.object_class_cost
        self.bbox_cost = config.bbox_cost
        self.giou_cost = config.giou_cost
        self.relation_class_cost = config.relation_class_cost

    def _validate_target(self, target: dict[str, torch.Tensor]) -> None:
        missing = {"class_labels", "boxes", "relations"} - target.keys()
        if missing:
            raise ValueError(f"Missing target keys: {sorted(missing)}")

        class_labels = target["class_labels"]
        boxes = target["boxes"]
        relations = target["relations"]
        if class_labels.ndim != 1:
            raise ValueError("class_labels must have shape [num_entities]")
        if boxes.ndim != 2 or boxes.shape != (class_labels.shape[0], 4):
            raise ValueError("boxes must have shape [num_entities, 4]")
        if relations.ndim != 2 or relations.shape[1] != 3:
            raise ValueError("relations must have shape [num_relations, 3]")
        if class_labels.numel() and (
            class_labels.min() < 0 or class_labels.max() >= self.num_object_labels
        ):
            raise ValueError(
                f"class_labels must be in [0, {self.num_object_labels - 1}]"
            )
        if not relations.numel():
            return

        entity_indices = relations[:, :2]
        if entity_indices.min() < 0 or entity_indices.max() >= class_labels.shape[0]:
            raise ValueError("relation entity indices are outside class_labels")
        predicates = relations[:, 2]
        if predicates.min() < 0 or predicates.max() >= self.num_relation_labels:
            raise ValueError(
                f"predicate IDs must be in [0, {self.num_relation_labels - 1}]"
            )

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        subject_probabilities = outputs["subject_logits"].softmax(-1)
        object_probabilities = outputs["object_logits"].softmax(-1)
        relation_probabilities = outputs["relation_logits"].softmax(-1)
        indices: list[tuple[torch.Tensor, torch.Tensor]] = []

        if len(labels) != subject_probabilities.shape[0]:
            raise ValueError("labels length must equal batch size")

        for batch_index, target in enumerate(labels):
            self._validate_target(target)
            relations = target["relations"].to(
                device=subject_probabilities.device,
                dtype=torch.long,
            )
            if relations.numel() == 0:
                empty = torch.empty(
                    0,
                    dtype=torch.long,
                    device=subject_probabilities.device,
                )
                indices.append((empty, empty))
                continue

            class_labels = target["class_labels"].to(
                device=subject_probabilities.device,
                dtype=torch.long,
            )
            boxes = target["boxes"].to(
                device=subject_probabilities.device,
                dtype=outputs["subject_boxes"].dtype,
            )
            subject_entity = relations[:, 0]
            object_entity = relations[:, 1]
            predicate = relations[:, 2]

            subject_classes = class_labels[subject_entity]
            object_classes = class_labels[object_entity]
            subject_boxes = boxes[subject_entity]
            object_boxes = boxes[object_entity]

            subject_class_cost = -subject_probabilities[
                batch_index, :, subject_classes
            ]
            object_class_cost = -object_probabilities[
                batch_index, :, object_classes
            ]
            relation_class_cost = -relation_probabilities[
                batch_index, :, predicate + 1
            ]
            subject_bbox_cost = torch.cdist(
                outputs["subject_boxes"][batch_index],
                subject_boxes,
                p=1,
            )
            object_bbox_cost = torch.cdist(
                outputs["object_boxes"][batch_index],
                object_boxes,
                p=1,
            )
            subject_giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(outputs["subject_boxes"][batch_index]),
                box_cxcywh_to_xyxy(subject_boxes),
            )
            object_giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(outputs["object_boxes"][batch_index]),
                box_cxcywh_to_xyxy(object_boxes),
            )

            cost = (
                self.object_class_cost
                * (subject_class_cost + object_class_cost)
                + self.relation_class_cost * relation_class_cost
                + self.bbox_cost * (subject_bbox_cost + object_bbox_cost)
                + self.giou_cost * (subject_giou_cost + object_giou_cost)
            )
            source, destination = linear_sum_assignment(
                cost.detach().float().cpu()
            )
            indices.append(
                (
                    torch.as_tensor(
                        source,
                        dtype=torch.long,
                        device=subject_probabilities.device,
                    ),
                    torch.as_tensor(
                        destination,
                        dtype=torch.long,
                        device=subject_probabilities.device,
                    ),
                )
            )

        return indices


class PsgtrLoss(nn.Module):
    """PSGTR matching and classification/box/mask losses."""

    def __init__(self, config: PsgtrLossConfig) -> None:
        super().__init__()
        self.config = config
        self.matcher = PsgtrHungarianMatcher(config)

    @staticmethod
    def _normalizer(
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> float:
        count = torch.tensor(
            [sum(source.numel() for source, _ in indices)],
            dtype=torch.float32,
            device=device,
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(count)
            count /= dist.get_world_size()
        return max(count.item(), 1.0)

    def loss_labels(
        self,
        outputs: dict[str, torch.Tensor],
        labels: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        device = outputs["subject_logits"].device
        batch_size, num_queries = outputs["subject_logits"].shape[:2]
        object_background = self.config.num_object_labels

        relation_targets = torch.zeros(
            (batch_size, num_queries),
            dtype=torch.long,
            device=device,
        )
        subject_positive_logits: list[torch.Tensor] = []
        object_positive_logits: list[torch.Tensor] = []
        subject_positive_targets: list[torch.Tensor] = []
        object_positive_targets: list[torch.Tensor] = []

        subject_all_targets = torch.full(
            (batch_size, num_queries),
            object_background,
            dtype=torch.long,
            device=device,
        )
        object_all_targets = subject_all_targets.clone()

        for batch_index, (source, destination) in enumerate(indices):
            if source.numel() == 0:
                continue
            target = labels[batch_index]
            relations = target["relations"].to(device=device, dtype=torch.long)
            class_labels = target["class_labels"].to(
                device=device,
                dtype=torch.long,
            )
            matched_relations = relations[destination]
            subject_targets = class_labels[matched_relations[:, 0]]
            object_targets = class_labels[matched_relations[:, 1]]

            relation_targets[batch_index, source] = matched_relations[:, 2] + 1
            subject_all_targets[batch_index, source] = subject_targets
            object_all_targets[batch_index, source] = object_targets
            subject_positive_logits.append(
                outputs["subject_logits"][batch_index, source]
            )
            object_positive_logits.append(
                outputs["object_logits"][batch_index, source]
            )
            subject_positive_targets.append(subject_targets)
            object_positive_targets.append(object_targets)

        relation_weights = torch.ones(
            self.config.num_relation_labels + 1,
            device=device,
        )
        relation_weights[0] = self.config.no_relation_coefficient
        relation_loss = F.cross_entropy(
            outputs["relation_logits"].transpose(1, 2),
            relation_targets,
            weight=relation_weights,
        )

        if self.config.train_object_background:
            object_weights = torch.ones(
                self.config.num_object_labels + 1,
                device=device,
            )
            object_weights[-1] = self.config.object_eos_coefficient
            subject_loss = F.cross_entropy(
                outputs["subject_logits"].transpose(1, 2),
                subject_all_targets,
                weight=object_weights,
            )
            object_loss = F.cross_entropy(
                outputs["object_logits"].transpose(1, 2),
                object_all_targets,
                weight=object_weights,
            )
        elif subject_positive_logits:
            subject_loss = F.cross_entropy(
                torch.cat(subject_positive_logits),
                torch.cat(subject_positive_targets),
            )
            object_loss = F.cross_entropy(
                torch.cat(object_positive_logits),
                torch.cat(object_positive_targets),
            )
        else:
            subject_loss = outputs["subject_logits"].sum() * 0
            object_loss = outputs["object_logits"].sum() * 0

        return {
            "loss_subject_ce": subject_loss,
            "loss_object_ce": object_loss,
            "loss_relation_ce": relation_loss,
        }

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        labels: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        normalizer: float,
    ) -> dict[str, torch.Tensor]:
        subject_predictions: list[torch.Tensor] = []
        object_predictions: list[torch.Tensor] = []
        subject_targets: list[torch.Tensor] = []
        object_targets: list[torch.Tensor] = []
        device = outputs["subject_boxes"].device

        for batch_index, (source, destination) in enumerate(indices):
            if source.numel() == 0:
                continue
            target = labels[batch_index]
            relations = target["relations"].to(device=device, dtype=torch.long)
            boxes = target["boxes"].to(
                device=device,
                dtype=outputs["subject_boxes"].dtype,
            )
            matched_relations = relations[destination]
            subject_predictions.append(
                outputs["subject_boxes"][batch_index, source]
            )
            object_predictions.append(
                outputs["object_boxes"][batch_index, source]
            )
            subject_targets.append(boxes[matched_relations[:, 0]])
            object_targets.append(boxes[matched_relations[:, 1]])

        if not subject_predictions:
            zero = outputs["subject_boxes"].sum() * 0
            return {
                "loss_subject_bbox": zero,
                "loss_object_bbox": zero,
                "loss_subject_giou": zero,
                "loss_object_giou": zero,
            }

        subject_prediction = torch.cat(subject_predictions)
        object_prediction = torch.cat(object_predictions)
        subject_target = torch.cat(subject_targets)
        object_target = torch.cat(object_targets)

        subject_bbox_loss = F.l1_loss(
            subject_prediction,
            subject_target,
            reduction="sum",
        ) / normalizer
        object_bbox_loss = F.l1_loss(
            object_prediction,
            object_target,
            reduction="sum",
        ) / normalizer
        subject_giou_loss = (
            1
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(subject_prediction),
                    box_cxcywh_to_xyxy(subject_target),
                )
            )
        ).sum() / normalizer
        object_giou_loss = (
            1
            - torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(object_prediction),
                    box_cxcywh_to_xyxy(object_target),
                )
            )
        ).sum() / normalizer

        return {
            "loss_subject_bbox": subject_bbox_loss,
            "loss_object_bbox": object_bbox_loss,
            "loss_subject_giou": subject_giou_loss,
            "loss_object_giou": object_giou_loss,
        }

    def loss_masks(
        self,
        outputs: dict[str, torch.Tensor],
        labels: list[dict[str, torch.Tensor]],
        indices: list[tuple[torch.Tensor, torch.Tensor]],
        normalizer: float,
    ) -> dict[str, torch.Tensor]:
        subject_loss = outputs["subject_masks"].sum() * 0
        object_loss = outputs["object_masks"].sum() * 0
        device = outputs["subject_masks"].device

        for batch_index, (source, destination) in enumerate(indices):
            if source.numel() == 0:
                continue
            target = labels[batch_index]
            if "masks" not in target:
                raise ValueError("Each label dictionary must contain masks")
            relations = target["relations"].to(device=device, dtype=torch.long)
            masks = target["masks"].to(
                device=device,
                dtype=outputs["subject_masks"].dtype,
            )
            if masks.ndim != 3 or masks.shape[0] != target["class_labels"].shape[0]:
                raise ValueError("masks must have shape [num_entities, height, width]")
            matched_relations = relations[destination]
            target_height, target_width = masks.shape[-2:]

            predicted_subject_masks = F.interpolate(
                outputs["subject_masks"][batch_index, source, None],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            predicted_object_masks = F.interpolate(
                outputs["object_masks"][batch_index, source, None],
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            subject_loss = subject_loss + dice_loss(
                predicted_subject_masks,
                masks[matched_relations[:, 0]],
                normalizer,
            )
            object_loss = object_loss + dice_loss(
                predicted_object_masks,
                masks[matched_relations[:, 1]],
                normalizer,
            )

        return {
            "loss_subject_dice": subject_loss,
            "loss_object_dice": object_loss,
        }

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: list[dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        indices = self.matcher(outputs, labels)
        normalizer = self._normalizer(indices, outputs["subject_logits"].device)

        loss_dict = self.loss_labels(outputs, labels, indices)
        loss_dict.update(self.loss_boxes(outputs, labels, indices, normalizer))
        loss_dict.update(self.loss_masks(outputs, labels, indices, normalizer))

        auxiliary_outputs = outputs.get("auxiliary_outputs") or []
        for layer_index, auxiliary_output in enumerate(auxiliary_outputs):
            auxiliary_indices = self.matcher(auxiliary_output, labels)
            auxiliary_normalizer = self._normalizer(
                auxiliary_indices,
                auxiliary_output["subject_logits"].device,
            )
            auxiliary_losses = self.loss_labels(
                auxiliary_output,
                labels,
                auxiliary_indices,
            )
            auxiliary_losses.update(
                self.loss_boxes(
                    auxiliary_output,
                    labels,
                    auxiliary_indices,
                    auxiliary_normalizer,
                )
            )
            loss_dict.update(
                {
                    f"{key}_{layer_index}": value
                    for key, value in auxiliary_losses.items()
                }
            )

        coefficients = {
            "loss_subject_ce": self.config.subject_class_loss_coefficient,
            "loss_object_ce": self.config.object_class_loss_coefficient,
            "loss_relation_ce": self.config.relation_class_loss_coefficient,
            "loss_subject_bbox": self.config.bbox_loss_coefficient,
            "loss_object_bbox": self.config.bbox_loss_coefficient,
            "loss_subject_giou": self.config.giou_loss_coefficient,
            "loss_object_giou": self.config.giou_loss_coefficient,
            "loss_subject_dice": self.config.dice_loss_coefficient,
            "loss_object_dice": self.config.dice_loss_coefficient,
        }
        total_loss = sum(
            value
            * coefficients[
                key.rsplit("_", 1)[0]
                if key.rsplit("_", 1)[-1].isdigit()
                else key
            ]
            for key, value in loss_dict.items()
        )
        return total_loss, loss_dict
