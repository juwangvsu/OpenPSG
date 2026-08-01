from __future__ import annotations

from typing import Any

from huggingface_hub.dataclasses import strict
from transformers import DetrConfig


@strict
class PsgtrConfig(DetrConfig):
    """Configuration for Panoptic Scene Graph Transformer.

    Every decoder query predicts a complete ``(subject, predicate, object)``
    triplet. Object background is the final object-logit index. Relation
    background/no-relation is index 0; zero-based dataset predicate IDs are
    shifted by one internally.
    """

    model_type = "psgtr"

    num_object_labels: int = 133
    num_relation_labels: int = 56

    object_class_cost: float = 1.0
    bbox_cost: float = 5.0
    giou_cost: float = 2.0
    relation_class_cost: float = 2.0

    subject_class_loss_coefficient: float = 1.0
    object_class_loss_coefficient: float = 1.0
    relation_class_loss_coefficient: float = 2.0
    bbox_loss_coefficient: float = 5.0
    giou_loss_coefficient: float = 2.0
    dice_loss_coefficient: float = 1.0

    object_eos_coefficient: float = 0.02
    no_relation_coefficient: float = 0.02
    train_object_background: bool = False

    relation_id2label: dict[int, str] | dict[str, str] | None = None
    relation_label2id: dict[str, int] | dict[str, str] | None = None

    def __post_init__(self, **kwargs: Any) -> None:
        if self.num_object_labels <= 0:
            raise ValueError("num_object_labels must be positive")
        if self.num_relation_labels <= 0:
            raise ValueError("num_relation_labels must be positive")
        if self.d_model % self.encoder_attention_heads != 0:
            raise ValueError("d_model must be divisible by encoder_attention_heads")
        if (self.d_model + self.encoder_attention_heads) % 8 != 0:
            raise ValueError(
                "d_model + encoder_attention_heads must be divisible by 8 "
                "for the DETR mask heads"
            )

        kwargs.pop("num_labels", None)
        super().__post_init__(num_labels=self.num_object_labels, **kwargs)
        if self.num_labels != self.num_object_labels:
            raise ValueError(
                "id2label must contain exactly num_object_labels entries"
            )

        if self.relation_id2label is None:
            self.relation_id2label = {
                index: f"relation_{index}"
                for index in range(self.num_relation_labels)
            }
        else:
            self.relation_id2label = {
                int(index): label
                for index, label in self.relation_id2label.items()
            }
        if len(self.relation_id2label) != self.num_relation_labels:
            raise ValueError(
                "relation_id2label must contain exactly num_relation_labels entries"
            )

        if self.relation_label2id is None:
            self.relation_label2id = {
                label: index
                for index, label in self.relation_id2label.items()
            }
        else:
            self.relation_label2id = {
                label: int(index)
                for label, index in self.relation_label2id.items()
            }
        expected = {
            label: index for index, label in self.relation_id2label.items()
        }
        if self.relation_label2id != expected:
            raise ValueError("relation_label2id must invert relation_id2label")

    @classmethod
    def from_detr_config(
        cls,
        config: DetrConfig,
        *,
        num_object_labels: int,
        num_relation_labels: int,
        **overrides: Any,
    ) -> "PsgtrConfig":
        values = config.to_dict()
        for key in (
            "model_type",
            "num_labels",
            "num_object_labels",
            "num_relation_labels",
            "id2label",
            "label2id",
            "relation_id2label",
            "relation_label2id",
        ):
            values.pop(key, None)
        values["architectures"] = ["PsgtrForPanopticSceneGraphGeneration"]
        values.update(overrides)
        return cls(
            num_object_labels=num_object_labels,
            num_relation_labels=num_relation_labels,
            **values,
        )
