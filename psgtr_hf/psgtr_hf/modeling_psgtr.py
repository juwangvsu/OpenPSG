from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.detr.modeling_detr import (
    DetrForSegmentation,
    DetrMaskHeadSmallConv,
    DetrMHAttentionMap,
    DetrModel,
    DetrPreTrainedModel,
)
from transformers.utils import ModelOutput

from .configuration_psgtr import PsgtrConfig
from .loss import PsgtrLoss, box_cxcywh_to_xyxy


@dataclass
class PsgtrOutput(ModelOutput):
    loss: torch.Tensor | None = None
    loss_dict: dict[str, torch.Tensor] | None = None
    subject_logits: torch.Tensor | None = None
    object_logits: torch.Tensor | None = None
    relation_logits: torch.Tensor | None = None
    subject_boxes: torch.Tensor | None = None
    object_boxes: torch.Tensor | None = None
    subject_masks: torch.Tensor | None = None
    object_masks: torch.Tensor | None = None
    auxiliary_outputs: list[dict[str, torch.Tensor]] | None = None
    last_hidden_state: torch.Tensor | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    decoder_attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    encoder_attentions: tuple[torch.Tensor, ...] | None = None


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        dimensions = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(source, target)
            for source, target in zip(dimensions[:-1], dimensions[1:])
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if index + 1 < len(self.layers):
                hidden_states = F.relu(hidden_states)
        return hidden_states


class PsgtrForPanopticSceneGraphGeneration(DetrPreTrainedModel):
    config_class = PsgtrConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"

    def __init__(self, config: PsgtrConfig) -> None:
        super().__init__(config)
        self.model = DetrModel(config)

        self.subject_class_embed = nn.Linear(
            config.d_model,
            config.num_object_labels + 1,
        )
        self.object_class_embed = nn.Linear(
            config.d_model,
            config.num_object_labels + 1,
        )
        self.relation_class_embed = nn.Linear(
            config.d_model,
            config.num_relation_labels + 1,
        )
        self.subject_bbox_embed = MLP(config.d_model, config.d_model, 4, 3)
        self.object_bbox_embed = MLP(config.d_model, config.d_model, 4, 3)

        attention_heads = config.encoder_attention_heads
        self.subject_bbox_attention = DetrMHAttentionMap(
            config.d_model,
            attention_heads,
            dropout=0.0,
        )
        self.object_bbox_attention = DetrMHAttentionMap(
            config.d_model,
            attention_heads,
            dropout=0.0,
        )

        intermediate_channels = self.model.backbone.intermediate_channel_sizes
        if len(intermediate_channels) < 4:
            raise ValueError(
                "PSGTR mask heads need a backbone exposing at least four feature levels"
            )
        mask_head_channels = config.d_model + attention_heads
        fpn_channels = intermediate_channels[::-1][-3:]
        self.subject_mask_head = DetrMaskHeadSmallConv(
            input_channels=mask_head_channels,
            fpn_channels=fpn_channels,
            hidden_size=config.d_model,
            activation_function=config.activation_function,
        )
        self.object_mask_head = DetrMaskHeadSmallConv(
            input_channels=mask_head_channels,
            fpn_channels=fpn_channels,
            hidden_size=config.d_model,
            activation_function=config.activation_function,
        )
        self.criterion = PsgtrLoss(config)
        self.post_init()

    @staticmethod
    def _copy_if_compatible(target: nn.Module, source: nn.Module) -> bool:
        target_state = target.state_dict()
        source_state = source.state_dict()
        if target_state.keys() != source_state.keys():
            return False
        if any(target_state[key].shape != source_state[key].shape for key in target_state):
            return False
        target.load_state_dict(source_state)
        return True

    @classmethod
    def from_detr_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_object_labels: int = 133,
        num_relation_labels: int = 56,
        config: PsgtrConfig | None = None,
        **pretrained_kwargs: Any,
    ) -> "PsgtrForPanopticSceneGraphGeneration":
        """Initialize PSGTR from a Hugging Face DETR segmentation checkpoint.

        Compatible DETR backbone/encoder/decoder weights, query embeddings,
        box heads, attention maps, and mask heads are copied. Predicate logits
        are always initialized from scratch. Object classifiers are copied only
        when the source and destination vocabularies have identical dimensions.
        """

        # PyTorch < 2.6 is blocked from loading legacy ``pytorch_model.bin``
        # checkpoints by recent Transformers releases. Force the safe format
        # even if a caller accidentally supplies ``use_safetensors=False``.
        pretrained_kwargs["use_safetensors"] = True

        # The main revision of this checkpoint only exposes a legacy .bin file.
        # Hugging Face PR revision 10 contains the equivalent safetensors file.
        if pretrained_model_name_or_path == "facebook/detr-resnet-50-panoptic":
            pretrained_kwargs.setdefault("revision", "refs/pr/10")

        detr = DetrForSegmentation.from_pretrained(
            pretrained_model_name_or_path,
            **pretrained_kwargs,
        )
        if config is None:
            config = PsgtrConfig.from_detr_config(
                detr.config,
                num_object_labels=num_object_labels,
                num_relation_labels=num_relation_labels,
                auxiliary_loss=True,
            )
        model = cls(config)

        target_state = model.model.state_dict()
        compatible_state = {
            key: value
            for key, value in detr.detr.model.state_dict().items()
            if key in target_state and value.shape == target_state[key].shape
        }
        model.model.load_state_dict(compatible_state, strict=False)

        model._copy_if_compatible(
            model.subject_class_embed,
            detr.detr.class_labels_classifier,
        )
        model._copy_if_compatible(
            model.object_class_embed,
            detr.detr.class_labels_classifier,
        )
        model._copy_if_compatible(
            model.subject_bbox_embed,
            detr.detr.bbox_predictor,
        )
        model._copy_if_compatible(
            model.object_bbox_embed,
            detr.detr.bbox_predictor,
        )
        model._copy_if_compatible(
            model.subject_bbox_attention,
            detr.bbox_attention,
        )
        model._copy_if_compatible(
            model.object_bbox_attention,
            detr.bbox_attention,
        )
        model._copy_if_compatible(model.subject_mask_head, detr.mask_head)
        model._copy_if_compatible(model.object_mask_head, detr.mask_head)
        return model

    def _prediction_heads(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "subject_logits": self.subject_class_embed(hidden_states),
            "object_logits": self.object_class_embed(hidden_states),
            "relation_logits": self.relation_class_embed(hidden_states),
            "subject_boxes": self.subject_bbox_embed(hidden_states).sigmoid(),
            "object_boxes": self.object_bbox_embed(hidden_states).sigmoid(),
        }

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        decoder_attention_mask: torch.FloatTensor | None = None,
        encoder_outputs: BaseModelOutput | tuple[torch.Tensor, ...] | torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        decoder_inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> PsgtrOutput | tuple[Any, ...]:
        if inputs_embeds is not None:
            raise ValueError(
                "inputs_embeds cannot be used because PSGTR segmentation needs "
                "the backbone's multi-scale feature maps"
            )
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        if output_attentions is not None:
            kwargs["output_attentions"] = output_attentions
        if output_hidden_states is not None:
            kwargs["output_hidden_states"] = output_hidden_states

        batch_size, _, image_height, image_width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = torch.ones(
                (batch_size, image_height, image_width),
                device=pixel_values.device,
                dtype=torch.long,
            )

        vision_features = self.model.backbone(pixel_values, pixel_mask)
        feature_map, mask = vision_features[-1]
        if mask is None:
            raise ValueError("The backbone did not return a feature mask")
        mask = mask.to(torch.bool)
        projected_feature_map = self.model.input_projection(feature_map)
        flattened_features = projected_feature_map.flatten(2).transpose(1, 2)
        spatial_position_embeddings = (
            self.model.position_embedding(
                shape=feature_map.shape,
                device=pixel_values.device,
                dtype=pixel_values.dtype,
                mask=mask,
            )
            .flatten(2)
            .transpose(1, 2)
        )
        flattened_mask = mask.flatten(1)

        if encoder_outputs is None:
            encoder_outputs = self.model.encoder(
                inputs_embeds=flattened_features,
                attention_mask=flattened_mask,
                spatial_position_embeddings=spatial_position_embeddings,
                **kwargs,
            )
        elif isinstance(encoder_outputs, torch.Tensor):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)
        elif not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        object_query_positions = self.model.query_position_embeddings.weight.unsqueeze(0).repeat(
            batch_size,
            1,
            1,
        )
        queries = (
            decoder_inputs_embeds
            if decoder_inputs_embeds is not None
            else torch.zeros_like(object_query_positions)
        )
        decoder_outputs = self.model.decoder(
            inputs_embeds=queries,
            attention_mask=decoder_attention_mask,
            spatial_position_embeddings=spatial_position_embeddings,
            object_queries_position_embeddings=object_query_positions,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=flattened_mask,
            **kwargs,
        )
        sequence_output = decoder_outputs.last_hidden_state
        predictions = self._prediction_heads(sequence_output)

        feature_height, feature_width = feature_map.shape[-2:]
        memory = encoder_outputs.last_hidden_state.transpose(1, 2).reshape(
            batch_size,
            self.config.d_model,
            feature_height,
            feature_width,
        )
        minimum = torch.finfo(memory.dtype).min
        additive_attention_mask = torch.where(
            mask.unsqueeze(1).unsqueeze(1),
            torch.zeros((), device=memory.device, dtype=memory.dtype),
            torch.full((), minimum, device=memory.device, dtype=memory.dtype),
        )
        subject_attention = self.subject_bbox_attention(
            sequence_output,
            memory,
            attention_mask=additive_attention_mask,
        )
        object_attention = self.object_bbox_attention(
            sequence_output,
            memory,
            attention_mask=additive_attention_mask,
        )
        fpn_features = [
            vision_features[2][0],
            vision_features[1][0],
            vision_features[0][0],
        ]
        subject_mask_logits = self.subject_mask_head(
            features=projected_feature_map,
            attention_masks=subject_attention,
            fpn_features=fpn_features,
        )
        object_mask_logits = self.object_mask_head(
            features=projected_feature_map,
            attention_masks=object_attention,
            fpn_features=fpn_features,
        )
        predictions["subject_masks"] = subject_mask_logits.view(
            batch_size,
            self.config.num_queries,
            subject_mask_logits.shape[-2],
            subject_mask_logits.shape[-1],
        )
        predictions["object_masks"] = object_mask_logits.view(
            batch_size,
            self.config.num_queries,
            object_mask_logits.shape[-2],
            object_mask_logits.shape[-1],
        )

        auxiliary_outputs: list[dict[str, torch.Tensor]] | None = None
        intermediate_hidden_states = decoder_outputs.intermediate_hidden_states
        if self.config.auxiliary_loss and intermediate_hidden_states is not None:
            auxiliary_outputs = [
                self._prediction_heads(hidden_state)
                for hidden_state in intermediate_hidden_states[:-1]
            ]
            predictions["auxiliary_outputs"] = auxiliary_outputs

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = self.criterion(predictions, labels)

        if not return_dict:
            values: tuple[Any, ...] = (
                predictions["subject_logits"],
                predictions["object_logits"],
                predictions["relation_logits"],
                predictions["subject_boxes"],
                predictions["object_boxes"],
                predictions["subject_masks"],
                predictions["object_masks"],
                auxiliary_outputs,
            )
            return ((loss, loss_dict) + values) if loss is not None else values

        return PsgtrOutput(
            loss=loss,
            loss_dict=loss_dict,
            subject_logits=predictions["subject_logits"],
            object_logits=predictions["object_logits"],
            relation_logits=predictions["relation_logits"],
            subject_boxes=predictions["subject_boxes"],
            object_boxes=predictions["object_boxes"],
            subject_masks=predictions["subject_masks"],
            object_masks=predictions["object_masks"],
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=sequence_output,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    @torch.no_grad()
    def post_process_triplets(
        self,
        outputs: PsgtrOutput,
        target_sizes: torch.Tensor | list[tuple[int, int]] | None = None,
        score_threshold: float = 0.0,
        top_k: int | None = None,
        mask_threshold: float | None = 0.5,
    ) -> list[dict[str, torch.Tensor]]:
        """Convert query predictions into per-image scene-graph triplets.

        ``target_sizes`` contains ``(height, width)``. Returned predicate IDs
        are zero-based dataset IDs, not the internal no-relation-shifted IDs.
        """

        required = (
            outputs.subject_logits,
            outputs.object_logits,
            outputs.relation_logits,
            outputs.subject_boxes,
            outputs.object_boxes,
            outputs.subject_masks,
            outputs.object_masks,
        )
        if any(value is None for value in required):
            raise ValueError("outputs do not contain complete PSGTR predictions")

        subject_probabilities = outputs.subject_logits.softmax(-1)[..., :-1]
        object_probabilities = outputs.object_logits.softmax(-1)[..., :-1]
        relation_probabilities = outputs.relation_logits.softmax(-1)[..., 1:]
        subject_scores, subject_labels = subject_probabilities.max(-1)
        object_scores, object_labels = object_probabilities.max(-1)
        relation_scores, relation_labels = relation_probabilities.max(-1)
        scores = (
            subject_scores * object_scores * relation_scores
        ).clamp(min=0).pow(1 / 3)

        batch_size = scores.shape[0]
        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("target_sizes length must equal batch size")

        results: list[dict[str, torch.Tensor]] = []
        for batch_index in range(batch_size):
            selected = torch.nonzero(
                scores[batch_index] >= score_threshold,
                as_tuple=False,
            ).flatten()
            if top_k is not None and selected.numel() > top_k:
                order = scores[batch_index, selected].topk(top_k).indices
                selected = selected[order]

            subject_boxes = box_cxcywh_to_xyxy(
                outputs.subject_boxes[batch_index, selected]
            )
            object_boxes = box_cxcywh_to_xyxy(
                outputs.object_boxes[batch_index, selected]
            )
            subject_masks = outputs.subject_masks[batch_index, selected]
            object_masks = outputs.object_masks[batch_index, selected]

            if target_sizes is not None:
                height, width = target_sizes[batch_index]
                height = int(height.item()) if isinstance(height, torch.Tensor) else int(height)
                width = int(width.item()) if isinstance(width, torch.Tensor) else int(width)
                scale = subject_boxes.new_tensor([width, height, width, height])
                subject_boxes = subject_boxes * scale
                object_boxes = object_boxes * scale
                subject_masks = F.interpolate(
                    subject_masks[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                object_masks = F.interpolate(
                    object_masks[:, None],
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]

            subject_masks = subject_masks.sigmoid()
            object_masks = object_masks.sigmoid()
            if mask_threshold is not None:
                subject_masks = subject_masks >= mask_threshold
                object_masks = object_masks >= mask_threshold

            results.append(
                {
                    "scores": scores[batch_index, selected],
                    "subject_scores": subject_scores[batch_index, selected],
                    "object_scores": object_scores[batch_index, selected],
                    "relation_scores": relation_scores[batch_index, selected],
                    "subject_labels": subject_labels[batch_index, selected],
                    "object_labels": object_labels[batch_index, selected],
                    "relation_labels": relation_labels[batch_index, selected],
                    "subject_boxes": subject_boxes,
                    "object_boxes": object_boxes,
                    "subject_masks": subject_masks,
                    "object_masks": object_masks,
                    "query_indices": selected,
                }
            )
        return results
