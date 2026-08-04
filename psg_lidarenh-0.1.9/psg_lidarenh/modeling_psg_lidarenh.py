from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.utils import ModelOutput

from psgtr_hf.modeling_psgtr import PsgtrForPanopticSceneGraphGeneration

from .configuration_psg_lidarenh import PsgLidarEnhConfig
from .lidar_encoder import GlobalLidarQueryFusion, LidarVoxelEncoder
from .training_modes import image_backbone_module


@dataclass
class PsgLidarEnhOutput(ModelOutput):
    loss: torch.Tensor | None = None
    loss_dict: dict[str, torch.Tensor] | None = None
    subject_logits: torch.Tensor | None = None
    object_logits: torch.Tensor | None = None
    relation_logits: torch.Tensor | None = None
    subject_boxes: torch.Tensor | None = None
    object_boxes: torch.Tensor | None = None
    subject_masks: torch.Tensor | None = None
    object_masks: torch.Tensor | None = None
    lidar_tokens: torch.Tensor | None = None
    lidar_token_mask: torch.Tensor | None = None
    last_hidden_state: torch.Tensor | None = None
    auxiliary_outputs: list[dict[str, torch.Tensor]] | None = None
    decoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    decoder_attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    encoder_last_hidden_state: torch.Tensor | None = None
    encoder_hidden_states: tuple[torch.Tensor, ...] | None = None
    encoder_attentions: tuple[torch.Tensor, ...] | None = None


class PsgLidarEnhForPanopticSceneGraphGeneration(
    PsgtrForPanopticSceneGraphGeneration
):
    """PSGTR whose 2D boxes and predicates receive global LiDAR context.

    Object class logits and image masks remain camera-only. No 3D prediction
    head, 3D box target, point label, or camera-LiDAR calibration is required.
    """

    config_class = PsgLidarEnhConfig

    def __init__(self, config: PsgLidarEnhConfig) -> None:
        super().__init__(config)
        self.lidar_encoder = LidarVoxelEncoder(
            config.lidar_input_dim,
            config.d_model,
            hidden_dim=config.lidar_hidden_dim,
            layers=config.lidar_mlp_layers,
            point_cloud_range=config.point_cloud_range,
            voxel_size=config.voxel_size,
            max_tokens=config.max_lidar_tokens,
        )
        self.lidar_query_fusion = GlobalLidarQueryFusion(
            config.d_model,
            config.lidar_attention_heads,
            ffn_dim=config.lidar_ffn_dim,
            attention_dropout=config.lidar_attention_dropout,
            dropout=config.lidar_fusion_dropout,
        )
        self.post_init()

    @classmethod
    def from_psgtr_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: PsgLidarEnhConfig | None = None,
        **pretrained_kwargs: Any,
    ) -> "PsgLidarEnhForPanopticSceneGraphGeneration":
        """Initialize a fresh PSGTR while retaining only pretrained ResNet weights.

        The transformer encoder/decoder, learned queries, class/relation/box
        heads, mask branch, and LiDAR branch remain newly initialized. Training
        code freezes the copied image backbone for this initialization mode.
        """

        pretrained_kwargs["use_safetensors"] = True
        source = PsgtrForPanopticSceneGraphGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **pretrained_kwargs,
        )
        if config is None:
            config = PsgLidarEnhConfig.from_psgtr_config(source.config)
        model = cls(config)
        source_backbone = image_backbone_module(source)
        target_backbone = image_backbone_module(model)
        result = target_backbone.load_state_dict(
            source_backbone.state_dict(),
            strict=True,
        )
        copied_tensors = len(source_backbone.state_dict())
        model.bootstrap_report = {
            "source": str(pretrained_model_name_or_path),
            "initialization": "pretrained_image_backbone_only",
            "copied_tensors": copied_tensors,
            "new_or_missing_tensors": list(result.missing_keys),
            "unexpected_tensors": list(result.unexpected_keys),
        }
        return model

    @classmethod
    def from_psgtr_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        config: PsgLidarEnhConfig | None = None,
        **pretrained_kwargs: Any,
    ) -> "PsgLidarEnhForPanopticSceneGraphGeneration":
        """Copy all compatible tensors from a safetensors PSGTR checkpoint."""

        pretrained_kwargs["use_safetensors"] = True
        source = PsgtrForPanopticSceneGraphGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **pretrained_kwargs,
        )
        if config is None:
            config = PsgLidarEnhConfig.from_psgtr_config(source.config)
        model = cls(config)
        target_state = model.state_dict()
        compatible = {
            name: value
            for name, value in source.state_dict().items()
            if name in target_state and target_state[name].shape == value.shape
        }
        result = model.load_state_dict(compatible, strict=False)
        model.bootstrap_report = {
            "source": str(pretrained_model_name_or_path),
            "initialization": "full_psgtr_pretrained",
            "copied_tensors": len(compatible),
            "new_or_missing_tensors": list(result.missing_keys),
            "unexpected_tensors": list(result.unexpected_keys),
        }
        return model

    def forward(
        self,
        pixel_values: torch.Tensor,
        lidar_points: list[torch.Tensor] | torch.Tensor | None = None,
        pixel_mask: torch.Tensor | None = None,
        lidar_point_mask: torch.Tensor | None = None,
        labels: list[dict[str, torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> PsgLidarEnhOutput:
        base_output = super().forward(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=None,
            return_dict=True,
            **kwargs,
        )
        queries = base_output.last_hidden_state
        if queries is None:
            raise RuntimeError("PSGTR did not return decoder hidden states")

        if lidar_points is None:
            fused_queries = queries
            lidar_tokens = None
            lidar_valid = None
        else:
            lidar_tokens, lidar_valid = self.lidar_encoder(
                lidar_points,
                lidar_point_mask,
            )
            fused_queries = self.lidar_query_fusion(
                queries,
                lidar_tokens,
                lidar_valid,
            )

        predictions: dict[str, Any] = {
            "subject_logits": base_output.subject_logits,
            "object_logits": base_output.object_logits,
            "relation_logits": self.relation_class_embed(fused_queries),
            "subject_boxes": self.subject_bbox_embed(fused_queries).sigmoid(),
            "object_boxes": self.object_bbox_embed(fused_queries).sigmoid(),
            "subject_masks": base_output.subject_masks,
            "object_masks": base_output.object_masks,
        }
        if base_output.auxiliary_outputs is not None:
            predictions["auxiliary_outputs"] = base_output.auxiliary_outputs

        loss = None
        loss_dict = None
        if labels is not None:
            loss, loss_dict = self.criterion(predictions, labels)

        return PsgLidarEnhOutput(
            loss=loss,
            loss_dict=loss_dict,
            subject_logits=predictions["subject_logits"],
            object_logits=predictions["object_logits"],
            relation_logits=predictions["relation_logits"],
            subject_boxes=predictions["subject_boxes"],
            object_boxes=predictions["object_boxes"],
            subject_masks=predictions["subject_masks"],
            object_masks=predictions["object_masks"],
            lidar_tokens=lidar_tokens,
            lidar_token_mask=lidar_valid,
            last_hidden_state=fused_queries,
            auxiliary_outputs=base_output.auxiliary_outputs,
            decoder_hidden_states=base_output.decoder_hidden_states,
            decoder_attentions=base_output.decoder_attentions,
            cross_attentions=base_output.cross_attentions,
            encoder_last_hidden_state=base_output.encoder_last_hidden_state,
            encoder_hidden_states=base_output.encoder_hidden_states,
            encoder_attentions=base_output.encoder_attentions,
        )
