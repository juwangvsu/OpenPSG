from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import torch
from torch import nn


def test_model_forward_and_psgtr_bootstrap(monkeypatch) -> None:
    transformers = types.ModuleType("transformers")
    transformers_utils = types.ModuleType("transformers.utils")

    class ModelOutput:
        pass

    transformers_utils.ModelOutput = ModelOutput
    transformers.utils = transformers_utils

    psgtr = types.ModuleType("psgtr_hf")
    configuration = types.ModuleType("psgtr_hf.configuration_psgtr")
    modeling = types.ModuleType("psgtr_hf.modeling_psgtr")

    class PsgtrConfig:
        def __init__(self, **kwargs):
            defaults = {
                "d_model": 8,
                "decoder_attention_heads": 2,
                "num_object_labels": 3,
                "num_relation_labels": 4,
            }
            defaults.update(kwargs)
            for key, value in defaults.items():
                setattr(self, key, value)

        def to_dict(self):
            return dict(self.__dict__)

    class FakeCriterion:
        def __call__(self, predictions, labels):
            loss = predictions["subject_boxes"].sum() * 0.0 + 1.0
            return loss, {"loss_fake": loss}

    class PsgtrForPanopticSceneGraphGeneration(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.subject_bbox_embed = nn.Linear(config.d_model, 4)
            self.object_bbox_embed = nn.Linear(config.d_model, 4)
            self.relation_class_embed = nn.Linear(
                config.d_model,
                config.num_relation_labels + 1,
            )
            self.criterion = FakeCriterion()

        def post_init(self):
            return None

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            del path, kwargs
            torch.manual_seed(3)
            return cls(PsgtrConfig())

        def forward(self, pixel_values, pixel_mask=None, labels=None, **kwargs):
            del pixel_mask, labels, kwargs
            batch = pixel_values.shape[0]
            query = torch.arange(
                batch * 5 * self.config.d_model,
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            ).reshape(batch, 5, self.config.d_model) / 100.0
            subject_logits = torch.ones(batch, 5, 4, device=pixel_values.device)
            object_logits = torch.full(
                (batch, 5, 4),
                2.0,
                device=pixel_values.device,
            )
            subject_masks = torch.ones(batch, 5, 2, 2, device=pixel_values.device)
            object_masks = torch.zeros(batch, 5, 2, 2, device=pixel_values.device)
            return SimpleNamespace(
                last_hidden_state=query,
                subject_logits=subject_logits,
                object_logits=object_logits,
                relation_logits=self.relation_class_embed(query),
                subject_boxes=self.subject_bbox_embed(query).sigmoid(),
                object_boxes=self.object_bbox_embed(query).sigmoid(),
                subject_masks=subject_masks,
                object_masks=object_masks,
                auxiliary_outputs=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            )

    configuration.PsgtrConfig = PsgtrConfig
    modeling.PsgtrForPanopticSceneGraphGeneration = (
        PsgtrForPanopticSceneGraphGeneration
    )
    psgtr.configuration_psgtr = configuration
    psgtr.modeling_psgtr = modeling

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(sys.modules, "psgtr_hf", psgtr)
    monkeypatch.setitem(sys.modules, "psgtr_hf.configuration_psgtr", configuration)
    monkeypatch.setitem(sys.modules, "psgtr_hf.modeling_psgtr", modeling)
    for name in (
        "psg_lidarenh.configuration_psg_lidarenh",
        "psg_lidarenh.modeling_psg_lidarenh",
    ):
        sys.modules.pop(name, None)

    config_module = importlib.import_module(
        "psg_lidarenh.configuration_psg_lidarenh"
    )
    model_module = importlib.import_module("psg_lidarenh.modeling_psg_lidarenh")
    config = config_module.PsgLidarEnhConfig(
        d_model=8,
        decoder_attention_heads=2,
        num_object_labels=3,
        num_relation_labels=4,
        lidar_attention_heads=2,
        lidar_hidden_dim=8,
        lidar_ffn_dim=16,
        point_cloud_range=(-10, -10, -3, 10, 10, 3),
        voxel_size=(1, 1, 1),
        max_lidar_tokens=8,
    )
    model = model_module.PsgLidarEnhForPanopticSceneGraphGeneration(config)
    pixels = torch.zeros(2, 3, 4, 4)
    points = [
        torch.tensor([[1.0, 2.0, 0.0, 0.5]]),
        torch.tensor([[2.0, 1.0, 0.0, 0.7]]),
    ]
    output = model(pixel_values=pixels, lidar_points=points, labels=[{}, {}])
    base = PsgtrForPanopticSceneGraphGeneration.forward(model, pixels)
    assert torch.equal(output.subject_logits, base.subject_logits)
    assert torch.equal(output.object_logits, base.object_logits)
    assert torch.equal(output.subject_masks, base.subject_masks)
    assert torch.equal(output.object_masks, base.object_masks)
    assert torch.equal(output.subject_boxes, base.subject_boxes)
    assert torch.equal(output.object_boxes, base.object_boxes)
    assert torch.equal(output.relation_logits, base.relation_logits)
    assert output.lidar_tokens is not None
    assert output.loss is not None
    assert not hasattr(output, "subject_boxes_3d")

    bootstrapped = (
        model_module.PsgLidarEnhForPanopticSceneGraphGeneration
        .from_psgtr_pretrained("checkpoint")
    )
    assert bootstrapped.bootstrap_report["copied_tensors"] > 0
    assert bootstrapped.lidar_query_fusion.gate.item() == 0.0
