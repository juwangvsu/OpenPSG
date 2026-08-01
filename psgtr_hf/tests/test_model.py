import pytest
import torch

pytest.importorskip("transformers")
from transformers import ResNetConfig

from psgtr_hf import PsgtrConfig, PsgtrForPanopticSceneGraphGeneration


def tiny_model() -> PsgtrForPanopticSceneGraphGeneration:
    backbone = ResNetConfig(
        num_channels=3,
        embedding_size=8,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 1, 1],
        out_features=["stage1", "stage2", "stage3", "stage4"],
    )
    config = PsgtrConfig(
        backbone_config=backbone,
        d_model=32,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        encoder_layers=1,
        decoder_layers=2,
        num_queries=5,
        num_object_labels=4,
        num_relation_labels=3,
        auxiliary_loss=True,
    )
    return PsgtrForPanopticSceneGraphGeneration(config)


def test_forward_backward_and_post_process() -> None:
    model = tiny_model()
    labels = [
        {
            "class_labels": torch.tensor([1, 2]),
            "boxes": torch.tensor(
                [[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]
            ),
            "masks": torch.stack(
                [torch.eye(64), torch.flip(torch.eye(64), [1])]
            ),
            "relations": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
    ]

    output = model(torch.randn(1, 3, 64, 64), labels=labels)
    assert output.subject_logits.shape == (1, 5, 5)
    assert output.object_logits.shape == (1, 5, 5)
    assert output.relation_logits.shape == (1, 5, 4)
    assert output.subject_boxes.shape == (1, 5, 4)
    assert output.object_boxes.shape == (1, 5, 4)
    assert torch.isfinite(output.loss)
    output.loss.backward()

    result = model.post_process_triplets(
        output,
        target_sizes=[(64, 64)],
        top_k=2,
    )
    assert len(result) == 1
    assert result[0]["scores"].shape == (2,)
    assert result[0]["subject_masks"].shape[-2:] == (64, 64)


class _CheckpointLoadSentinel(RuntimeError):
    pass


def test_detr_loader_forces_safetensors_and_safe_revision(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        raise _CheckpointLoadSentinel

    monkeypatch.setattr(
        "psgtr_hf.modeling_psgtr.DetrForSegmentation.from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(_CheckpointLoadSentinel):
        PsgtrForPanopticSceneGraphGeneration.from_detr_pretrained(
            "facebook/detr-resnet-50-panoptic",
            use_safetensors=False,
        )

    assert captured["model_name"] == "facebook/detr-resnet-50-panoptic"
    assert captured["use_safetensors"] is True
    assert captured["revision"] == "refs/pr/10"


def test_detr_loader_preserves_explicit_safetensors_revision(monkeypatch) -> None:
    captured = {}

    def fake_from_pretrained(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        raise _CheckpointLoadSentinel

    monkeypatch.setattr(
        "psgtr_hf.modeling_psgtr.DetrForSegmentation.from_pretrained",
        fake_from_pretrained,
    )

    with pytest.raises(_CheckpointLoadSentinel):
        PsgtrForPanopticSceneGraphGeneration.from_detr_pretrained(
            "facebook/detr-resnet-50-panoptic",
            revision="my-safetensors-revision",
        )

    assert captured["use_safetensors"] is True
    assert captured["revision"] == "my-safetensors-revision"
