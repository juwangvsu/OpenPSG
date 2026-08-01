from __future__ import annotations

import torch
from transformers import ResNetConfig

from psgtr_hf import PsgtrConfig, PsgtrForPanopticSceneGraphGeneration


def main() -> None:
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
        num_queries=10,
        num_object_labels=4,
        num_relation_labels=3,
        auxiliary_loss=True,
    )
    model = PsgtrForPanopticSceneGraphGeneration(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    pixel_values = torch.randn(1, 3, 64, 64)
    masks = torch.zeros(2, 64, 64)
    masks[0, 8:32, 8:28] = 1
    masks[1, 30:56, 34:58] = 1
    labels = [
        {
            "class_labels": torch.tensor([1, 2], dtype=torch.long),
            "boxes": torch.tensor(
                [[0.28125, 0.3125, 0.3125, 0.375], [0.71875, 0.671875, 0.375, 0.40625]],
                dtype=torch.float32,
            ),
            "masks": masks,
            "relations": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }
    ]

    outputs = model(pixel_values=pixel_values, labels=labels)
    optimizer.zero_grad(set_to_none=True)
    outputs.loss.backward()
    optimizer.step()

    print(f"loss={outputs.loss.detach().item():.4f}")
    print({name: round(value.detach().item(), 4) for name, value in outputs.loss_dict.items()})


if __name__ == "__main__":
    main()
