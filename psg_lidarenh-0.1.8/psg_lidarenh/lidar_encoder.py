from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    layers: int,
) -> nn.Sequential:
    modules: list[nn.Module] = []
    current = input_dim
    for _ in range(max(0, layers - 1)):
        modules.extend((nn.Linear(current, hidden_dim), nn.GELU()))
        current = hidden_dim
    modules.append(nn.Linear(current, output_dim))
    return nn.Sequential(*modules)


class LidarVoxelEncoder(nn.Module):
    """Convert each variable-size LiDAR cloud into global voxel tokens.

    Every occupied voxel is represented by mean point features, normalized
    voxel center, and log point count. No camera projection or 3D annotation is
    used. This makes the fusion global by design.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 256,
        layers: int = 3,
        point_cloud_range: Sequence[float],
        voxel_size: Sequence[float],
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        if input_dim < 3:
            raise ValueError("input_dim must include x, y, and z")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.max_tokens = int(max_tokens)
        self.register_buffer(
            "range_minimum",
            torch.tensor(point_cloud_range[:3], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "range_maximum",
            torch.tensor(point_cloud_range[3:], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "voxel_size",
            torch.tensor(voxel_size, dtype=torch.float32),
            persistent=False,
        )
        self.token_mlp = _mlp(input_dim + 4, hidden_dim, output_dim, layers)

    def _tokenize_one(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 2 or points.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected points [P, {self.input_dim}], got {tuple(points.shape)}"
            )
        dtype = next(self.token_mlp.parameters()).dtype
        points = points.to(dtype=dtype)
        finite = torch.isfinite(points).all(dim=-1)
        xyz = points[:, :3]
        inside = ((xyz >= self.range_minimum) & (xyz < self.range_maximum)).all(-1)
        points = points[finite & inside]
        if points.numel() == 0:
            # Keep the same encoder parameters in the autograd graph on every
            # DDP rank. A standalone learned empty-token parameter would be
            # unused for normal non-empty KITTI-360 scans and causes DDP's
            # "Expected to have finished reduction" failure.
            empty_features = points.new_zeros((1, self.input_dim + 4))
            return self.token_mlp(empty_features)

        xyz = points[:, :3]
        coordinates = torch.floor(
            (xyz - self.range_minimum) / self.voxel_size
        ).to(torch.long)
        unique, inverse, counts = torch.unique(
            coordinates,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        voxel_count = unique.shape[0]
        feature_sum = points.new_zeros((voxel_count, self.input_dim))
        feature_sum.index_add_(0, inverse, points)
        mean_features = feature_sum / counts[:, None].to(points.dtype)

        center = self.range_minimum + (unique.to(points.dtype) + 0.5) * self.voxel_size
        extent = (self.range_maximum - self.range_minimum).clamp_min(1e-6)
        normalized_center = 2.0 * (center - self.range_minimum) / extent - 1.0
        density = counts.to(points.dtype).log1p().unsqueeze(-1)
        features = torch.cat((mean_features, normalized_center, density), dim=-1)

        if voxel_count > self.max_tokens:
            keep = counts.argsort(descending=True)[: self.max_tokens]
            features = features[keep]
        return self.token_mlp(features)

    def forward(
        self,
        point_clouds: Sequence[torch.Tensor] | torch.Tensor,
        point_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(point_clouds, torch.Tensor):
            if point_clouds.ndim != 3:
                raise ValueError("Padded point clouds must have shape [B, P, F]")
            clouds: list[torch.Tensor] = []
            for batch_index in range(point_clouds.shape[0]):
                cloud = point_clouds[batch_index]
                if point_mask is not None:
                    cloud = cloud[point_mask[batch_index].to(torch.bool)]
                clouds.append(cloud)
        else:
            clouds = list(point_clouds)
        if not clouds:
            raise ValueError("At least one point cloud is required")

        encoded = [self._tokenize_one(cloud) for cloud in clouds]
        token_count = max(tokens.shape[0] for tokens in encoded)
        batch = encoded[0].new_zeros((len(encoded), token_count, self.output_dim))
        valid = torch.zeros(
            (len(encoded), token_count),
            dtype=torch.bool,
            device=encoded[0].device,
        )
        for index, tokens in enumerate(encoded):
            batch[index, : tokens.shape[0]] = tokens
            valid[index, : tokens.shape[0]] = True
        return batch, valid


class GlobalLidarQueryFusion(nn.Module):
    """Cross-attend every PSGTR triplet query to all LiDAR tokens."""

    def __init__(
        self,
        d_model: int,
        attention_heads: int,
        *,
        ffn_dim: int,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            d_model,
            attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        queries: torch.Tensor,
        lidar_tokens: torch.Tensor,
        lidar_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        context, _ = self.attention(
            queries,
            lidar_tokens,
            lidar_tokens,
            key_padding_mask=~lidar_valid_mask,
            need_weights=False,
        )
        context = self.attention_norm(context)
        context = self.ffn_norm(context + self.dropout(self.ffn(context)))
        return queries + torch.tanh(self.gate) * context
