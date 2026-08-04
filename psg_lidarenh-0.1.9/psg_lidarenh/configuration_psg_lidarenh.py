from __future__ import annotations

from typing import Any, Sequence

from psgtr_hf.configuration_psgtr import PsgtrConfig


class PsgLidarEnhConfig(PsgtrConfig):
    """PSGTR configuration with a global LiDAR-token fusion branch."""

    model_type = "psg_lidarenh"

    def __init__(
        self,
        *,
        lidar_input_dim: int = 4,
        lidar_hidden_dim: int = 256,
        lidar_mlp_layers: int = 3,
        lidar_attention_heads: int | None = None,
        lidar_attention_dropout: float = 0.0,
        lidar_ffn_dim: int = 1024,
        lidar_fusion_dropout: float = 0.1,
        point_cloud_range: Sequence[float] = (
            -50.0,
            -50.0,
            -5.0,
            50.0,
            50.0,
            3.0,
        ),
        voxel_size: Sequence[float] = (0.5, 0.5, 0.5),
        max_lidar_tokens: int = 4096,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.lidar_input_dim = int(lidar_input_dim)
        self.lidar_hidden_dim = int(lidar_hidden_dim)
        self.lidar_mlp_layers = int(lidar_mlp_layers)
        self.lidar_attention_heads = int(
            lidar_attention_heads or self.decoder_attention_heads
        )
        self.lidar_attention_dropout = float(lidar_attention_dropout)
        self.lidar_ffn_dim = int(lidar_ffn_dim)
        self.lidar_fusion_dropout = float(lidar_fusion_dropout)
        self.point_cloud_range = tuple(float(value) for value in point_cloud_range)
        self.voxel_size = tuple(float(value) for value in voxel_size)
        self.max_lidar_tokens = int(max_lidar_tokens)
        self._validate_lidar_fields()

    def _validate_lidar_fields(self) -> None:
        if self.lidar_input_dim < 3:
            raise ValueError("lidar_input_dim must include x, y, and z")
        if self.lidar_mlp_layers < 1:
            raise ValueError("lidar_mlp_layers must be positive")
        if self.d_model % self.lidar_attention_heads:
            raise ValueError("d_model must be divisible by lidar_attention_heads")
        if len(self.point_cloud_range) != 6:
            raise ValueError("point_cloud_range must contain six values")
        if len(self.voxel_size) != 3 or any(value <= 0 for value in self.voxel_size):
            raise ValueError("voxel_size must contain three positive values")
        if self.max_lidar_tokens <= 0:
            raise ValueError("max_lidar_tokens must be positive")
        if any(
            low >= high
            for low, high in zip(
                self.point_cloud_range[:3],
                self.point_cloud_range[3:],
            )
        ):
            raise ValueError("point_cloud_range minima must be below maxima")

    @classmethod
    def from_psgtr_config(
        cls,
        config: PsgtrConfig,
        **overrides: Any,
    ) -> "PsgLidarEnhConfig":
        values = config.to_dict()
        values.pop("model_type", None)
        values.update(overrides)
        return cls(**values)
