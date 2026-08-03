from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.4"

__all__ = [
    "Kitti360Converter",
    "LidarManifestDataset",
    "LidarSceneGraphCollator",
    "LidarVoxelEncoder",
    "PsgLidarEnhConfig",
    "PsgLidarEnhForPanopticSceneGraphGeneration",
    "PsgLidarEnhOutput",
]

if TYPE_CHECKING:
    from .configuration_psg_lidarenh import PsgLidarEnhConfig
    from .dataset import LidarManifestDataset, LidarSceneGraphCollator
    from .kitti360 import Kitti360Converter
    from .lidar_encoder import LidarVoxelEncoder
    from .modeling_psg_lidarenh import (
        PsgLidarEnhForPanopticSceneGraphGeneration,
        PsgLidarEnhOutput,
    )


def __getattr__(name: str) -> Any:
    if name == "PsgLidarEnhConfig":
        from .configuration_psg_lidarenh import PsgLidarEnhConfig

        return PsgLidarEnhConfig
    if name in {
        "PsgLidarEnhForPanopticSceneGraphGeneration",
        "PsgLidarEnhOutput",
    }:
        from .modeling_psg_lidarenh import (
            PsgLidarEnhForPanopticSceneGraphGeneration,
            PsgLidarEnhOutput,
        )

        return {
            "PsgLidarEnhForPanopticSceneGraphGeneration": (
                PsgLidarEnhForPanopticSceneGraphGeneration
            ),
            "PsgLidarEnhOutput": PsgLidarEnhOutput,
        }[name]
    if name == "LidarVoxelEncoder":
        from .lidar_encoder import LidarVoxelEncoder

        return LidarVoxelEncoder
    if name in {"LidarManifestDataset", "LidarSceneGraphCollator"}:
        from .dataset import LidarManifestDataset, LidarSceneGraphCollator

        return {
            "LidarManifestDataset": LidarManifestDataset,
            "LidarSceneGraphCollator": LidarSceneGraphCollator,
        }[name]
    if name == "Kitti360Converter":
        from .kitti360 import Kitti360Converter

        return Kitti360Converter
    raise AttributeError(name)
