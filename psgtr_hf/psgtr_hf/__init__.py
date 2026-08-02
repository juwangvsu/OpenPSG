from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.5.0"

__all__ = [
    "PsgtrConfig",
    "OpenPsgDataset",
    "OpenPsgMetadata",
    "PsgCollator",
    "PsgImageTransforms",
    "PsgtrForPanopticSceneGraphGeneration",
    "build_openpsg_dataloaders",
    "PsgtrHungarianMatcher",
    "PsgtrLoss",
    "PsgtrOutput",
    "PsgEvaluationAccumulator",
]

if TYPE_CHECKING:
    from .configuration_psgtr import PsgtrConfig
    from .dataset import (
        OpenPsgDataset,
        OpenPsgMetadata,
        PsgCollator,
        PsgImageTransforms,
        build_openpsg_dataloaders,
    )
    from .loss import PsgtrHungarianMatcher, PsgtrLoss
    from .metrics import PsgEvaluationAccumulator
    from .modeling_psgtr import (
        PsgtrForPanopticSceneGraphGeneration,
        PsgtrOutput,
    )


def __getattr__(name: str) -> Any:
    if name in {
        "OpenPsgDataset",
        "OpenPsgMetadata",
        "PsgCollator",
        "PsgImageTransforms",
        "build_openpsg_dataloaders",
    }:
        from .dataset import (
            OpenPsgDataset,
            OpenPsgMetadata,
            PsgCollator,
            PsgImageTransforms,
            build_openpsg_dataloaders,
        )

        return {
            "OpenPsgDataset": OpenPsgDataset,
            "OpenPsgMetadata": OpenPsgMetadata,
            "PsgCollator": PsgCollator,
            "PsgImageTransforms": PsgImageTransforms,
            "build_openpsg_dataloaders": build_openpsg_dataloaders,
        }[name]
    if name == "PsgtrConfig":
        from .configuration_psgtr import PsgtrConfig

        return PsgtrConfig
    if name in {"PsgtrHungarianMatcher", "PsgtrLoss"}:
        from .loss import PsgtrHungarianMatcher, PsgtrLoss

        return {
            "PsgtrHungarianMatcher": PsgtrHungarianMatcher,
            "PsgtrLoss": PsgtrLoss,
        }[name]
    if name == "PsgEvaluationAccumulator":
        from .metrics import PsgEvaluationAccumulator

        return PsgEvaluationAccumulator
    if name in {"PsgtrForPanopticSceneGraphGeneration", "PsgtrOutput"}:
        from .modeling_psgtr import (
            PsgtrForPanopticSceneGraphGeneration,
            PsgtrOutput,
        )

        return {
            "PsgtrForPanopticSceneGraphGeneration": (
                PsgtrForPanopticSceneGraphGeneration
            ),
            "PsgtrOutput": PsgtrOutput,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
