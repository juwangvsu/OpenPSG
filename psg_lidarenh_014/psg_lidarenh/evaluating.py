from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from .evaluation import evaluate_model
from .modeling_psg_lidarenh import PsgLidarEnhForPanopticSceneGraphGeneration
from .training import (
    build_dataset,
    initialize_distributed,
    make_eval_loader,
    sample_subset,
    seed_everything,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a LiDAR-enhanced PSGTR checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Dataset root containing coco/, or the coco directory itself; "
            "both forms are accepted"
        ),
    )
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--lidar-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "both"), default="both")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--recall-k", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_checkpoint(path: Path) -> Path:
    marker = path / "last_checkpoint"
    if marker.is_file():
        return Path(marker.read_text(encoding="utf-8").strip())
    return path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device, rank, local_rank, world_size = initialize_distributed(args.device)
    seed_everything(args.seed + rank)
    checkpoint = resolve_checkpoint(args.checkpoint)
    model = PsgLidarEnhForPanopticSceneGraphGeneration.from_pretrained(
        checkpoint,
        use_safetensors=True,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    requested = (
        ("train", "validation") if args.split == "both" else (args.split,)
    )
    report = {"checkpoint": str(checkpoint.resolve()), "splits": {}}
    sample_report: dict[str, list[int]] = {}
    for offset, split in enumerate(requested):
        dataset = build_dataset(args, split, training=False)
        subset, indices = sample_subset(dataset, args.samples, args.seed + 1000 + offset)
        loader = make_eval_loader(
            subset,
            args.batch_size,
            args.num_workers,
            device,
            rank,
            world_size,
        )
        report["splits"][split] = evaluate_model(
            model,
            loader,
            device,
            dataset.base_dataset.metadata,
            amp=args.amp,
            recall_ks=args.recall_k,
        )
        sample_report[split] = indices
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        name = checkpoint.name
        (args.output_dir / f"evaluation-{name}.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / f"evaluation-{name}-samples.json").write_text(
            json.dumps(sample_report, indent=2) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
