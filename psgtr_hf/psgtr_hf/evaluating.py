from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Subset

from .dataset import OpenPsgDataset, PsgCollator, PsgImageTransforms, seed_worker
from .inference import resolve_checkpoint
from .training import (
    distributed_barrier,
    evaluate_psg,
    initialize_distributed,
    is_distributed,
    is_main_process,
    select_random_evaluation_indices,
    set_seed,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one PSGTR checkpoint on a deterministic random subset of "
            "the OpenPSG train split, validation split, or both."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Exact checkpoint directory, or a training output directory containing "
            "last_checkpoint."
        ),
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotation-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "both"),
        default="both",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of random images evaluated per requested split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument(
        "--recall-k",
        type=int,
        nargs="+",
        default=[20, 50, 100],
    )
    parser.add_argument("--entity-score-threshold", type=float, default=0.25)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--thing-nms-threshold", type=float, default=0.8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.min_size <= 0 or args.max_size <= 0:
        parser.error("--min-size and --max-size must be positive")
    if not args.recall_k or any(value <= 0 for value in args.recall_k):
        parser.error("--recall-k must contain positive values")
    args.recall_k = sorted(set(args.recall_k))
    for name in (
        "entity_score_threshold",
        "mask_threshold",
        "iou_threshold",
        "thing_nms_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    return args


def requested_splits(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("train", "validation")
    return (value,)


def build_evaluation_loader(
    args: argparse.Namespace,
    *,
    split: str,
    rank: int,
    world_size: int,
) -> tuple[OpenPsgDataset, DataLoader, list[int]]:
    transforms = PsgImageTransforms(
        training=False,
        min_size_choices=(args.min_size,),
        max_size=args.max_size,
    )
    dataset = OpenPsgDataset(
        args.annotation_file,
        args.data_root,
        split=split,
        transforms=transforms,
        randomize_duplicate_relations=False,
    )
    split_seed = args.seed + (1 if split == "validation" else 0)
    selected_indices = select_random_evaluation_indices(
        len(dataset),
        args.samples,
        split_seed,
    )

    # Each selected image is assigned to exactly one rank. Unlike
    # DistributedSampler, this does not pad the sample list with duplicates.
    local_indices = selected_indices[rank::world_size]
    generator = torch.Generator().manual_seed(split_seed + rank)
    loader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=PsgCollator(),
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return dataset, loader, selected_indices


def _selection_record(
    dataset: OpenPsgDataset,
    selected_indices: Sequence[int],
) -> list[dict[str, int]]:
    return [
        {
            "dataset_index": int(index),
            "image_id": int(dataset.samples[index]["image_id"]),
        }
        for index in selected_indices
    ]


def _format_metrics(split: str, result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    recalls = " ".join(
        f"R@{k}={100 * metrics[f'predicate_recall_at_{k}']:.2f} "
        f"mR@{k}={100 * metrics[f'predicate_mean_recall_at_{k}']:.2f}"
        for k in sorted(
            int(name.removeprefix("predicate_recall_at_"))
            for name in metrics
            if name.startswith("predicate_recall_at_")
        )
    )
    return (
        f"split={split} samples={int(metrics['evaluated_images'])} "
        f"loss={metrics['loss']:.4f} "
        f"PQ={100 * metrics['pq']:.2f} "
        f"SQ={100 * metrics['sq']:.2f} "
        f"RQ={100 * metrics['rq']:.2f} "
        f"PQ_th={100 * metrics['pq_th']:.2f} "
        f"PQ_st={100 * metrics['pq_st']:.2f} "
        f"{recalls}"
    )


def _report_name(checkpoint: Path) -> str:
    return f"evaluation-{checkpoint.name}.json"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .modeling_psgtr import PsgtrForPanopticSceneGraphGeneration

    checkpoint = resolve_checkpoint(args.checkpoint)
    device, rank, _, world_size = initialize_distributed(args)
    try:
        set_seed(args.seed + rank)
        if is_main_process():
            args.output_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier()

        loaders: dict[str, DataLoader] = {}
        selections: dict[str, list[dict[str, int]]] = {}
        metadata = None
        for split in requested_splits(args.split):
            dataset, loader, indices = build_evaluation_loader(
                args,
                split=split,
                rank=rank,
                world_size=world_size,
            )
            if metadata is None:
                metadata = dataset.metadata
            elif dataset.metadata != metadata:
                raise ValueError("Train and validation metadata do not match")
            loaders[split] = loader
            selections[split] = _selection_record(dataset, indices)

        if metadata is None:
            raise RuntimeError("No evaluation split was selected")

        model = PsgtrForPanopticSceneGraphGeneration.from_pretrained(
            checkpoint,
            use_safetensors=True,
        )
        if model.config.num_object_labels != len(metadata.object_classes):
            raise ValueError("Checkpoint object vocabulary does not match the dataset")
        if model.config.num_relation_labels != len(metadata.predicate_classes):
            raise ValueError("Checkpoint predicate vocabulary does not match the dataset")
        model.to(device)

        amp_enabled = args.amp and device.type == "cuda"
        results: dict[str, Any] = {}
        for split in requested_splits(args.split):
            result = evaluate_psg(
                model,
                loaders[split],
                device,
                metadata,
                amp=amp_enabled,
                recall_ks=args.recall_k,
                entity_score_threshold=args.entity_score_threshold,
                mask_threshold=args.mask_threshold,
                iou_threshold=args.iou_threshold,
                thing_nms_threshold=args.thing_nms_threshold,
            )
            results[split] = result
            if is_main_process():
                print(_format_metrics(split, result), flush=True)

        if is_main_process():
            selection_path = args.output_dir / f"evaluation-{checkpoint.name}-samples.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint),
                        "seed": args.seed,
                        "requested_samples_per_split": args.samples,
                        "splits": selections,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            report = {
                "checkpoint": str(checkpoint),
                "device": str(device),
                "world_size": world_size,
                "parameters": {
                    "split": args.split,
                    "samples_per_split": args.samples,
                    "seed": args.seed,
                    "batch_size_per_process": args.batch_size,
                    "min_size": args.min_size,
                    "max_size": args.max_size,
                    "recall_k": args.recall_k,
                    "entity_score_threshold": args.entity_score_threshold,
                    "mask_threshold": args.mask_threshold,
                    "iou_threshold": args.iou_threshold,
                    "thing_nms_threshold": args.thing_nms_threshold,
                    "amp": amp_enabled,
                },
                "sample_selection_file": str(selection_path),
                "results": results,
            }
            report_path = args.output_dir / _report_name(checkpoint)
            report_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"saved={report_path}", flush=True)
        distributed_barrier()
    finally:
        if is_distributed():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
