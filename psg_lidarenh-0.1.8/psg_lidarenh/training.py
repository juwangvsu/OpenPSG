from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from psgtr_hf.dataset import OpenPsgDataset, PsgImageTransforms

from .dataset import (
    LidarManifestDataset,
    LidarSceneGraphCollator,
    resolve_openpsg_data_root,
)
from .evaluation import evaluate_model
from .modeling_psg_lidarenh import (
    PsgLidarEnhForPanopticSceneGraphGeneration,
)


class RankPartitionSampler(Sampler[int]):
    def __init__(self, length: int, rank: int, world_size: int) -> None:
        self.indices = list(range(length))[rank::world_size]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train global LiDAR-enhanced PSGTR")
    parser.add_argument("--psgtr-checkpoint", type=Path)
    parser.add_argument("--resume", type=str)
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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--eval-recall-k", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.resume is None and args.psgtr_checkpoint is None:
        parser.error("Provide --psgtr-checkpoint for bootstrap or --resume")
    if args.eval_every <= 0 or args.eval_samples <= 0:
        parser.error("Evaluation cadence and sample count must be positive")
    return args


def initialize_distributed(argument: str) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if argument.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA distributed training requested but unavailable")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            dist.init_process_group("nccl")
        else:
            device = torch.device(argument)
            dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(argument)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
    return device, rank, local_rank, world_size


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_resume(value: str | None, output_dir: Path) -> Path | None:
    if value is None:
        return None
    if value != "auto":
        return Path(value)
    marker = output_dir / "last_checkpoint"
    if not marker.is_file():
        return None
    return Path(marker.read_text(encoding="utf-8").strip())


def build_dataset(
    args: argparse.Namespace,
    split: str,
    *,
    training: bool,
) -> LidarManifestDataset:
    base = OpenPsgDataset(
        args.annotation_file,
        resolve_openpsg_data_root(args.data_root),
        split=split,
        transforms=PsgImageTransforms(training=training),
        filter_empty_relations=True,
        deduplicate_relations=not training,
    )
    return LidarManifestDataset(base, args.lidar_manifest)


def sample_subset(dataset: Any, count: int, seed: int) -> tuple[Subset, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    size = min(int(count), len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices), indices


def make_eval_loader(
    dataset: Any,
    batch_size: int,
    workers: int,
    device: torch.device,
    rank: int,
    world_size: int,
) -> DataLoader:
    sampler = RankPartitionSampler(len(dataset), rank, world_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        collate_fn=LidarSceneGraphCollator(),
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    epoch: int,
    global_step: int,
) -> Path:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    checkpoint = output_dir / f"checkpoint-{epoch:04d}"
    temporary = output_dir / f".{checkpoint.name}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    unwrapped.save_pretrained(temporary, safe_serialization=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
        },
        temporary / "training_state.pt",
    )
    shutil.rmtree(checkpoint, ignore_errors=True)
    temporary.rename(checkpoint)
    (output_dir / "last_checkpoint").write_text(
        str(checkpoint.resolve()) + "\n",
        encoding="utf-8",
    )
    return checkpoint


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device, rank, local_rank, world_size = initialize_distributed(args.device)
    seed_everything(args.seed + rank)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    train_dataset = build_dataset(args, "train", training=True)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=LidarSceneGraphCollator(),
    )

    train_eval_full = build_dataset(args, "train", training=False)
    val_eval_full = build_dataset(args, "validation", training=False)
    train_eval, train_eval_indices = sample_subset(
        train_eval_full,
        args.eval_samples,
        args.seed + 1001,
    )
    val_eval, val_eval_indices = sample_subset(
        val_eval_full,
        args.eval_samples,
        args.seed + 2001,
    )
    # Evaluation is intentionally rank-zero-only. The fixed subsets are small,
    # while distributed metric reduction is fragile when per-rank samples
    # produce different optional loss keys. Other ranks wait at the epoch
    # barrier and resume training after rank zero finishes evaluation.
    train_eval_loader = make_eval_loader(
        train_eval,
        args.eval_batch_size,
        args.num_workers,
        device,
        0,
        1,
    )
    val_eval_loader = make_eval_loader(
        val_eval,
        args.eval_batch_size,
        args.num_workers,
        device,
        0,
        1,
    )

    resume = resolve_resume(args.resume, args.output_dir)
    start_epoch = 1
    global_step = 0
    if resume is not None:
        model = PsgLidarEnhForPanopticSceneGraphGeneration.from_pretrained(
            resume,
            use_safetensors=True,
        )
    else:
        model = PsgLidarEnhForPanopticSceneGraphGeneration.from_psgtr_pretrained(
            args.psgtr_checkpoint,
        )
    metadata = train_dataset.base_dataset.metadata
    expected_objects = len(metadata.thing_classes) + len(metadata.stuff_classes)
    if model.config.num_object_labels != expected_objects:
        raise ValueError(
            "Checkpoint object vocabulary does not match converted dataset: "
            f"{model.config.num_object_labels} != {expected_objects}"
        )
    if model.config.num_relation_labels != len(metadata.predicate_classes):
        raise ValueError("Checkpoint predicate vocabulary does not match dataset")
    model.to(device)

    backbone_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "model.backbone" in name:
            backbone_parameters.append(parameter)
        else:
            other_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": other_parameters, "lr": args.lr},
            {"params": backbone_parameters, "lr": args.backbone_lr},
        ],
        weight_decay=args.weight_decay,
    )
    if resume is not None:
        state_path = resume / "training_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = int(state["epoch"]) + 1
            global_step = int(state.get("global_step", 0))

    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    if rank == 0:
        sample_report = {
            "seed": args.seed,
            "train_indices": train_eval_indices,
            "validation_indices": val_eval_indices,
        }
        (args.output_dir / "evaluation-samples.json").write_text(
            json.dumps(sample_report, indent=2) + "\n",
            encoding="utf-8",
        )

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros(1, dtype=torch.float64, device=device)
        batch_count = torch.zeros(1, dtype=torch.float64, device=device)
        for batch_index, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
            lidar_points = [
                points.to(device, non_blocking=True)
                for points in batch["lidar_points"]
            ]
            labels = [
                {
                    key: value.to(device, non_blocking=True)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in target.items()
                }
                for target in batch["labels"]
            ]
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            synchronization = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel) and not should_step
                else nullcontext()
            )
            with synchronization:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    output = model(
                        pixel_values=pixel_values,
                        pixel_mask=pixel_mask,
                        lidar_points=lidar_points,
                        labels=labels,
                    )
                    if output.loss is None:
                        raise RuntimeError("Model returned no loss")
                    scaled_loss = output.loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            total_loss += output.loss.detach().to(torch.float64)
            batch_count += 1

        if world_size > 1:
            dist.all_reduce(total_loss)
            dist.all_reduce(batch_count)
        train_loss = float((total_loss / batch_count.clamp_min(1)).item())
        report: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
        }
        if epoch % args.eval_every == 0 and rank == 0:
            print(
                f"epoch={epoch:03d} evaluating train "
                f"samples={len(train_eval)} on rank=0",
                flush=True,
            )
            report["train_evaluation"] = evaluate_model(
                model,
                train_eval_loader,
                device,
                metadata,
                amp=args.amp,
                recall_ks=args.eval_recall_k,
                reduce_across_processes=False,
                progress_label=f"epoch={epoch:03d} train-eval",
            )
            print(
                f"epoch={epoch:03d} evaluating validation "
                f"samples={len(val_eval)} on rank=0",
                flush=True,
            )
            report["validation_evaluation"] = evaluate_model(
                model,
                val_eval_loader,
                device,
                metadata,
                amp=args.amp,
                recall_ks=args.eval_recall_k,
                reduce_across_processes=False,
                progress_label=f"epoch={epoch:03d} validation-eval",
                progress_every=1,
            )
        if rank == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.5f} step={global_step}",
                flush=True,
            )
            if epoch % args.save_every == 0 or epoch == args.epochs:
                report["checkpoint"] = str(
                    save_checkpoint(
                        model,
                        optimizer,
                        args.output_dir,
                        epoch,
                        global_step,
                    )
                )
            if epoch % args.eval_every == 0:
                path = args.output_dir / f"evaluation-epoch-{epoch:04d}.json"
                path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                with (args.output_dir / "evaluation-history.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as stream:
                    stream.write(json.dumps(report) + "\n")
        if world_size > 1:
            dist.barrier(device_ids=[local_rank] if device.type == "cuda" else None)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
