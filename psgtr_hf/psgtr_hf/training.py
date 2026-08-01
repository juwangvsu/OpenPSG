from __future__ import annotations

import argparse
import os
import json
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from .dataset import OpenPsgDataset, build_openpsg_dataloaders


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PSGTR on OpenPSG annotations")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--annotation-file",
        type=Path,
        default=Path("data/psg/psg_train_val.json"),
    )
    parser.add_argument(
        "--model",
        default="facebook/detr-resnet-50-panoptic",
        help="Hugging Face DETR segmentation checkpoint or local directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work_dirs/psgtr_hf"))
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory, or 'auto' to use output-dir/last_checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Micro-batch size per GPU/process.",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-drop", type=int, default=40)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--train-min-sizes", type=int, nargs="+", default=list(range(480, 801, 32)))
    parser.add_argument("--validation-min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=1333)
    parser.add_argument("--crop-probability", type=float, default=0.5)
    parser.add_argument("--flip-probability", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("epochs and batch-size must be positive; num-workers cannot be negative")
    if args.gradient_accumulation_steps <= 0:
        parser.error("gradient-accumulation-steps must be positive")
    if args.log_every <= 0 or args.save_every <= 0 or args.save_total_limit <= 0:
        parser.error("log-every, save-every, and save-total-limit must be positive")
    if args.lr_drop <= 0 or args.max_grad_norm <= 0:
        parser.error("lr-drop and max-grad-norm must be positive")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_distributed(args: argparse.Namespace) -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA distributed training was requested but CUDA is unavailable")
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device(args.device)
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    return device, rank, local_rank, world_size


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed() or dist.get_rank() == 0


def reduce_metrics(
    sums: dict[str, float],
    count: int,
    device: torch.device,
) -> dict[str, float]:
    names = sorted(sums)
    if not is_distributed():
        return {name: sums[name] / max(count, 1) for name in names}

    values = torch.tensor(
        [*(sums[name] for name in names), float(count)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    total_count = max(float(values[-1].item()), 1.0)
    return {name: float(values[index].item()) / total_count for index, name in enumerate(names)}


def unwrap_model(model: nn.Module) -> nn.Module:
    while True:
        if isinstance(model, DistributedDataParallel):
            model = model.module
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue
        return model


def move_labels(
    labels: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    return [
        {key: value.to(device, non_blocking=True) for key, value in target.items()}
        for target in labels
    ]


def make_optimizer(
    model: nn.Module,
    *,
    lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    backbone = []
    other = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("model.backbone") else other).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": other, "lr": lr},
            {"params": backbone, "lr": backbone_lr},
        ],
        weight_decay=weight_decay,
    )


def _amp_context(device: torch.device, enabled: bool):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    epoch: int,
    global_step: int,
    accumulation_steps: int,
    max_grad_norm: float,
    amp: bool,
    log_every: int,
    log: bool = True,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    count = 0
    start = time.monotonic()
    total_batches = len(loader) if hasattr(loader, "__len__") else None

    for batch_index, batch in enumerate(loader):
        group_size = accumulation_steps
        if total_batches is not None:
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, total_batches - group_start)
        should_step = (batch_index + 1) % accumulation_steps == 0
        if total_batches is not None and batch_index + 1 == total_batches:
            should_step = True

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = move_labels(batch["labels"], device)
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not should_step
            else nullcontext()
        )
        with sync_context:
            with _amp_context(device, amp):
                outputs = model(
                    pixel_values=pixel_values,
                    pixel_mask=pixel_mask,
                    labels=labels,
                )
                if outputs.loss is None:
                    raise RuntimeError("Model did not return a training loss")
                loss = outputs.loss / group_size
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            scaler.scale(loss).backward()

        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        values = {"loss": float(outputs.loss.detach())}
        if outputs.loss_dict:
            values.update({name: float(value.detach()) for name, value in outputs.loss_dict.items()})
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value
        count += 1

        if log and (batch_index + 1) % log_every == 0:
            elapsed = time.monotonic() - start
            average = sums["loss"] / count
            rate = count / max(elapsed, 1e-9)
            print(
                f"epoch={epoch:03d} batch={batch_index + 1} "
                f"loss={average:.4f} batches_per_second={rate:.3f}",
                flush=True,
            )

    return reduce_metrics(sums, count, device), global_step


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        pixel_mask = batch["pixel_mask"].to(device, non_blocking=True)
        labels = move_labels(batch["labels"], device)
        with _amp_context(device, amp):
            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
        if outputs.loss is None:
            raise RuntimeError("Model did not return a validation loss")
        values = {"loss": float(outputs.loss.detach())}
        if outputs.loss_dict:
            values.update({name: float(value.detach()) for name, value in outputs.loss_dict.items()})
        for name, value in values.items():
            sums[name] = sums.get(name, 0.0) + value
        count += 1
    return reduce_metrics(sums, count, device)


def _checkpoint_directories(output_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in output_dir.glob("checkpoint-*")
            if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    output_dir: Path,
    *,
    epoch: int,
    global_step: int,
    metadata: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    checkpoint = output_dir / f"checkpoint-{epoch:04d}"
    temporary = output_dir / f".{checkpoint.name}.tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)
    unwrapped = unwrap_model(model)
    unwrapped.save_pretrained(temporary)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        },
        temporary / "training_state.pt",
    )
    (temporary / "dataset_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (temporary / "training_args.json").write_text(
        json.dumps(serializable_args, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(checkpoint, ignore_errors=True)
    temporary.rename(checkpoint)
    (output_dir / "last_checkpoint").write_text(str(checkpoint.resolve()) + "\n")
    return checkpoint


def resolve_resume(args: argparse.Namespace) -> Path | None:
    if args.resume is None:
        return None
    if args.resume != "auto":
        return Path(args.resume)
    marker = args.output_dir / "last_checkpoint"
    if not marker.is_file():
        raise FileNotFoundError(f"No automatic checkpoint marker at {marker}")
    return Path(marker.read_text(encoding="utf-8").strip())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from .modeling_psgtr import PsgtrForPanopticSceneGraphGeneration

    device, rank, local_rank, world_size = initialize_distributed(args)
    set_seed(args.seed + rank)
    if is_main_process():
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    train_loader, validation_loader = build_openpsg_dataloaders(
        args.annotation_file,
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        train_min_sizes=args.train_min_sizes,
        validation_min_size=args.validation_min_size,
        max_size=args.max_size,
        crop_probability=args.crop_probability,
        flip_probability=args.flip_probability,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        distributed=world_size > 1,
        rank=rank,
        world_size=world_size,
    )
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, OpenPsgDataset):
        raise TypeError("Unexpected training dataset type")
    metadata = train_dataset.metadata
    resume = resolve_resume(args)
    if resume is None:
        model = PsgtrForPanopticSceneGraphGeneration.from_detr_pretrained(
            args.model,
            num_object_labels=len(metadata.object_classes),
            num_relation_labels=len(metadata.predicate_classes),
        )
        model.config.id2label = dict(enumerate(metadata.object_classes))
        model.config.label2id = {
            label: index for index, label in enumerate(metadata.object_classes)
        }
        model.config.relation_id2label = dict(enumerate(metadata.predicate_classes))
        model.config.relation_label2id = {
            label: index for index, label in enumerate(metadata.predicate_classes)
        }
    else:
        model = PsgtrForPanopticSceneGraphGeneration.from_pretrained(resume)
        if model.config.num_object_labels != len(metadata.object_classes):
            raise ValueError("Checkpoint object vocabulary does not match the dataset")
        if model.config.num_relation_labels != len(metadata.predicate_classes):
            raise ValueError("Checkpoint predicate vocabulary does not match the dataset")

    model.to(device)
    optimizer = make_optimizer(
        model,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_drop,
        gamma=0.1,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    global_step = 0
    if resume is not None:
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])

    if args.compile:
        model = torch.compile(model)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
        )

    if is_main_process():
        effective_batch_size = (
            args.batch_size * world_size * args.gradient_accumulation_steps
        )
        print(
            f"train_samples={len(train_loader.dataset)} "
            f"validation_samples={len(validation_loader.dataset)} "
            f"objects={len(metadata.object_classes)} predicates={len(metadata.predicate_classes)} "
            f"world_size={world_size} per_gpu_batch={args.batch_size} "
            f"gradient_accumulation={args.gradient_accumulation_steps} "
            f"effective_batch={effective_batch_size} device={device}",
            flush=True,
        )
    for epoch in range(start_epoch, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_metrics, global_step = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch=epoch,
            global_step=global_step,
            accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            amp=amp_enabled,
            log_every=args.log_every,
            log=is_main_process(),
        )
        validation_metrics = evaluate(model, validation_loader, device, amp=amp_enabled)
        scheduler.step()
        if is_main_process():
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
                f"validation_loss={validation_metrics['loss']:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )

            if epoch % args.save_every == 0 or epoch == args.epochs:
                checkpoint = save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args.output_dir,
                    epoch=epoch,
                    global_step=global_step,
                    metadata=metadata.to_dict(),
                    args=args,
                )
                checkpoints = _checkpoint_directories(args.output_dir)
                for old in checkpoints[: max(0, len(checkpoints) - args.save_total_limit)]:
                    if old != checkpoint:
                        shutil.rmtree(old)
                print(f"saved={checkpoint}", flush=True)
        if is_distributed():
            dist.barrier()

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
