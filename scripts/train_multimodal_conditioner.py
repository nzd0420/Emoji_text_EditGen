#!/usr/bin/env python3
"""Train the multimodal emoji instruction encoder with NVIDIA GPU optimizations."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from emoji_editing import (
    EmojiEditCollator,
    EmojiEditMultimodalConfig,
    EmojiEditMultimodalDataset,
    EmojiEditMultimodalEncoder,
    build_label_vocab_from_csv,
    save_label_vocab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair-csv",
        type=Path,
        default=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),
        help="Path to the generated edit-pair CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/multimodal_conditioner"),
        help="Directory for checkpoints, vocab, and logs.",
    )
    parser.add_argument("--text-model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--vision-model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-scale", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fusion-dim", type=int, default=768)
    parser.add_argument("--num-query-tokens", type=int, default=8)
    parser.add_argument("--fusion-layers", type=int, default=4)
    parser.add_argument("--fusion-heads", type=int, default=12)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--freeze-text-backbone", action="store_true")
    parser.add_argument("--freeze-vision-backbone", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--contrastive-loss-weight", type=float, default=1.0)
    parser.add_argument("--emotion-loss-weight", type=float, default=0.35)
    parser.add_argument("--vendor-loss-weight", type=float, default=0.35)
    parser.add_argument("--task-loss-weight", type=float, default=0.2)
    parser.add_argument("--sentiment-loss-weight", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def rank0_print(*values: object) -> None:
    if is_main_process():
        print(*values)


def setup_distributed() -> tuple[torch.device, int]:
    if is_distributed():
        torch.distributed.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        return device, local_rank

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(0)
    return device, 0


def cleanup_distributed() -> None:
    if is_distributed() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloader(
    dataset: EmojiEditMultimodalDataset,
    collator: EmojiEditCollator,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    sampler = None
    if is_distributed():
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False)
        shuffle = False

    persistent_workers = num_workers > 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collator,
        drop_last=False,
    )
    return loader, sampler


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> AdamW:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    return AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


def build_scheduler(optimizer: AdamW, total_steps: int, warmup_ratio: float, min_lr_scale: float) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_scale + (1.0 - min_lr_scale) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    if hasattr(batch, "to_device"):
        return batch.to_device(device)
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
) -> dict[str, float]:
    model.eval()
    autocast_dtype = get_autocast_dtype(precision)
    total_loss = 0.0
    total_contrastive = 0.0
    total_emotion_acc = 0.0
    total_vendor_acc = 0.0
    total_task_acc = 0.0
    total_samples = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                source_images=batch["source_images"],
                target_images=batch["target_images"],
                task_type_ids=batch["task_type_ids"],
                target_vendor_ids=batch["target_vendor_ids"],
                target_emotion_ids=batch["target_emotion_ids"],
                target_sentiment_ids=batch["target_sentiment_ids"],
            )

        batch_size = int(batch["input_ids"].size(0))
        total_samples += batch_size
        total_loss += float(outputs["loss"].detach().item()) * batch_size
        total_contrastive += float(outputs["contrastive_loss"].detach().item()) * batch_size
        total_emotion_acc += float(
            (outputs["emotion_logits"].argmax(dim=-1) == batch["target_emotion_ids"]).float().sum().item()
        )
        total_vendor_acc += float(
            (outputs["vendor_logits"].argmax(dim=-1) == batch["target_vendor_ids"]).float().sum().item()
        )
        total_task_acc += float(
            (outputs["task_logits"].argmax(dim=-1) == batch["task_type_ids"]).float().sum().item()
        )

    if is_distributed():
        stats = torch.tensor(
            [
                total_loss,
                total_contrastive,
                total_emotion_acc,
                total_vendor_acc,
                total_task_acc,
                float(total_samples),
            ],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_loss, total_contrastive, total_emotion_acc, total_vendor_acc, total_task_acc, total_samples = (
            stats.tolist()
        )

    sample_denom = max(1.0, float(total_samples))
    return {
        "val_loss": total_loss / sample_denom,
        "val_contrastive_loss": total_contrastive / sample_denom,
        "val_emotion_acc": total_emotion_acc / sample_denom,
        "val_vendor_acc": total_vendor_acc / sample_denom,
        "val_task_acc": total_task_acc / sample_denom,
    }


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    args: argparse.Namespace,
    filename: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "args": vars(args),
        "model_config": asdict(unwrapped.config),
    }
    torch.save(payload, output_dir / filename)


def maybe_resume(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int, float]:
    if args.resume_from is None:
        return 0, 0, float("inf")

    checkpoint = torch.load(args.resume_from, map_location=device)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rank0_print(f"Resumed from {args.resume_from}")
    return int(checkpoint["epoch"]), int(checkpoint["global_step"]), float(checkpoint["best_val_loss"])


def main() -> int:
    args = parse_args()
    device, local_rank = setup_distributed()
    set_seed(args.seed + local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_label_vocab_from_csv(args.pair_csv)
    if is_main_process():
        save_label_vocab(vocab, output_dir / "label_vocab.json")

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, use_fast=True)
    collator = EmojiEditCollator(tokenizer=tokenizer, max_length=args.max_length, pad_to_multiple_of=8)

    train_dataset = EmojiEditMultimodalDataset(
        pair_csv_path=args.pair_csv,
        split="train",
        vocab=vocab,
        image_size=args.image_size,
        max_samples=args.max_train_samples,
    )
    val_dataset = EmojiEditMultimodalDataset(
        pair_csv_path=args.pair_csv,
        split="val",
        vocab=vocab,
        image_size=args.image_size,
        max_samples=args.max_val_samples,
    )

    train_loader, train_sampler = create_dataloader(
        dataset=train_dataset,
        collator=collator,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )
    val_loader, _ = create_dataloader(
        dataset=val_dataset,
        collator=collator,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
    )

    model_config = EmojiEditMultimodalConfig(
        text_model_name=args.text_model_name,
        vision_model_name=args.vision_model_name,
        attn_implementation=args.attn_implementation,
        fusion_dim=args.fusion_dim,
        num_query_tokens=args.num_query_tokens,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        freeze_text_backbone=args.freeze_text_backbone,
        freeze_vision_backbone=args.freeze_vision_backbone,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        contrastive_loss_weight=args.contrastive_loss_weight,
        emotion_loss_weight=args.emotion_loss_weight,
        vendor_loss_weight=args.vendor_loss_weight,
        task_loss_weight=args.task_loss_weight,
        sentiment_loss_weight=args.sentiment_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    model = EmojiEditMultimodalEncoder(
        config=model_config,
        num_emotions=len(vocab.emotions),
        num_sentiments=len(vocab.sentiments),
        num_task_types=len(vocab.task_types),
        num_vendors=len(vocab.vendors),
    )
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=args.warmup_ratio,
        min_lr_scale=args.min_lr_scale,
    )

    start_epoch, global_step, best_val_loss = maybe_resume(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    if is_main_process():
        (output_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "model_config": asdict(model_config),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    autocast_dtype = get_autocast_dtype(args.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=args.precision == "fp16")

    rank0_print(
        f"Train samples={len(train_dataset)} | Val samples={len(val_dataset)} | "
        f"Train batch={args.train_batch_size} | Precision={args.precision}"
    )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            source_images = batch["source_images"].to(memory_format=torch.channels_last)
            target_images = batch["target_images"].to(memory_format=torch.channels_last)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=autocast_dtype is not None):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    source_images=source_images,
                    target_images=target_images,
                    task_type_ids=batch["task_type_ids"],
                    target_vendor_ids=batch["target_vendor_ids"],
                    target_emotion_ids=batch["target_emotion_ids"],
                    target_sentiment_ids=batch["target_sentiment_ids"],
                )
                loss = outputs["loss"] / args.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                loss_value = float(outputs["loss"].detach().item())
                running_loss += loss_value
                running_steps += 1

                if global_step % args.log_interval == 0 and running_steps > 0:
                    avg_loss = running_loss / running_steps
                    lr = scheduler.get_last_lr()[0]
                    rank0_print(
                        f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                        f"loss={avg_loss:.4f} lr={lr:.6e}"
                    )
                    running_loss = 0.0
                    running_steps = 0

        metrics = evaluate(model=model, dataloader=val_loader, device=device, precision=args.precision)
        epoch_time = time.time() - epoch_start
        rank0_print(
            f"epoch={epoch + 1} done in {epoch_time:.1f}s | "
            f"val_loss={metrics['val_loss']:.4f} | "
            f"val_emotion_acc={metrics['val_emotion_acc']:.4f} | "
            f"val_vendor_acc={metrics['val_vendor_acc']:.4f} | "
            f"val_task_acc={metrics['val_task_acc']:.4f}"
        )

        if is_main_process():
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                global_step=global_step,
                best_val_loss=best_val_loss,
                args=args,
                filename="latest.pt",
            )
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    args=args,
                    filename="best.pt",
                )

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
