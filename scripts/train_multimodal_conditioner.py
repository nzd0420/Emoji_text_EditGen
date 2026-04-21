#!/usr/bin/env python3
"""Train the multimodal emoji instruction encoder with NVIDIA GPU optimizations."""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
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


@dataclass
class MultimodalTrainConfig:
    pair_csv: Path
    output_dir: Path
    text_model_name: str
    vision_model_name: str
    attn_implementation: str
    image_size: int
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    num_workers: int
    prefetch_factor: int
    epochs: int
    lr: float
    min_lr_scale: float
    weight_decay: float
    warmup_ratio: float
    grad_accum_steps: int
    max_grad_norm: float
    fusion_dim: int
    num_query_tokens: int
    fusion_layers: int
    fusion_heads: int
    mlp_ratio: float
    dropout: float
    freeze_text_backbone: bool
    freeze_vision_backbone: bool
    disable_gradient_checkpointing: bool
    contrastive_loss_weight: float
    emotion_loss_weight: float
    vendor_loss_weight: float
    task_loss_weight: float
    sentiment_loss_weight: float
    label_smoothing: float
    precision: str
    seed: int
    log_interval: int
    compile_model: bool
    resume_from: Path | None
    max_train_samples: int | None
    max_val_samples: int | None


# 在这里修改多模态编码器训练配置。
TRAIN_CONFIG = MultimodalTrainConfig(
    pair_csv=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),  # 训练样本表路径。
    output_dir=Path("artifacts/multimodal_conditioner"),  # 模型、日志和词表输出目录。
    text_model_name="intfloat/multilingual-e5-base",  # 文本编码器。
    vision_model_name="openai/clip-vit-base-patch32",  # 视觉编码器。
    attn_implementation="sdpa",  # 注意力实现方式，NVIDIA 新卡建议保留 sdpa。
    image_size=256,  # 输入图像尺寸。
    max_length=96,  # 指令最大 token 长度。
    train_batch_size=64,  # 训练 batch size。
    eval_batch_size=128,  # 验证 batch size。
    num_workers=8,  # DataLoader worker 数量。
    prefetch_factor=4,  # 每个 worker 预取批次数。
    epochs=10,  # 训练轮数。
    lr=2e-4,  # 初始学习率。
    min_lr_scale=0.1,  # 余弦退火的最低学习率比例。
    weight_decay=0.01,  # AdamW 权重衰减。
    warmup_ratio=0.03,  # 预热步数占比。
    grad_accum_steps=1,  # 梯度累积步数。
    max_grad_norm=1.0,  # 梯度裁剪上限。
    fusion_dim=768,  # 融合层隐藏维度。
    num_query_tokens=8,  # 查询 token 数量。
    fusion_layers=4,  # 融合层层数。
    fusion_heads=12,  # 融合层注意力头数。
    mlp_ratio=4.0,  # 融合层 FFN 扩展比例。
    dropout=0.0,  # 融合层 dropout。
    freeze_text_backbone=False,  # 是否冻结文本 backbone。
    freeze_vision_backbone=False,  # 是否冻结视觉 backbone。
    disable_gradient_checkpointing=False,  # 设为 True 时关闭梯度检查点。
    contrastive_loss_weight=1.0,  # 对比学习损失权重。
    emotion_loss_weight=0.35,  # 情绪分类损失权重。
    vendor_loss_weight=0.35,  # 平台风格分类损失权重。
    task_loss_weight=0.2,  # 任务类型分类损失权重。
    sentiment_loss_weight=0.1,  # 情感极性分类损失权重。
    label_smoothing=0.0,  # 分类标签平滑。
    precision="fp16",  # RTX 40 系列优先试 bf16；不稳定时改 fp16。
    seed=42,  # 随机种子。
    log_interval=50,  # 每隔多少个优化步打印一次日志。
    compile_model=True,  # 是否启用 torch.compile。
    resume_from=None,  # 恢复训练的 checkpoint 路径，不续训时保持 None。
    max_train_samples=None,  # 调试时可限制训练样本数。
    max_val_samples=None,  # 调试时可限制验证样本数。
)


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
    config: MultimodalTrainConfig,
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
        "config": asdict(config),
        "model_config": asdict(unwrapped.config),
    }
    torch.save(payload, output_dir / filename)


def maybe_resume(
    config: MultimodalTrainConfig,
    model: nn.Module,
    optimizer: AdamW,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, int, float]:
    if config.resume_from is None:
        return 0, 0, float("inf")

    checkpoint = torch.load(config.resume_from, map_location=device)
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rank0_print(f"Resumed from {config.resume_from}")
    return int(checkpoint["epoch"]), int(checkpoint["global_step"]), float(checkpoint["best_val_loss"])


def main() -> int:
    config = TRAIN_CONFIG
    device, local_rank = setup_distributed()
    set_seed(config.seed + local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_label_vocab_from_csv(config.pair_csv)
    if is_main_process():
        save_label_vocab(vocab, output_dir / "label_vocab.json")

    tokenizer = AutoTokenizer.from_pretrained(config.text_model_name, use_fast=True)
    collator = EmojiEditCollator(tokenizer=tokenizer, max_length=config.max_length, pad_to_multiple_of=8)

    train_dataset = EmojiEditMultimodalDataset(
        pair_csv_path=config.pair_csv,
        split="train",
        vocab=vocab,
        image_size=config.image_size,
        max_samples=config.max_train_samples,
    )
    val_dataset = EmojiEditMultimodalDataset(
        pair_csv_path=config.pair_csv,
        split="val",
        vocab=vocab,
        image_size=config.image_size,
        max_samples=config.max_val_samples,
    )

    train_loader, train_sampler = create_dataloader(
        dataset=train_dataset,
        collator=collator,
        batch_size=config.train_batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=True,
    )
    val_loader, _ = create_dataloader(
        dataset=val_dataset,
        collator=collator,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        shuffle=False,
    )

    model_config = EmojiEditMultimodalConfig(
        text_model_name=config.text_model_name,
        vision_model_name=config.vision_model_name,
        attn_implementation=config.attn_implementation,
        fusion_dim=config.fusion_dim,
        num_query_tokens=config.num_query_tokens,
        fusion_layers=config.fusion_layers,
        fusion_heads=config.fusion_heads,
        mlp_ratio=config.mlp_ratio,
        dropout=config.dropout,
        freeze_text_backbone=config.freeze_text_backbone,
        freeze_vision_backbone=config.freeze_vision_backbone,
        gradient_checkpointing=not config.disable_gradient_checkpointing,
        contrastive_loss_weight=config.contrastive_loss_weight,
        emotion_loss_weight=config.emotion_loss_weight,
        vendor_loss_weight=config.vendor_loss_weight,
        task_loss_weight=config.task_loss_weight,
        sentiment_loss_weight=config.sentiment_loss_weight,
        label_smoothing=config.label_smoothing,
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

    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)

    if is_distributed():
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = build_optimizer(model, lr=config.lr, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.grad_accum_steps))
    total_steps = max(1, steps_per_epoch * config.epochs)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=config.warmup_ratio,
        min_lr_scale=config.min_lr_scale,
    )

    start_epoch, global_step, best_val_loss = maybe_resume(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    if is_main_process():
        (output_dir / "train_config.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "model_config": asdict(model_config),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    autocast_dtype = get_autocast_dtype(config.precision)
    scaler = torch.cuda.amp.GradScaler(enabled=config.precision == "fp16")

    rank0_print(
        f"Train samples={len(train_dataset)} | Val samples={len(val_dataset)} | "
        f"Train batch={config.train_batch_size} | Precision={config.precision}"
    )

    for epoch in range(start_epoch, config.epochs):
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
                loss = outputs["loss"] / config.grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % config.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
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

                if global_step % config.log_interval == 0 and running_steps > 0:
                    avg_loss = running_loss / running_steps
                    lr = scheduler.get_last_lr()[0]
                    rank0_print(
                        f"epoch={epoch + 1}/{config.epochs} step={global_step} "
                        f"loss={avg_loss:.4f} lr={lr:.6e}"
                    )
                    running_loss = 0.0
                    running_steps = 0

        metrics = evaluate(model=model, dataloader=val_loader, device=device, precision=config.precision)
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
                config=config,
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
                    config=config,
                    filename="best.pt",
                )

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
