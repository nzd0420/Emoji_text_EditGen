#!/usr/bin/env python3
"""Train a diffusion-based emoji editor with text-guided image conditioning."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel

from emoji_editing.diffusion_data import EmojiDiffusionCollator, EmojiDiffusionEditDataset
from emoji_editing.prompting import DEFAULT_NEGATIVE_PROMPT, PromptBuildConfig


@dataclass
class DiffusionTrainConfig:
    pretrained_model_name_or_path: str
    pair_csv: Path
    output_dir: Path
    resolution: int
    train_batch_size: int
    eval_batch_size: int
    dataloader_num_workers: int
    epochs: int
    max_train_steps: int | None
    learning_rate: float
    scale_lr: bool
    lr_scheduler: str
    lr_warmup_steps: int
    snr_gamma: float | None
    adam_beta1: float
    adam_beta2: float
    adam_weight_decay: float
    adam_epsilon: float
    max_grad_norm: float
    gradient_accumulation_steps: int
    mixed_precision: str
    seed: int
    rank: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: tuple[str, ...]
    train_text_encoder_lora: bool
    conditioning_dropout_prob: float
    gradient_checkpointing: bool
    enable_xformers_memory_efficient_attention: bool
    allow_tf32: bool
    use_8bit_adam: bool
    checkpointing_steps: int
    checkpoints_total_limit: int
    resume_from_checkpoint: Path | None
    validation_steps: int
    num_validation_images: int
    validation_inference_steps: int
    validation_guidance_scale: float
    validation_image_guidance_scale: float
    max_train_samples: int | None
    max_val_samples: int | None
    validation_seed: int
    report_to: str | None


# 在这里修改 diffusion 编辑器训练配置。
TRAIN_CONFIG = DiffusionTrainConfig(
    pretrained_model_name_or_path="timbrooks/instruct-pix2pix",  # 底座模型。
    pair_csv=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),  # 训练样本表路径。
    output_dir=Path("artifacts/emoji_diffusion_editor"),  # LoRA、日志和验证图输出目录。
    resolution=256,  # 训练分辨率。
    train_batch_size=24,  # 单卡训练 batch size。
    eval_batch_size=8,  # 预留给后续扩展验证批处理。
    dataloader_num_workers=8,  # DataLoader worker 数量。
    epochs=12,  # 训练轮数。
    max_train_steps=None,  # 固定总步数时填整数；None 表示按 epochs 自动推算。
    learning_rate=1e-4,  # 学习率。
    scale_lr=False,  # 是否按总 batch 自动缩放学习率。
    lr_scheduler="cosine",  # 学习率调度器类型。
    lr_warmup_steps=500,  # 预热步数。
    snr_gamma=None,  # 启用 Min-SNR Loss 时填写数值，例如 5.0。
    adam_beta1=0.9,  # AdamW beta1。
    adam_beta2=0.999,  # AdamW beta2。
    adam_weight_decay=1e-2,  # AdamW 权重衰减。
    adam_epsilon=1e-8,  # AdamW epsilon。
    max_grad_norm=1.0,  # 梯度裁剪上限。
    gradient_accumulation_steps=1,  # 梯度累积步数。
    mixed_precision="bf16",  # RTX 40 系列优先试 bf16；不稳定时改 fp16。
    seed=42,  # 随机种子。
    rank=16,  # LoRA rank。
    lora_alpha=16,  # LoRA alpha。
    lora_dropout=0.0,  # LoRA dropout。
    lora_target_modules=("to_q", "to_k", "to_v", "to_out.0"),  # UNet LoRA 注入位置。
    train_text_encoder_lora=True,  # 是否同时训练文本编码器 LoRA。
    conditioning_dropout_prob=0.05,  # 条件 dropout 概率。
    gradient_checkpointing=True,  # 是否启用梯度检查点。
    enable_xformers_memory_efficient_attention=False,  # 安装 xformers 后可改成 True。
    allow_tf32=True,  # NVIDIA Ampere/Ada/Hopper 推荐开启。
    use_8bit_adam=False,  # 安装 bitsandbytes 后可改成 True。
    checkpointing_steps=1000,  # 每隔多少优化步保存一次 checkpoint。
    checkpoints_total_limit=3,  # 最多保留多少个历史 checkpoint。
    resume_from_checkpoint=None,  # 从某个 checkpoint 目录恢复训练。
    validation_steps=500,  # 每隔多少步跑一次验证渲染。
    num_validation_images=6,  # 每次验证渲染多少张图。
    validation_inference_steps=30,  # 验证推理步数。
    validation_guidance_scale=5.0,  # 验证文本 guidance。
    validation_image_guidance_scale=1.8,  # 验证图像 guidance。
    max_train_samples=None,  # 调试时可限制训练样本数。
    max_val_samples=128,  # 调试时可限制验证样本数。
    validation_seed=1234,  # 验证随机种子。
    report_to=None,  # 例如 "wandb"；不用日志平台时保持 None。
)


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def maybe_enable_xformers(unet: Any, text_encoder: Any, enabled: bool) -> None:
    if not enabled:
        return
    try:
        unet.enable_xformers_memory_efficient_attention()
        if hasattr(text_encoder, "enable_xformers_memory_efficient_attention"):
            text_encoder.enable_xformers_memory_efficient_attention()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("xFormers was requested but could not be enabled.") from exc


def make_optimizer(config: DiffusionTrainConfig, params_to_optimize: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if config.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bitsandbytes is required when use_8bit_adam=True.") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = AdamW
    return optimizer_cls(
        params_to_optimize,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.adam_weight_decay,
        eps=config.adam_epsilon,
    )


def compute_snr(noise_scheduler: Any, timesteps: torch.Tensor) -> torch.Tensor:
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=timesteps.device, dtype=torch.float32)
    sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod[timesteps]) ** 0.5
    return (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2


def apply_conditioning_dropout(
    encoder_hidden_states: torch.Tensor,
    original_image_embeds: torch.Tensor,
    text_encoder: Any,
    tokenizer: Any,
    device: torch.device,
    dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout_prob <= 0:
        return encoder_hidden_states, original_image_embeds

    batch_size = encoder_hidden_states.shape[0]
    text_keep_mask = (torch.rand(batch_size, device=device) > dropout_prob).view(batch_size, 1, 1)
    image_keep_mask = (torch.rand(batch_size, device=device) > dropout_prob).view(batch_size, 1, 1, 1)

    null_tokens = tokenizer(
        [""] * batch_size,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    null_hidden_states = text_encoder(
        input_ids=null_tokens.input_ids.to(device),
        attention_mask=null_tokens.attention_mask.to(device),
        return_dict=True,
    ).last_hidden_state
    encoder_hidden_states = torch.where(text_keep_mask, encoder_hidden_states, null_hidden_states)
    original_image_embeds = original_image_embeds * image_keep_mask
    return encoder_hidden_states, original_image_embeds


def unwrap(accelerator: Accelerator, model: Any) -> Any:
    return accelerator.unwrap_model(model)


def save_lora_weights(
    accelerator: Accelerator,
    unet: Any,
    text_encoder: Any,
    output_dir: Path,
    train_text_encoder_lora: bool,
) -> None:
    from diffusers import StableDiffusionInstructPix2PixPipeline
    from diffusers.utils import convert_state_dict_to_diffusers

    try:
        from peft import get_peft_model_state_dict
    except ImportError:  # pragma: no cover
        from peft.utils import get_peft_model_state_dict  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    unet_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap(accelerator, unet)))
    text_state = None
    if train_text_encoder_lora:
        text_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrap(accelerator, text_encoder)))
    StableDiffusionInstructPix2PixPipeline.save_lora_weights(
        save_directory=str(output_dir),
        unet_lora_layers=unet_state,
        text_encoder_lora_layers=text_state,
    )


def cleanup_old_checkpoints(checkpoints_dir: Path, limit: int) -> None:
    if limit is None or limit <= 0 or not checkpoints_dir.exists():
        return
    checkpoints = sorted(
        [path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda item: int(item.name.split("-")[-1]),
    )
    while len(checkpoints) > limit:
        shutil.rmtree(checkpoints[0], ignore_errors=True)
        checkpoints.pop(0)


def make_image_grid(images: list[Image.Image], captions: list[str], cell_size: int) -> Image.Image:
    cols = 2
    rows = math.ceil(len(images) / cols)
    caption_height = 54
    grid = Image.new("RGB", (cols * cell_size, rows * (cell_size + caption_height)), color=(248, 245, 238))
    draw = ImageDraw.Draw(grid)
    for idx, image in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = col * cell_size
        y = row * (cell_size + caption_height)
        preview = image.convert("RGB").resize((cell_size, cell_size), resample=Image.LANCZOS)
        grid.paste(preview, (x, y))
        draw.text((x + 8, y + cell_size + 8), captions[idx][:72], fill=(28, 28, 28))
    return grid


@torch.no_grad()
def run_validation(
    accelerator: Accelerator,
    config: DiffusionTrainConfig,
    vae: Any,
    text_encoder: Any,
    tokenizer: Any,
    unet: Any,
    validation_dataset: EmojiDiffusionEditDataset,
    step: int,
    weight_dtype: torch.dtype,
) -> None:
    if not accelerator.is_main_process or len(validation_dataset) == 0:
        return

    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline

    samples = [validation_dataset[idx] for idx in range(min(config.num_validation_images, len(validation_dataset)))]
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        config.pretrained_model_name_or_path,
        vae=unwrap(accelerator, vae),
        text_encoder=unwrap(accelerator, text_encoder),
        tokenizer=tokenizer,
        unet=unwrap(accelerator, unet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(config.validation_seed)
    rendered_images: list[Image.Image] = []
    captions: list[str] = []
    for sample in samples:
        source_image = ((sample.source_pixel_values.clamp(-1, 1) + 1.0) * 127.5).byte()
        source_pil = Image.fromarray(source_image.permute(1, 2, 0).cpu().numpy())
        result = pipeline(
            prompt=sample.prompt,
            image=source_pil,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=config.validation_inference_steps,
            guidance_scale=config.validation_guidance_scale,
            image_guidance_scale=config.validation_image_guidance_scale,
            generator=generator,
        ).images[0]
        rendered_images.append(result)
        captions.append(f"{sample.source_name} -> {sample.target_name}")

    validation_dir = config.output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    make_image_grid(rendered_images, captions, cell_size=config.resolution).save(validation_dir / f"step_{step:06d}.png")
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    config = TRAIN_CONFIG
    project_config = ProjectConfiguration(project_dir=str(config.output_dir), logging_dir=str(config.output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        (config.output_dir / "train_args.json").write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if config.seed is not None:
        set_seed(config.seed)
    if config.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()

    unet.add_adapter(
        LoraConfig(
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(config.lora_target_modules),
        )
    )
    if config.train_text_encoder_lora:
        text_encoder.add_adapter(
            LoraConfig(
                r=config.rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
        )

    maybe_enable_xformers(unet, text_encoder, config.enable_xformers_memory_efficient_attention)

    prompt_config = PromptBuildConfig()
    train_dataset = EmojiDiffusionEditDataset(
        pair_csv_path=config.pair_csv,
        split="train",
        resolution=config.resolution,
        prompt_config=prompt_config,
        max_samples=config.max_train_samples,
    )
    validation_dataset = EmojiDiffusionEditDataset(
        pair_csv_path=config.pair_csv,
        split="val",
        resolution=config.resolution,
        prompt_config=prompt_config,
        max_samples=config.max_val_samples,
    )
    collator = EmojiDiffusionCollator(tokenizer=tokenizer, max_length=tokenizer.model_max_length)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=config.dataloader_num_workers > 0,
    )
    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]
    if config.train_text_encoder_lora:
        params_to_optimize.extend(param for param in text_encoder.parameters() if param.requires_grad)

    if config.scale_lr:
        config.learning_rate = (
            config.learning_rate
            * config.gradient_accumulation_steps
            * config.train_batch_size
            * accelerator.num_processes
        )

    optimizer = make_optimizer(config, params_to_optimize)
    steps_per_epoch = math.ceil(len(train_dataloader) / config.gradient_accumulation_steps)
    if config.max_train_steps is None:
        config.max_train_steps = config.epochs * steps_per_epoch
    else:
        config.epochs = math.ceil(config.max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=config.max_train_steps * accelerator.num_processes,
    )

    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = get_weight_dtype(accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)
    if not config.train_text_encoder_lora:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    first_epoch = 0
    if config.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming from {config.resume_from_checkpoint}")
        accelerator.load_state(str(config.resume_from_checkpoint))
        state_path = Path(config.resume_from_checkpoint) / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            global_step = int(state.get("global_step", 0))
            first_epoch = int(state.get("epoch", 0))

    progress_bar = tqdm(range(global_step, config.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training")

    for epoch in range(first_epoch, config.epochs):
        unet.train()
        text_encoder.train(config.train_text_encoder_lora)

        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                edited_pixel_values = batch["edited_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                original_pixel_values = batch["original_pixel_values"].to(accelerator.device, dtype=weight_dtype)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)

                latents = vae.encode(edited_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                original_image_embeds = vae.encode(original_pixel_values).latent_dist.mode() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).last_hidden_state
                encoder_hidden_states, original_image_embeds = apply_conditioning_dropout(
                    encoder_hidden_states=encoder_hidden_states,
                    original_image_embeds=original_image_embeds,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    device=accelerator.device,
                    dropout_prob=config.conditioning_dropout_prob,
                )

                model_pred = unet(
                    torch.cat([noisy_latents, original_image_embeds], dim=1),
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True,
                ).sample

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                if config.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    weights = torch.stack([snr, config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        weights = weights / snr
                    else:
                        weights = weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, loss.ndim))) * weights
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=float(loss.detach().item()), lr=lr_scheduler.get_last_lr()[0])

                if accelerator.is_main_process and global_step % config.checkpointing_steps == 0:
                    checkpoints_dir = config.output_dir / "checkpoints"
                    checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    (checkpoint_dir / "trainer_state.json").write_text(
                        json.dumps({"global_step": global_step, "epoch": epoch}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    save_lora_weights(accelerator, unet, text_encoder, checkpoint_dir / "lora", config.train_text_encoder_lora)
                    cleanup_old_checkpoints(checkpoints_dir, config.checkpoints_total_limit)

                if global_step % config.validation_steps == 0:
                    run_validation(
                        accelerator=accelerator,
                        config=config,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        validation_dataset=validation_dataset,
                        step=global_step,
                        weight_dtype=weight_dtype,
                    )
                    if accelerator.is_main_process:
                        save_lora_weights(accelerator, unet, text_encoder, config.output_dir / "lora_latest", config.train_text_encoder_lora)

            if global_step >= config.max_train_steps:
                break
        if global_step >= config.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(accelerator, unet, text_encoder, config.output_dir / "lora_final", config.train_text_encoder_lora)
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
