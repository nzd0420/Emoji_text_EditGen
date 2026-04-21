#!/usr/bin/env python3
"""Train a diffusion-based emoji editor with text-guided image conditioning."""

from __future__ import annotations

import argparse
import json
import math
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-model-name-or-path", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--pair-csv", type=Path, default=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/emoji_diffusion_editor"))
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--dataloader-num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--scale-lr", action="store_true")
    parser.add_argument("--lr-scheduler", default="cosine", choices=["constant", "cosine", "cosine_with_restarts", "polynomial", "linear"])
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--snr-gamma", type=float, default=None)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target-modules", default="to_q,to_k,to_v,to_out.0")
    parser.add_argument("--train-text-encoder-lora", action="store_true")
    parser.add_argument("--conditioning-dropout-prob", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--enable-xformers-memory-efficient-attention", action="store_true")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--use-8bit-adam", action="store_true")
    parser.add_argument("--checkpointing-steps", type=int, default=1000)
    parser.add_argument("--checkpoints-total-limit", type=int, default=3)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--validation-steps", type=int, default=500)
    parser.add_argument("--num-validation-images", type=int, default=6)
    parser.add_argument("--validation-inference-steps", type=int, default=30)
    parser.add_argument("--validation-guidance-scale", type=float, default=5.0)
    parser.add_argument("--validation-image-guidance-scale", type=float, default=1.8)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--validation-seed", type=int, default=1234)
    parser.add_argument("--report-to", default=None)
    return parser.parse_args()


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


def make_optimizer(args: argparse.Namespace, params_to_optimize: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("bitsandbytes is required for --use-8bit-adam.") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = AdamW
    return optimizer_cls(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


def parse_target_modules(csv_text: str) -> list[str]:
    return [value.strip() for value in csv_text.split(",") if value.strip()]


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
    args: argparse.Namespace,
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

    samples = [validation_dataset[idx] for idx in range(min(args.num_validation_images, len(validation_dataset)))]
    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
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

    generator = torch.Generator(device=accelerator.device).manual_seed(args.validation_seed)
    rendered_images: list[Image.Image] = []
    captions: list[str] = []
    for sample in samples:
        source_image = ((sample.source_pixel_values.clamp(-1, 1) + 1.0) * 127.5).byte()
        source_pil = Image.fromarray(source_image.permute(1, 2, 0).cpu().numpy())
        result = pipeline(
            prompt=sample.prompt,
            image=source_pil,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=args.validation_inference_steps,
            guidance_scale=args.validation_guidance_scale,
            image_guidance_scale=args.validation_image_guidance_scale,
            generator=generator,
        ).images[0]
        rendered_images.append(result)
        captions.append(f"{sample.source_name} -> {sample.target_name}")

    validation_dir = args.output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    make_image_grid(rendered_images, captions, cell_size=args.resolution).save(validation_dir / f"step_{step:06d}.png")
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    project_config = ProjectConfiguration(project_dir=str(args.output_dir), logging_dir=str(args.output_dir / "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=project_config,
    )

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "train_args.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    if args.seed is not None:
        set_seed(args.seed)
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from peft import LoraConfig

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if hasattr(text_encoder, "gradient_checkpointing_enable"):
            text_encoder.gradient_checkpointing_enable()

    unet.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_target_modules(args.lora_target_modules),
        )
    )
    if args.train_text_encoder_lora:
        text_encoder.add_adapter(
            LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
        )

    maybe_enable_xformers(unet, text_encoder, args.enable_xformers_memory_efficient_attention)

    prompt_config = PromptBuildConfig()
    train_dataset = EmojiDiffusionEditDataset(
        pair_csv_path=args.pair_csv,
        split="train",
        resolution=args.resolution,
        prompt_config=prompt_config,
        max_samples=args.max_train_samples,
    )
    validation_dataset = EmojiDiffusionEditDataset(
        pair_csv_path=args.pair_csv,
        split="val",
        resolution=args.resolution,
        prompt_config=prompt_config,
        max_samples=args.max_val_samples,
    )
    collator = EmojiDiffusionCollator(tokenizer=tokenizer, max_length=tokenizer.model_max_length)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collator,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=args.dataloader_num_workers > 0,
    )
    params_to_optimize = [param for param in unet.parameters() if param.requires_grad]
    if args.train_text_encoder_lora:
        params_to_optimize.extend(param for param in text_encoder.parameters() if param.requires_grad)

    if args.scale_lr:
        args.learning_rate = args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = make_optimizer(args, params_to_optimize)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.epochs * steps_per_epoch
    else:
        args.epochs = math.ceil(args.max_train_steps / steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = get_weight_dtype(accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder_lora:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Resuming from {args.resume_from_checkpoint}")
        accelerator.load_state(str(args.resume_from_checkpoint))
        state_path = Path(args.resume_from_checkpoint) / "trainer_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            global_step = int(state.get("global_step", 0))
            first_epoch = int(state.get("epoch", 0))

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, desc="Training")

    for epoch in range(first_epoch, args.epochs):
        unet.train()
        text_encoder.train(args.train_text_encoder_lora)

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
                    dropout_prob=args.conditioning_dropout_prob,
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

                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        weights = weights / snr
                    else:
                        weights = weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, loss.ndim))) * weights
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix(loss=float(loss.detach().item()), lr=lr_scheduler.get_last_lr()[0])

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    checkpoints_dir = args.output_dir / "checkpoints"
                    checkpoint_dir = checkpoints_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    (checkpoint_dir / "trainer_state.json").write_text(
                        json.dumps({"global_step": global_step, "epoch": epoch}, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    save_lora_weights(accelerator, unet, text_encoder, checkpoint_dir / "lora", args.train_text_encoder_lora)
                    cleanup_old_checkpoints(checkpoints_dir, args.checkpoints_total_limit)

                if global_step % args.validation_steps == 0:
                    run_validation(
                        accelerator=accelerator,
                        args=args,
                        vae=vae,
                        text_encoder=text_encoder,
                        tokenizer=tokenizer,
                        unet=unet,
                        validation_dataset=validation_dataset,
                        step=global_step,
                        weight_dtype=weight_dtype,
                    )
                    if accelerator.is_main_process:
                        save_lora_weights(accelerator, unet, text_encoder, args.output_dir / "lora_latest", args.train_text_encoder_lora)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora_weights(accelerator, unet, text_encoder, args.output_dir / "lora_final", args.train_text_encoder_lora)
    accelerator.end_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
