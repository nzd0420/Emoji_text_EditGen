"""Inference helpers for the emoji diffusion editor."""

from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from .catalog import EmojiCatalogEntry, catalog_lookup, entries_for_vendor, load_vendor_catalog, vendors_from_catalog
from .image_utils import WHITE_BACKGROUND, prepare_emoji_image
from .prompting import DEFAULT_NEGATIVE_PROMPT, PromptBuildConfig, build_inference_prompt


def _infer_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_input_image(
    image: Image.Image,
    resolution: int,
    background_rgb: tuple[int, int, int] = WHITE_BACKGROUND,
    trim_foreground: bool = True,
    trim_margin_ratio: float = 0.08,
) -> Image.Image:
    return prepare_emoji_image(
        image,
        resolution=resolution,
        background_rgb=background_rgb,
        trim_foreground=trim_foreground,
        trim_margin_ratio=trim_margin_ratio,
        interpolation=Image.LANCZOS,
    )


@lru_cache(maxsize=2)
def load_editor_pipeline(
    base_model: str,
    lora_path: str | None,
    precision: str,
    device: str | None,
    scheduler_name: str = "euler_a",
    enable_xformers: bool = False,
) -> Any:
    from diffusers import (
        DPMSolverMultistepScheduler,
        EulerAncestralDiscreteScheduler,
        StableDiffusionInstructPix2PixPipeline,
    )

    torch_dtype = _infer_dtype(precision)
    resolved_device = _resolve_device(device)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )
    if scheduler_name == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    if lora_path:
        pipe.load_lora_weights(lora_path)

    if enable_xformers and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    pipe.to(resolved_device)
    pipe.set_progress_bar_config(disable=True)
    if resolved_device.startswith("cuda"):
        pipe.unet.to(memory_format=torch.channels_last)
    return pipe


def edit_emoji_image(
    source_image: Image.Image,
    instruction: str,
    base_model: str,
    lora_path: str | None,
    precision: str = "fp16",
    device: str | None = None,
    source_name: str | None = None,
    source_vendor: str | None = None,
    steps: int = 30,
    guidance_scale: float = 4.5,
    image_guidance_scale: float = 1.8,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = -1,
    resolution: int = 256,
    trim_foreground: bool = True,
    trim_margin_ratio: float = 0.08,
    scheduler_name: str = "euler_a",
    extra_style_hint: str | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    pipe = load_editor_pipeline(
        base_model=base_model,
        lora_path=lora_path,
        precision=precision,
        device=device,
        scheduler_name=scheduler_name,
    )
    prepared = prepare_input_image(
        source_image,
        resolution=resolution,
        trim_foreground=trim_foreground,
        trim_margin_ratio=trim_margin_ratio,
    )
    prompt = build_inference_prompt(
        instruction=instruction,
        source_name=source_name,
        source_vendor=source_vendor,
        extra_style_hint=extra_style_hint,
        config=PromptBuildConfig(),
    )

    actual_seed = seed if seed >= 0 else random.randint(0, 2**31 - 1)
    generator = None
    if _resolve_device(device).startswith("cuda"):
        generator = torch.Generator(device=_resolve_device(device)).manual_seed(actual_seed)
    else:
        generator = torch.Generator().manual_seed(actual_seed)

    result = pipe(
        prompt=prompt,
        image=prepared,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        negative_prompt=negative_prompt,
        generator=generator,
    ).images[0]
    metadata = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": actual_seed,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "image_guidance_scale": image_guidance_scale,
        "resolution": resolution,
        "trim_foreground": trim_foreground,
        "trim_margin_ratio": trim_margin_ratio,
        "source_name": source_name,
        "source_vendor": source_vendor,
    }
    return result, metadata


def load_ui_catalog(vendor_index_csv: str | Path) -> dict[str, Any]:
    entries = load_vendor_catalog(vendor_index_csv)
    return {
        "entries": entries,
        "lookup": catalog_lookup(entries),
        "vendors": vendors_from_catalog(entries),
    }


def choices_for_vendor(entries: list[EmojiCatalogEntry], vendor: str) -> list[tuple[str, str]]:
    vendor_entries = entries_for_vendor(entries, vendor)
    return [(entry.display_name, entry.key) for entry in vendor_entries]
