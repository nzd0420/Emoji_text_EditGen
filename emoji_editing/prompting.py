"""Prompt construction for emoji diffusion training and inference."""

from __future__ import annotations

import random
from dataclasses import dataclass


DEFAULT_NEGATIVE_PROMPT = (
    "photorealistic, realistic face, human skin, body, hands, extra limbs, text, letters, logo, "
    "watermark, sticker border, frame, cropped face, duplicate face, distorted eyes, asymmetrical eyes, "
    "deformed emoji, blurry, noisy, low detail, jpeg artifacts, background clutter"
)

INFERENCE_STYLE_HINT = (
    "clean centered emoji icon, simple plain background, crisp platform-style shading, no text"
)

INFERENCE_PRESERVATION_HINT = (
    "Preserve the source emoji identity, same face shape, centered composition, original color palette, "
    "and platform style unless explicitly changed"
)


@dataclass(frozen=True)
class PromptBuildConfig:
    """Controls how structured prompts are composed."""

    prefix: str = "emoji icon edit"
    style_hint: str = "clean centered emoji icon, simple plain background, crisp platform-style shading"
    structured_prompt_probability: float = 0.8
    include_attribute_delta_probability: float = 0.6


def build_training_prompt(
    row: dict[str, str],
    config: PromptBuildConfig,
    rng: random.Random | None = None,
) -> str:
    """Compose a robust training prompt from the CSV metadata."""

    if rng is None:
        rng = random

    raw_instruction = row["instruction"].strip()
    source_name = row["source_name"].strip()
    target_name = row["target_name"].strip()
    source_vendor = row["source_vendor"].strip()
    target_vendor = row["target_vendor"].strip()
    source_emotion = row["source_emotion"].strip()
    target_emotion = row["target_emotion"].strip()
    task_type = row["task_type"].replace("_", " ").strip()
    attribute_delta = row.get("attribute_delta", "").strip()

    if rng.random() > config.structured_prompt_probability:
        return f"Instruction: {raw_instruction} {config.prefix}. {config.style_hint}."

    # 指令是最关键信息，放在最前面，避免 CLIP 77-token 截断时被丢弃。
    segments = [
        f"Instruction: {raw_instruction}",
        f"{config.prefix}.",
        config.style_hint + ".",
        f"Task: {task_type}.",
        f"Source emoji: {source_name}.",
        f"Source style: {source_vendor}.",
        f"Target expression: {target_name}.",
        f"Target emotion: {target_emotion}.",
    ]

    if source_vendor != target_vendor:
        segments.append(f"Target style: {target_vendor}.")
    elif source_emotion != target_emotion:
        segments.append(f"Keep the original {source_vendor} emoji style.")

    if attribute_delta and rng.random() < config.include_attribute_delta_probability:
        segments.append(f"Requested change: {attribute_delta.replace('|', '; ')}.")

    return " ".join(segment for segment in segments if segment)


def build_inference_prompt(
    instruction: str,
    source_name: str | None = None,
    source_vendor: str | None = None,
    extra_style_hint: str | None = None,
    config: PromptBuildConfig | None = None,
) -> str:
    """Compose an inference prompt from UI inputs."""

    if config is None:
        config = PromptBuildConfig()

    # 与训练保持一致：指令前置，确保 77-token 截断时不会丢失。
    segments = [
        f"Instruction: {instruction.strip()}",
        f"{config.prefix}.",
        INFERENCE_STYLE_HINT + ".",
        INFERENCE_PRESERVATION_HINT + ".",
    ]
    if source_name:
        segments.append(f"Source emoji: {source_name}.")
    if source_vendor:
        segments.append(f"Source style: {source_vendor}. Keep this platform style unless the instruction asks for another style.")
    if extra_style_hint:
        segments.append(extra_style_hint.strip().rstrip(".") + ".")
    return " ".join(segment for segment in segments if segment)
