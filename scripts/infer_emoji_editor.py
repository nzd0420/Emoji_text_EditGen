#!/usr/bin/env python3
"""Run the emoji diffusion editor with an in-file editable config block."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from emoji_editing.catalog import load_vendor_catalog
from emoji_editing.diffusion_inference import edit_emoji_image


@dataclass(frozen=True)
class InferenceConfig:
    base_model: str
    lora_path: Path
    input_image: Path | None
    vendor_index_csv: Path
    vendor: str
    emoji_key: str | None
    instruction: str
    precision: str
    device: str | None
    steps: int
    guidance_scale: float
    image_guidance_scale: float
    seed: int
    resolution: int
    scheduler: str
    extra_style_hint: str | None
    output_image: Path
    output_metadata: Path


# 在这里修改推理脚本配置。
INFER_CONFIG = InferenceConfig(
    base_model="timbrooks/instruct-pix2pix",  # 推理底座模型。
    lora_path=Path("artifacts/emoji_diffusion_editor/lora_final"),  # 训练完成后的 LoRA 目录。
    input_image=None,  # 填本地图片路径时会优先使用该图片；保持 None 时走内置 emoji。
    vendor_index_csv=Path("data/interim/emoji_editing/metadata/vendor_image_index.csv"),  # 内置 emoji 索引表。
    vendor="Apple",  # 当 input_image 为 None 时，默认选择哪个平台风格。
    emoji_key=None,  # 指定某个内置 emoji 的 key；保持 None 时自动选该 vendor 的第一张。
    instruction="Add sunglasses and make the face more confident.",  # 自然语言编辑指令。
    precision="fp16",  # RTX 单卡推理通常先用 fp16。
    device=None,  # 强制设备，例如 'cuda:0'；保持 None 时自动选择。
    steps=30,  # 推理步数。
    guidance_scale=4.5,  # 文本 guidance 强度。
    image_guidance_scale=1.8,  # 源图像保持强度。
    seed=-1,  # -1 表示随机；填固定整数可复现。
    resolution=256,  # 推理分辨率。
    scheduler="euler_a",  # 采样器，可选 'euler_a' 或 'dpm'。
    extra_style_hint=None,  # 额外风格提示词，不需要时保持 None。
    output_image=Path("artifacts/emoji_editor_output.png"),  # 输出图片路径。
    output_metadata=Path("artifacts/emoji_editor_output.json"),  # 输出 metadata 路径。
)


def resolve_source(config: InferenceConfig) -> tuple[Image.Image, str | None, str | None]:
    if config.input_image is not None:
        with Image.open(config.input_image) as image:
            return image.convert("RGBA").copy(), None, None

    entries = load_vendor_catalog(config.vendor_index_csv)
    lookup = {entry.key: entry for entry in entries}
    if config.emoji_key is None:
        vendor_entries = [entry for entry in entries if entry.vendor == config.vendor]
        if not vendor_entries:
            raise ValueError(f"No entries found for vendor {config.vendor}")
        entry = vendor_entries[0]
    else:
        entry = lookup[config.emoji_key]

    with Image.open(entry.image_path) as image:
        return image.convert("RGBA").copy(), entry.name, entry.vendor


def main() -> int:
    config = INFER_CONFIG
    source_image, source_name, source_vendor = resolve_source(config)
    result, metadata = edit_emoji_image(
        source_image=source_image,
        instruction=config.instruction,
        base_model=config.base_model,
        lora_path=config.lora_path if config.lora_path.exists() else None,
        precision=config.precision,
        device=config.device,
        source_name=source_name,
        source_vendor=source_vendor,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        image_guidance_scale=config.image_guidance_scale,
        seed=config.seed,
        resolution=config.resolution,
        scheduler_name=config.scheduler,
        extra_style_hint=config.extra_style_hint,
    )
    config.output_image.parent.mkdir(parents=True, exist_ok=True)
    config.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    result.save(config.output_image)
    config.output_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(config.output_image)
    print(config.output_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
