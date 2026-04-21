"""Dataset utilities for instruction-guided emoji diffusion editing."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .prompting import PromptBuildConfig, build_training_prompt


def _read_rows(csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(csv_path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _pad_to_square(image: Image.Image, background_rgb: tuple[int, int, int]) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGBA", (side, side), background_rgb + (255,))
    x = (side - width) // 2
    y = (side - height) // 2
    canvas.alpha_composite(image, dest=(x, y))
    return canvas


def load_editor_image(
    image_path: str | Path,
    resolution: int,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
    interpolation: int = Image.LANCZOS,
) -> Tensor:
    path = Path(image_path)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        squared = _pad_to_square(rgba, background_rgb=background_rgb)
        resized = squared.resize((resolution, resolution), resample=interpolation).convert("RGB")

    array = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


@dataclass
class DiffusionExample:
    """Single image-edit training example."""

    prompt: str
    source_pixel_values: Tensor
    target_pixel_values: Tensor
    source_name: str
    source_vendor: str
    target_name: str
    target_vendor: str
    pair_id: str


class EmojiDiffusionEditDataset(Dataset[DiffusionExample]):
    """Loads source/target emoji edit pairs for InstructPix2Pix training."""

    def __init__(
        self,
        pair_csv_path: str | Path,
        split: str,
        resolution: int = 256,
        prompt_config: PromptBuildConfig | None = None,
        max_samples: int | None = None,
        background_rgb: tuple[int, int, int] = (255, 255, 255),
        interpolation: int = Image.LANCZOS,
    ) -> None:
        rows = [row for row in _read_rows(pair_csv_path) if row["split"] == split]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows
        self.split = split
        self.resolution = resolution
        self.prompt_config = prompt_config or PromptBuildConfig()
        self.background_rgb = background_rgb
        self.interpolation = interpolation

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> DiffusionExample:
        row = self.rows[index]
        local_rng = random.Random(f"{row['pair_id']}::{self.split}::{index}")
        prompt = build_training_prompt(row=row, config=self.prompt_config, rng=local_rng)
        source_pixel_values = load_editor_image(
            row["source_image_path"],
            resolution=self.resolution,
            background_rgb=self.background_rgb,
            interpolation=self.interpolation,
        )
        target_pixel_values = load_editor_image(
            row["target_image_path"],
            resolution=self.resolution,
            background_rgb=self.background_rgb,
            interpolation=self.interpolation,
        )
        return DiffusionExample(
            prompt=prompt,
            source_pixel_values=source_pixel_values,
            target_pixel_values=target_pixel_values,
            source_name=row["source_name"],
            source_vendor=row["source_vendor"],
            target_name=row["target_name"],
            target_vendor=row["target_vendor"],
            pair_id=row["pair_id"],
        )


class EmojiDiffusionCollator:
    """Tokenizes prompts and batches image-edit examples."""

    def __init__(self, tokenizer: Any, max_length: int = 77) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: list[DiffusionExample]) -> dict[str, Any]:
        tokenized = self.tokenizer(
            [example.prompt for example in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "pair_ids": [example.pair_id for example in examples],
            "prompts": [example.prompt for example in examples],
            "source_names": [example.source_name for example in examples],
            "target_names": [example.target_name for example in examples],
            "source_vendors": [example.source_vendor for example in examples],
            "target_vendors": [example.target_vendor for example in examples],
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "original_pixel_values": torch.stack([example.source_pixel_values for example in examples]),
            "edited_pixel_values": torch.stack([example.target_pixel_values for example in examples]),
        }
