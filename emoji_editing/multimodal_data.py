"""Dataset and batching utilities for multimodal emoji edit training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .image_utils import WHITE_BACKGROUND
from .io_utils import read_csv_rows

CLIP_RGB_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_RGB_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class MultimodalLabelVocab:
    """Stable label vocabulary used by datasets and training scripts."""

    emotions: list[str]
    sentiments: list[str]
    task_types: list[str]
    vendors: list[str]

    @property
    def emotion_to_id(self) -> dict[str, int]:
        return {value: idx for idx, value in enumerate(self.emotions)}

    @property
    def sentiment_to_id(self) -> dict[str, int]:
        return {value: idx for idx, value in enumerate(self.sentiments)}

    @property
    def task_type_to_id(self) -> dict[str, int]:
        return {value: idx for idx, value in enumerate(self.task_types)}

    @property
    def vendor_to_id(self) -> dict[str, int]:
        return {value: idx for idx, value in enumerate(self.vendors)}


@dataclass
class MultimodalSample:
    """Single dataset item before tokenization."""

    pair_id: str
    task_type: str
    split: str
    instruction: str
    attribute_delta: str
    source_image: Tensor
    target_image: Tensor
    source_vendor_id: int
    target_vendor_id: int
    source_emotion_id: int
    target_emotion_id: int
    source_sentiment_id: int
    target_sentiment_id: int
    task_type_id: int
    source_name: str
    target_name: str
    source_unicode_slug: str
    target_unicode_slug: str


class MultimodalBatch(dict):
    """Dictionary batch with CUDA-friendly transfer and pinning helpers."""

    def pin_memory(self) -> "MultimodalBatch":
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.pin_memory()
        return self

    def to_device(self, device: torch.device) -> "MultimodalBatch":
        moved = MultimodalBatch()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        return moved


def build_label_vocab_from_rows(rows: list[dict[str, str]]) -> MultimodalLabelVocab:
    emotions = sorted({row["source_emotion"] for row in rows} | {row["target_emotion"] for row in rows})
    sentiments = sorted({row["source_sentiment"] for row in rows} | {row["target_sentiment"] for row in rows})
    task_types = sorted({row["task_type"] for row in rows})
    vendors = sorted({row["source_vendor"] for row in rows} | {row["target_vendor"] for row in rows})
    return MultimodalLabelVocab(
        emotions=emotions,
        sentiments=sentiments,
        task_types=task_types,
        vendors=vendors,
    )


def build_label_vocab_from_csv(csv_path: str | Path) -> MultimodalLabelVocab:
    return build_label_vocab_from_rows(read_csv_rows(csv_path))


def save_label_vocab(vocab: MultimodalLabelVocab, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(vocab), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_and_normalize_image(
    image_path: str | Path,
    image_size: int,
    background_rgb: tuple[int, int, int],
    rgb_mean: tuple[float, float, float],
    rgb_std: tuple[float, float, float],
) -> Tensor:
    path = Path(image_path)
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
        rgba = rgba.resize((image_size, image_size), resample=Image.BICUBIC)
        background = Image.new("RGBA", rgba.size, background_rgb + (255,))
        composite = Image.alpha_composite(background, rgba).convert("RGB")

    array = np.asarray(composite, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(rgb_mean, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(rgb_std, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


class EmojiEditMultimodalDataset(Dataset[MultimodalSample]):
    """Loads source/target emoji pairs plus language instructions."""

    def __init__(
        self,
        pair_csv_path: str | Path,
        split: str,
        vocab: MultimodalLabelVocab,
        image_size: int = 224,
        max_samples: int | None = None,
        background_rgb: tuple[int, int, int] = WHITE_BACKGROUND,
        rgb_mean: tuple[float, float, float] = CLIP_RGB_MEAN,
        rgb_std: tuple[float, float, float] = CLIP_RGB_STD,
    ) -> None:
        rows = [row for row in read_csv_rows(pair_csv_path) if row["split"] == split]
        if max_samples is not None:
            rows = rows[:max_samples]
        self.rows = rows
        self.split = split
        self.vocab = vocab
        self.image_size = image_size
        self.background_rgb = background_rgb
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std

        self._emotion_to_id = vocab.emotion_to_id
        self._sentiment_to_id = vocab.sentiment_to_id
        self._task_type_to_id = vocab.task_type_to_id
        self._vendor_to_id = vocab.vendor_to_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> MultimodalSample:
        row = self.rows[index]
        source_image = _load_and_normalize_image(
            row["source_image_path"],
            image_size=self.image_size,
            background_rgb=self.background_rgb,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        )
        target_image = _load_and_normalize_image(
            row["target_image_path"],
            image_size=self.image_size,
            background_rgb=self.background_rgb,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        )

        return MultimodalSample(
            pair_id=row["pair_id"],
            task_type=row["task_type"],
            split=row["split"],
            instruction=row["instruction"],
            attribute_delta=row["attribute_delta"],
            source_image=source_image,
            target_image=target_image,
            source_vendor_id=self._vendor_to_id[row["source_vendor"]],
            target_vendor_id=self._vendor_to_id[row["target_vendor"]],
            source_emotion_id=self._emotion_to_id[row["source_emotion"]],
            target_emotion_id=self._emotion_to_id[row["target_emotion"]],
            source_sentiment_id=self._sentiment_to_id[row["source_sentiment"]],
            target_sentiment_id=self._sentiment_to_id[row["target_sentiment"]],
            task_type_id=self._task_type_to_id[row["task_type"]],
            source_name=row["source_name"],
            target_name=row["target_name"],
            source_unicode_slug=row["source_unicode_slug"],
            target_unicode_slug=row["target_unicode_slug"],
        )


class EmojiEditCollator:
    """Fast tokenizer-backed collator for GPU-friendly rectangular batches."""

    def __init__(
        self,
        tokenizer: Any,
        max_length: int = 96,
        pad_to_multiple_of: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, samples: list[MultimodalSample]) -> MultimodalBatch:
        tokenized = self.tokenizer(
            [sample.instruction for sample in samples],
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = MultimodalBatch(
            pair_ids=[sample.pair_id for sample in samples],
            instructions=[sample.instruction for sample in samples],
            attribute_deltas=[sample.attribute_delta for sample in samples],
            source_names=[sample.source_name for sample in samples],
            target_names=[sample.target_name for sample in samples],
            source_unicode_slugs=[sample.source_unicode_slug for sample in samples],
            target_unicode_slugs=[sample.target_unicode_slug for sample in samples],
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            source_images=torch.stack([sample.source_image for sample in samples], dim=0),
            target_images=torch.stack([sample.target_image for sample in samples], dim=0),
            source_vendor_ids=torch.tensor([sample.source_vendor_id for sample in samples], dtype=torch.long),
            target_vendor_ids=torch.tensor([sample.target_vendor_id for sample in samples], dtype=torch.long),
            source_emotion_ids=torch.tensor([sample.source_emotion_id for sample in samples], dtype=torch.long),
            target_emotion_ids=torch.tensor([sample.target_emotion_id for sample in samples], dtype=torch.long),
            source_sentiment_ids=torch.tensor([sample.source_sentiment_id for sample in samples], dtype=torch.long),
            target_sentiment_ids=torch.tensor([sample.target_sentiment_id for sample in samples], dtype=torch.long),
            task_type_ids=torch.tensor([sample.task_type_id for sample in samples], dtype=torch.long),
        )
        return batch
