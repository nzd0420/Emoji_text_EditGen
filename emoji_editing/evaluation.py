"""Quantitative metrics for the emoji diffusion editor.

提供两类可复现的定量指标，用于把"做了一个能跑的系统"变成"有对照的研究"：

- CLIP 图文 / 图图相似度：衡量编辑结果是否符合指令、是否接近目标图、是否保持源图身份。
- LPIPS 感知距离（可选，需安装 ``lpips``）：衡量结果与目标图的感知差异。

所有函数都接受一组 PIL 图像并按 batch 计算，返回 **逐样本** 的指标列表，方便上层
按 task_type 分组聚合。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np
import torch
from PIL import Image

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def resolve_device(device: str | None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_clip(model_name: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


def _pooler_output(outputs: object) -> torch.Tensor:
    pooled = getattr(outputs, "pooler_output", None)
    if isinstance(pooled, torch.Tensor):
        return pooled
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1 and isinstance(outputs[1], torch.Tensor):
        return outputs[1]
    raise TypeError(f"CLIP backbone returned unsupported output type: {type(outputs)!r}")


@torch.no_grad()
def _clip_image_features(
    images: Sequence[Image.Image],
    model_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model, processor = _load_clip(model_name, device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        batch = [img.convert("RGB") for img in images[start : start + batch_size]]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        vision_outputs = model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
        feats = model.visual_projection(_pooler_output(vision_outputs))
        chunks.append(torch.nn.functional.normalize(feats, dim=-1))
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def _clip_text_features(
    texts: Sequence[str],
    model_name: str,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    model, processor = _load_clip(model_name, device)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        inputs = processor(
            text=batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            return_dict=True,
        )
        feats = model.text_projection(_pooler_output(text_outputs))
        chunks.append(torch.nn.functional.normalize(feats, dim=-1))
    return torch.cat(chunks, dim=0)


def clip_image_similarity(
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    model_name: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
    batch_size: int = 32,
) -> list[float]:
    """逐样本的 CLIP 图像余弦相似度（两个等长图像序列对位比较）。"""
    if len(images_a) != len(images_b):
        raise ValueError("images_a 与 images_b 长度必须一致。")
    if not images_a:
        return []
    dev = resolve_device(device)
    fa = _clip_image_features(images_a, model_name, dev, batch_size)
    fb = _clip_image_features(images_b, model_name, dev, batch_size)
    return (fa * fb).sum(dim=-1).cpu().tolist()


def clip_text_alignment(
    images: Sequence[Image.Image],
    texts: Sequence[str],
    model_name: str = DEFAULT_CLIP_MODEL,
    device: str | None = None,
    batch_size: int = 32,
) -> list[float]:
    """逐样本的 CLIP 图文余弦相似度（结果图与编辑指令的对齐度）。"""
    if len(images) != len(texts):
        raise ValueError("images 与 texts 长度必须一致。")
    if not images:
        return []
    dev = resolve_device(device)
    fi = _clip_image_features(images, model_name, dev, batch_size)
    ft = _clip_text_features(texts, model_name, dev, batch_size)
    return (fi * ft).sum(dim=-1).cpu().tolist()


def lpips_available() -> bool:
    try:
        import lpips  # noqa: F401

        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def _load_lpips(device: str):
    import lpips

    return lpips.LPIPS(net="alex").to(device).eval()


def _pil_batch_to_tensor(images: Sequence[Image.Image], device: str) -> torch.Tensor:
    # LPIPS 期望 [-1, 1] 范围的 NCHW 张量；要求同一批图尺寸一致（评估前已统一到 resolution）。
    array = np.stack([np.asarray(img.convert("RGB"), dtype=np.float32) for img in images])
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2) / 127.5 - 1.0
    return tensor.to(device)


@torch.no_grad()
def lpips_distance(
    images_a: Sequence[Image.Image],
    images_b: Sequence[Image.Image],
    device: str | None = None,
    batch_size: int = 16,
) -> list[float]:
    """逐样本 LPIPS 感知距离（越小越接近）。需安装 ``lpips``，否则请先用 lpips_available 判断。"""
    if len(images_a) != len(images_b):
        raise ValueError("images_a 与 images_b 长度必须一致。")
    if not images_a:
        return []
    dev = resolve_device(device)
    net = _load_lpips(dev)
    scores: list[float] = []
    for start in range(0, len(images_a), batch_size):
        ta = _pil_batch_to_tensor(images_a[start : start + batch_size], dev)
        tb = _pil_batch_to_tensor(images_b[start : start + batch_size], dev)
        dist = net(ta, tb)
        scores.extend(dist.view(-1).cpu().tolist())
    return scores


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")
