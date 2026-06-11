"""Shared image helpers used by the diffusion data and inference paths."""

from __future__ import annotations

import numpy as np
from PIL import Image

WHITE_BACKGROUND: tuple[int, int, int] = (255, 255, 255)


def pad_to_square(
    image: Image.Image,
    background_rgb: tuple[int, int, int] = WHITE_BACKGROUND,
) -> Image.Image:
    """Center an image on a square RGBA canvas filled with ``background_rgb``."""

    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGBA", (side, side), background_rgb + (255,))
    x = (side - width) // 2
    y = (side - height) // 2
    canvas.alpha_composite(image.convert("RGBA"), dest=(x, y))
    return canvas


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if not np.any(mask):
        return None
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def foreground_bbox(
    image: Image.Image,
    alpha_threshold: int = 8,
    background_threshold: int = 12,
) -> tuple[int, int, int, int] | None:
    """Detect the visible emoji region from alpha or a solid corner background."""

    rgba = image.convert("RGBA")
    array = np.asarray(rgba)
    alpha = array[..., 3]

    if np.any(alpha <= alpha_threshold):
        return _mask_bbox(alpha > alpha_threshold)

    height, width = alpha.shape
    corner = max(1, min(width, height) // 16)
    rgb = array[..., :3].astype(np.int16)
    corner_samples = np.concatenate(
        [
            rgb[:corner, :corner].reshape(-1, 3),
            rgb[:corner, -corner:].reshape(-1, 3),
            rgb[-corner:, :corner].reshape(-1, 3),
            rgb[-corner:, -corner:].reshape(-1, 3),
        ],
        axis=0,
    )
    background_rgb = np.median(corner_samples, axis=0)
    max_channel_delta = np.max(np.abs(rgb - background_rgb), axis=2)
    return _mask_bbox(max_channel_delta > background_threshold)


def crop_foreground(
    image: Image.Image,
    margin_ratio: float = 0.08,
    alpha_threshold: int = 8,
    background_threshold: int = 12,
) -> Image.Image:
    """Crop transparent or solid-color borders while keeping a small margin."""

    rgba = image.convert("RGBA")
    bbox = foreground_bbox(
        rgba,
        alpha_threshold=alpha_threshold,
        background_threshold=background_threshold,
    )
    if bbox is None:
        return rgba

    left, top, right, bottom = bbox
    width, height = rgba.size
    foreground_side = max(right - left, bottom - top)
    margin = int(round(foreground_side * margin_ratio)) if margin_ratio > 0 else 0

    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width, right + margin)
    bottom = min(height, bottom + margin)
    return rgba.crop((left, top, right, bottom))


def prepare_emoji_image(
    image: Image.Image,
    resolution: int,
    background_rgb: tuple[int, int, int] = WHITE_BACKGROUND,
    trim_foreground: bool = True,
    trim_margin_ratio: float = 0.08,
    interpolation: int = Image.LANCZOS,
) -> Image.Image:
    """Crop optional borders, pad to square, resize, and composite to RGB."""

    rgba = image.convert("RGBA")
    if trim_foreground:
        rgba = crop_foreground(rgba, margin_ratio=trim_margin_ratio)
    squared = pad_to_square(rgba, background_rgb=background_rgb)
    return squared.resize((resolution, resolution), resample=interpolation).convert("RGB")
