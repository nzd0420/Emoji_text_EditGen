"""Shared image helpers used by the diffusion data and inference paths."""

from __future__ import annotations

from PIL import Image

# 默认贴底背景色（白色），emoji 透明区域会合成到该背景上。
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
