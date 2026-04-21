#!/usr/bin/env python3
"""Run the emoji diffusion editor from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from emoji_editing.catalog import load_vendor_catalog
from emoji_editing.diffusion_inference import edit_emoji_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="timbrooks/instruct-pix2pix")
    parser.add_argument("--lora-path", default="artifacts/emoji_diffusion_editor/lora_final")
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--vendor-index-csv", type=Path, default=Path("data/interim/emoji_editing/metadata/vendor_image_index.csv"))
    parser.add_argument("--vendor", default="Apple")
    parser.add_argument("--emoji-key", default=None, help="Catalog key like Apple::1. Used when --input-image is not provided.")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--image-guidance-scale", type=float, default=1.8)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--scheduler", choices=["euler_a", "dpm"], default="euler_a")
    parser.add_argument("--extra-style-hint", default=None)
    parser.add_argument("--output-image", type=Path, default=Path("artifacts/emoji_editor_output.png"))
    parser.add_argument("--output-metadata", type=Path, default=Path("artifacts/emoji_editor_output.json"))
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> tuple[Image.Image, str | None, str | None]:
    if args.input_image is not None:
        with Image.open(args.input_image) as image:
            return image.convert("RGBA").copy(), None, None

    entries = load_vendor_catalog(args.vendor_index_csv)
    lookup = {entry.key: entry for entry in entries}
    if args.emoji_key is None:
        vendor_entries = [entry for entry in entries if entry.vendor == args.vendor]
        if not vendor_entries:
            raise ValueError(f"No entries found for vendor {args.vendor}")
        entry = vendor_entries[0]
    else:
        entry = lookup[args.emoji_key]

    with Image.open(entry.image_path) as image:
        return image.convert("RGBA").copy(), entry.name, entry.vendor


def main() -> int:
    args = parse_args()
    source_image, source_name, source_vendor = resolve_source(args)
    result, metadata = edit_emoji_image(
        source_image=source_image,
        instruction=args.instruction,
        base_model=args.base_model,
        lora_path=args.lora_path if Path(args.lora_path).exists() else None,
        precision=args.precision,
        device=args.device,
        source_name=source_name,
        source_vendor=source_vendor,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        seed=args.seed,
        resolution=args.resolution,
        scheduler_name=args.scheduler,
        extra_style_hint=args.extra_style_hint,
    )
    args.output_image.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output_image)
    args.output_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_image)
    print(args.output_metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
