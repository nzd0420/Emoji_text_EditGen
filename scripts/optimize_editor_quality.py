#!/usr/bin/env python3
"""Compare 60k checkpoints and search inference guidance settings.

This script runs the two practical quality tuning passes for the emoji editor:

1. Compare saved 60k LoRA checkpoints on the same sampled test rows.
2. Pick a best checkpoint by CLIP-based metrics, then run a guidance grid.

Outputs are written under ``artifacts/quality_search`` by default:

- ``checkpoint_metrics.csv``
- ``checkpoint_comparison.png``
- ``best_checkpoint.json``
- ``guidance_metrics.csv``
- ``guidance_grid.png``
- ``best_guidance.json``

The script is intentionally config-file style, matching the rest of the repo.
Edit ``SEARCH_CONFIG`` below, then run:

``python scripts/optimize_editor_quality.py``
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

# Let this script import emoji_editing from any working directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emoji_editing.diffusion_inference import edit_emoji_image, load_editor_pipeline, prepare_input_image
from emoji_editing.evaluation import clip_image_similarity, clip_text_alignment, mean
from emoji_editing.io_utils import read_csv_rows
from emoji_editing.prompting import DEFAULT_NEGATIVE_PROMPT


@dataclass(frozen=True)
class QualitySearchConfig:
    pair_csv: Path
    base_model: str
    candidate_lora_paths: tuple[Path, ...]
    split: str
    num_samples: int
    subset_seed: int
    generation_seed: int
    resolution: int
    precision: str
    device: str | None
    checkpoint_steps: int
    checkpoint_guidance_scale: float
    checkpoint_image_guidance_scale: float
    checkpoint_scheduler: str
    guidance_steps: int
    guidance_scales: tuple[float, ...]
    image_guidance_scales: tuple[float, ...]
    schedulers: tuple[str, ...]
    clip_model: str
    output_dir: Path


SEARCH_CONFIG = QualitySearchConfig(
    pair_csv=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),
    base_model="timbrooks/instruct-pix2pix",
    candidate_lora_paths=(
        Path("artifacts/emoji_diffusion_editor_60k/checkpoints/checkpoint-52000/lora"),
        Path("artifacts/emoji_diffusion_editor_60k/checkpoints/checkpoint-54000/lora"),
        Path("artifacts/emoji_diffusion_editor_60k/checkpoints/checkpoint-56000/lora"),
        Path("artifacts/emoji_diffusion_editor_60k/checkpoints/checkpoint-58000/lora"),
        Path("artifacts/emoji_diffusion_editor_60k/checkpoints/checkpoint-60000/lora"),
        Path("artifacts/emoji_diffusion_editor_60k/lora_final"),
    ),
    split="test",
    num_samples=6,
    subset_seed=0,
    generation_seed=1234,
    resolution=256,
    precision="fp16",
    device="cuda:0",
    checkpoint_steps=35,
    checkpoint_guidance_scale=4.5,
    checkpoint_image_guidance_scale=2.2,
    checkpoint_scheduler="dpm",
    guidance_steps=40,
    guidance_scales=(3.5, 4.5, 5.5),
    image_guidance_scales=(1.6, 2.2, 2.8),
    schedulers=("euler_a", "dpm"),
    clip_model="openai/clip-vit-base-patch32",
    output_dir=Path("artifacts/quality_search"),
)


def lora_label(path: Path) -> str:
    if path.name == "lora":
        return path.parent.name
    return path.name


def combo_label(scheduler: str, guidance: float, image_guidance: float) -> str:
    return f"{scheduler}_t{guidance:g}_i{image_guidance:g}"


def stratified_sample(rows: list[dict[str, str]], num_samples: int, seed: int) -> list[dict[str, str]]:
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row["task_type"], []).append(row)

    rng = random.Random(seed)
    tasks = sorted(by_task)
    per_task = max(1, math.ceil(num_samples / max(1, len(tasks))))
    picked: list[dict[str, str]] = []
    for task in tasks:
        pool = list(by_task[task])
        rng.shuffle(pool)
        picked.extend(pool[:per_task])
    rng.shuffle(picked)
    return picked[:num_samples]


def open_rgba(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_score(record: dict[str, float]) -> float:
    # Target similarity matters most; source similarity protects identity; text is a weaker CLIP signal.
    return (
        record["clip_image_to_target"]
        + 0.35 * record["clip_text_alignment"]
        + 0.20 * record["clip_image_to_source"]
    )


def score_results(
    results: list[Image.Image],
    targets: list[Image.Image],
    sources: list[Image.Image],
    instructions: list[str],
    config: QualitySearchConfig,
) -> dict[str, float]:
    clip_text = mean(clip_text_alignment(results, instructions, model_name=config.clip_model, device=config.device))
    clip_target = mean(clip_image_similarity(results, targets, model_name=config.clip_model, device=config.device))
    clip_source = mean(clip_image_similarity(results, sources, model_name=config.clip_model, device=config.device))
    record = {
        "clip_text_alignment": clip_text,
        "clip_image_to_target": clip_target,
        "clip_image_to_source": clip_source,
    }
    record["score"] = metric_score(record)
    return record


def render_cell(image: Image.Image, caption: str, cell_size: int, caption_height: int) -> Image.Image:
    cell = Image.new("RGB", (cell_size, cell_size + caption_height), color=(250, 249, 246))
    preview = image.convert("RGB").resize((cell_size, cell_size), resample=Image.LANCZOS)
    cell.paste(preview, (0, 0))
    draw = ImageDraw.Draw(cell)
    draw.rectangle((0, cell_size, cell_size, cell_size + caption_height), fill=(245, 243, 238))
    draw.text((8, cell_size + 8), caption[:64], fill=(28, 28, 28))
    return cell


def make_sheet(
    columns: list[str],
    rows: list[list[Image.Image]],
    row_labels: list[str],
    output_path: Path,
    cell_size: int = 160,
    caption_height: int = 46,
) -> None:
    width = len(columns) * cell_size
    header_height = 36
    height = header_height + len(rows) * (cell_size + caption_height)
    sheet = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for col_idx, title in enumerate(columns):
        x = col_idx * cell_size
        draw.rectangle((x, 0, x + cell_size, header_height), fill=(32, 31, 28))
        draw.text((x + 8, 10), title[:24], fill=(255, 255, 255))

    for row_idx, images in enumerate(rows):
        y = header_height + row_idx * (cell_size + caption_height)
        for col_idx, image in enumerate(images):
            caption = row_labels[row_idx] if col_idx == 0 else columns[col_idx]
            sheet.paste(render_cell(image, caption, cell_size, caption_height), (col_idx * cell_size, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def clear_pipeline_cache() -> None:
    load_editor_pipeline.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_for_rows(
    rows: list[dict[str, str]],
    lora_path: Path,
    config: QualitySearchConfig,
    steps: int,
    guidance_scale: float,
    image_guidance_scale: float,
    scheduler: str,
    output_subdir: Path,
) -> list[Image.Image]:
    output_subdir.mkdir(parents=True, exist_ok=True)
    results: list[Image.Image] = []
    for index, row in enumerate(tqdm(rows, desc=output_subdir.name, leave=False)):
        source = open_rgba(row["source_image_path"])
        result, metadata = edit_emoji_image(
            source_image=source,
            instruction=row["instruction"],
            base_model=config.base_model,
            lora_path=str(lora_path),
            precision=config.precision,
            device=config.device,
            source_name=row["source_name"],
            source_vendor=row["source_vendor"],
            steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            seed=config.generation_seed + index,
            resolution=config.resolution,
            scheduler_name=scheduler,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        )
        filename = f"{index:02d}_{row['pair_id']}.png"
        result.save(output_subdir / filename)
        save_json(output_subdir / f"{index:02d}_{row['pair_id']}.json", metadata)
        results.append(result)
    return results


def compare_checkpoints(
    rows: list[dict[str, str]],
    sources: list[Image.Image],
    targets: list[Image.Image],
    instructions: list[str],
    config: QualitySearchConfig,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    outputs_by_label: dict[str, list[Image.Image]] = {}
    valid_candidates = [path for path in config.candidate_lora_paths if path.exists()]
    if not valid_candidates:
        raise FileNotFoundError("No candidate LoRA checkpoint directories exist.")

    for lora_path in valid_candidates:
        label = lora_label(lora_path)
        print(f"[stage] checkpoint {label}", flush=True)
        generated = generate_for_rows(
            rows=rows,
            lora_path=lora_path,
            config=config,
            steps=config.checkpoint_steps,
            guidance_scale=config.checkpoint_guidance_scale,
            image_guidance_scale=config.checkpoint_image_guidance_scale,
            scheduler=config.checkpoint_scheduler,
            output_subdir=config.output_dir / "checkpoint_images" / label,
        )
        scores = score_results(generated, targets, sources, instructions, config)
        record = {
            "label": label,
            "lora_path": str(lora_path),
            "steps": config.checkpoint_steps,
            "guidance_scale": config.checkpoint_guidance_scale,
            "image_guidance_scale": config.checkpoint_image_guidance_scale,
            "scheduler": config.checkpoint_scheduler,
            **scores,
        }
        records.append(record)
        outputs_by_label[label] = generated
        clear_pipeline_cache()

    records.sort(key=lambda item: float(item["score"]), reverse=True)
    metrics_path = config.output_dir / "checkpoint_metrics.csv"
    write_csv(
        metrics_path,
        records,
        [
            "label",
            "lora_path",
            "steps",
            "guidance_scale",
            "image_guidance_scale",
            "scheduler",
            "clip_text_alignment",
            "clip_image_to_target",
            "clip_image_to_source",
            "score",
        ],
    )

    columns = ["source", "target", *[record["label"] for record in records]]
    sheet_rows: list[list[Image.Image]] = []
    row_labels: list[str] = []
    for row_index, row in enumerate(rows):
        sheet_rows.append(
            [
                sources[row_index],
                targets[row_index],
                *[outputs_by_label[record["label"]][row_index] for record in records],
            ]
        )
        row_labels.append(f"{row['task_type']} {row['pair_id']}")
    make_sheet(columns, sheet_rows, row_labels, config.output_dir / "checkpoint_comparison.png")

    best = records[0]
    save_json(
        config.output_dir / "best_checkpoint.json",
        {
            "best_checkpoint": best,
            "selection_note": "Best by CLIP target + text + source composite score. Inspect checkpoint_comparison.png visually.",
        },
    )
    print(f"[best] checkpoint={best['label']} score={best['score']:.4f}", flush=True)
    return Path(best["lora_path"]), records, best


def search_guidance_grid(
    rows: list[dict[str, str]],
    sources: list[Image.Image],
    targets: list[Image.Image],
    instructions: list[str],
    best_lora_path: Path,
    config: QualitySearchConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    first_row_outputs: list[tuple[str, Image.Image]] = []

    for scheduler in config.schedulers:
        for guidance_scale in config.guidance_scales:
            for image_guidance_scale in config.image_guidance_scales:
                label = combo_label(scheduler, guidance_scale, image_guidance_scale)
                print(f"[stage] guidance {label}", flush=True)
                generated = generate_for_rows(
                    rows=rows,
                    lora_path=best_lora_path,
                    config=config,
                    steps=config.guidance_steps,
                    guidance_scale=guidance_scale,
                    image_guidance_scale=image_guidance_scale,
                    scheduler=scheduler,
                    output_subdir=config.output_dir / "guidance_images" / label,
                )
                scores = score_results(generated, targets, sources, instructions, config)
                record = {
                    "label": label,
                    "lora_path": str(best_lora_path),
                    "steps": config.guidance_steps,
                    "guidance_scale": guidance_scale,
                    "image_guidance_scale": image_guidance_scale,
                    "scheduler": scheduler,
                    **scores,
                }
                records.append(record)
                first_row_outputs.append((label, generated[0]))

    clear_pipeline_cache()
    records.sort(key=lambda item: float(item["score"]), reverse=True)
    write_csv(
        config.output_dir / "guidance_metrics.csv",
        records,
        [
            "label",
            "lora_path",
            "steps",
            "guidance_scale",
            "image_guidance_scale",
            "scheduler",
            "clip_text_alignment",
            "clip_image_to_target",
            "clip_image_to_source",
            "score",
        ],
    )

    # Make a readable visual grid for the first sampled row.
    grid_columns = ["source", "target", *[label for label, _ in first_row_outputs]]
    make_sheet(
        grid_columns,
        [[sources[0], targets[0], *[image for _, image in first_row_outputs]]],
        [f"{rows[0]['task_type']} {rows[0]['pair_id']}"],
        config.output_dir / "guidance_grid.png",
        cell_size=150,
        caption_height=52,
    )

    best = records[0]
    save_json(
        config.output_dir / "best_guidance.json",
        {
            "best_guidance": best,
            "selection_note": "Best by CLIP target + text + source composite score. Inspect guidance_grid.png visually.",
        },
    )
    print(f"[best] guidance={best['label']} score={best['score']:.4f}", flush=True)
    return records


def main() -> int:
    config = SEARCH_CONFIG
    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.output_dir / "search_config.json", asdict(config))

    if config.device and config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"SEARCH_CONFIG.device={config.device!r}, but torch.cuda.is_available() is False. "
            "Fix CUDA visibility or set device=None/cpu for a slow CPU run."
        )

    rows_all = [row for row in read_csv_rows(config.pair_csv) if row["split"] == config.split]
    if not rows_all:
        raise ValueError(f"No rows found for split={config.split}")
    rows = stratified_sample(rows_all, config.num_samples, config.subset_seed)
    instructions = [row["instruction"] for row in rows]
    sources = [prepare_input_image(open_rgba(row["source_image_path"]), resolution=config.resolution) for row in rows]
    targets = [prepare_input_image(open_rgba(row["target_image_path"]), resolution=config.resolution) for row in rows]

    print(f"[stage] sampled {len(rows)} rows from split={config.split}", flush=True)
    best_lora_path, _, _ = compare_checkpoints(rows, sources, targets, instructions, config)
    search_guidance_grid(rows, sources, targets, instructions, best_lora_path, config)
    print(f"[done] outputs: {config.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
