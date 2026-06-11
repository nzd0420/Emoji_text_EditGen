#!/usr/bin/env python3
"""Sweep image_guidance_scale to reveal the instruction-vs-fidelity trade-off.

固定微调后的模型，扫描不同的 ``image_guidance_scale``，在测试子集上测两个指标：

- clip_text_alignment  : 结果 ↔ 指令的对齐度（越高越"服从指令"）。
- clip_image_to_source : 结果 ↔ 源图的相似度（越高越"保持原图")。

这正是 InstructPix2Pix 论文里经典的权衡曲线：图像 guidance 越大越保守（贴近原图、
但少改动），越小越激进（改得多、但容易跑偏）。输出 CSV，并在装了 matplotlib 时画图。

在 SWEEP_CONFIG 里改配置，然后 ``python scripts/sweep_guidance.py``。
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

# 让脚本从任意工作目录都能 import emoji_editing。
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from emoji_editing.diffusion_inference import edit_emoji_image, prepare_input_image
from emoji_editing.evaluation import clip_image_similarity, clip_text_alignment, mean
from emoji_editing.io_utils import read_csv_rows


@dataclass(frozen=True)
class SweepConfig:
    pair_csv: Path
    base_model: str
    lora_path: Path | None
    split: str
    num_samples: int
    resolution: int
    precision: str
    device: str | None
    steps: int
    guidance_scale: float
    image_guidance_scales: tuple[float, ...]
    seed: int
    subset_seed: int
    clip_model: str
    output_csv: Path
    output_plot: Path


# 在这里修改扫描配置。
SWEEP_CONFIG = SweepConfig(
    pair_csv=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),  # 样本表。
    base_model="timbrooks/instruct-pix2pix",  # 底座模型。
    lora_path=Path("artifacts/emoji_diffusion_editor_60k/lora_final"),  # 微调 LoRA；设 None 则扫 zeroshot。
    split="test",  # 评估划分。
    num_samples=80,  # 每个 guidance 值评估的样本数（分层均分）。
    resolution=256,  # 分辨率。
    precision="fp16",  # 推理精度。
    device="cuda:0",  # 强制使用第一张 GPU。
    steps=30,  # 推理步数。
    guidance_scale=5.0,  # 固定的文本 guidance。
    image_guidance_scales=(1.0, 1.5, 2.0, 2.5),  # 要扫描的图像 guidance 列表。
    seed=1234,  # 固定种子。
    subset_seed=0,  # 子集种子。
    clip_model="openai/clip-vit-base-patch32",  # CLIP 评测模型。
    output_csv=Path("artifacts/evaluation/guidance_sweep.csv"),  # 曲线数据。
    output_plot=Path("artifacts/evaluation/guidance_sweep.png"),  # 可选曲线图。
)


def stratified_sample(rows: list[dict], num_samples: int, seed: int) -> list[dict]:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
    tasks = sorted(by_task)
    per_task = max(1, num_samples // len(tasks))
    rng = random.Random(seed)
    picked: list[dict] = []
    for task in tasks:
        pool = list(by_task[task])
        rng.shuffle(pool)
        picked.extend(pool[:per_task])
    rng.shuffle(picked)
    return picked


def maybe_plot(records: list[dict], config: SweepConfig) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] 未安装 matplotlib，跳过画图，仅输出 CSV。", flush=True)
        return

    xs = [r["clip_image_to_source"] for r in records]
    ys = [r["clip_text_alignment"] for r in records]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(xs, ys, marker="o", color="#c0552f")
    for record in records:
        ax.annotate(
            f"g={record['image_guidance_scale']}",
            (record["clip_image_to_source"], record["clip_text_alignment"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("CLIP image similarity to source (保持原图 →)")
    ax.set_ylabel("CLIP text alignment (服从指令 ↑)")
    ax.set_title("Image-guidance trade-off")
    fig.tight_layout()
    config.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_plot, dpi=150)
    print(f"[done] 曲线图: {config.output_plot}", flush=True)


def main() -> int:
    config = SWEEP_CONFIG
    if config.lora_path is not None and not config.lora_path.exists():
        raise FileNotFoundError(f"未找到 LoRA 目录 {config.lora_path}；设为 None 可扫描 zeroshot。")

    rows_all = [row for row in read_csv_rows(config.pair_csv) if row["split"] == config.split]
    rows = stratified_sample(rows_all, config.num_samples, config.subset_seed)
    instructions = [row["instruction"] for row in rows]
    sources = [
        prepare_input_image(_open_rgba(row["source_image_path"]), resolution=config.resolution) for row in rows
    ]
    print(f"[stage] 扫描 {len(config.image_guidance_scales)} 个 guidance 值，每个 {len(rows)} 个样本。", flush=True)

    lora_path = str(config.lora_path) if config.lora_path is not None else None
    records: list[dict] = []
    for image_guidance in config.image_guidance_scales:
        results: list[Image.Image] = []
        for row in tqdm(rows, desc=f"image_guidance={image_guidance}", leave=False):
            result, _ = edit_emoji_image(
                source_image=_open_rgba(row["source_image_path"]),
                instruction=row["instruction"],
                base_model=config.base_model,
                lora_path=lora_path,
                precision=config.precision,
                device=config.device,
                source_name=row["source_name"],
                source_vendor=row["source_vendor"],
                steps=config.steps,
                guidance_scale=config.guidance_scale,
                image_guidance_scale=image_guidance,
                seed=config.seed,
                resolution=config.resolution,
            )
            results.append(result)
        clip_t = mean(clip_text_alignment(results, instructions, model_name=config.clip_model, device=config.device))
        clip_i_src = mean(
            clip_image_similarity(results, sources, model_name=config.clip_model, device=config.device)
        )
        records.append(
            {
                "image_guidance_scale": image_guidance,
                "clip_text_alignment": clip_t,
                "clip_image_to_source": clip_i_src,
            }
        )
        print(
            f"[result] image_guidance={image_guidance:<4} "
            f"clip_text_alignment={clip_t:.4f}  clip_image_to_source={clip_i_src:.4f}",
            flush=True,
        )

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with config.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_guidance_scale", "clip_text_alignment", "clip_image_to_source"])
        for record in records:
            writer.writerow(
                [record["image_guidance_scale"], f"{record['clip_text_alignment']:.6f}", f"{record['clip_image_to_source']:.6f}"]
            )
    print(f"[done] 曲线数据: {config.output_csv}", flush=True)
    maybe_plot(records, config)
    return 0


def _open_rgba(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGBA").copy()


if __name__ == "__main__":
    raise SystemExit(main())
