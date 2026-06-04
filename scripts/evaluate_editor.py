#!/usr/bin/env python3
"""Quantitatively compare baselines vs. the LoRA-finetuned emoji editor.

对同一批测试 pair，跑三套方案并用 CLIP / LPIPS 定量打分，输出对比表：

- ``copy``     : 直接返回源图（no-op 启发式，image-similarity 的天然下界）。
- ``zeroshot`` : 未微调的预训练 InstructPix2Pix（强 baseline）。
- ``lora``     : 在自建数据上微调后的模型（我们的 Main Approach）。

指标（均为逐样本计算后按 task_type 聚合）：
- clip_text_alignment  : 结果图 ↔ 编辑指令的 CLIP 相似度（越高越符合指令）。
- clip_image_to_target : 结果图 ↔ 目标图的 CLIP 相似度（越高越接近 ground truth）。
- clip_image_to_source : 结果图 ↔ 源图的 CLIP 相似度（越高越保持源图身份）。
- lpips_to_target      : 结果图 ↔ 目标图的 LPIPS 感知距离（越低越好，需安装 lpips）。

在 EVAL_CONFIG 里改配置，然后 ``python scripts/evaluate_editor.py``。
"""

from __future__ import annotations

import csv
import json
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

from emoji_editing.diffusion_inference import edit_emoji_image, load_editor_pipeline, prepare_input_image
from emoji_editing.evaluation import (
    clip_image_similarity,
    clip_text_alignment,
    lpips_available,
    lpips_distance,
    mean,
)
from emoji_editing.io_utils import read_csv_rows


@dataclass(frozen=True)
class EvalConfig:
    pair_csv: Path
    base_model: str
    lora_path: Path
    split: str
    num_samples: int
    resolution: int
    precision: str
    device: str | None
    steps: int
    guidance_scale: float
    image_guidance_scale: float
    seed: int
    subset_seed: int
    clip_model: str
    use_lpips: bool
    output_csv: Path
    output_json: Path


# 在这里修改评估配置。
EVAL_CONFIG = EvalConfig(
    pair_csv=Path("data/interim/emoji_editing/metadata/all_edit_pairs.csv"),  # 样本表。
    base_model="timbrooks/instruct-pix2pix",  # 底座（也用作 zeroshot baseline）。
    lora_path=Path("artifacts/emoji_diffusion_editor/lora_final"),  # 训练产出的 LoRA。
    split="test",  # 在哪个划分上评估。
    num_samples=160,  # 评估样本数（按 task_type 分层均分）；时间紧可调小。
    resolution=256,  # 与训练一致的分辨率。
    precision="fp16",  # 单卡推理用 fp16。
    device=None,  # None 时自动选 cuda/cpu。
    steps=30,  # 推理步数。
    guidance_scale=5.0,  # 文本 guidance。
    image_guidance_scale=1.8,  # 图像保持 guidance。
    seed=1234,  # 固定种子，保证可复现。
    subset_seed=0,  # 子集采样种子。
    clip_model="openai/clip-vit-base-patch32",  # CLIP 评测模型。
    use_lpips=True,  # 安装了 lpips 才会真正计算，否则自动跳过。
    output_csv=Path("artifacts/evaluation/editor_metrics.csv"),  # 扁平结果表。
    output_json=Path("artifacts/evaluation/editor_metrics.json"),  # 含 task_type 细分的完整结果。
)

METRICS = ("clip_text_alignment", "clip_image_to_target", "clip_image_to_source", "lpips_to_target")
HIGHER_BETTER = {
    "clip_text_alignment": True,
    "clip_image_to_target": True,
    "clip_image_to_source": True,
    "lpips_to_target": False,
}


def stratified_sample(rows: list[dict], num_samples: int, seed: int) -> list[dict]:
    """按 task_type 分组均分采样，保证每种编辑任务都被覆盖。"""
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


def load_prepared(path: str, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        source = image.convert("RGBA").copy()
    return prepare_input_image(source, resolution=resolution)


def generate_results(method: str, rows: list[dict], config: EvalConfig) -> list[Image.Image]:
    results: list[Image.Image] = []
    for row in tqdm(rows, desc=f"generate[{method}]", leave=False):
        if method == "copy":
            results.append(load_prepared(row["source_image_path"], config.resolution))
            continue
        with Image.open(row["source_image_path"]) as image:
            source = image.convert("RGBA").copy()
        lora_path = str(config.lora_path) if method == "lora" else None
        result, _ = edit_emoji_image(
            source_image=source,
            instruction=row["instruction"],
            base_model=config.base_model,
            lora_path=lora_path,
            precision=config.precision,
            device=config.device,
            source_name=row["source_name"],
            source_vendor=row["source_vendor"],
            steps=config.steps,
            guidance_scale=config.guidance_scale,
            image_guidance_scale=config.image_guidance_scale,
            seed=config.seed,
            resolution=config.resolution,
        )
        results.append(result)
    return results


def aggregate_by_task(values: list[float], task_types: list[str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for value, task in zip(values, task_types):
        groups[task].append(value)
        groups["overall"].append(value)
    return {scope: mean(scores) for scope, scores in groups.items()}


def score_method(
    results: list[Image.Image],
    targets: list[Image.Image],
    sources: list[Image.Image],
    instructions: list[str],
    task_types: list[str],
    config: EvalConfig,
    compute_lpips: bool,
) -> dict[str, dict[str, float]]:
    clip_t = clip_text_alignment(results, instructions, model_name=config.clip_model, device=config.device)
    clip_i_target = clip_image_similarity(results, targets, model_name=config.clip_model, device=config.device)
    clip_i_source = clip_image_similarity(results, sources, model_name=config.clip_model, device=config.device)
    scored = {
        "clip_text_alignment": aggregate_by_task(clip_t, task_types),
        "clip_image_to_target": aggregate_by_task(clip_i_target, task_types),
        "clip_image_to_source": aggregate_by_task(clip_i_source, task_types),
    }
    if compute_lpips:
        lpips_target = lpips_distance(results, targets, device=config.device)
        scored["lpips_to_target"] = aggregate_by_task(lpips_target, task_types)
    else:
        scored["lpips_to_target"] = {"overall": float("nan")}
    return scored


def print_overall_table(report: dict[str, dict[str, dict[str, float]]], methods: list[str]) -> None:
    header = f"{'method':<10}" + "".join(f"{m:>22}" for m in METRICS)
    print("\n=== Overall (按全部样本平均) ===")
    print(header)
    print("-" * len(header))
    for method in methods:
        cells = []
        for metric in METRICS:
            value = report[method][metric].get("overall", float("nan"))
            arrow = "↑" if HIGHER_BETTER[metric] else "↓"
            cells.append(f"{value:>20.4f}{arrow}")
        print(f"{method:<10}" + "".join(cells))
    print("（↑ 越高越好，↓ 越低越好；lpips 需安装 lpips 包）")


def write_outputs(report: dict, methods: list[str], task_types: list[str], config: EvalConfig) -> None:
    config.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "split": config.split,
            "num_samples_requested": config.num_samples,
            "num_samples_used": len(task_types),
            "steps": config.steps,
            "guidance_scale": config.guidance_scale,
            "image_guidance_scale": config.image_guidance_scale,
            "seed": config.seed,
            "task_type_counts": {t: task_types.count(t) for t in sorted(set(task_types))},
        },
        "methods": methods,
        "metrics": report,
    }
    config.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scopes = sorted({scope for method in methods for metric in METRICS for scope in report[method][metric]})
    with config.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "scope", *METRICS])
        for method in methods:
            for scope in scopes:
                writer.writerow(
                    [method, scope, *[f"{report[method][metric].get(scope, float('nan')):.6f}" for metric in METRICS]]
                )


def main() -> int:
    config = EVAL_CONFIG
    if not torch.cuda.is_available():
        print("[warn] 未检测到 CUDA，推理会非常慢。", flush=True)

    rows_all = [row for row in read_csv_rows(config.pair_csv) if row["split"] == config.split]
    if not rows_all:
        raise ValueError(f"split={config.split} 下没有样本，请检查 {config.pair_csv}")
    rows = stratified_sample(rows_all, config.num_samples, config.subset_seed)
    task_types = [row["task_type"] for row in rows]
    instructions = [row["instruction"] for row in rows]
    print(f"[stage] 评估 {len(rows)} 个样本，task_type 分布: "
          f"{ {t: task_types.count(t) for t in sorted(set(task_types))} }", flush=True)

    # 参考图只需准备一次（与具体方案无关）。
    targets = [load_prepared(row["target_image_path"], config.resolution) for row in rows]
    sources = [load_prepared(row["source_image_path"], config.resolution) for row in rows]

    methods = ["copy", "zeroshot"]
    if config.lora_path.exists():
        methods.append("lora")
    else:
        print(f"[warn] 未找到 LoRA 目录 {config.lora_path}，仅评估 copy 与 zeroshot baseline。", flush=True)

    compute_lpips = config.use_lpips and lpips_available()
    if config.use_lpips and not compute_lpips:
        print("[warn] 未安装 lpips，跳过 LPIPS 指标（pip install lpips 可启用）。", flush=True)

    report: dict[str, dict[str, dict[str, float]]] = {}
    for method in methods:
        print(f"[stage] 生成方案 {method} ...", flush=True)
        results = generate_results(method, rows, config)
        report[method] = score_method(
            results, targets, sources, instructions, task_types, config, compute_lpips
        )
        # 切换方案前清掉 pipeline 缓存，避免 zeroshot 与 lora 两套权重同时占显存。
        if method in ("zeroshot", "lora"):
            load_editor_pipeline.cache_clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_overall_table(report, methods)
    write_outputs(report, methods, task_types, config)
    print(f"\n[done] 明细 JSON: {config.output_json}")
    print(f"[done] 扁平 CSV: {config.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
