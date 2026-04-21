"""Download and extract public Kaggle datasets for the emoji editing project."""

from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    handle: str
    archive_name: str
    extract_dir: str
    description: str


@dataclass(frozen=True)
class DownloadConfig:
    project_root: Path
    force_redownload: bool


# 在这里修改下载脚本配置。
DOWNLOAD_CONFIG = DownloadConfig(
    project_root=Path(__file__).resolve().parents[1],  # 项目根目录，通常不需要改。
    force_redownload=False,  # 设为 True 时会重新下载并重新解压所有数据。
)


DATASETS = [
    DatasetSpec(
        handle="subinium/emojiimage-dataset",
        archive_name="subinium_emojiimage_dataset.zip",
        extract_dir="full_emoji_image_dataset",
        description="Vendor-specific emoji images for paired editing and style transfer.",
    ),
    DatasetSpec(
        handle="ajabkhan21/complete-unicode-emoji-dataset-emojis-with-meaning",
        archive_name="ajabkhan21_complete_unicode_emoji_dataset_emojis_with_meaning.zip",
        extract_dir="unicode_emoji_meanings",
        description="Unicode emoji strings and official descriptions for text supervision.",
    ),
    DatasetSpec(
        handle="rtatman/emojinet",
        archive_name="rtatman_emojinet.zip",
        extract_dir="emojinet",
        description="Emoji lexical senses and context metadata for richer instruction templates.",
    ),
    DatasetSpec(
        handle="shuvokumarbasak4004/emojis-list-unicode-image-dataset",
        archive_name="shuvokumarbasak4004_emojis_list_unicode_image_dataset.zip",
        extract_dir="unicode_emoji_image_dataset",
        description="Uniformly rendered 256x256 Unicode emoji images for canonical targets.",
    ),
]


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def download_archive(spec: DatasetSpec, archives_dir: Path, force: bool) -> Path:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec.handle}"
    archive_path = archives_dir / spec.archive_name
    if archive_path.exists() and not force:
        print(f"[skip] archive exists: {archive_path}")
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[download] {spec.handle}")
    print(f"           -> {archive_path}")
    with urllib.request.urlopen(url, timeout=120) as response, tmp_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(archive_path)
    print(f"[saved] {archive_path.name} ({format_size(archive_path.stat().st_size)})")
    return archive_path


def extract_archive(spec: DatasetSpec, archive_path: Path, extracts_root: Path, force: bool) -> Path:
    extract_path = extracts_root / spec.extract_dir
    if extract_path.exists() and any(extract_path.iterdir()) and not force:
        print(f"[skip] extracted dir exists: {extract_path}")
        return extract_path

    ensure_clean_dir(extract_path)
    print(f"[extract] {archive_path.name}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_path)
    file_count = sum(1 for p in extract_path.rglob("*") if p.is_file())
    print(f"[ready] {extract_path} ({file_count} files)")
    return extract_path


def main() -> int:
    config = DOWNLOAD_CONFIG
    project_root = config.project_root.resolve()
    raw_kaggle_dir = project_root / "data" / "raw" / "kaggle"
    archives_dir = raw_kaggle_dir / "archives"
    interim_dir = project_root / "data" / "interim"
    processed_dir = project_root / "data" / "processed"

    archives_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Kaggle raw dir: {raw_kaggle_dir}")
    print()

    for spec in DATASETS:
        print(f"== {spec.extract_dir} ==")
        print(spec.description)
        archive_path = download_archive(spec, archives_dir, force=config.force_redownload)
        extract_archive(spec, archive_path, raw_kaggle_dir, force=config.force_redownload)
        print()

    print("All requested Kaggle datasets are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
