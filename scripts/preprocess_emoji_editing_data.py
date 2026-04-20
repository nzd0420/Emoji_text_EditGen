"""Build a curated emoji editing dataset from the downloaded Kaggle sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


VENDORS = [
    "Apple",
    "Google",
    "Facebook",
    "Windows",
    "Twitter",
    "JoyPixels",
    "Samsung",
    "Gmail",
    "SoftBank",
    "DoCoMo",
    "KDDI",
]

INCLUDE_EXACT_NAMES = {
    "rolling on the floor laughing",
    "star-struck",
    "exploding head",
}

EXCLUDE_NAME_KEYWORDS = {
    "cat",
    "monkey",
    "skull",
    "poo",
    "clown",
    "ogre",
    "goblin",
    "ghost",
    "alien",
    "robot",
    "person",
    "man",
    "woman",
    "dog face",
    "tiger face",
    "horse face",
    "cow face",
    "pig face",
    "mouse face",
    "rabbit face",
    "dragon face",
    "moon",
    "sun with face",
    "wind",
}

ATTRIBUTE_FIELDS = [
    "emotion_primary",
    "sentiment",
    "has_tears",
    "has_sweat",
    "has_hearts",
    "has_tongue",
    "has_glasses",
    "has_mask",
    "has_halo",
    "has_horns",
    "has_hat",
    "has_temperature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root where data/raw is located.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild processed images and metadata outputs from scratch.",
    )
    return parser.parse_args()


def normalize_unicode(value: str) -> str:
    parts = re.findall(r"U\+[0-9A-F]+", value.upper())
    return "-".join(parts)


def sanitize_name(value: str) -> str:
    return value.replace("⊛", "").replace("  ", " ").strip()


def strip_unicode_version_prefix(value: str) -> str:
    return re.sub(r"^E\d+(?:\.\d+)?\s+", "", value).strip()


def slug_from_unicode(unicode_slug: str) -> str:
    return unicode_slug.lower().replace("+", "").replace("-", "_")


def remove_ds_store(root: Path) -> None:
    for path in root.rglob(".DS_Store"):
        path.unlink()


def ensure_clean_dir(path: Path, force: bool) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def load_unicode_descriptions(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_rows(path)
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        emoji = row["emoji"].strip()
        description = strip_unicode_version_prefix(row["description"])
        lookup[(emoji, "")] = {
            "emoji": emoji,
            "unicode_description": description,
        }
    return lookup


def load_unicode_image_lookup(csv_path: Path, image_root: Path) -> dict[tuple[str, str], dict[str, str]]:
    rows = read_csv_rows(csv_path)
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        unicode_slug = normalize_unicode(row["Emoji Unicode"])
        emoji_char = row["Emoji Character"].strip()
        image_path = image_root / unicode_slug / "emoji.png"
        if not image_path.exists():
            continue
        payload = {
            "unicode_slug": unicode_slug,
            "unicode_name": row["Emoji Name"].strip(":"),
            "image_path": str(image_path),
        }
        lookup[(emoji_char, "")] = payload
        lookup[("", unicode_slug)] = payload
        lookup[(emoji_char, unicode_slug)] = payload
    return lookup


def load_emojinet_lookup(path: Path) -> dict[str, dict[str, str]]:
    items = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, dict[str, str]] = {}
    for item in items:
        unicode_slug = normalize_unicode(item.get("unicode", ""))
        if not unicode_slug:
            continue
        keywords = item.get("keywords") or []
        if isinstance(keywords, list):
            keywords_text = "|".join(str(value).strip() for value in keywords if str(value).strip())
        else:
            keywords_text = str(keywords).strip()
        lookup[unicode_slug] = {
            "emojinet_name": sanitize_name(str(item.get("name", "")).strip()),
            "emojinet_category": str(item.get("category") or "").strip(),
            "emojinet_shortcode": str(item.get("shortcode") or "").strip(),
            "emojinet_definition": str(item.get("definition") or "").strip(),
            "emojinet_keywords": keywords_text,
        }
    return lookup


def is_human_face_emoji(name: str) -> bool:
    lowered = name.lower().strip()
    if lowered in INCLUDE_EXACT_NAMES:
        return True
    if "face" not in lowered:
        return False
    return not any(token in lowered for token in EXCLUDE_NAME_KEYWORDS)


def infer_emotion(name: str) -> str:
    text = name.lower()
    rules = [
        ("love", ("heart", "kiss")),
        ("joy", ("laugh", "grinning", "beaming", "party", "partying", "zany", "smiling", "hugging", "savoring")),
        ("cool", ("sunglasses", "nerd", "monocle", "cowboy")),
        ("thinking", ("thinking", "raised eyebrow", "shushing", "hand over mouth", "zipper-mouth", "lying")),
        ("neutral", ("neutral", "expressionless", "without mouth", "rolling eyes", "smirk", "unamused", "relieved", "pensive", "confused")),
        ("surprised", ("open mouth", "hushed", "astonished", "flushed", "pleading", "exhaling", "in clouds", "spiral eyes", "exploding")),
        ("sad", ("frowning", "anguished", "crying", "disappointed", "downcast", "weary", "tired", "persevering", "confounded", "sad", "sleepy", "sleeping", "yawning")),
        ("fear", ("fear", "screaming", "anxious", "worried")),
        ("angry", ("angry", "pouting", "steam", "symbols on mouth", "horns")),
        ("sick", ("mask", "thermometer", "bandage", "nauseated", "vomiting", "sneezing", "hot", "cold", "woozy", "knocked-out", "drooling")),
        ("playful", ("tongue", "upside-down", "money-mouth")),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "other"


def infer_sentiment(name: str, emotion: str) -> str:
    text = name.lower()
    if emotion in {"joy", "love", "cool", "playful"}:
        return "positive"
    if emotion in {"sad", "fear", "angry", "sick"}:
        return "negative"
    if emotion == "surprised":
        return "mixed"
    if any(token in text for token in ("neutral", "expressionless", "without mouth", "thinking")):
        return "neutral"
    return "mixed"


def extract_attributes(name: str) -> dict[str, int | str]:
    text = name.lower()
    emotion = infer_emotion(text)
    attributes: dict[str, int | str] = {
        "emotion_primary": emotion,
        "sentiment": infer_sentiment(text, emotion),
        "has_tears": int("tear" in text or "cry" in text),
        "has_sweat": int("sweat" in text),
        "has_hearts": int("heart" in text),
        "has_tongue": int("tongue" in text),
        "has_glasses": int(any(token in text for token in ("sunglasses", "nerd", "monocle"))),
        "has_mask": int("mask" in text),
        "has_halo": int("halo" in text),
        "has_horns": int("horns" in text),
        "has_hat": int("hat" in text),
        "has_temperature": int(any(token in text for token in ("thermometer", "hot", "cold"))),
    }
    return attributes


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def deterministic_template(options: list[str], *parts: str) -> str:
    seed = sum(sum(ord(char) for char in part) for part in parts)
    return options[seed % len(options)]


def semantic_instruction(target_name: str, source_name: str) -> str:
    options = [
        "Turn this emoji into {target}.",
        "Edit this emoji so it becomes {target}.",
        "Keep the emoji style, but change the expression to {target}.",
        "Modify this emoji to look like {target}.",
    ]
    template = deterministic_template(options, source_name, target_name)
    return template.format(target=target_name)


def style_instruction(target_vendor: str, emoji_name: str) -> str:
    options = [
        "Keep the same emoji, but render it in {vendor} style.",
        "Change only the visual style to {vendor} while preserving the same expression.",
        "Restyle this {emoji_name} as a {vendor} emoji.",
    ]
    template = deterministic_template(options, target_vendor, emoji_name)
    return template.format(vendor=target_vendor, emoji_name=emoji_name)


def attribute_delta(source: dict[str, object], target: dict[str, object]) -> str:
    deltas: list[str] = []
    if source["name"] != target["name"]:
        deltas.append(f"name:{source['name']}->{target['name']}")
    if source["emotion_primary"] != target["emotion_primary"]:
        deltas.append(f"emotion:{source['emotion_primary']}->{target['emotion_primary']}")
    if source["sentiment"] != target["sentiment"]:
        deltas.append(f"sentiment:{source['sentiment']}->{target['sentiment']}")
    for key in ATTRIBUTE_FIELDS[2:]:
        if source[key] != target[key]:
            deltas.append(f"{key}:{source[key]}->{target[key]}")
    return "|".join(deltas)


def convert_rgba_copy(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        converted = image.convert("RGBA")
        converted.save(target_path)


def assign_split(*parts: object) -> str:
    key = "||".join(str(part) for part in parts)
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def build_catalogs(project_root: Path, force: bool) -> dict[str, Path]:
    raw_root = project_root / "data" / "raw" / "kaggle"
    interim_root = project_root / "data" / "interim" / "emoji_editing"
    metadata_root = interim_root / "metadata"
    processed_root = project_root / "data" / "processed" / "emoji_editing"
    processed_vendor_root = processed_root / "images" / "vendor_rgba72"
    processed_unicode_root = processed_root / "images" / "unicode_256"

    ensure_clean_dir(metadata_root, force)
    ensure_clean_dir(processed_vendor_root, force)
    ensure_clean_dir(processed_unicode_root, force)
    remove_ds_store(raw_root)

    full_emoji_csv = raw_root / "full_emoji_image_dataset" / "full_emoji.csv"
    unicode_meaning_csv = raw_root / "unicode_emoji_meanings" / "unicode_emojis_full.csv.csv"
    emojinet_json = raw_root / "emojinet" / "emojis.json"
    unicode_image_csv = raw_root / "unicode_emoji_image_dataset" / "emoji_dataset.csv"
    unicode_image_root = raw_root / "unicode_emoji_image_dataset" / "emoji" / "emoji"
    vendor_image_root = raw_root / "full_emoji_image_dataset" / "image"

    unicode_meanings = load_unicode_descriptions(unicode_meaning_csv)
    emojinet_lookup = load_emojinet_lookup(emojinet_json)
    unicode_image_lookup = load_unicode_image_lookup(unicode_image_csv, unicode_image_root)

    catalog_rows: list[dict[str, object]] = []
    face_rows: list[dict[str, object]] = []
    vendor_image_rows: list[dict[str, object]] = []
    processed_vendor_path_by_key: dict[tuple[int, str], str] = {}
    processed_unicode_path_by_row_id: dict[int, str] = {}

    with full_emoji_csv.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = int(row["#"])
            emoji_char = row["emoji"].strip()
            unicode_slug = normalize_unicode(row["unicode"])
            unicode_key = (emoji_char, "")
            emoji_name = sanitize_name(row["name"])
            is_face = is_human_face_emoji(emoji_name)
            merged = {
                "row_id": row_id,
                "emoji": emoji_char,
                "unicode": row["unicode"].strip(),
                "unicode_slug": unicode_slug,
                "name": emoji_name,
                "unicode_description": "",
                "emojinet_name": "",
                "emojinet_category": "",
                "emojinet_shortcode": "",
                "emojinet_definition": "",
                "emojinet_keywords": "",
                "canonical_unicode_image_path": "",
            }
            if unicode_key in unicode_meanings:
                merged["unicode_description"] = unicode_meanings[unicode_key]["unicode_description"]
            elif (emoji_char, unicode_slug) in unicode_meanings:
                merged["unicode_description"] = unicode_meanings[(emoji_char, unicode_slug)]["unicode_description"]

            if unicode_slug in emojinet_lookup:
                merged.update(emojinet_lookup[unicode_slug])

            unicode_image = unicode_image_lookup.get((emoji_char, unicode_slug)) or unicode_image_lookup.get((emoji_char, "")) or unicode_image_lookup.get(("", unicode_slug))
            if is_face and unicode_image:
                source_path = Path(unicode_image["image_path"])
                target_name = f"{row_id:04d}_{slug_from_unicode(unicode_slug)}.png"
                target_path = processed_unicode_root / target_name
                if force or not target_path.exists():
                    convert_rgba_copy(source_path, target_path)
                merged["canonical_unicode_image_path"] = str(target_path)
                processed_unicode_path_by_row_id[row_id] = str(target_path)

            available_vendors: list[str] = []
            for vendor in VENDORS:
                source_path = vendor_image_root / vendor / f"{row_id}.png"
                has_image = int(source_path.exists())
                merged[f"has_{vendor.lower()}"] = has_image
                if not has_image:
                    continue
                available_vendors.append(vendor)
                if is_face:
                    target_name = f"{row_id:04d}_{slug_from_unicode(unicode_slug)}.png"
                    target_path = processed_vendor_root / vendor / target_name
                    if force or not target_path.exists():
                        convert_rgba_copy(source_path, target_path)
                    processed_vendor_path_by_key[(row_id, vendor)] = str(target_path)
                    vendor_image_rows.append(
                        {
                            "row_id": row_id,
                            "emoji": emoji_char,
                            "unicode_slug": unicode_slug,
                            "name": emoji_name,
                            "vendor": vendor,
                            "raw_image_path": str(source_path),
                            "processed_image_path": str(target_path),
                        }
                    )

            merged["available_vendor_count"] = len(available_vendors)
            merged["available_vendors"] = "|".join(available_vendors)
            catalog_rows.append(merged)

            if is_face:
                attributes = extract_attributes(emoji_name)
                face_row = {
                    **merged,
                    **attributes,
                }
                face_rows.append(face_row)

    face_row_by_id = {int(row["row_id"]): row for row in face_rows}

    style_pairs: list[dict[str, object]] = []
    style_pair_id = 1
    for face_row in face_rows:
        row_id = int(face_row["row_id"])
        available = [vendor for vendor in VENDORS if (row_id, vendor) in processed_vendor_path_by_key]
        for source_vendor in available:
            for target_vendor in available:
                if source_vendor == target_vendor:
                    continue
                style_pairs.append(
                    {
                        "pair_id": f"style_{style_pair_id:06d}",
                        "task_type": "style_transfer",
                        "split": assign_split("style", row_id, source_vendor, target_vendor),
                        "source_row_id": row_id,
                        "source_vendor": source_vendor,
                        "source_emoji": face_row["emoji"],
                        "source_name": face_row["name"],
                        "source_unicode_slug": face_row["unicode_slug"],
                        "source_image_path": processed_vendor_path_by_key[(row_id, source_vendor)],
                        "source_emotion": face_row["emotion_primary"],
                        "source_sentiment": face_row["sentiment"],
                        "target_row_id": row_id,
                        "target_vendor": target_vendor,
                        "target_emoji": face_row["emoji"],
                        "target_name": face_row["name"],
                        "target_unicode_slug": face_row["unicode_slug"],
                        "target_image_path": processed_vendor_path_by_key[(row_id, target_vendor)],
                        "target_emotion": face_row["emotion_primary"],
                        "target_sentiment": face_row["sentiment"],
                        "instruction": style_instruction(target_vendor, str(face_row["name"])),
                        "attribute_delta": f"vendor:{source_vendor}->{target_vendor}",
                    }
                )
                style_pair_id += 1

    semantic_pairs: list[dict[str, object]] = []
    semantic_pair_id = 1
    rows_by_vendor: dict[str, list[dict[str, object]]] = defaultdict(list)
    for face_row in face_rows:
        row_id = int(face_row["row_id"])
        for vendor in VENDORS:
            if (row_id, vendor) in processed_vendor_path_by_key:
                rows_by_vendor[vendor].append(face_row)

    for vendor, rows in rows_by_vendor.items():
        ordered_rows = sorted(rows, key=lambda item: int(item["row_id"]))
        for source_row in ordered_rows:
            source_row_id = int(source_row["row_id"])
            for target_row in ordered_rows:
                target_row_id = int(target_row["row_id"])
                if source_row_id == target_row_id:
                    continue
                semantic_pairs.append(
                    {
                        "pair_id": f"semantic_{semantic_pair_id:06d}",
                        "task_type": "semantic_edit",
                        "split": assign_split("semantic", vendor, source_row_id, target_row_id),
                        "source_row_id": source_row_id,
                        "source_vendor": vendor,
                        "source_emoji": source_row["emoji"],
                        "source_name": source_row["name"],
                        "source_unicode_slug": source_row["unicode_slug"],
                        "source_image_path": processed_vendor_path_by_key[(source_row_id, vendor)],
                        "source_emotion": source_row["emotion_primary"],
                        "source_sentiment": source_row["sentiment"],
                        "target_row_id": target_row_id,
                        "target_vendor": vendor,
                        "target_emoji": target_row["emoji"],
                        "target_name": target_row["name"],
                        "target_unicode_slug": target_row["unicode_slug"],
                        "target_image_path": processed_vendor_path_by_key[(target_row_id, vendor)],
                        "target_emotion": target_row["emotion_primary"],
                        "target_sentiment": target_row["sentiment"],
                        "instruction": semantic_instruction(str(target_row["name"]), str(source_row["name"])),
                        "attribute_delta": attribute_delta(source_row, target_row),
                    }
                )
                semantic_pair_id += 1

    all_pairs = style_pairs + semantic_pairs

    catalog_fieldnames = [
        "row_id",
        "emoji",
        "unicode",
        "unicode_slug",
        "name",
        "unicode_description",
        "emojinet_name",
        "emojinet_category",
        "emojinet_shortcode",
        "emojinet_definition",
        "emojinet_keywords",
        "canonical_unicode_image_path",
        "available_vendor_count",
        "available_vendors",
    ] + [f"has_{vendor.lower()}" for vendor in VENDORS]

    face_fieldnames = catalog_fieldnames + ATTRIBUTE_FIELDS

    pair_fieldnames = [
        "pair_id",
        "task_type",
        "split",
        "source_row_id",
        "source_vendor",
        "source_emoji",
        "source_name",
        "source_unicode_slug",
        "source_image_path",
        "source_emotion",
        "source_sentiment",
        "target_row_id",
        "target_vendor",
        "target_emoji",
        "target_name",
        "target_unicode_slug",
        "target_image_path",
        "target_emotion",
        "target_sentiment",
        "instruction",
        "attribute_delta",
    ]

    write_csv(metadata_root / "emoji_catalog.csv", catalog_rows, catalog_fieldnames)
    write_csv(metadata_root / "face_emoji_catalog.csv", face_rows, face_fieldnames)
    write_csv(
        metadata_root / "vendor_image_index.csv",
        vendor_image_rows,
        ["row_id", "emoji", "unicode_slug", "name", "vendor", "raw_image_path", "processed_image_path"],
    )
    write_csv(metadata_root / "style_transfer_pairs.csv", style_pairs, pair_fieldnames)
    write_csv(metadata_root / "semantic_edit_pairs.csv", semantic_pairs, pair_fieldnames)
    write_csv(metadata_root / "all_edit_pairs.csv", all_pairs, pair_fieldnames)

    emotion_counts = Counter(str(row["emotion_primary"]) for row in face_rows)
    split_counts = Counter(str(row["split"]) for row in all_pairs)
    task_split_counts: dict[str, dict[str, int]] = {}
    for task_type in ("style_transfer", "semantic_edit"):
        task_rows = [row for row in all_pairs if str(row["task_type"]) == task_type]
        task_split_counts[task_type] = dict(Counter(str(row["split"]) for row in task_rows))
    vendor_face_counts = {
        vendor: sum(1 for row in face_rows if (int(row["row_id"]), vendor) in processed_vendor_path_by_key)
        for vendor in VENDORS
    }
    stats = {
        "emoji_catalog_count": len(catalog_rows),
        "face_emoji_count": len(face_rows),
        "vendor_image_count": len(vendor_image_rows),
        "style_transfer_pair_count": len(style_pairs),
        "semantic_edit_pair_count": len(semantic_pairs),
        "all_edit_pair_count": len(all_pairs),
        "emotion_distribution": dict(sorted(emotion_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "task_split_distribution": task_split_counts,
        "vendor_face_counts": vendor_face_counts,
        "canonical_unicode_image_count": len(processed_unicode_path_by_row_id),
    }
    (metadata_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metadata_root": metadata_root,
        "processed_vendor_root": processed_vendor_root,
        "processed_unicode_root": processed_unicode_root,
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    outputs = build_catalogs(project_root, force=args.force)
    print(f"Metadata written to: {outputs['metadata_root']}")
    print(f"Processed vendor images: {outputs['processed_vendor_root']}")
    print(f"Processed canonical unicode images: {outputs['processed_unicode_root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
