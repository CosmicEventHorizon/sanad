#!/usr/bin/env python3
"""Prepare shared-handwriting and numeric PP-OCRv6 recognition datasets.

All English/French annotated handwriting crops feed the shared model. Numeric/date crops and
the production-shaped fiche-number set additionally feed the numeric specialization.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FORM_SOURCE = Path("/home/anon/Documents/Projects/Poly-AI-Hackathon-The-Online-Midwife/synthetic_training_data")
FICHE_SOURCE = ROOT / "dataset/prepared/fiche_numbers/trocr"
OUTPUT = ROOT / "dataset/prepared/ppocrv6_recognition"
DICTIONARY = ROOT / ".tools/PaddleOCR/ppocr/utils/dict/ppocrv6_dict.txt"
MAX_TEXT_LENGTH = 40
NUMERIC_TYPES = {
    "abortions", "adult_weight", "age", "baby_length", "birth_gestation", "birth_weight",
    "bp", "early_age", "fhr", "fundal", "future_date", "gestation", "glucose", "head",
    "height", "hemoglobin", "late_age", "living", "newborn_weight", "parity", "past_date",
    "pulse", "recent_date", "small_int", "temperature",
}
NUMERIC_PATTERN = re.compile(r"[+-]?\d+(?:[.,/:-]\d+)*(?:%)?")


def normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFC", value)
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return " ".join(text.replace("\t", " ").replace("\n", " ").split())


def numeric_text(value: str) -> str:
    match = NUMERIC_PATTERN.search(value.replace(" ", ""))
    return match.group(0) if match else ""


def split_for(group: str) -> str:
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite prepared dataset: {OUTPUT}")
    if not DICTIONARY.exists():
        raise FileNotFoundError(DICTIONARY)

    records: dict[str, dict[str, list[dict[str, str]]]] = {
        task: {split: [] for split in ("train", "val", "test")}
        for task in ("shared", "numeric")
    }
    groups: dict[str, dict[str, set[str]]] = {
        task: {split: set() for split in ("train", "val", "test")}
        for task in records
    }
    source_counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    def add(task: str, split: str, relative: Path, label: str, group: str, source_kind: str) -> None:
        label = normalized_text(label)
        if not label:
            skipped[f"{task}:empty"] += 1
            return
        if len(label) > MAX_TEXT_LENGTH:
            skipped[f"{task}:too_long"] += 1
            return
        records[task][split].append(
            {
                "file_name": relative.as_posix(),
                "text": label,
                "group_key": group,
                "source_kind": source_kind,
            }
        )
        groups[task][split].add(group)

    for annotations in sorted(FORM_SOURCE.glob("*/annotations.jsonl")):
        form_id = annotations.parent.name
        image_root = annotations.parent / "images"
        for line in annotations.read_text().splitlines():
            page = json.loads(line)
            if page.get("language") not in {"en", "fr"}:
                continue
            source_image = image_root / page["image_filename"]
            group = f"form:{form_id}:{page['image_filename']}"
            split = split_for(group)
            with Image.open(source_image) as image:
                width, height = image.size
                for field_id, field in page.get("fields_annotation", {}).items():
                    rendered = normalized_text(str(field.get("rendered_text", "")))
                    if not rendered or not any(character.isalnum() for character in rendered):
                        skipped["shared:non_text"] += 1
                        continue
                    box = field.get("bbox")
                    if not box or len(box) != 4:
                        skipped["shared:no_bbox"] += 1
                        continue
                    left, top, right, bottom = box
                    pixel_box = (
                        max(0, round(left * width)),
                        max(0, round(top * height)),
                        min(width, round(right * width)),
                        min(height, round(bottom * height)),
                    )
                    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                        skipped["shared:bad_bbox"] += 1
                        continue
                    name = safe_name(f"{form_id}__{Path(page['image_filename']).stem}__{field_id}.png")
                    relative = Path("images") / split / "forms" / name
                    destination = OUTPUT / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    image.crop(pixel_box).convert("RGB").save(destination)
                    add("shared", split, relative, rendered, group, "annotated_form_crop")
                    source_counts["annotated_form_crop"] += 1
                    field_type = str(field.get("type", ""))
                    number = numeric_text(rendered) if field_type in NUMERIC_TYPES else ""
                    if number:
                        add("numeric", split, relative, number, group, "annotated_numeric_crop")
                        source_counts["annotated_numeric_crop"] += 1

    for split in ("train", "val", "test"):
        metadata_path = FICHE_SOURCE / split / "metadata.csv"
        with metadata_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                source = FICHE_SOURCE / split / row["file_name"]
                name = safe_name(f"fiche__{split}__{Path(row['file_name']).name}")
                relative = Path("images") / split / "fiche" / name
                destination = OUTPUT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                group = f"fiche:{row['group_key']}"
                kind = row["source_kind"]
                add("shared", split, relative, row["text"], group, kind)
                add("numeric", split, relative, row["text"], group, kind)
                source_counts[kind] += 1

    dictionary = {line.rstrip("\n") for line in DICTIONARY.read_text(encoding="utf-8").splitlines()}
    dictionary.add(" ")
    unsupported = sorted(
        {
            character
            for task_records in records.values()
            for split_records in task_records.values()
            for row in split_records
            for character in row["text"]
            if character not in dictionary
        }
    )
    if unsupported:
        raise RuntimeError(f"Labels contain characters missing from PP-OCRv6 dictionary: {unsupported}")

    overlaps: dict[str, dict[str, int]] = {}
    for task, task_records in records.items():
        overlaps[task] = {
            "train_val": len(groups[task]["train"] & groups[task]["val"]),
            "train_test": len(groups[task]["train"] & groups[task]["test"]),
            "val_test": len(groups[task]["val"] & groups[task]["test"]),
        }
        if any(overlaps[task].values()):
            raise RuntimeError(f"{task} group leakage: {overlaps[task]}")
        for split, split_records in task_records.items():
            label_path = OUTPUT / f"{task}_{split}.txt"
            label_path.write_text(
                "".join(f"{row['file_name']}\t{row['text']}\n" for row in split_records),
                encoding="utf-8",
            )
            metadata_path = OUTPUT / f"{task}_{split}.csv"
            with metadata_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file_name", "text", "group_key", "source_kind"])
                writer.writeheader()
                writer.writerows(split_records)

    report = {
        "source": str(FORM_SOURCE),
        "fiche_source": str(FICHE_SOURCE),
        "output": str(OUTPUT),
        "languages": ["en", "fr"],
        "arabic_excluded": True,
        "max_text_length": MAX_TEXT_LENGTH,
        "splits": {
            task: {split: len(rows) for split, rows in task_records.items()}
            for task, task_records in records.items()
        },
        "group_overlap": overlaps,
        "source_counts": dict(source_counts),
        "skipped": dict(skipped),
        "dictionary": str(DICTIONARY),
        "unsupported_characters": unsupported,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
