#!/usr/bin/env python3
"""Extract leakage-safe Latin numeric field crops from generated maternal forms.

The crop is saved at its detected/annotated line aspect ratio. TrOCRProcessor performs the same
direct 384x384 resize used by Android and by the reference implementation during training.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path("/home/anon/Documents/Projects/Poly-AI-Hackathon-The-Online-Midwife/synthetic_training_data")
OUTPUT = ROOT / "dataset/prepared/numbers/trocr"
DATE_TYPES = {"past_date", "future_date", "recent_date"}
NUMERIC_PATTERN = re.compile(r"[+-]?\d+(?:[.,/:-]\d+)*(?:%)?")


def numeric_target(text: str) -> str:
    match = NUMERIC_PATTERN.search(text.replace(" ", ""))
    return match.group(0) if match else ""


def split_for(target: str) -> str:
    """Keep every rendering of the same target value in one split."""
    bucket = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.rglob("metadata.csv")):
        raise SystemExit(f"Refusing to overwrite prepared dataset: {OUTPUT}")
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    formats: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for annotation_path in sorted(SOURCE.glob("*/annotations.jsonl")):
        form = annotation_path.parent.name
        image_root = annotation_path.parent / "images"
        for line in annotation_path.read_text().splitlines():
            record = json.loads(line)
            if record.get("language") not in {"en", "fr"}:
                continue
            image_path = image_root / record["image_filename"]
            with Image.open(image_path) as page:
                width, height = page.size
                for field_id, field in record.get("fields_annotation", {}).items():
                    field_type = str(field.get("type", ""))
                    if field_type in DATE_TYPES:
                        continue
                    rendered = str(field.get("rendered_text", ""))
                    target = numeric_target(rendered)
                    if not target or not target[0].isdigit() and target[0] not in "+-":
                        continue
                    box = field.get("bbox")
                    if not box or len(box) != 4:
                        continue
                    left, top, right, bottom = box
                    pixel_box = (
                        max(0, round(left * width)),
                        max(0, round(top * height)),
                        min(width, round(right * width)),
                        min(height, round(bottom * height)),
                    )
                    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
                        continue
                    split = split_for(target)
                    name = safe_name(f"{form}__{Path(record['image_filename']).stem}__{field_id}.png")
                    relative = Path("images") / name
                    destination = OUTPUT / split / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    page.crop(pixel_box).convert("RGB").save(destination)
                    rows[split].append(
                        {
                            "file_name": str(relative),
                            "text": target,
                            "language": record["language"],
                            "format": field_type,
                            "group_key": target,
                            "source_image": str(image_path),
                        }
                    )
                    formats[field_type] += 1
                    languages[record["language"]] += 1

    fieldnames = ["file_name", "text", "language", "format", "group_key", "source_image"]
    for split, split_rows in rows.items():
        with (OUTPUT / split / "metadata.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
    groups = {split: {row["group_key"] for row in split_rows} for split, split_rows in rows.items()}
    overlap = {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Numeric target leakage: {overlap}")
    report = {
        "source": str(SOURCE),
        "output": str(OUTPUT),
        "preprocessing": "Original line crop; TrOCRProcessor direct resize to 384x384",
        "scripts": ["en", "fr"],
        "date_fields_excluded": True,
        "splits": {split: len(split_rows) for split, split_rows in rows.items()},
        "unique_targets": {split: len(values) for split, values in groups.items()},
        "group_overlap": overlap,
        "formats": dict(formats),
        "languages": dict(languages),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
