#!/usr/bin/env python3
"""Prepare leakage-safe Latin training data and preserve Arabic samples separately."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FORM_SOURCE = ROOT / "dataset/01_pregnancy_postpartum_monitoring"
NUMBER_SOURCE = ROOT / "dataset/numbers"
OUTPUT_ROOT = ROOT / "dataset/prepared"
FORM_OUTPUT = OUTPUT_ROOT / "forms"
NUMBER_OUTPUT = OUTPUT_ROOT / "numbers"
SPLITS = ("train", "val", "test")
SEED = 20261114

# Printed-label boxes on the canonical, undistorted form. Coordinates are
# normalized [left, top, right, bottom]. The dataset's recorded homography is
# applied to these rectangles to create a true quadrilateral per image.
PRINTED_ANCHORS = [
    ("form_number", (0.084, 0.119, 0.176, 0.133)),
    ("region", (0.084, 0.160, 0.140, 0.174)),
    ("province", (0.537, 0.160, 0.611, 0.174)),
    ("health_facility_name", (0.083, 0.202, 0.296, 0.218)),
    ("health_facility_type", (0.083, 0.245, 0.300, 0.261)),
    ("facility_mobility", (0.166, 0.379, 0.447, 0.396)),
    ("patient_full_name", (0.083, 0.445, 0.278, 0.463)),
    ("high_risk_pregnancy", (0.122, 0.521, 0.340, 0.539)),
    ("risk_factors", (0.098, 0.547, 0.381, 0.566)),
    ("other_risk_factors", (0.483, 0.663, 0.601, 0.681)),
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def grouped_split(
    rows: list[dict[str, str]],
    group_key: str,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)

    keys = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest(),
    )
    targets = [round(len(rows) * ratios[0]), round(len(rows) * ratios[1])]
    result = {name: [] for name in SPLITS}
    split_index = 0
    for key in keys:
        if split_index == 0 and len(result["train"]) >= targets[0]:
            split_index = 1
        if split_index == 1 and len(result["val"]) >= targets[1]:
            split_index = 2
        result[SPLITS[split_index]].extend(groups[key])
    return result


def transform_point(
    x: float,
    y: float,
    homography: list[list[float]],
) -> tuple[float, float]:
    denominator = homography[2][0] * x + homography[2][1] * y + homography[2][2]
    return (
        (homography[0][0] * x + homography[0][1] * y + homography[0][2]) / denominator,
        (homography[1][0] * x + homography[1][1] * y + homography[1][2]) / denominator,
    )


def obb_label(
    image_size: tuple[int, int],
    homography: list[list[float]],
) -> list[str]:
    width, height = image_size
    labels = []
    for class_id, (_, (left, top, right, bottom)) in enumerate(PRINTED_ANCHORS):
        canonical = [
            (left * width, top * height),
            (right * width, top * height),
            (right * width, bottom * height),
            (left * width, bottom * height),
        ]
        transformed = [transform_point(x, y, homography) for x, y in canonical]
        normalized = [
            coordinate
            for x, y in transformed
            for coordinate in (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))
        ]
        labels.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in normalized))
    return labels


def prepare_forms() -> dict:
    rows = load_csv(FORM_SOURCE / "labels.csv")
    annotations = {
        item["image_filename"]: item
        for item in map(json.loads, (FORM_SOURCE / "annotations.jsonl").read_text(encoding="utf-8").splitlines())
    }
    latin = [dict(row, source_group=row["random_seed"]) for row in rows if row["language"] != "ar"]
    arabic = [row for row in rows if row["language"] == "ar"]
    split_rows = grouped_split(latin, "source_group")

    arabic_dir = FORM_OUTPUT / "arabic"
    for row in arabic:
        link_or_copy(FORM_SOURCE / "images" / row["image_filename"], arabic_dir / "images" / row["image_filename"])
    write_csv(arabic_dir / "labels.csv", arabic, list(rows[0]))
    with (arabic_dir / "annotations.jsonl").open("w", encoding="utf-8") as handle:
        for row in arabic:
            handle.write(json.dumps(annotations[row["image_filename"]], ensure_ascii=False) + "\n")

    yolo_root = FORM_OUTPUT / "yolo_obb"
    manifest_fields = list(latin[0]) + ["prepared_split"]
    canonical_size: tuple[int, int] | None = None
    for split, items in split_rows.items():
        manifest = []
        for row in items:
            filename = row["image_filename"]
            source = FORM_SOURCE / "images" / filename
            with Image.open(source) as image:
                size = image.size
            canonical_size = canonical_size or size
            if size != canonical_size:
                raise ValueError(f"Unexpected image size for {filename}: {size}")
            link_or_copy(source, yolo_root / "images" / split / filename)
            homography = annotations[filename]["distortions"]["homography"]
            label_path = yolo_root / "labels" / split / f"{Path(filename).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(obb_label(size, homography)) + "\n", encoding="utf-8")
            manifest.append(dict(row, prepared_split=split))
        write_csv(FORM_OUTPUT / "manifests" / f"{split}.csv", manifest, manifest_fields)

    yaml_lines = [
        f"path: {yolo_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {index}: {name}" for index, (name, _) in enumerate(PRINTED_ANCHORS)]
    (yolo_root / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "source_samples": len(rows),
        "latin_samples": len(latin),
        "arabic_samples": len(arabic),
        "splits": {name: len(items) for name, items in split_rows.items()},
        "languages": dict(Counter(row["language"] for row in latin)),
        "profiles": dict(Counter(row["profile"] for row in latin)),
        "classes": [name for name, _ in PRINTED_ANCHORS],
        "image_size": canonical_size,
    }


def prepare_numbers() -> dict:
    rows = load_csv(NUMBER_SOURCE / "labels.csv")
    latin = [row for row in rows if row["language"] != "ar"]
    arabic = [row for row in rows if row["language"] == "ar"]
    split_rows = grouped_split(latin, "group_key")

    arabic_dir = NUMBER_OUTPUT / "arabic"
    for row in arabic:
        link_or_copy(NUMBER_SOURCE / row["image"], arabic_dir / "images" / row["image"])
    write_csv(arabic_dir / "labels.csv", arabic, list(rows[0]))
    with (arabic_dir / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for row in arabic:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    trocr_root = NUMBER_OUTPUT / "trocr"
    metadata_fields = ["file_name", "text", "language", "format", "group_key", "source_image"]
    for split, items in split_rows.items():
        metadata = []
        for row in items:
            source_relative = Path(row["image"])
            unique_name = f"{source_relative.parent.name}_{source_relative.name}"
            destination = trocr_root / split / "images" / unique_name
            link_or_copy(NUMBER_SOURCE / source_relative, destination)
            metadata.append(
                {
                    "file_name": f"images/{unique_name}",
                    "text": row["text"],
                    "language": row["language"],
                    "format": row["format"],
                    "group_key": row["group_key"],
                    "source_image": row["image"],
                }
            )
        write_csv(trocr_root / split / "metadata.csv", metadata, metadata_fields)

    overlaps = {}
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        first_groups = {row["group_key"] for row in split_rows[first]}
        second_groups = {row["group_key"] for row in split_rows[second]}
        overlaps[f"{first}_{second}"] = len(first_groups & second_groups)
    if any(overlaps.values()):
        raise RuntimeError(f"Group leakage detected: {overlaps}")

    return {
        "source_samples": len(rows),
        "latin_samples": len(latin),
        "arabic_samples": len(arabic),
        "latin_groups": len({row["group_key"] for row in latin}),
        "splits": {name: len(items) for name, items in split_rows.items()},
        "languages": dict(Counter(row["language"] for row in latin)),
        "formats": dict(Counter(row["format"] for row in latin)),
        "group_overlap": overlaps,
    }


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    report = {
        "seed": SEED,
        "policy": "80/10/10 deterministic grouped split; Arabic preserved separately and excluded from training",
        "forms": prepare_forms(),
        "numbers": prepare_numbers(),
    }
    report_path = OUTPUT_ROOT / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
