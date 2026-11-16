#!/usr/bin/env python3
"""Build a leakage-safe fiche-number OCR dataset from production pipeline crops.

The ten real crops remain held-out. Synthetic examples reuse their pipeline-produced crop
backgrounds and geometry, render the exact YYYY-NNN-NNN contract, and apply camera noise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = ROOT / "ai/runs/pipeline_diagnostic_two_detector"
OUTPUT = ROOT / "dataset/prepared/fiche_numbers/trocr"
GROUND_TRUTH = ROOT / "ai/config/fiche_number_ground_truth.json"
FONT_ROOT = Path("/home/anon/Documents/Projects/Poly-AI-Hackathon-The-Online-Midwife/synthetic_generation_assets/fonts")
FONTS = [FONT_ROOT / "Caveat.ttf", FONT_ROOT / "Kalam.ttf", FONT_ROOT / "PatrickHand.ttf"]
FIELDNAMES = ["file_name", "text", "format", "group_key", "source_kind", "source_crop"]


def split_for(target: str) -> str:
    bucket = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def blank_background(source: Image.Image) -> Image.Image:
    """Remove handwriting while preserving the crop's paper texture and bottom rule."""
    array = cv2.cvtColor(np.asarray(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    height, _ = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    top_region = gray[: max(1, round(height * 0.83))]
    threshold = min(175, int(np.median(top_region) * 0.72))
    mask[: top_region.shape[0]] = (top_region < threshold).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    cleaned = cv2.inpaint(array, mask, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))


def render_code(background: Image.Image, target: str, font_path: Path, rng: random.Random) -> Image.Image:
    width, height = background.size
    scale = 4
    font = ImageFont.truetype(str(font_path), rng.randint(31, 39) * scale)
    scratch = Image.new("L", (width * scale, height * scale * 3), 0)
    draw = ImageDraw.Draw(scratch)
    draw.text((8 * scale, -8 * scale), target, font=font, fill=255, stroke_width=0)
    box = scratch.getbbox()
    if box is None:
        raise RuntimeError("Rendered fiche number is empty")
    glyph = scratch.crop(box)
    desired_height = rng.uniform(0.66, 0.88) * height * scale
    desired_width = rng.uniform(0.43, 0.62) * width * scale
    resize_scale = min(desired_width / glyph.width, desired_height / glyph.height)
    glyph = glyph.resize(
        (max(1, round(glyph.width * resize_scale)), max(1, round(glyph.height * resize_scale))),
        Image.Resampling.LANCZOS,
    )
    angle = rng.uniform(-1.4, 1.4)
    glyph = glyph.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    glyph = glyph.resize((max(1, round(glyph.width / scale)), max(1, round(glyph.height / scale))), Image.Resampling.LANCZOS)
    canvas = background.copy()
    x = rng.randint(5, max(5, min(25, width - glyph.width - 1)))
    y = rng.randint(0, max(0, min(4, height - glyph.height - 2)))
    ink = rng.choice([(12, 42, 112), (10, 55, 145), (18, 28, 52), (32, 36, 42), (22, 65, 130)])
    color = Image.new("RGB", canvas.size, ink)
    mask = Image.new("L", canvas.size, 0)
    mask.paste(glyph, (x, y))
    canvas.paste(color, mask=mask)
    return canvas


def augment(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    if rng.random() < 0.9:
        points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        jitter = np.float32([[rng.uniform(-2, 2), rng.uniform(-1, 1)] for _ in range(4)])
        matrix = cv2.getPerspectiveTransform(points, points + jitter)
        array = cv2.warpPerspective(
            cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.82, 1.16))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.82, 1.24))
    if rng.random() < 0.65:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.05, 0.75)))
    array = np.asarray(image).astype(np.float32)
    sigma = rng.uniform(0.0, 4.0)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def unique_targets(count: int, excluded: set[str], rng: random.Random) -> list[str]:
    targets: set[str] = set()
    while len(targets) < count:
        year = 2026 if rng.random() < 0.72 else rng.randint(2023, 2030)
        target = f"{year:04d}-{rng.randint(0, 999):03d}-{rng.randint(0, 999):03d}"
        if target not in excluded:
            targets.add(target)
    return sorted(targets)


def write_metadata(split: str, rows: list[dict[str, str]]) -> None:
    path = OUTPUT / split / "metadata.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identities", type=int, default=3000, help="Unique synthetic fiche numbers")
    parser.add_argument("--variations", type=int, default=2, help="Camera variations per number")
    parser.add_argument("--seed", type=int, default=20261115)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"Refusing to overwrite prepared dataset: {OUTPUT}")
    missing_fonts = [str(path) for path in FONTS if not path.exists()]
    if missing_fonts:
        raise FileNotFoundError(f"Missing handwriting fonts: {missing_fonts}")

    truth: dict[str, str] = json.loads(GROUND_TRUTH.read_text())
    real_crops: list[tuple[Path, str]] = []
    for form, target in truth.items():
        crop = PIPELINE_ROOT / form / "crops/00_form_number.png"
        if not crop.exists():
            raise FileNotFoundError(crop)
        real_crops.append((crop, target))
    backgrounds = [(path, blank_background(Image.open(path).convert("RGB"))) for path, _ in real_crops]

    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(args.seed)
    targets = unique_targets(args.identities, set(truth.values()), rng)
    for target_index, target in enumerate(targets):
        split = split_for(target)
        for variation in range(args.variations):
            source_path, background = backgrounds[rng.randrange(len(backgrounds))]
            font_path = FONTS[rng.randrange(len(FONTS))]
            sample_rng = random.Random(f"{args.seed}:{target}:{variation}")
            image = augment(render_code(background, target, font_path, sample_rng), sample_rng)
            name = f"synthetic_{target_index:05d}_{variation}_{target}.png"
            relative = Path("images") / name
            destination = OUTPUT / split / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
            rows[split].append(
                {
                    "file_name": str(relative),
                    "text": target,
                    "format": "fiche_number",
                    "group_key": target,
                    "source_kind": "synthetic_pipeline_crop",
                    "source_crop": str(source_path),
                }
            )

    for index, (source, target) in enumerate(real_crops):
        relative = Path("images") / f"real_{index:02d}_{source.parents[1].name}.png"
        destination = OUTPUT / "test" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows["test"].append(
            {
                "file_name": str(relative),
                "text": target,
                "format": "fiche_number",
                "group_key": f"real:{source.parents[1].name}",
                "source_kind": "real_pipeline_crop",
                "source_crop": str(source),
            }
        )

    for split, split_rows in rows.items():
        write_metadata(split, split_rows)
    groups = {split: {row["group_key"] for row in split_rows if row["source_kind"] != "real_pipeline_crop"} for split, split_rows in rows.items()}
    overlaps = {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Fiche-number leakage detected: {overlaps}")
    report = {
        "format": "YYYY-NNN-NNN",
        "pipeline_crop_root": str(PIPELINE_ROOT),
        "real_examples": len(real_crops),
        "real_examples_usage": "held-out test only",
        "synthetic_identities": len(targets),
        "variations_per_identity": args.variations,
        "splits": {split: len(split_rows) for split, split_rows in rows.items()},
        "source_kinds": dict(Counter(row["source_kind"] for split_rows in rows.values() for row in split_rows)),
        "group_overlap": overlaps,
        "preprocessing": "Production pipeline crop geometry; TrOCRProcessor direct resize to 384x384",
        "fonts": [str(path) for path in FONTS],
        "seed": args.seed,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
