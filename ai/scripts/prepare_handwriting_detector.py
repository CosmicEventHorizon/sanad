#!/usr/bin/env python3
"""Prepare a shared, one-class YOLO OBB handwriting-line dataset.

Only Latin-script synthetic pages are included. The source annotations already contain the
post-distortion bounding boxes, so this script never needs access to organizer/test labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/home/anon/Documents/Projects/Poly-AI-Hackathon-The-Online-Midwife/synthetic_training_data")
DEFAULT_OUTPUT = ROOT / "dataset/prepared/handwriting_yolo_obb"


def split_for(filename: str) -> str:
    """Keep every generated page intact while producing a deterministic 80/10/10 split."""
    index = int(Path(filename).stem.rsplit("_", 1)[-1])
    bucket = index % 10
    return "test" if bucket == 0 else "val" if bucket == 1 else "train"


def obb_line(box: list[float]) -> str:
    x1, y1, x2, y2 = box
    coordinates = (x1, y1, x2, y1, x2, y2, x1, y2)
    return "0 " + " ".join(f"{value:.8f}" for value in coordinates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    boxes = 0
    for annotation_path in sorted(args.source.glob("*/annotations.jsonl")):
        image_root = annotation_path.parent / "images"
        for raw in annotation_path.read_text().splitlines():
            record = json.loads(raw)
            if record.get("language") not in {"en", "fr"}:
                continue
            filename = record["image_filename"]
            source_image = image_root / filename
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            labels = []
            for field in record.get("fields_annotation", {}).values():
                if not str(field.get("rendered_text", "")).strip():
                    continue
                box = field.get("bbox")
                if not box or len(box) != 4 or not all(0 <= value <= 1 for value in box):
                    continue
                labels.append(obb_line(box))
            if not labels:
                continue
            split = split_for(filename)
            unique_name = f"{annotation_path.parent.name}__{filename}"
            image_path = args.output / "images" / split / unique_name
            label_path = args.output / "labels" / split / f"{Path(unique_name).stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if not image_path.exists():
                image_path.symlink_to(source_image.resolve())
            label_path.write_text("\n".join(labels) + "\n")
            counts[split] += 1
            boxes += len(labels)

    dataset = {
        "path": str(args.output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "handwriting_line"},
    }
    (args.output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    summary = {
        "source": str(args.source),
        "output": str(args.output),
        "scripts": ["en", "fr"],
        "pages": dict(counts),
        "handwriting_boxes": boxes,
        "split_policy": "numeric source page index modulo 10; 80/10/10",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
