#!/usr/bin/env python3
"""Evaluate the packaged numeric post-processing contract on the held-out test set."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import evaluate
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "ai/config/trocr_numbers.json").read_text())
MODEL = ROOT / CONFIG.get("output_directory", "ai/runs/trocr_numbers") / "best"
DATA = ROOT / "dataset/prepared/numbers/trocr/test"
REPORT = ROOT / "ai/reports/trocr_numbers_metrics.json"


def normalize(text: str) -> str:
    match = re.search(r"[+-]?\d+(?:[.,/:-]\d+)*(?:%)?", text.replace(" ", ""))
    return match.group(0) if match else ""


def main() -> None:
    formats = set(CONFIG["target_formats"])
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["format"] in formats and any(c.isdigit() for c in normalize(row["text"]))]

    processor = TrOCRProcessor.from_pretrained(str(MODEL))
    model = VisionEncoderDecoderModel.from_pretrained(str(MODEL)).to("cuda").eval()
    predictions: list[str] = []
    references: list[str] = []
    with torch.inference_mode():
        for offset in range(0, len(rows), 8):
            batch = rows[offset : offset + 8]
            images = [Image.open(DATA / row["file_name"]).convert("RGB") for row in batch]
            pixels = processor(images=images, return_tensors="pt").pixel_values.to("cuda")
            generated = model.generate(pixels, max_length=CONFIG["generation_max_length"], num_beams=2)
            predictions.extend(normalize(text) for text in processor.batch_decode(generated, skip_special_tokens=True))
            references.extend(normalize(row["text"]) for row in batch)

    exact = sum(prediction == reference for prediction, reference in zip(predictions, references)) / len(references)
    metrics = {
        "samples": len(references),
        "cer": float(evaluate.load("cer").compute(predictions=predictions, references=references)),
        "exact_match": exact,
        "post_processing": "Keep digits and configured numeric separators; remove visible unit text.",
        "examples": [
            {"reference": reference, "prediction": prediction}
            for reference, prediction in list(zip(references, predictions))[:20]
        ],
    }
    report = json.loads(REPORT.read_text())
    report["test_pipeline"] = metrics
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
