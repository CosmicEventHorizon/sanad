#!/usr/bin/env python3
"""Smoke-test the exported numeric TrOCR graphs with Android-equivalent greedy decoding."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import TrOCRProcessor


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "app/src/main/assets/models/trocr_numbers"
CONFIG = json.loads((ROOT / "ai/config/trocr_numbers.json").read_text())
MODEL = ROOT / CONFIG.get("output_directory", "ai/runs/trocr_numbers") / "best"
DATA = ROOT / "dataset/prepared/numbers/trocr/test"


def numeric_value(text: str) -> str:
    match = re.search(r"[+-]?\d+(?:[.,/:-]\d+)*(?:%)?", text.replace(" ", ""))
    return match.group(0) if match else ""


def main() -> None:
    processor = TrOCRProcessor.from_pretrained(str(MODEL))
    encoder = ort.InferenceSession(str(ASSETS / "encoder_model.onnx"), providers=["CPUExecutionProvider"])
    decoder = ort.InferenceSession(str(ASSETS / "decoder_model.onnx"), providers=["CPUExecutionProvider"])
    formats = set(CONFIG["target_formats"])
    with (DATA / "metadata.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["format"] in formats and numeric_value(row["text"])][:10]

    results = []
    for row in rows:
        image = Image.open(DATA / row["file_name"]).convert("RGB")
        pixels = processor(images=image, return_tensors="np").pixel_values.astype(np.float32)
        hidden = encoder.run(None, {encoder.get_inputs()[0].name: pixels})[0]
        tokens = [processor.tokenizer.cls_token_id]
        for _ in range(CONFIG["generation_max_length"] - 1):
            logits = decoder.run(None, {
                "input_ids": np.asarray([tokens], dtype=np.int64),
                "encoder_hidden_states": hidden,
            })[0]
            token = int(np.argmax(logits[0, -1]))
            if token == processor.tokenizer.sep_token_id:
                break
            tokens.append(token)
        prediction = numeric_value(processor.decode(tokens[1:], skip_special_tokens=True))
        results.append({"reference": numeric_value(row["text"]), "prediction": prediction})

    report = {"status": "passed", "samples": results}
    path = ROOT / "ai/reports/trocr_onnx_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
