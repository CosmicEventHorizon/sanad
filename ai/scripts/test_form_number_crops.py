#!/usr/bin/env python3
"""Run the fine-tuned numeric TrOCR checkpoint on diagnostic form-number crops."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CROPS = ROOT / "ai/runs/pipeline_diagnostic_two_detector"
DEFAULT_MODEL = ROOT / "ai/runs/trocr_numbers_square_crops/best"
DEFAULT_REPORT = DEFAULT_CROPS / "form_number_trocr_results.json"
DEFAULT_SHEET = DEFAULT_CROPS / "form_number_trocr_contact_sheet.png"


def sequence_confidences(output, eos_token_id: int) -> list[float]:
    """Return geometric-mean greedy token probability for each generated sequence."""
    generated = output.sequences[:, 1:]
    token_probabilities = [
        torch.softmax(logits.float(), dim=-1).gather(1, generated[:, step : step + 1]).squeeze(1)
        for step, logits in enumerate(output.scores)
    ]
    confidences: list[float] = []
    for row in range(generated.shape[0]):
        probabilities: list[float] = []
        for step, probability_batch in enumerate(token_probabilities):
            token = int(generated[row, step])
            if token == eos_token_id:
                break
            probabilities.append(max(float(probability_batch[row]), 1e-12))
        confidences.append(math.exp(sum(math.log(value) for value in probabilities) / len(probabilities)) if probabilities else 0.0)
    return confidences


def save_contact_sheet(results: list[dict], destination: Path) -> None:
    width = 900
    row_height = 118
    sheet = Image.new("RGB", (width, row_height * len(results)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, result in enumerate(results):
        top = index * row_height
        image = Image.open(result["crop_path"]).convert("RGB")
        scale = min(860 / image.width, 72 / image.height)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))))
        sheet.paste(image, (20, top + 34))
        draw.text(
            (20, top + 8),
            f"{result['form']}  ->  {result['prediction']!r}  confidence={result['confidence']:.3f}",
            fill="black",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-root", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--contact-sheet", type=Path, default=DEFAULT_SHEET)
    args = parser.parse_args()

    crop_paths = sorted(args.crops_root.glob("*/crops/00_form_number.png"))
    if not crop_paths:
        raise FileNotFoundError(f"No form-number crops found under {args.crops_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(args.model, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device).eval()
    images = [Image.open(path).convert("RGB") for path in crop_paths]
    pixels = processor(images=images, return_tensors="pt").pixel_values.to(device)
    with torch.inference_mode():
        output = model.generate(
            pixels,
            max_length=32,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )
    predictions = processor.batch_decode(output.sequences, skip_special_tokens=True)
    confidences = sequence_confidences(output, processor.tokenizer.sep_token_id)
    results = [
        {
            "form": path.parents[1].name,
            "crop_path": str(path),
            "width": image.width,
            "height": image.height,
            "prediction": prediction.strip(),
            "confidence": confidence,
        }
        for path, image, prediction, confidence in zip(crop_paths, images, predictions, confidences)
    ]
    report = {
        "model": str(args.model),
        "preprocessing": "TrOCRProcessor direct resize to 384x384",
        "decoding": "greedy",
        "results": results,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    save_contact_sheet(results, args.contact_sheet)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
