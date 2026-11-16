#!/usr/bin/env python3
"""Export pretrained text and fine-tuned numeric TrOCR graphs for mobile ORT."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from optimum.exporters.onnx import main_export
from transformers import TrOCRProcessor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "ai/config/trocr_numbers.json").read_text())
TEXT_MODEL = "microsoft/trocr-small-handwritten"
NUMBER_MODEL = ROOT / CONFIG.get("output_directory", "ai/runs/trocr_numbers") / "best"
EXPORT_ROOT = ROOT / "ai/runs/onnx"
ASSET_ROOT = ROOT / "app/src/main/assets/models"


def export_model(source: str | Path, name: str, include_encoder: bool = True) -> dict:
    export_path = EXPORT_ROOT / name
    asset_path = ASSET_ROOT / name
    if export_path.exists():
        shutil.rmtree(export_path)
    main_export(
        model_name_or_path=str(source),
        output=export_path,
        task="image-to-text",
        opset=18,
        device="cpu",
        do_validation=True,
    )
    processor = TrOCRProcessor.from_pretrained(str(source))
    (export_path / "vocab.json").write_text(
        json.dumps(processor.tokenizer.get_vocab(), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if asset_path.exists():
        shutil.rmtree(asset_path)
    shutil.copytree(export_path, asset_path)
    if not include_encoder:
        for path in asset_path.glob("encoder_model.onnx*"):
            path.unlink()
    return {
        "source": str(source),
        "asset_directory": str(asset_path.relative_to(ROOT)),
        "shared_encoder": not include_encoder,
        "files": sorted(str(path.relative_to(asset_path)) for path in asset_path.rglob("*") if path.is_file()),
    }


def main() -> None:
    if not NUMBER_MODEL.exists():
        raise FileNotFoundError("Train the numeric TrOCR model before export")
    report = {
        "text": export_model(TEXT_MODEL, "trocr"),
        "numbers": export_model(NUMBER_MODEL, "trocr_numbers", include_encoder=True),
    }
    report_path = ROOT / "ai/reports/trocr_onnx_export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
