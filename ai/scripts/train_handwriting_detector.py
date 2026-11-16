#!/usr/bin/env python3
"""Train, evaluate and export the shared handwriting YOLO11n-OBB detector."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "ai/config/handwriting_yolo11n_obb.yaml"
DATA_PATH = ROOT / "dataset/prepared/handwriting_yolo_obb/dataset.yaml"
RUNS_PATH = ROOT / "ai/runs"
REPORT_PATH = ROOT / "ai/reports/handwriting_yolo_obb_metrics.json"
ASSET_PATH = ROOT / "app/src/main/assets/models/handwriting.onnx"


def metric(metrics, name: str) -> float | None:
    value = metrics.results_dict.get(name)
    return float(value) if value is not None else None


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    pretrained = config.pop("model")
    model = YOLO(pretrained)
    model.train(
        data=str(DATA_PATH),
        project=str(RUNS_PATH),
        name="shared_handwriting_yolo11n_obb",
        exist_ok=True,
        **config,
    )

    best_path = RUNS_PATH / "shared_handwriting_yolo11n_obb/weights/best.pt"
    best = YOLO(str(best_path))
    validation = best.val(data=str(DATA_PATH), split="val", imgsz=config["imgsz"], batch=config["batch"], device=0)
    test = best.val(data=str(DATA_PATH), split="test", imgsz=config["imgsz"], batch=config["batch"], device=0)
    exported = Path(best.export(format="onnx", imgsz=config["imgsz"], opset=17, simplify=True, dynamic=False, batch=1, device=0))
    ASSET_PATH.parent.mkdir(parents=True, exist_ok=True)
    ASSET_PATH.write_bytes(exported.read_bytes())

    report = {
        "model": "YOLO11n-OBB",
        "classes": ["handwriting_line"],
        "checkpoint": str(best_path.relative_to(ROOT)),
        "onnx_asset": str(ASSET_PATH.relative_to(ROOT)),
        "validation": {
            "precision": metric(validation, "metrics/precision(B)"),
            "recall": metric(validation, "metrics/recall(B)"),
            "map50": metric(validation, "metrics/mAP50(B)"),
            "map50_95": metric(validation, "metrics/mAP50-95(B)"),
        },
        "test": {
            "precision": metric(test, "metrics/precision(B)"),
            "recall": metric(test, "metrics/recall(B)"),
            "map50": metric(test, "metrics/mAP50(B)"),
            "map50_95": metric(test, "metrics/mAP50-95(B)"),
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
