#!/usr/bin/env python3
"""Calculate exact-match and character-error metrics for PP-OCR predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def normalized(value: str) -> str:
    return " ".join(value.strip().split())


def calculate(rows: list[dict[str, object]]) -> dict[str, object]:
    characters = sum(len(str(row["target"])) for row in rows)
    errors = sum(edit_distance(str(row["target"]), str(row["prediction"])) for row in rows)
    exact = sum(row["target"] == row["prediction"] for row in rows)
    return {
        "samples": len(rows),
        "characters": characters,
        "character_errors": errors,
        "cer": errors / characters if characters else 0.0,
        "exact_matches": exact,
        "exact_match_accuracy": exact / len(rows) if rows else 0.0,
        "mean_confidence": (
            sum(float(row["confidence"]) for row in rows) / len(rows) if rows else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions: dict[str, tuple[str, float]] = {}
    for line in args.predictions.read_text(encoding="utf-8").splitlines():
        path, prediction, confidence = line.rsplit("\t", 2)
        relative = Path(path).resolve().relative_to(args.data_dir.resolve()).as_posix()
        predictions[relative] = (normalized(prediction), float(confidence))

    evaluated: list[dict[str, object]] = []
    missing: list[str] = []
    with args.metadata.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            file_name = source["file_name"]
            if file_name not in predictions:
                missing.append(file_name)
                continue
            prediction, confidence = predictions[file_name]
            evaluated.append(
                {
                    "file_name": file_name,
                    "target": normalized(source["text"]),
                    "prediction": prediction,
                    "confidence": confidence,
                    "source_kind": source["source_kind"],
                }
            )

    if missing:
        raise RuntimeError(f"Missing predictions for {len(missing)} crops, first: {missing[0]}")

    by_source: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        by_source[str(row["source_kind"])].append(row)

    report = {
        "predictions": str(args.predictions),
        "metadata": str(args.metadata),
        "overall": calculate(evaluated),
        "by_source_kind": {key: calculate(value) for key, value in sorted(by_source.items())},
        "examples": evaluated[:100],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
