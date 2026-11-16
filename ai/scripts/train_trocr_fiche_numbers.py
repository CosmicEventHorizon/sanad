#!/usr/bin/env python3
"""Fine-tune numeric TrOCR for the pregnancy form's YYYY-NNN-NNN fiche number."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import evaluate
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "ai/config/trocr_fiche_numbers.json").read_text())
DATA_ROOT = ROOT / CONFIG["dataset_directory"]
OUTPUT = ROOT / CONFIG["output_directory"]
REPORT = ROOT / CONFIG["report"]
PATTERN = re.compile(r"\d{4}-\d{3}-\d{3}")


def normalize_fiche(text: str) -> str:
    compact = text.replace(" ", "").replace("–", "-").replace("—", "-")
    match = PATTERN.search(compact)
    return match.group(0) if match else compact


class FicheNumberDataset(Dataset):
    def __init__(self, split: str, processor: TrOCRProcessor):
        self.root = DATA_ROOT / split
        with (self.root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["format"] == CONFIG["format"]]
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["file_name"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["text"],
            padding="max_length",
            max_length=CONFIG["max_target_length"],
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    set_seed(CONFIG["seed"])
    processor = TrOCRProcessor.from_pretrained(CONFIG["base_model"], use_fast=False)
    initial_model = ROOT / CONFIG["initial_model"]
    model = VisionEncoderDecoderModel.from_pretrained(str(initial_model))
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = CONFIG["generation_max_length"]
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 0
    model.config.length_penalty = 1.0
    model.config.num_beams = 2
    if CONFIG.get("freeze_encoder", False):
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False

    train_dataset = FicheNumberDataset("train", processor)
    validation_dataset = FicheNumberDataset("val", processor)
    test_dataset = FicheNumberDataset("test", processor)
    cer_metric = evaluate.load("cer")

    def decode(prediction_ids, label_ids) -> tuple[list[str], list[str]]:
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        labels = np.array(label_ids)
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predictions = [normalize_fiche(text) for text in processor.batch_decode(prediction_ids, skip_special_tokens=True)]
        references = [normalize_fiche(text) for text in processor.batch_decode(labels, skip_special_tokens=True)]
        return predictions, references

    def score(predictions: list[str], references: list[str]) -> dict[str, float | int]:
        return {
            "samples": len(references),
            "cer": float(cer_metric.compute(predictions=predictions, references=references)),
            "exact_match": sum(left == right for left, right in zip(predictions, references)) / max(len(references), 1),
        }

    def compute_metrics(prediction) -> dict[str, float]:
        predictions, references = decode(prediction.predictions, prediction.label_ids)
        metrics = score(predictions, references)
        return {"cer": float(metrics["cer"]), "exact_match": float(metrics["exact_match"])}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT),
        num_train_epochs=CONFIG["epochs"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_ratio=CONFIG["warmup_ratio"],
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=CONFIG["generation_max_length"],
        generation_num_beams=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        seed=CONFIG["seed"],
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    train_result = trainer.train()
    test_result = trainer.predict(test_dataset, metric_key_prefix="test")
    predictions, references = decode(test_result.predictions, test_result.label_ids)
    per_source: dict[str, dict[str, float | int | list[dict[str, str]]]] = {}
    source_kinds = sorted({row["source_kind"] for row in test_dataset.rows})
    for source_kind in source_kinds:
        indices = [index for index, row in enumerate(test_dataset.rows) if row["source_kind"] == source_kind]
        source_predictions = [predictions[index] for index in indices]
        source_references = [references[index] for index in indices]
        per_source[source_kind] = score(source_predictions, source_references) | {
            "results": [
                {
                    "file_name": test_dataset.rows[index]["file_name"],
                    "expected": references[index],
                    "prediction": predictions[index],
                }
                for index in indices
            ]
        }

    final_path = OUTPUT / "best"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    report = {
        "base_model": CONFIG["base_model"],
        "initial_model": CONFIG["initial_model"],
        "target_pattern": CONFIG["target_pattern"],
        "preprocessing": "Production pipeline-shaped crop; TrOCRProcessor direct resize to 384x384, matching Android inference",
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "train": {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "test": {key: float(value) for key, value in test_result.metrics.items() if isinstance(value, (int, float))},
        "test_by_source": per_source,
        "checkpoint": str(final_path.relative_to(ROOT)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
