#!/usr/bin/env python3
"""Fine-tune TrOCR-small on leakage-safe Latin numeric handwriting samples."""

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
CONFIG = json.loads((ROOT / "ai/config/trocr_numbers.json").read_text())
DATA_ROOT = ROOT / "dataset/prepared/numbers/trocr"
OUTPUT = ROOT / CONFIG.get("output_directory", "ai/runs/trocr_numbers")
REPORT = ROOT / "ai/reports/trocr_numbers_metrics.json"


def normalize_numeric(text: str) -> str:
    match = re.search(r"[+-]?\d+(?:[.,/:-]\d+)*(?:%)?", text.replace(" ", ""))
    return match.group(0) if match else ""


class NumberDataset(Dataset):
    def __init__(self, split: str, processor: TrOCRProcessor, max_length: int):
        self.root = DATA_ROOT / split
        with (self.root / "metadata.csv").open(newline="", encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        allowed_formats = set(CONFIG.get("target_formats", []))
        self.rows = []
        for row in source_rows:
            if allowed_formats and row["format"] not in allowed_formats:
                continue
            target = normalize_numeric(row["text"])
            if not any(character.isdigit() for character in target):
                continue
            self.rows.append({**row, "target_text": target})
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image = Image.open(self.root / row["file_name"]).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            row["target_text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    set_seed(CONFIG["seed"])
    initial_model = ROOT / CONFIG["initial_model"] if CONFIG.get("initial_model") else CONFIG["base_model"]
    processor = TrOCRProcessor.from_pretrained(CONFIG["base_model"])
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

    train_dataset = NumberDataset("train", processor, CONFIG["max_target_length"])
    validation_dataset = NumberDataset("val", processor, CONFIG["max_target_length"])
    test_dataset = NumberDataset("test", processor, CONFIG["max_target_length"])
    cer_metric = evaluate.load("cer")

    def compute_metrics(prediction) -> dict[str, float]:
        prediction_ids = prediction.predictions
        if isinstance(prediction_ids, tuple):
            prediction_ids = prediction_ids[0]
        label_ids = np.array(prediction.label_ids)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        predicted_text = [normalize_numeric(text) for text in processor.batch_decode(prediction_ids, skip_special_tokens=True)]
        label_text = processor.batch_decode(label_ids, skip_special_tokens=True)
        exact = sum(pred == expected for pred, expected in zip(predicted_text, label_text)) / max(len(label_text), 1)
        return {
            "cer": float(cer_metric.compute(predictions=predicted_text, references=label_text)),
            "exact_match": float(exact),
        }

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

    final_path = OUTPUT / "best"
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    report = {
        "base_model": CONFIG["base_model"],
        "initial_model": str(initial_model),
        "preprocessing": "Original detected/annotated line crop; TrOCRProcessor direct resize to 384x384, matching Android inference",
        "encoder_frozen": CONFIG.get("freeze_encoder", False),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "target_normalization": {
            "formats": CONFIG.get("target_formats", []),
            "characters": CONFIG.get("target_characters"),
            "description": "Units are visible context but targets contain only numeric content and meaningful separators.",
        },
        "characters": sorted({character for dataset in (train_dataset, validation_dataset, test_dataset) for row in dataset.rows for character in row["target_text"]}),
        "train": {key: float(value) for key, value in train_result.metrics.items() if isinstance(value, (int, float))},
        "test": {key: float(value) for key, value in test_result.metrics.items() if isinstance(value, (int, float))},
        "checkpoint": str(final_path.relative_to(ROOT)),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
