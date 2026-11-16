#!/usr/bin/env python3
"""Run the Android form pipeline phase-by-phase and emit a visual HTML report."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass
class Detection:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float
    confidence: float
    class_id: int = 0


class TrOcr:
    def __init__(self, assets: Path) -> None:
        self.assets = assets
        self.sessions: dict[Path, ort.InferenceSession] = {}
        self.vocabularies: dict[Path, dict[int, str]] = {}

    def _session(self, path: Path) -> ort.InferenceSession:
        if path not in self.sessions:
            self.sessions[path] = ort.InferenceSession(str(path))
        return self.sessions[path]

    @staticmethod
    def _decode_byte_bpe(pieces: list[str]) -> str:
        if any("▁" in piece for piece in pieces):
            return "".join(pieces).replace("▁", " ")
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        unicode_values = byte_values.copy()
        extra = 0
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value)
                unicode_values.append(256 + extra)
                extra += 1
        reverse = {chr(unicode_value): byte_value for byte_value, unicode_value in zip(byte_values, unicode_values)}
        encoded = "".join(pieces)
        return bytes(reverse[character] for character in encoded if character in reverse).decode("utf-8", errors="replace")

    def recognize(self, crop: np.ndarray, directory: str, max_length: int = 32) -> dict[str, Any]:
        model_dir = self.assets / directory
        encoder_dir = model_dir if (model_dir / "encoder_model.onnx").exists() else self.assets / "models/trocr"
        encoder = self._session(encoder_dir / "encoder_model.onnx")
        decoder = self._session(model_dir / "decoder_model.onnx")
        resized = cv2.resize(crop, (384, 384), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pixels = rgb.astype(np.float32).transpose(2, 0, 1)[None] / 127.5 - 1.0
        hidden = encoder.run(None, {encoder.get_inputs()[0].name: pixels})[0]
        generation = json.loads((model_dir / "generation_config.json").read_text())
        generated = [int(generation["decoder_start_token_id"])]
        probabilities: list[float] = []
        for _ in range(1, max_length):
            logits = decoder.run(
                None,
                {
                    "input_ids": np.asarray([generated], dtype=np.int64),
                    "encoder_hidden_states": hidden,
                },
            )[0][0, -1]
            token = int(np.argmax(logits))
            shifted = logits.astype(np.float64) - float(logits[token])
            probabilities.append(float(1.0 / max(np.exp(shifted).sum(), 1e-12)))
            if token == int(generation["eos_token_id"]):
                break
            generated.append(token)
        if model_dir not in self.vocabularies:
            vocabulary = json.loads((model_dir / "vocab.json").read_text())
            self.vocabularies[model_dir] = {int(value): key for key, value in vocabulary.items()}
        pieces = [self.vocabularies[model_dir].get(token, "") for token in generated[1:]]
        pieces = [piece for piece in pieces if not (piece.startswith("<") and piece.endswith(">"))]
        text = self._decode_byte_bpe(pieces).strip()
        return {
            "text": text,
            "confidence": float(np.mean(probabilities)) if probabilities else 0.0,
            "tokens": generated,
            "pieces": pieces,
            "ocr_input": resized,
        }


def load_image(path: Path) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def letterbox(source: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = source.shape[:2]
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    left = (size - resized_width) / 2.0
    top = (size - resized_height) / 2.0
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    x = round(left)
    y = round(top)
    output[y : y + resized_height, x : x + resized_width] = resized
    return output, scale, left, top


def axis_iou(first: Detection, second: Detection) -> float:
    left = max(first.center_x - first.width / 2, second.center_x - second.width / 2)
    top = max(first.center_y - first.height / 2, second.center_y - second.height / 2)
    right = min(first.center_x + first.width / 2, second.center_x + second.width / 2)
    bottom = min(first.center_y + first.height / 2, second.center_y + second.height / 2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0 else 0.0


def run_detector(
    source: np.ndarray,
    session: ort.InferenceSession,
    config: dict[str, Any],
    expected_classes: int,
) -> tuple[list[Detection], np.ndarray, tuple[int, ...]]:
    size = int(config["input"]["width"])
    prepared, scale, left, top = letterbox(source, size)
    tensor = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    output = session.run(None, {session.get_inputs()[0].name: tensor})[0][0]
    class_count = output.shape[0] - 5
    if class_count != expected_classes:
        raise ValueError(f"Detector has {class_count} classes, expected {expected_classes}")
    candidates: list[Detection] = []
    for candidate in range(output.shape[1]):
        scores = output[4 : 4 + class_count, candidate]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < float(config["confidence_threshold"]):
            continue
        detection = Detection(
            center_x=(float(output[0, candidate]) - left) / scale,
            center_y=(float(output[1, candidate]) - top) / scale,
            width=float(output[2, candidate]) / scale,
            height=float(output[3, candidate]) / scale,
            angle=float(output[-1, candidate]),
            confidence=confidence,
            class_id=class_id,
        )
        if detection.width >= 2 and detection.height >= 2:
            candidates.append(detection)
    threshold = float(config.get("iou_threshold", 0.45))
    maximum = int(config.get("maximum_detections", 150))
    kept: list[Detection] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if len(kept) >= maximum:
            break
        if not any(item.class_id == candidate.class_id and axis_iou(item, candidate) > threshold for item in kept):
            kept.append(candidate)
    return kept, prepared, tuple(output.shape)


def local_center(detection: Detection, anchor: Detection) -> tuple[float, float]:
    cosine = math.cos(anchor.angle)
    sine = math.sin(anchor.angle)
    dx = detection.center_x - anchor.center_x
    dy = detection.center_y - anchor.center_y
    return (
        0.5 + (dx * cosine + dy * sine) / max(anchor.width, 1.0),
        0.5 + (-dx * sine + dy * cosine) / max(anchor.height, 1.0),
    )


def association_cost(region: list[float], anchor: Detection, handwriting: Detection) -> float:
    x, y = local_center(handwriting, anchor)
    x1, y1, x2, y2 = region
    region_width = max(x2 - x1, 0.2)
    region_height = max(y2 - y1, 0.5)
    outside_x = max(x1 - x, 0.0) + max(x - x2, 0.0)
    outside_y = max(y1 - y, 0.0) + max(y - y2, 0.0)
    outside = math.hypot(outside_x / region_width, outside_y / region_height)
    if outside > 0.65:
        return 10.0
    expected_x = (x1 + x2) / 2
    expected_y = (y1 + y2) / 2
    center = math.hypot((x - expected_x) / region_width, (y - expected_y) / region_height)
    angle = abs((handwriting.angle - anchor.angle + math.pi / 2) % math.pi - math.pi / 2)
    return outside * 3 + center * 0.35 + angle * 0.08 + (1 - handwriting.confidence) * 0.15


def hungarian(cost: np.ndarray) -> list[int]:
    rows, columns = cost.shape
    row_potential = np.zeros(rows + 1, dtype=np.float64)
    column_potential = np.zeros(columns + 1, dtype=np.float64)
    matched_row = np.zeros(columns + 1, dtype=np.int32)
    path = np.zeros(columns + 1, dtype=np.int32)
    for row in range(1, rows + 1):
        matched_row[0] = row
        current_column = 0
        minimum = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = matched_row[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                reduced = cost[current_row - 1, column - 1] - row_potential[current_row] - column_potential[column]
                if reduced < minimum[column]:
                    minimum[column] = reduced
                    path[column] = current_column
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    row_potential[matched_row[column]] += delta
                    column_potential[column] -= delta
                else:
                    minimum[column] -= delta
            current_column = next_column
            if matched_row[current_column] == 0:
                break
        while current_column != 0:
            previous = path[current_column]
            matched_row[current_column] = matched_row[previous]
            current_column = previous
    result = [-1] * rows
    for column in range(1, columns + 1):
        if matched_row[column] > 0:
            result[matched_row[column] - 1] = column - 1
    return result


def match_handwriting(
    anchors_by_class: dict[int, Detection],
    anchors: list[dict[str, Any]],
    handwriting: list[Detection],
) -> tuple[dict[str, Detection], dict[str, float]]:
    fields = [
        anchor
        for anchor in anchors
        if anchor["field_type"] in {"text", "number", "numeric"} and int(anchor["class_id"]) in anchors_by_class
    ]
    if not fields or not handwriting:
        return {}, {}
    costs = np.full((len(fields), len(handwriting) + len(fields)), 2.5, dtype=np.float64)
    for row, field in enumerate(fields):
        detected_anchor = anchors_by_class[int(field["class_id"])]
        for column, hand in enumerate(handwriting):
            costs[row, column] = association_cost(field["relative_crop"], detected_anchor, hand)
    assignment = hungarian(costs)
    matches: dict[str, Detection] = {}
    scores: dict[str, float] = {}
    for row, column in enumerate(assignment):
        if 0 <= column < len(handwriting) and costs[row, column] < 2.5:
            field_id = fields[row]["field_id"]
            matches[field_id] = handwriting[column]
            scores[field_id] = float(costs[row, column])
    return matches, scores


def local_corners(detection: Detection, relative: list[float]) -> np.ndarray:
    ux = math.cos(detection.angle)
    uy = math.sin(detection.angle)
    vx = -uy
    vy = ux

    def point(x: float, y: float) -> tuple[float, float]:
        return (
            detection.center_x + (x - 0.5) * detection.width * ux + (y - 0.5) * detection.height * vx,
            detection.center_y + (x - 0.5) * detection.width * uy + (y - 0.5) * detection.height * vy,
        )

    x1, y1, x2, y2 = relative
    return np.asarray([point(x1, y1), point(x2, y1), point(x2, y2), point(x1, y2)], dtype=np.float32)


def rectify(source: np.ndarray, detection: Detection, relative: list[float]) -> tuple[np.ndarray, np.ndarray]:
    corners = local_corners(detection, relative)
    width = max(32, int(np.linalg.norm(corners[0] - corners[1])))
    height = max(24, int(np.linalg.norm(corners[1] - corners[2])))
    destination = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, destination)
    crop = cv2.warpPerspective(
        source,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return crop, corners


def ink_ratio(crop: np.ndarray, trim_border: bool) -> float:
    height, width = crop.shape[:2]
    margin_x = max(1, width // 5) if trim_border else 0
    margin_y = max(1, height // 5) if trim_border else 0
    usable = crop[margin_y : height - margin_y, margin_x : width - margin_x]
    if usable.size == 0:
        return 0.0
    gray = cv2.cvtColor(usable, cv2.COLOR_BGR2GRAY).astype(np.float32)
    threshold = min(190.0, float(gray.mean()) * 0.78)
    return float(np.mean(gray < threshold))


def checkbox_confidence(ratio: float, threshold: float) -> float:
    return float(np.clip(0.5 + min(0.49, abs(ratio - threshold) / max(threshold, 0.01)), 0.0, 0.99))


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def draw_detection_overlay(image: np.ndarray, detections: dict[int, Detection], anchors: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    for anchor in anchors:
        class_id = int(anchor["class_id"])
        detection = detections.get(class_id)
        if detection is None:
            continue
        corners = local_corners(detection, [0.0, 0.0, 1.0, 1.0]).astype(np.int32)
        cv2.polylines(overlay, [corners], True, (0, 210, 0), 3, cv2.LINE_AA)
        label = f"{class_id}:{anchor['field_id']} {detection.confidence:.2f}"
        location = tuple(corners[np.argmin(corners[:, 1])])
        cv2.putText(overlay, label, location, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 240), 2, cv2.LINE_AA)
    return overlay


def draw_crop_overlay(image: np.ndarray, detections: dict[int, Detection], anchors: list[dict[str, Any]]) -> np.ndarray:
    overlay = image.copy()
    colors = [(255, 0, 0), (0, 180, 255), (255, 0, 255), (0, 220, 0)]
    for anchor in anchors:
        detection = detections.get(int(anchor["class_id"]))
        if detection is None:
            continue
        regions = [(anchor["field_id"], anchor["relative_crop"])]
        regions += [(f"{anchor['field_id']}:{option['id']}", option["relative_crop"]) for option in anchor.get("options", [])]
        for index, (label, relative) in enumerate(regions):
            corners = local_corners(detection, relative).astype(np.int32)
            color = colors[index % len(colors)]
            cv2.polylines(overlay, [corners], True, color, 3, cv2.LINE_AA)
            cv2.putText(overlay, label, tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)
    return overlay


def rectify_detection(source: np.ndarray, detection: Detection, padding: float) -> tuple[np.ndarray, np.ndarray]:
    horizontal = max(4.0, detection.height * padding)
    vertical = max(3.0, detection.height * padding)
    padded = Detection(
        detection.center_x,
        detection.center_y,
        detection.width + horizontal * 2,
        detection.height + vertical * 2,
        detection.angle,
        detection.confidence,
        detection.class_id,
    )
    return rectify(source, padded, [0.0, 0.0, 1.0, 1.0])


def draw_handwriting_overlay(image: np.ndarray, handwriting: list[Detection]) -> np.ndarray:
    overlay = image.copy()
    for index, detection in enumerate(handwriting):
        corners = local_corners(detection, [0.0, 0.0, 1.0, 1.0]).astype(np.int32)
        cv2.polylines(overlay, [corners], True, (255, 0, 220), 3, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"H{index} {detection.confidence:.2f}",
            tuple(corners[np.argmin(corners[:, 1])]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 0, 180),
            2,
            cv2.LINE_AA,
        )
    return overlay


def draw_assignment_overlay(
    image: np.ndarray,
    anchors_by_class: dict[int, Detection],
    anchors: list[dict[str, Any]],
    matches: dict[str, Detection],
    scores: dict[str, float],
) -> np.ndarray:
    overlay = image.copy()
    for anchor in anchors:
        field_id = anchor["field_id"]
        if field_id not in matches:
            continue
        printed = anchors_by_class[int(anchor["class_id"])]
        hand = matches[field_id]
        search = local_corners(printed, anchor["relative_crop"]).astype(np.int32)
        detected = local_corners(hand, [0.0, 0.0, 1.0, 1.0]).astype(np.int32)
        cv2.polylines(overlay, [search], True, (0, 170, 255), 2, cv2.LINE_AA)
        cv2.polylines(overlay, [detected], True, (255, 0, 220), 3, cv2.LINE_AA)
        start = (round(printed.center_x), round(printed.center_y))
        end = (round(hand.center_x), round(hand.center_y))
        cv2.line(overlay, start, end, (40, 190, 40), 2, cv2.LINE_AA)
        cv2.putText(
            overlay,
            f"{field_id} cost={scores[field_id]:.2f}",
            tuple(detected[np.argmin(detected[:, 1])]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 40, 220),
            2,
            cv2.LINE_AA,
        )
    return overlay


def correct_perspective(
    source: np.ndarray,
    detections: dict[int, Detection],
    anchors: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[int, Detection], np.ndarray | None]:
    reference_ids = [0, 2, 7, 9]
    if any(class_id not in detections for class_id in reference_ids):
        return source, detections, None
    height, width = source.shape[:2]
    by_id = {int(anchor["class_id"]): anchor for anchor in anchors}
    source_points = np.asarray(
        [[detections[class_id].center_x, detections[class_id].center_y] for class_id in reference_ids],
        dtype=np.float32,
    )
    destination_points = []
    for class_id in reference_ids:
        x1, y1, x2, y2 = by_id[class_id]["canonical_anchor"]
        destination_points.append([((x1 + x2) * 0.5) * width, ((y1 + y2) * 0.5) * height])
    destination_points = np.asarray(destination_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    corrected = cv2.warpPerspective(
        source,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    canonical: dict[int, Detection] = {}
    for anchor in anchors:
        class_id = int(anchor["class_id"])
        if class_id not in detections:
            continue
        x1, y1, x2, y2 = anchor["canonical_anchor"]
        canonical[class_id] = Detection(
            center_x=((x1 + x2) * 0.5) * width,
            center_y=((y1 + y2) * 0.5) * height,
            width=(x2 - x1) * width,
            height=(y2 - y1) * height,
            angle=0.0,
            confidence=detections[class_id].confidence,
        )
    return corrected, canonical, matrix


def make_contact_sheet(output: Path, form_name: str, rows: list[dict[str, Any]]) -> None:
    width = 1900
    row_height = 245
    canvas = Image.new("RGB", (width, 90 + row_height * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=20)
    small = ImageFont.load_default(size=16)
    draw.text((20, 20), f"Sanad crop diagnostic: {form_name}", fill="black", font=font)
    draw.text(
        (20, 52),
        "Left: anchor-relative search region   Middle: actual matched handwriting OBB crop   Next: direct 384x384 TrOCR input",
        fill="black",
        font=small,
    )
    for index, row in enumerate(rows):
        y = 90 + index * row_height
        draw.line((0, y, width, y), fill=(210, 210, 210), width=2)
        draw.text((18, y + 10), row["field_id"], fill="black", font=font)
        search_path = row.get("search_path", row["crop_path"])
        search = Image.open(search_path).convert("RGB")
        search.thumbnail((430, 170))
        canvas.paste(search, (18, y + 52))
        crop = Image.open(row["crop_path"]).convert("RGB")
        crop.thumbnail((430, 170))
        canvas.paste(crop, (470, y + 52))
        if row.get("ocr_input_path"):
            ocr = Image.open(row["ocr_input_path"]).convert("RGB")
            ocr.thumbnail((190, 170))
            canvas.paste(ocr, (920, y + 52))
        details = [
            f"anchor confidence: {row['anchor_confidence']:.4f}",
            f"handwriting confidence: {row.get('handwriting_confidence', 0.0):.4f}",
            f"assignment cost: {row.get('assignment_cost', 0.0):.4f}",
            f"crop size: {row['crop_width']} x {row['crop_height']}",
            f"value: {row.get('value', '')!r}",
            f"field confidence: {row.get('field_confidence', 0.0):.4f}",
        ]
        if row.get("tokens"):
            details.append(f"tokens: {row['tokens']}")
        draw.multiline_text((1140, y + 52), "\n".join(details), fill="black", font=small, spacing=7)
    canvas.save(output)


def process_form(
    image_path: Path,
    output_dir: Path,
    anchor_detector: ort.InferenceSession,
    handwriting_detector: ort.InferenceSession,
    anchors: list[dict[str, Any]],
    spec: dict[str, Any],
    recognizer: TrOcr,
) -> dict[str, Any]:
    form_dir = output_dir / image_path.stem
    form_dir.mkdir(parents=True, exist_ok=True)
    source = load_image(image_path)
    save_image(form_dir / "00_input.png", source)
    anchor_config = spec["pipeline"]["anchor_detection"]
    handwriting_config = spec["pipeline"]["handwriting_detection"]
    anchor_candidates, anchor_input, anchor_shape = run_detector(source, anchor_detector, anchor_config, len(anchors))
    handwriting, handwriting_input, handwriting_shape = run_detector(source, handwriting_detector, handwriting_config, 1)
    save_image(form_dir / "01_anchor_letterboxed.png", anchor_input)
    save_image(form_dir / "02_handwriting_letterboxed.png", handwriting_input)
    anchors_by_class: dict[int, Detection] = {}
    for candidate in anchor_candidates:
        previous = anchors_by_class.get(candidate.class_id)
        if previous is None or candidate.confidence > previous.confidence:
            anchors_by_class[candidate.class_id] = candidate
    matches, assignment_scores = match_handwriting(anchors_by_class, anchors, handwriting)
    save_image(form_dir / "01_anchor_detections.png", draw_detection_overlay(source, anchors_by_class, anchors))
    save_image(form_dir / "02_handwriting_detections.png", draw_handwriting_overlay(source, handwriting))
    save_image(
        form_dir / "03_field_assignments.png",
        draw_assignment_overlay(source, anchors_by_class, anchors, matches, assignment_scores),
    )
    save_image(form_dir / "04_anchor_search_regions.png", draw_crop_overlay(source, anchors_by_class, anchors))

    minimum_ink = float(spec["pipeline"]["checkbox_recognition"]["minimum_ink_ratio"])
    text_directory = spec["pipeline"]["handwriting_recognition"]["model_directory"]
    numeric_directory = spec["pipeline"]["numeric_recognition"]["model_directory"]
    fields: list[dict[str, Any]] = []
    contact_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        class_id = int(anchor["class_id"])
        field_id = anchor["field_id"]
        field_type = anchor["field_type"]
        detected_anchor = anchors_by_class.get(class_id)
        if detected_anchor is None:
            fields.append({"field_id": field_id, "type": field_type, "value": "", "confidence": 0.0, "failure": "anchor_not_detected"})
            continue
        search_crop, search_corners = rectify(source, detected_anchor, anchor["relative_crop"])
        search_path = form_dir / "search_regions" / f"{class_id:02d}_{field_id}.png"
        save_image(search_path, search_crop)
        crop_path = form_dir / "crops" / f"{class_id:02d}_{field_id}.png"
        row: dict[str, Any] = {
            "field_id": field_id,
            "type": field_type,
            "anchor_confidence": detected_anchor.confidence,
            "search_width": int(search_crop.shape[1]),
            "search_height": int(search_crop.shape[0]),
            "search_corners": search_corners.tolist(),
            "search_path": str(search_path),
            "crop_path": str(crop_path),
        }
        if field_type == "boolean":
            save_image(crop_path, search_crop)
            row.update(crop_width=int(search_crop.shape[1]), crop_height=int(search_crop.shape[0]), crop_corners=search_corners.tolist())
            ratio = ink_ratio(search_crop, True)
            value = str(ratio >= minimum_ink).lower()
            confidence = checkbox_confidence(ratio, minimum_ink)
            row.update(value=value, field_confidence=confidence, ink_ratio=ratio, ink_threshold=minimum_ink)
        elif field_type in {"single_choice", "multiple_choice"}:
            save_image(crop_path, search_crop)
            row.update(crop_width=int(search_crop.shape[1]), crop_height=int(search_crop.shape[0]), crop_corners=search_corners.tolist())
            option_scores: dict[str, float] = {}
            option_details: list[dict[str, Any]] = []
            for option in anchor.get("options", []):
                option_crop, option_corners = rectify(source, detected_anchor, option["relative_crop"])
                option_path = form_dir / "crops" / f"{class_id:02d}_{field_id}__{option['id']}.png"
                save_image(option_path, option_crop)
                option_threshold = float(option.get("ink_threshold", minimum_ink))
                ratio = ink_ratio(option_crop, bool(option.get("trim_border", True)))
                option_scores[option["id"]] = ratio
                option_details.append(
                    {
                        "id": option["id"],
                        "ink_ratio": ratio,
                        "threshold": option_threshold,
                        "selected": ratio >= option_threshold,
                        "crop_path": str(option_path.relative_to(output_dir)),
                        "crop_corners": option_corners.tolist(),
                    }
                )
            selected = [option for option in option_details if option["selected"]]
            if field_type == "single_choice" and selected:
                selected = [max(selected, key=lambda item: item["ink_ratio"])]
            value = ",".join(option["id"] for option in selected)
            option_confidence = min(
                (checkbox_confidence(item["ink_ratio"], item["threshold"]) for item in option_details),
                default=0.0,
            )
            row.update(
                value=value,
                field_confidence=detected_anchor.confidence * option_confidence,
                option_scores=option_scores,
                options=option_details,
            )
        else:
            hand = matches.get(field_id)
            if hand is None:
                save_image(crop_path, search_crop)
                row.update(
                    value="",
                    field_confidence=0.0,
                    handwriting_confidence=0.0,
                    assignment_cost=0.0,
                    crop_width=int(search_crop.shape[1]),
                    crop_height=int(search_crop.shape[0]),
                    crop_corners=search_corners.tolist(),
                    failure="no_handwriting_match",
                )
                fields.append(
                    {key: value for key, value in row.items() if key not in {"crop_path", "search_path", "field_confidence"}}
                    | {
                        "confidence": 0.0,
                        "crop_path": str(crop_path.relative_to(output_dir)),
                        "search_path": str(search_path.relative_to(output_dir)),
                    }
                )
                contact_rows.append(row)
                continue
            crop, corners = rectify_detection(source, hand, float(handwriting_config.get("crop_padding", 0.15)))
            save_image(crop_path, crop)
            directory = numeric_directory if field_type in {"number", "numeric"} else text_directory
            result = recognizer.recognize(crop, directory)
            ocr_path = form_dir / "ocr_inputs" / f"{class_id:02d}_{field_id}_384.png"
            save_image(ocr_path, result.pop("ocr_input"))
            value = result["text"]
            confidence = min(detected_anchor.confidence, hand.confidence) * float(result["confidence"])
            row.update(
                value=value,
                field_confidence=confidence,
                handwriting_confidence=hand.confidence,
                assignment_cost=assignment_scores[field_id],
                crop_width=int(crop.shape[1]),
                crop_height=int(crop.shape[0]),
                crop_corners=corners.tolist(),
                ocr_confidence=result["confidence"],
                tokens=result["tokens"],
                pieces=result["pieces"],
                ocr_input_path=str(ocr_path),
            )
        fields.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"crop_path", "search_path", "ocr_input_path", "field_confidence"}
            }
            | {
                "confidence": row["field_confidence"],
                "crop_path": str(crop_path.relative_to(output_dir)),
                "search_path": str(search_path.relative_to(output_dir)),
            }
        )
        contact_rows.append(row)

    make_contact_sheet(form_dir / "05_crop_contact_sheet.png", image_path.name, contact_rows)
    report = {
        "image": str(image_path),
        "source_size": {"width": source.shape[1], "height": source.shape[0]},
        "anchor_detector_output_shape": list(anchor_shape),
        "handwriting_detector_output_shape": list(handwriting_shape),
        "accepted_anchor_count": len(anchors_by_class),
        "handwriting_detection_count": len(handwriting),
        "matched_text_field_count": len(matches),
        "anchors": [detection.__dict__ for detection in anchor_candidates],
        "handwriting": [detection.__dict__ for detection in handwriting],
        "assignments": {
            field_id: {"cost": assignment_scores[field_id], **detection.__dict__}
            for field_id, detection in matches.items()
        },
        "fields": fields,
    }
    (form_dir / "pipeline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def write_html(output_dir: Path, reports: list[dict[str, Any]]) -> None:
    sections = []
    for report in reports:
        stem = Path(report["image"]).stem
        rows = []
        for field in report["fields"]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(field['field_id'])}</td>"
                f"<td>{html.escape(str(field.get('value', '')))}</td>"
                f"<td>{field.get('anchor_confidence', 0):.3f}</td>"
                f"<td>{field.get('confidence', 0):.3f}</td>"
                f"<td>{field.get('crop_width', '')}×{field.get('crop_height', '')}</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{html.escape(Path(report['image']).name)}</h2>"
            f"<p>Anchors: {report['accepted_anchor_count']}/10; handwriting detections: {report['handwriting_detection_count']}; "
            f"matched text fields: {report['matched_text_field_count']}/6</p>"
            f"<div class='grid'><figure><img src='{stem}/00_input.png'><figcaption>0. Input</figcaption></figure>"
            f"<figure><img src='{stem}/01_anchor_detections.png'><figcaption>1. Printed-label YOLO OBB</figcaption></figure>"
            f"<figure><img src='{stem}/02_handwriting_detections.png'><figcaption>2. Shared handwriting YOLO OBB</figcaption></figure>"
            f"<figure><img src='{stem}/03_field_assignments.png'><figcaption>3. One-to-one field assignments</figcaption></figure>"
            f"<figure><img src='{stem}/04_anchor_search_regions.png'><figcaption>4. Anchor-relative search regions (not OCR crops)</figcaption></figure></div>"
            f"<figure><img class='sheet' src='{stem}/05_crop_contact_sheet.png'><figcaption>5. Search regions, actual handwriting crops, and TrOCR inputs</figcaption></figure>"
            "<table><thead><tr><th>Field</th><th>Output</th><th>Anchor conf.</th><th>Final conf.</th><th>Crop</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table><p><a href='{stem}/pipeline.json'>Full JSON diagnostics</a></p></section>"
        )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Sanad pipeline diagnostic</title>
<style>body{{font:16px system-ui;margin:24px;max-width:1500px}}section{{border-top:3px solid #166b64;padding-top:20px;margin-top:40px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}figure{{margin:8px}}img{{width:100%;max-height:700px;object-fit:contain;background:#eee}}img.sheet{{max-height:none}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #bbb;padding:8px;text-align:left}}</style></head>
<body><h1>Sanad Android-equivalent two-detector diagnostic</h1><p>Each section shows printed anchors, shared handwriting detections, assignment geometry, actual OBB crops, the exact directly resized 384×384 TrOCR input, and decoded values.</p>{''.join(sections)}</body></html>"""
    (output_dir / "index.html").write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("app/src/main/assets"))
    parser.add_argument("--output", type=Path, default=Path("ai/runs/pipeline_diagnostic"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads((args.assets / "forms/pregnancy_monitoring/spec.json").read_text())
    anchors = json.loads((args.assets / spec["anchors"]).read_text())["classes"]
    anchor_model = args.assets / spec["pipeline"]["anchor_detection"]["model"]
    handwriting_model = args.assets / spec["pipeline"]["handwriting_detection"]["model"]
    anchor_detector = ort.InferenceSession(str(anchor_model))
    handwriting_detector = ort.InferenceSession(str(handwriting_model))
    recognizer = TrOcr(args.assets)
    reports = [
        process_form(path, args.output, anchor_detector, handwriting_detector, anchors, spec, recognizer)
        for path in args.images
    ]
    (args.output / "summary.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
    write_html(args.output, reports)
    for report in reports:
        values = {field["field_id"]: field.get("value", "") for field in report["fields"]}
        print(
            Path(report["image"]).name,
            f"anchors={report['accepted_anchor_count']}/10",
            f"hands={report['handwriting_detection_count']}",
            f"matches={report['matched_text_field_count']}/6",
            values,
        )
    print(f"Report: {(args.output / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
