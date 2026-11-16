# Sanad

Sanad is an offline-first Android prototype for digitizing standardized maternal healthcare forms. It captures or imports a form, runs a form-specific ONNX pipeline, highlights uncertain fields for review, and stores verified patient visits locally in Room.

## Architecture

The single `app` module is organized into Clean Architecture packages:

- `presentation`: Jetpack Compose screens, navigation, ViewModels, UI state, theme and components
- `application`: workflow Services and DTOs
- `domain`: pure Kotlin models, rules, and `I`-prefixed interfaces
- `infrastructure`: Room repositories, CameraX, ONNX Runtime processing, and future sync adapter
- `di`: lightweight `AppContainer`

Dependencies point inward. Infrastructure implements domain interfaces and is assembled only by the container.

## Requirements

- JDK 21
- Android SDK 35 and Build Tools 35.0.0
- Android device or emulator running Android 8.0 (API 26) or newer

Create a local, untracked `local.properties` file containing `sdk.dir=/path/to/your/Android/Sdk` if Gradle cannot discover your SDK automatically.

## Build and test

```bash
./gradlew test
./gradlew assembleDebug
```

The debug APK is generated at `app/build/outputs/apk/debug/app-debug.apk`.

Large datasets, checkpoints, trained weights, ONNX graphs, local virtual environments, and APK outputs are intentionally excluded from Git. Run the preparation, training, and export commands below to recreate the AI assets before building an AI-enabled APK. The Android source still compiles without checking those binaries into version control, but on-device form processing requires the exported files at the configured asset paths.

## Run

Connect a device with USB debugging enabled or start an emulator, then run:

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.sanad.app/.MainActivity
```

Open the folder in VS Code and install the recommended Kotlin and Gradle extensions. The app is also compatible with Android Studio.

## AI training pipeline

The Python tooling uses a local virtual environment and keeps generated runs out of version control. The dataset preparation command preserves Arabic samples and annotations separately, excludes them from Latin training, and performs deterministic grouped 80/10/10 splits.

```bash
uv venv --python 3.11 .venv-ai
uv pip install --python .venv-ai/bin/python torch torchvision ultralytics \
  transformers datasets evaluate jiwer optimum optimum-onnx onnx onnxruntime-gpu \
  opencv-python pillow pyyaml accelerate sentencepiece

.venv-ai/bin/python ai/scripts/prepare_datasets.py
.venv-ai/bin/python ai/scripts/train_yolo_obb.py
.venv-ai/bin/python ai/scripts/prepare_handwriting_detector.py
.venv-ai/bin/python ai/scripts/train_handwriting_detector.py
.venv-ai/bin/python ai/scripts/train_trocr_numbers.py
.venv-ai/bin/python ai/scripts/export_trocr_onnx.py
```

Training configuration lives in `ai/config`. Metrics are written to `ai/reports`; best checkpoints stay in the ignored `ai/runs` directory. Successful export creates the locally ignored Android assets at:

- `app/src/main/assets/forms/pregnancy_monitoring/anchors.onnx`
- `app/src/main/assets/models/handwriting.onnx`
- `app/src/main/assets/models/trocr/`
- `app/src/main/assets/models/trocr_numbers/`

The YOLO configuration is sized for an RTX 2060 6 GB (960 px, batch 4, AMP). The app packages `.onnx` files uncompressed and executes them with ONNX Runtime without a network connection.

Current held-out results:

- YOLO11n-OBB: 0.9992 precision, 1.0000 recall, 0.9950 mAP50, and 0.9942 mAP50–95.
- Numeric TrOCR: 0.3504 CER and 0.2566 exact-match accuracy on leakage-safe unseen value groups. It is suitable for demonstrating the review-first workflow, not autonomous clinical transcription; uncertain output must be verified by a healthcare worker.

The numeric model learns digits and meaningful numeric separators while treating printed or handwritten units as surrounding context. The original annotations remain unchanged in the prepared dataset. Direct exported-graph checks are available in `ai/reports/trocr_onnx_validation.json`.

## Implementation notes

- Form bundles live under `app/src/main/assets/forms/<form_id>`. Each bundle owns its template, anchor mapping, pipeline specification, and printed-label YOLO OBB detector. The handwriting YOLO OBB detector and TrOCR readers are shared under `app/src/main/assets/models`.
- Printed-label geometry constrains one-to-one assignment of shared handwriting detections to fields. OCR crops are rectified from the matched handwriting OBB itself, while boolean marks continue to use label-relative ink density. Missing or uncertain output is flagged for review rather than guessed.
- Room is the source of truth; scanning, review, search, details, and language switching work offline.
- `IRecordSyncService` deliberately has a non-networked placeholder implementation. A remote adapter can be added without changing application or presentation code.
