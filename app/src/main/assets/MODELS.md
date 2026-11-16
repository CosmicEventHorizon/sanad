# Local model assets

Model binaries are intentionally excluded from Git. Generate them with the scripts under `ai/scripts`; export writes each form-specific printed-anchor detector to `forms/<form_id>/anchors.onnx`, the shared handwriting-line detector to `models/handwriting.onnx`, and shared TrOCR graphs under `models/`.

The JSON form definitions remain versioned so additional form bundles can reuse the shared handwriting detector and OCR readers without committing large weights.
