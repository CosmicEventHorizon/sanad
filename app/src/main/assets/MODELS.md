# Local model assets

Model binaries are intentionally excluded from Git. Generate them with the scripts under `ai/scripts`; the export step writes the form-specific anchor detector to `forms/<form_id>/anchors.onnx` and shared TrOCR graphs under `models/`.

The JSON form definitions remain versioned so additional form bundles can be configured without committing large weights.
