package com.sanad.app.domain.form

/**
 * Paths are relative to Android's assets root. A form owns its anchor detector,
 * while handwriting recognition assets may be shared by bundles.
 */
data class FormAssetBundle(
    val formId: String,
    val titleStringResource: String,
    val templatePath: String,
    val anchorsPath: String,
    val anchorModelPath: String,
    val handwritingRecognitionDirectory: String,
    val numericRecognitionDirectory: String,
)
