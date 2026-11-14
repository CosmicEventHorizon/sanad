package com.sanad.app.domain.model

data class FormTemplate(
    val id: String,
    val displayName: String,
    val version: Int,
    val fieldKeys: List<String>,
)

data class RecognizedField(
    val key: String,
    val value: String,
    val confidence: Float,
)

data class ProcessedForm(
    val formType: String,
    val sourceImageUri: String,
    val fields: List<RecognizedField>,
)
