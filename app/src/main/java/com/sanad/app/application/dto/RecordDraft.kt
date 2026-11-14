package com.sanad.app.application.dto

import com.sanad.app.domain.model.ProcessedForm

data class RecordDraft(
    val patientId: String = "",
    val patientName: String = "",
    val dateOfBirth: String = "",
    val phone: String = "",
    val village: String = "",
    val visitDate: String = "",
    val gestationalAgeWeeks: String = "",
    val bloodPressure: String = "",
    val weightKg: String = "",
    val notes: String = "",
    val formFields: Map<String, String> = emptyMap(),
    val formType: String = "Antenatal care form",
    val sourceImageUri: String? = null,
    val lowConfidenceKeys: Set<String> = emptySet(),
) {
    companion object {
        fun from(form: ProcessedForm): RecordDraft {
            val values = form.fields.associate { it.key to it.value }
            val isPregnancyForm = form.formType == "pregnancy_monitoring"
            return RecordDraft(
                patientId = values["patientId"] ?: values["form_number"].orEmpty(),
                patientName = values["patientName"] ?: values["patient_full_name"].orEmpty(),
                dateOfBirth = values["dateOfBirth"].orEmpty(),
                phone = values["phone"].orEmpty(),
                village = values["village"].orEmpty(),
                visitDate = values["visitDate"].orEmpty().ifBlank {
                    if (isPregnancyForm) java.text.SimpleDateFormat("yyyy-MM-dd", java.util.Locale.US).format(java.util.Date()) else ""
                },
                gestationalAgeWeeks = values["gestationalAgeWeeks"].orEmpty(),
                bloodPressure = values["bloodPressure"].orEmpty(),
                weightKg = values["weightKg"].orEmpty(),
                notes = values["notes"].orEmpty(),
                formFields = if (isPregnancyForm) values else emptyMap(),
                formType = form.formType,
                sourceImageUri = form.sourceImageUri,
                lowConfidenceKeys = form.fields.filter { it.confidence < 0.8f }.mapTo(mutableSetOf()) { it.key },
            )
        }
    }
}
