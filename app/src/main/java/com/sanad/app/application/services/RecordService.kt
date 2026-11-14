package com.sanad.app.application.services

import com.sanad.app.application.dto.RecordDraft
import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.domain.model.Patient
import com.sanad.app.domain.repository.IMedicalRecordRepository
import java.util.UUID

class RecordService(
    private val repository: IMedicalRecordRepository,
    private val patientService: PatientService,
) {
    fun observeRecords(query: String = "") = if (query.isBlank()) repository.observeAll() else repository.search(query.trim())

    suspend fun get(id: String) = repository.findById(id)

    suspend fun saveVerified(draft: RecordDraft): String {
        require(draft.patientId.isNotBlank()) { "Patient ID is required" }
        require(draft.patientName.isNotBlank()) { "Patient name is required" }
        require(draft.visitDate.isNotBlank()) { "Visit date is required" }
        patientService.save(
            Patient(draft.patientId.trim(), draft.patientName.trim(), draft.dateOfBirth.trim(), draft.phone.trim(), draft.village.trim())
        )
        val id = UUID.randomUUID().toString()
        repository.save(
            MedicalRecord(
                id = id,
                patientId = draft.patientId.trim(),
                patientName = draft.patientName.trim(),
                visitDate = draft.visitDate.trim(),
                gestationalAgeWeeks = draft.gestationalAgeWeeks.trim(),
                bloodPressure = draft.bloodPressure.trim(),
                weightKg = draft.weightKg.trim(),
                notes = draft.notes.trim(),
                formType = draft.formType,
                formData = draft.formFields.mapValues { it.value.trim() },
                sourceImageUri = draft.sourceImageUri,
            )
        )
        return id
    }
}
