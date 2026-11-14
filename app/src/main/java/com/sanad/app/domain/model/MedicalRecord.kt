package com.sanad.app.domain.model

data class MedicalRecord(
    val id: String,
    val patientId: String,
    val patientName: String,
    val visitDate: String,
    val gestationalAgeWeeks: String,
    val bloodPressure: String,
    val weightKg: String,
    val notes: String,
    val formType: String,
    val formData: Map<String, String> = emptyMap(),
    val sourceImageUri: String? = null,
    val createdAt: Long = System.currentTimeMillis(),
    val syncStatus: SyncStatus = SyncStatus.PENDING,
)

enum class SyncStatus { PENDING, SYNCED, FAILED }
