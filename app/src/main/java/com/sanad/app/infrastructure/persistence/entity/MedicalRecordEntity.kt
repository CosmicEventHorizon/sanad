package com.sanad.app.infrastructure.persistence.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(tableName = "medical_records", indices = [Index("patientId"), Index("visitDate")])
data class MedicalRecordEntity(
    @PrimaryKey val id: String,
    val patientId: String,
    val patientName: String,
    val visitDate: String,
    val gestationalAgeWeeks: String,
    val bloodPressure: String,
    val weightKg: String,
    val notes: String,
    val formType: String,
    val formDataJson: String,
    val sourceImageUri: String?,
    val createdAt: Long,
    val syncStatus: String,
)
