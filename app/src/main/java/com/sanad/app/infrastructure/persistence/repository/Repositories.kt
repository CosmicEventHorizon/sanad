package com.sanad.app.infrastructure.persistence.repository

import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.domain.model.Patient
import com.sanad.app.domain.model.SyncStatus
import com.sanad.app.domain.repository.IMedicalRecordRepository
import com.sanad.app.domain.repository.IPatientRepository
import com.sanad.app.infrastructure.persistence.dao.IMedicalRecordDao
import com.sanad.app.infrastructure.persistence.dao.IPatientDao
import com.sanad.app.infrastructure.persistence.entity.MedicalRecordEntity
import com.sanad.app.infrastructure.persistence.entity.PatientEntity
import kotlinx.coroutines.flow.map
import org.json.JSONObject

class PatientRepository(private val dao: IPatientDao) : IPatientRepository {
    override suspend fun save(patient: Patient) = dao.upsert(patient.toEntity())
    override suspend fun findById(id: String) = dao.findById(id)?.toDomain()
    override fun observeAll() = dao.observeAll().map { rows -> rows.map { it.toDomain() } }
}

class MedicalRecordRepository(private val dao: IMedicalRecordDao) : IMedicalRecordRepository {
    override suspend fun save(record: MedicalRecord) = dao.upsert(record.toEntity())
    override suspend fun findById(id: String) = dao.findById(id)?.toDomain()
    override fun observeAll() = dao.observeAll().map { rows -> rows.map { it.toDomain() } }
    override fun search(query: String) = dao.search(query).map { rows -> rows.map { it.toDomain() } }
}

private fun Patient.toEntity() = PatientEntity(id, fullName, dateOfBirth, phone, village, createdAt)
private fun PatientEntity.toDomain() = Patient(id, fullName, dateOfBirth, phone, village, createdAt)

private fun MedicalRecord.toEntity() = MedicalRecordEntity(
    id, patientId, patientName, visitDate, gestationalAgeWeeks, bloodPressure,
    weightKg, notes, formType, JSONObject(formData).toString(), sourceImageUri, createdAt, syncStatus.name,
)

private fun MedicalRecordEntity.toDomain() = MedicalRecord(
    id, patientId, patientName, visitDate, gestationalAgeWeeks, bloodPressure,
    weightKg, notes, formType, jsonMap(formDataJson), sourceImageUri, createdAt, SyncStatus.valueOf(syncStatus),
)

private fun jsonMap(value: String): Map<String, String> = runCatching {
    val json = JSONObject(value)
    json.keys().asSequence().associateWith { json.optString(it) }
}.getOrDefault(emptyMap())
