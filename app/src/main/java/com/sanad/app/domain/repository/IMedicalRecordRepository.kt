package com.sanad.app.domain.repository

import com.sanad.app.domain.model.MedicalRecord
import kotlinx.coroutines.flow.Flow

interface IMedicalRecordRepository {
    suspend fun save(record: MedicalRecord)
    suspend fun findById(id: String): MedicalRecord?
    fun observeAll(): Flow<List<MedicalRecord>>
    fun search(query: String): Flow<List<MedicalRecord>>
}
