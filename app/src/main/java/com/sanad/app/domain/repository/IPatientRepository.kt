package com.sanad.app.domain.repository

import com.sanad.app.domain.model.Patient
import kotlinx.coroutines.flow.Flow

interface IPatientRepository {
    suspend fun save(patient: Patient)
    suspend fun findById(id: String): Patient?
    fun observeAll(): Flow<List<Patient>>
}
