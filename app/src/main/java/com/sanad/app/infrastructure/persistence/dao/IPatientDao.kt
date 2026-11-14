package com.sanad.app.infrastructure.persistence.dao

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert
import com.sanad.app.infrastructure.persistence.entity.PatientEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface IPatientDao {
    @Upsert suspend fun upsert(patient: PatientEntity)
    @Query("SELECT * FROM patients WHERE id = :id") suspend fun findById(id: String): PatientEntity?
    @Query("SELECT * FROM patients ORDER BY fullName") fun observeAll(): Flow<List<PatientEntity>>
}
