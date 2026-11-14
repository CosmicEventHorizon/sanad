package com.sanad.app.infrastructure.persistence.dao

import androidx.room.Dao
import androidx.room.Query
import androidx.room.Upsert
import com.sanad.app.infrastructure.persistence.entity.MedicalRecordEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface IMedicalRecordDao {
    @Upsert suspend fun upsert(record: MedicalRecordEntity)
    @Query("SELECT * FROM medical_records WHERE id = :id") suspend fun findById(id: String): MedicalRecordEntity?
    @Query("SELECT * FROM medical_records ORDER BY createdAt DESC") fun observeAll(): Flow<List<MedicalRecordEntity>>
    @Query("""SELECT * FROM medical_records
        WHERE patientId LIKE '%' || :query || '%'
        OR patientName LIKE '%' || :query || '%'
        OR visitDate LIKE '%' || :query || '%'
        OR formType LIKE '%' || :query || '%'
        ORDER BY createdAt DESC""")
    fun search(query: String): Flow<List<MedicalRecordEntity>>
}
