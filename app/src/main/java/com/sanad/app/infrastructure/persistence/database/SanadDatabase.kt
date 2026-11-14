package com.sanad.app.infrastructure.persistence.database

import androidx.room.Database
import androidx.room.RoomDatabase
import com.sanad.app.infrastructure.persistence.dao.IMedicalRecordDao
import com.sanad.app.infrastructure.persistence.dao.IPatientDao
import com.sanad.app.infrastructure.persistence.entity.MedicalRecordEntity
import com.sanad.app.infrastructure.persistence.entity.PatientEntity

@Database(entities = [PatientEntity::class, MedicalRecordEntity::class], version = 2, exportSchema = false)
abstract class SanadDatabase : RoomDatabase() {
    abstract fun patientDao(): IPatientDao
    abstract fun medicalRecordDao(): IMedicalRecordDao
}
