package com.sanad.app.di

import android.content.Context
import androidx.room.Room
import com.sanad.app.application.services.PatientService
import com.sanad.app.application.services.RecordService
import com.sanad.app.application.services.ScanService
import com.sanad.app.application.services.ScanWorkflowService
import com.sanad.app.domain.repository.IMedicalRecordRepository
import com.sanad.app.domain.repository.IPatientRepository
import com.sanad.app.domain.service.IFieldExtractionService
import com.sanad.app.domain.service.IFormClassificationService
import com.sanad.app.domain.service.IFormRecognitionService
import com.sanad.app.infrastructure.ocr.PlaceholderFieldExtractionService
import com.sanad.app.infrastructure.ocr.PlaceholderFormClassificationService
import com.sanad.app.infrastructure.ocr.PlaceholderFormRecognitionService
import com.sanad.app.infrastructure.ocr.OnDeviceFormProcessingService
import com.sanad.app.infrastructure.assets.AndroidFormAssetCatalog
import com.sanad.app.infrastructure.persistence.database.SanadDatabase
import com.sanad.app.infrastructure.persistence.repository.MedicalRecordRepository
import com.sanad.app.infrastructure.persistence.repository.PatientRepository

class AppContainer(context: Context) {
    private val database = Room.databaseBuilder(context, SanadDatabase::class.java, "sanad.db")
        .fallbackToDestructiveMigration()
        .build()

    val formAssetCatalog = AndroidFormAssetCatalog(context.assets)

    val patientRepository: IPatientRepository = PatientRepository(database.patientDao())
    val medicalRecordRepository: IMedicalRecordRepository = MedicalRecordRepository(database.medicalRecordDao())

    val classificationService: IFormClassificationService = PlaceholderFormClassificationService()
    val recognitionService: IFormRecognitionService = PlaceholderFormRecognitionService()
    val extractionService: IFieldExtractionService = PlaceholderFieldExtractionService()
    val formProcessingService = OnDeviceFormProcessingService(context.applicationContext)

    val patientService = PatientService(patientRepository)
    val recordService = RecordService(medicalRecordRepository, patientService)
    val scanService = ScanService(classificationService, recognitionService, extractionService, formProcessingService)
    val scanWorkflowService = ScanWorkflowService(scanService)
}
