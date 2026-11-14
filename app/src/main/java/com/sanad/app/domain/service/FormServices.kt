package com.sanad.app.domain.service

import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.domain.model.RecognizedField

interface IFormClassificationService {
    suspend fun classify(imageUri: String): String
}

interface IFormRecognitionService {
    suspend fun recognizeText(imageUri: String): String
}

interface IFieldExtractionService {
    suspend fun extract(rawText: String, formType: String): List<RecognizedField>
}

/** Processes a captured form image end-to-end using the form's configured assets. */
interface IFormProcessingService {
    suspend fun process(imageUri: String, formType: String): List<RecognizedField>
}

interface IConfidenceScoringService {
    fun needsReview(confidence: Float): Boolean
}

interface IRecordSyncService {
    suspend fun sync(records: List<MedicalRecord>): Result<Int>
}
