package com.sanad.app.application.services

import com.sanad.app.domain.model.RecognizedField
import com.sanad.app.domain.service.IFieldExtractionService
import com.sanad.app.domain.service.IFormClassificationService
import com.sanad.app.domain.service.IFormRecognitionService
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class ScanServiceTest {
    @Test
    fun `process orchestrates classification recognition and extraction`() = runTest {
        val service = ScanService(
            classificationService = object : IFormClassificationService { override suspend fun classify(imageUri: String) = "ANC" },
            recognitionService = object : IFormRecognitionService { override suspend fun recognizeText(imageUri: String) = "raw" },
            extractionService = object : IFieldExtractionService {
                override suspend fun extract(rawText: String, formType: String) = listOf(RecognizedField("patientId", "P-1", .9f))
            },
        )
        val result = service.process("file://form.jpg")
        assertEquals("ANC", result.formType)
        assertEquals("P-1", result.fields.single().value)
    }

    @Test(expected = IllegalArgumentException::class)
    fun `process rejects blank image`() = runTest {
        ScanService(
            object : IFormClassificationService { override suspend fun classify(imageUri: String) = "" },
            object : IFormRecognitionService { override suspend fun recognizeText(imageUri: String) = "" },
            object : IFieldExtractionService { override suspend fun extract(rawText: String, formType: String) = emptyList<RecognizedField>() },
        ).process("")
    }
}
