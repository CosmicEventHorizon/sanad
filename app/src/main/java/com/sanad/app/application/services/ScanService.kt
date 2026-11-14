package com.sanad.app.application.services

import com.sanad.app.domain.model.ProcessedForm
import com.sanad.app.domain.service.IFieldExtractionService
import com.sanad.app.domain.service.IFormClassificationService
import com.sanad.app.domain.service.IFormRecognitionService
import com.sanad.app.domain.service.IFormProcessingService

class ScanService(
    private val classificationService: IFormClassificationService,
    private val recognitionService: IFormRecognitionService,
    private val extractionService: IFieldExtractionService,
    private val formProcessingService: IFormProcessingService? = null,
) {
    suspend fun process(imageUri: String): ProcessedForm {
        require(imageUri.isNotBlank()) { "An image is required" }
        val formType = classificationService.classify(imageUri)
        formProcessingService?.let {
            return ProcessedForm(formType, imageUri, it.process(imageUri, formType))
        }
        val rawText = recognitionService.recognizeText(imageUri)
        return ProcessedForm(formType, imageUri, extractionService.extract(rawText, formType))
    }
}
