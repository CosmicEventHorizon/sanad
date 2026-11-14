package com.sanad.app.infrastructure.ocr

import com.sanad.app.domain.model.RecognizedField
import com.sanad.app.domain.service.IConfidenceScoringService
import com.sanad.app.domain.service.IFieldExtractionService
import com.sanad.app.domain.service.IFormClassificationService
import com.sanad.app.domain.service.IFormRecognitionService
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PlaceholderFormClassificationService : IFormClassificationService {
    override suspend fun classify(imageUri: String): String = "pregnancy_monitoring"
}

class PlaceholderFormRecognitionService : IFormRecognitionService {
    override suspend fun recognizeText(imageUri: String): String {
        delay(650)
        return "placeholder-recognized-text"
    }
}

class PlaceholderFieldExtractionService : IFieldExtractionService {
    override suspend fun extract(rawText: String, formType: String): List<RecognizedField> {
        val today = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        return listOf(
            RecognizedField("patientId", "SAN-1042", 0.96f),
            RecognizedField("patientName", "Amina Diallo", 0.93f),
            RecognizedField("dateOfBirth", "1996-08-14", 0.88f),
            RecognizedField("phone", "+221 77 000 21 45", 0.67f),
            RecognizedField("village", "Ndioum", 0.72f),
            RecognizedField("visitDate", today, 0.98f),
            RecognizedField("gestationalAgeWeeks", "28", 0.91f),
            RecognizedField("bloodPressure", "120/80", 0.62f),
            RecognizedField("weightKg", "64", 0.86f),
            RecognizedField("notes", "Routine prenatal visit", 0.74f),
        )
    }
}

class ConfidenceScoringService(private val threshold: Float = 0.8f) : IConfidenceScoringService {
    override fun needsReview(confidence: Float) = confidence < threshold
}
