package com.sanad.app.application.dto

import com.sanad.app.domain.model.ProcessedForm
import com.sanad.app.domain.model.RecognizedField
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RecordDraftTest {
    @Test
    fun `pregnancy form preserves stable form fields and review flags`() {
        val draft = RecordDraft.from(
            ProcessedForm(
                formType = "pregnancy_monitoring",
                sourceImageUri = "file://scan.jpg",
                fields = listOf(
                    RecognizedField("form_number", "S1-42", 0.96f),
                    RecognizedField("patient_full_name", "Amina Diallo", 0.61f),
                    RecognizedField("risk_factors", "anemia,hypertension", 0.9f),
                ),
            )
        )

        assertEquals("S1-42", draft.patientId)
        assertEquals("Amina Diallo", draft.patientName)
        assertEquals("anemia,hypertension", draft.formFields["risk_factors"])
        assertTrue("patient_full_name" in draft.lowConfidenceKeys)
    }
}
