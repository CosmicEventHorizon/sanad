package com.sanad.app.presentation.review

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sanad.app.application.dto.RecordDraft
import com.sanad.app.application.services.RecordService
import com.sanad.app.application.services.ScanWorkflowService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class ReviewUiState(
    val draft: RecordDraft? = null,
    val saving: Boolean = false,
    val savedRecordId: String? = null,
    val error: String? = null,
)

class ReviewViewModel(
    workflow: ScanWorkflowService,
    private val recordService: RecordService,
) : ViewModel() {
    private val _state = MutableStateFlow(ReviewUiState(draft = workflow.latestForm.value?.let(RecordDraft::from)))
    val state = _state.asStateFlow()

    fun update(key: String, value: String) {
        val draft = _state.value.draft ?: return
        if (key in draft.formFields) {
            _state.value = _state.value.copy(
                draft = draft.copy(
                    formFields = draft.formFields + (key to value),
                    patientId = if (key == "form_number") value else draft.patientId,
                    patientName = if (key == "patient_full_name") value else draft.patientName,
                    lowConfidenceKeys = draft.lowConfidenceKeys - key,
                ),
                error = null,
            )
            return
        }
        val updated = when (key) {
            "patientId" -> draft.copy(patientId = value)
            "patientName" -> draft.copy(patientName = value)
            "dateOfBirth" -> draft.copy(dateOfBirth = value)
            "phone" -> draft.copy(phone = value)
            "village" -> draft.copy(village = value)
            "visitDate" -> draft.copy(visitDate = value)
            "gestationalAgeWeeks" -> draft.copy(gestationalAgeWeeks = value)
            "bloodPressure" -> draft.copy(bloodPressure = value)
            "weightKg" -> draft.copy(weightKg = value)
            "notes" -> draft.copy(notes = value)
            else -> draft
        }.copy(lowConfidenceKeys = draft.lowConfidenceKeys - key)
        _state.value = _state.value.copy(draft = updated, error = null)
    }

    fun save() {
        val draft = _state.value.draft ?: return
        viewModelScope.launch {
            _state.value = _state.value.copy(saving = true, error = null)
            runCatching { recordService.saveVerified(draft) }
                .onSuccess { _state.value = _state.value.copy(saving = false, savedRecordId = it) }
                .onFailure { _state.value = _state.value.copy(saving = false, error = it.message) }
        }
    }
}
