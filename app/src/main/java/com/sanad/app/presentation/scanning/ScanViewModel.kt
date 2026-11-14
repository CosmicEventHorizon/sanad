package com.sanad.app.presentation.scanning

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sanad.app.application.services.ScanWorkflowService
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class ScanUiState {
    data object Ready : ScanUiState()
    data object Processing : ScanUiState()
    data object Complete : ScanUiState()
    data class Error(val message: String) : ScanUiState()
}

class ScanViewModel(private val workflow: ScanWorkflowService) : ViewModel() {
    private val _state = MutableStateFlow<ScanUiState>(ScanUiState.Ready)
    val state = _state.asStateFlow()

    fun process(uri: String) {
        if (_state.value == ScanUiState.Processing) return
        viewModelScope.launch {
            _state.value = ScanUiState.Processing
            _state.value = runCatching { workflow.process(uri) }
                .fold(onSuccess = { ScanUiState.Complete }, onFailure = { ScanUiState.Error(it.message.orEmpty()) })
        }
    }

    fun reset() { _state.value = ScanUiState.Ready }
}
