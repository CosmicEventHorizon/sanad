package com.sanad.app.application.services

import com.sanad.app.domain.model.ProcessedForm
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

class ScanWorkflowService(private val scanService: ScanService) {
    private val _latestForm = MutableStateFlow<ProcessedForm?>(null)
    val latestForm = _latestForm.asStateFlow()

    suspend fun process(imageUri: String): ProcessedForm = scanService.process(imageUri).also { _latestForm.value = it }
    fun clear() { _latestForm.value = null }
}
