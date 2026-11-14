package com.sanad.app.presentation

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.sanad.app.di.AppContainer
import com.sanad.app.presentation.records.RecordsViewModel
import com.sanad.app.presentation.review.ReviewViewModel
import com.sanad.app.presentation.scanning.ScanViewModel

class SanadViewModelFactory(private val container: AppContainer) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T = when {
        modelClass.isAssignableFrom(ScanViewModel::class.java) -> ScanViewModel(container.scanWorkflowService)
        modelClass.isAssignableFrom(ReviewViewModel::class.java) -> ReviewViewModel(container.scanWorkflowService, container.recordService)
        modelClass.isAssignableFrom(RecordsViewModel::class.java) -> RecordsViewModel(container.recordService)
        else -> error("Unknown ViewModel: ${modelClass.name}")
    } as T
}
