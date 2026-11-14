package com.sanad.app.presentation.records

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.sanad.app.application.services.RecordService
import com.sanad.app.domain.model.MedicalRecord
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@OptIn(ExperimentalCoroutinesApi::class)
class RecordsViewModel(private val service: RecordService) : ViewModel() {
    private val _query = MutableStateFlow("")
    val query = _query.asStateFlow()
    val records = _query.flatMapLatest(service::observeRecords)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())
    private val _selected = MutableStateFlow<MedicalRecord?>(null)
    val selected = _selected.asStateFlow()

    fun setQuery(value: String) { _query.value = value }
    fun load(id: String) { viewModelScope.launch { _selected.value = service.get(id) } }
}
