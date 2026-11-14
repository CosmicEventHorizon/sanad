package com.sanad.app.presentation.records

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Wysiwyg
import androidx.compose.material.icons.rounded.CloudOff
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sanad.app.R
import com.sanad.app.presentation.components.ScreenHeader
import com.sanad.app.presentation.components.SectionTitle

@Composable
fun RecordDetailScreen(id: String, viewModel: RecordsViewModel, onBack: () -> Unit) {
    val record by viewModel.selected.collectAsStateWithLifecycle()
    LaunchedEffect(id) { viewModel.load(id) }
    Column(Modifier.fillMaxSize()) {
        Surface(shadowElevation = 1.dp) { ScreenHeader(R.string.record_details, onBack) }
        record?.let { item ->
            Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp)) {
                Card(Modifier.fillMaxWidth(), shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
                    Column(Modifier.padding(20.dp)) {
                        Text(item.patientName, style = MaterialTheme.typography.headlineMedium, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        Text(item.patientId, style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onPrimaryContainer)
                        Row(Modifier.padding(top = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Rounded.CloudOff, null, tint = MaterialTheme.colorScheme.primary)
                            Text(stringResource(R.string.sync_pending), modifier = Modifier.padding(start = 8.dp), color = MaterialTheme.colorScheme.primary)
                        }
                    }
                }
                Spacer(Modifier.height(24.dp))
                if (item.formData.isNotEmpty()) {
                    SectionTitle(stringResource(R.string.form_section_general_information), Icons.Rounded.Person)
                    detailKeys.take(7).forEach { key ->
                        DetailRow(stringResource(detailLabel(key)), item.formData[key].orEmpty())
                    }
                    Spacer(Modifier.height(18.dp))
                    SectionTitle(stringResource(R.string.form_section_pregnancy_risk), Icons.AutoMirrored.Rounded.Wysiwyg)
                    detailKeys.drop(7).forEach { key ->
                        DetailRow(stringResource(detailLabel(key)), item.formData[key].orEmpty())
                    }
                } else {
                    SectionTitle(stringResource(R.string.patient_details), Icons.Rounded.Person)
                    DetailRow(stringResource(R.string.patient_id), item.patientId)
                    DetailRow(stringResource(R.string.patient_name), item.patientName)
                    Spacer(Modifier.height(18.dp))
                    SectionTitle(stringResource(R.string.visit_details), Icons.AutoMirrored.Rounded.Wysiwyg)
                    DetailRow(stringResource(R.string.visit_date), item.visitDate)
                    DetailRow(stringResource(R.string.gestational_age), item.gestationalAgeWeeks)
                    DetailRow(stringResource(R.string.blood_pressure), item.bloodPressure)
                    DetailRow(stringResource(R.string.weight), item.weightKg)
                    DetailRow(stringResource(R.string.form_type), item.formType)
                    DetailRow(stringResource(R.string.notes), item.notes)
                }
            }
        }
    }
}

private val detailKeys = listOf(
    "form_number", "region", "province", "health_facility_name", "health_facility_type",
    "facility_mobility", "patient_full_name", "high_risk_pregnancy", "risk_factors", "other_risk_factors",
)

private fun detailLabel(key: String): Int = when (key) {
    "form_number" -> R.string.form_field_form_number
    "region" -> R.string.form_field_region
    "province" -> R.string.form_field_province
    "health_facility_name" -> R.string.form_field_health_facility_name
    "health_facility_type" -> R.string.form_field_health_facility_type
    "facility_mobility" -> R.string.form_field_facility_mobility
    "patient_full_name" -> R.string.form_field_patient_full_name
    "high_risk_pregnancy" -> R.string.form_field_high_risk_pregnancy
    "risk_factors" -> R.string.form_field_risk_factors
    else -> R.string.form_field_other_risk_factors
}

@Composable
private fun DetailRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth().padding(vertical = 10.dp), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.weight(.48f))
        Text(value.ifBlank { stringResource(R.string.unknown) }, style = MaterialTheme.typography.bodyLarge, modifier = Modifier.weight(.52f))
    }
}
