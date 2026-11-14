package com.sanad.app.presentation.review

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.rounded.Wysiwyg
import androidx.compose.material.icons.rounded.Person
import androidx.compose.material.icons.rounded.WarningAmber
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import androidx.annotation.StringRes
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sanad.app.R
import com.sanad.app.application.dto.RecordDraft
import com.sanad.app.presentation.components.ScreenHeader
import com.sanad.app.presentation.components.SectionTitle
import com.sanad.app.presentation.theme.WarmAmber

@Composable
fun ReviewScreen(viewModel: ReviewViewModel, onBack: () -> Unit, onSaved: (String) -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(state.savedRecordId) { state.savedRecordId?.let(onSaved) }
    val draft = state.draft
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background).imePadding()) {
        Surface(shadowElevation = 1.dp) { ScreenHeader(R.string.review_title, onBack) }
        if (draft == null) {
            Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
                Text(stringResource(R.string.processing_failed))
            }
        } else {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 20.dp)) {
                Spacer(Modifier.height(20.dp))
                Text(stringResource(R.string.review_subtitle), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(24.dp))
                if (draft.formFields.isNotEmpty()) {
                    SectionTitle(stringResource(R.string.form_section_general_information), Icons.Rounded.Person)
                    pregnancyKeys.take(7).forEach { key ->
                        ReviewField(key, draft.formFields[key].orEmpty(), pregnancyLabel(key), draft, viewModel::update)
                    }
                    Spacer(Modifier.height(12.dp))
                    SectionTitle(stringResource(R.string.form_section_pregnancy_risk), Icons.AutoMirrored.Rounded.Wysiwyg)
                    pregnancyKeys.drop(7).forEach { key ->
                        ReviewField(key, draft.formFields[key].orEmpty(), pregnancyLabel(key), draft, viewModel::update)
                    }
                } else {
                    SectionTitle(stringResource(R.string.patient_details), Icons.Rounded.Person)
                    ReviewField("patientId", draft.patientId, R.string.patient_id, draft, viewModel::update)
                    ReviewField("patientName", draft.patientName, R.string.patient_name, draft, viewModel::update)
                    ReviewField("dateOfBirth", draft.dateOfBirth, R.string.date_of_birth, draft, viewModel::update)
                    ReviewField("phone", draft.phone, R.string.phone, draft, viewModel::update, KeyboardType.Phone)
                    ReviewField("village", draft.village, R.string.village, draft, viewModel::update)
                    Spacer(Modifier.height(12.dp))
                    SectionTitle(stringResource(R.string.visit_details), Icons.AutoMirrored.Rounded.Wysiwyg)
                    ReviewField("visitDate", draft.visitDate, R.string.visit_date, draft, viewModel::update)
                    ReviewField("gestationalAgeWeeks", draft.gestationalAgeWeeks, R.string.gestational_age, draft, viewModel::update, KeyboardType.Number)
                    ReviewField("bloodPressure", draft.bloodPressure, R.string.blood_pressure, draft, viewModel::update)
                    ReviewField("weightKg", draft.weightKg, R.string.weight, draft, viewModel::update, KeyboardType.Decimal)
                    ReviewField("notes", draft.notes, R.string.notes, draft, viewModel::update, minLines = 3)
                }
                state.error?.let { Text(stringResource(R.string.required_fields), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodyMedium) }
                Spacer(Modifier.height(24.dp))
            }
            Surface(shadowElevation = 8.dp) {
                Button(
                    onClick = viewModel::save,
                    enabled = !state.saving,
                    modifier = Modifier.fillMaxWidth().padding(20.dp).height(56.dp).navigationBarsPadding(),
                    shape = RoundedCornerShape(16.dp),
                ) {
                    if (state.saving) CircularProgressIndicator(Modifier.height(22.dp), strokeWidth = 2.dp, color = Color.White)
                    else Text(stringResource(R.string.save_record))
                }
            }
        }
    }
}

private val pregnancyKeys = listOf(
    "form_number", "region", "province", "health_facility_name", "health_facility_type",
    "facility_mobility", "patient_full_name", "high_risk_pregnancy", "risk_factors", "other_risk_factors",
)

@StringRes
private fun pregnancyLabel(key: String): Int = when (key) {
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
private fun ReviewField(
    key: String,
    value: String,
    label: Int,
    draft: RecordDraft,
    onValue: (String, String) -> Unit,
    keyboardType: KeyboardType = KeyboardType.Text,
    minLines: Int = 1,
) {
    val flagged = key in draft.lowConfidenceKeys
    Column(Modifier.fillMaxWidth().padding(bottom = 14.dp)) {
        OutlinedTextField(
            value = value,
            onValueChange = { onValue(key, it) },
            label = { Text(stringResource(label)) },
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(14.dp),
            minLines = minLines,
            keyboardOptions = KeyboardOptions(keyboardType = keyboardType),
            colors = if (flagged) OutlinedTextFieldDefaults.colors(
                focusedBorderColor = WarmAmber,
                unfocusedBorderColor = WarmAmber,
                focusedLabelColor = MaterialTheme.colorScheme.onSurface,
            ) else OutlinedTextFieldDefaults.colors(),
        )
        if (flagged) {
            Row(Modifier.padding(start = 4.dp, top = 5.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(5.dp)) {
                Icon(Icons.Rounded.WarningAmber, null, tint = WarmAmber, modifier = Modifier.height(16.dp))
                Text(stringResource(R.string.needs_review), style = MaterialTheme.typography.labelLarge, color = Color(0xFF8B5B00))
            }
        }
    }
}
