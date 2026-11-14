package com.sanad.app.presentation.records

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.CloudOff
import androidx.compose.material.icons.rounded.FolderOff
import androidx.compose.material.icons.rounded.Search
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.sanad.app.R
import com.sanad.app.domain.model.MedicalRecord
import com.sanad.app.presentation.components.ScreenHeader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

private enum class RecordFilter { ALL, MONTH, PENDING }

@Composable
fun RecordsScreen(viewModel: RecordsViewModel, onBack: () -> Unit, onRecord: (String) -> Unit) {
    val records by viewModel.records.collectAsStateWithLifecycle()
    val query by viewModel.query.collectAsStateWithLifecycle()
    var filter by remember { mutableStateOf(RecordFilter.ALL) }
    val currentMonth = remember { SimpleDateFormat("yyyy-MM", Locale.US).format(Date()) }
    val visible = records.filter { filter != RecordFilter.MONTH || it.visitDate.startsWith(currentMonth) }
    Column(Modifier.fillMaxSize()) {
        Surface(shadowElevation = 1.dp) { ScreenHeader(R.string.records_title, onBack) }
        Column(Modifier.padding(horizontal = 20.dp)) {
            Text(stringResource(R.string.records_subtitle), color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(16.dp))
            OutlinedTextField(
                value = query,
                onValueChange = viewModel::setQuery,
                leadingIcon = { Icon(Icons.Rounded.Search, null) },
                placeholder = { Text(stringResource(R.string.search_records)) },
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(16.dp),
                singleLine = true,
            )
            Row(Modifier.fillMaxWidth().padding(vertical = 12.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                FilterChip(filter == RecordFilter.ALL, { filter = RecordFilter.ALL }, { Text(stringResource(R.string.all_records)) })
                FilterChip(filter == RecordFilter.MONTH, { filter = RecordFilter.MONTH }, { Text(stringResource(R.string.this_month)) })
                FilterChip(filter == RecordFilter.PENDING, { filter = RecordFilter.PENDING }, { Text(stringResource(R.string.pending_sync)) })
            }
        }
        if (visible.isEmpty()) EmptyRecords(query.isNotBlank())
        else LazyColumn(Modifier.fillMaxSize(), contentPadding = androidx.compose.foundation.layout.PaddingValues(start = 20.dp, end = 20.dp, bottom = 28.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(visible, key = { it.id }) { RecordCard(it) { onRecord(it.id) } }
        }
    }
}

@Composable
private fun RecordCard(record: MedicalRecord, onClick: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth().clickable(onClick = onClick), shape = RoundedCornerShape(18.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface), elevation = CardDefaults.cardElevation(1.dp)) {
        Row(Modifier.fillMaxWidth().padding(17.dp), verticalAlignment = Alignment.CenterVertically) {
            Surface(shape = CircleShape, color = MaterialTheme.colorScheme.primaryContainer, modifier = Modifier.size(48.dp)) {
                Box(contentAlignment = Alignment.Center) { Text(record.patientName.take(1).uppercase(), fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.primary) }
            }
            Column(Modifier.weight(1f).padding(horizontal = 13.dp)) {
                Text(record.patientName, style = MaterialTheme.typography.titleMedium, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text("${record.patientId}  •  ${record.visitDate}", style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Rounded.CloudOff, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(14.dp))
                    Text(stringResource(R.string.sync_pending), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.primary, modifier = Modifier.padding(start = 5.dp))
                }
            }
            Icon(Icons.Rounded.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun EmptyRecords(hasQuery: Boolean) {
    Column(Modifier.fillMaxSize().padding(40.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.Rounded.FolderOff, null, tint = MaterialTheme.colorScheme.outline, modifier = Modifier.size(60.dp))
        Text(stringResource(if (hasQuery) R.string.no_results else R.string.no_records), style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(top = 18.dp))
        if (!hasQuery) Text(stringResource(R.string.no_records_body), color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
    }
}
