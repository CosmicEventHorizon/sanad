package com.sanad.app.presentation.scanning

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.DocumentScanner
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
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

@Composable
fun ProcessingScreen(imageUri: String, viewModel: ScanViewModel, onComplete: () -> Unit, onBack: () -> Unit) {
    val state by viewModel.state.collectAsStateWithLifecycle()
    LaunchedEffect(imageUri) { viewModel.process(imageUri) }
    LaunchedEffect(state) { if (state == ScanUiState.Complete) onComplete() }
    Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        if (state is ScanUiState.Error) {
            Icon(Icons.Rounded.DocumentScanner, null, tint = MaterialTheme.colorScheme.error)
            Text(stringResource(R.string.processing_failed), style = MaterialTheme.typography.titleLarge, modifier = Modifier.padding(vertical = 18.dp))
            val message = (state as ScanUiState.Error).message
            if (message.isNotBlank()) {
                Text(message, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(bottom = 18.dp))
            }
            Button(onClick = { viewModel.reset(); viewModel.process(imageUri) }) { Text(stringResource(R.string.try_again)) }
        } else {
            CircularProgressIndicator()
            Text(stringResource(R.string.processing), style = MaterialTheme.typography.headlineMedium, modifier = Modifier.padding(top = 24.dp))
            Text(stringResource(R.string.processing_detail), color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
        }
    }
}
