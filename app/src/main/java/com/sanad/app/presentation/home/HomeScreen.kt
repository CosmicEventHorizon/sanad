package com.sanad.app.presentation.home

import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatDelegate
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material.icons.rounded.ChevronRight
import androidx.compose.material.icons.rounded.FolderOpen
import androidx.compose.material.icons.rounded.PhotoLibrary
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.core.os.LocaleListCompat
import com.sanad.app.R
import com.sanad.app.presentation.components.OfflineBadge
import com.sanad.app.presentation.components.SanadBrand

@Composable
fun HomeScreen(onScan: () -> Unit, onUpload: (String) -> Unit, onRecords: () -> Unit) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.GetContent()) { uri -> uri?.let { onUpload(it.toString()) } }
    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 20.dp).navigationBarsPadding(),
    ) {
        Spacer(Modifier.height(20.dp))
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.SpaceBetween) {
            SanadBrand()
            LanguageToggle()
        }
        Spacer(Modifier.height(28.dp))
        OfflineBadge()
        Spacer(Modifier.height(14.dp))
        Text(stringResource(R.string.home_greeting), style = MaterialTheme.typography.displaySmall, color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(8.dp))
        Text(stringResource(R.string.home_subtitle), style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(28.dp))
        PrimaryActionCard(Icons.Rounded.CameraAlt, R.string.scan_form, R.string.scan_form_hint, MaterialTheme.colorScheme.primary, onScan)
        Spacer(Modifier.height(14.dp))
        PrimaryActionCard(Icons.Rounded.PhotoLibrary, R.string.upload_image, R.string.upload_image_hint, MaterialTheme.colorScheme.secondary) { picker.launch("image/*") }
        Spacer(Modifier.height(20.dp))
        SecondaryActionCard(onRecords)
        Spacer(Modifier.height(32.dp))
        Text(stringResource(R.string.day1_footer), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.align(Alignment.CenterHorizontally))
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun LanguageToggle() {
    val current = AppCompatDelegate.getApplicationLocales().get(0)?.language ?: java.util.Locale.getDefault().language
    Surface(shape = RoundedCornerShape(12.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(Modifier.padding(3.dp)) {
            listOf("en" to R.string.english, "fr" to R.string.french).forEach { (code, label) ->
                val active = current == code
                Surface(
                    modifier = Modifier.clickable { AppCompatDelegate.setApplicationLocales(LocaleListCompat.forLanguageTags(code)) },
                    color = if (active) MaterialTheme.colorScheme.surface else Color.Transparent,
                    shape = RoundedCornerShape(9.dp),
                    tonalElevation = if (active) 2.dp else 0.dp,
                ) { Text(stringResource(label), style = MaterialTheme.typography.labelLarge, color = if (active) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(horizontal = 10.dp, vertical = 7.dp)) }
            }
        }
    }
}

@Composable
private fun PrimaryActionCard(icon: ImageVector, title: Int, subtitle: Int, color: Color, onClick: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth().height(112.dp).clickable(onClick = onClick),
        shape = RoundedCornerShape(22.dp),
        colors = CardDefaults.cardColors(containerColor = color),
    ) {
        Row(Modifier.fillMaxSize().padding(20.dp), verticalAlignment = Alignment.CenterVertically) {
            Surface(shape = RoundedCornerShape(16.dp), color = Color.White.copy(alpha = .16f)) { Icon(icon, null, tint = Color.White, modifier = Modifier.padding(14.dp).size(28.dp)) }
            Column(Modifier.weight(1f).padding(horizontal = 16.dp)) {
                Text(stringResource(title), style = MaterialTheme.typography.titleLarge, color = Color.White)
                Spacer(Modifier.height(3.dp))
                Text(stringResource(subtitle), style = MaterialTheme.typography.bodyMedium, color = Color.White.copy(alpha = .82f))
            }
            Icon(Icons.Rounded.ChevronRight, null, tint = Color.White, modifier = Modifier.size(28.dp))
        }
    }
}

@Composable
private fun SecondaryActionCard(onClick: () -> Unit) {
    Card(modifier = Modifier.fillMaxWidth().height(92.dp).clickable(onClick = onClick), shape = RoundedCornerShape(20.dp), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface), elevation = CardDefaults.cardElevation(1.dp)) {
        Row(Modifier.fillMaxSize().padding(18.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(52.dp), contentAlignment = Alignment.Center) { Icon(Icons.Rounded.FolderOpen, null, tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(30.dp)) }
            Column(Modifier.weight(1f).padding(horizontal = 12.dp)) {
                Text(stringResource(R.string.saved_records), style = MaterialTheme.typography.titleMedium)
                Text(stringResource(R.string.saved_records_hint), style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            Icon(Icons.Rounded.ChevronRight, null, tint = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}
