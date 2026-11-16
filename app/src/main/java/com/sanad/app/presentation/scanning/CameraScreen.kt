package com.sanad.app.presentation.scanning

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.CameraAlt
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.sanad.app.R
import com.sanad.app.presentation.components.ScreenHeader
import java.io.File

@Composable
fun CameraScreen(onBack: () -> Unit, onCaptured: (String) -> Unit) {
    val context = LocalContext.current
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) permission.launch(Manifest.permission.CAMERA) }
    Column(Modifier.fillMaxSize().background(Color(0xFF101B1C))) {
        Surface(color = MaterialTheme.colorScheme.surface) { ScreenHeader(R.string.scan_title, onBack) }
        if (granted) CameraContent(onCaptured) else PermissionContent { permission.launch(Manifest.permission.CAMERA) }
    }
}

@Composable
private fun CameraContent(onCaptured: (String) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var capture by remember { mutableStateOf<ImageCapture?>(null) }
    var captureError by remember { mutableStateOf(false) }
    Box(Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx ->
                PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FILL_CENTER }
            },
            modifier = Modifier.fillMaxSize(),
            update = { previewView ->
                val providerFuture = ProcessCameraProvider.getInstance(context)
                providerFuture.addListener({
                    val provider = providerFuture.get()
                    val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
                    capture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY).build()
                    runCatching {
                        provider.unbindAll()
                        provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                    }
                }, ContextCompat.getMainExecutor(context))
            },
        )
        Box(Modifier.align(Alignment.Center).fillMaxWidth(.88f).height(430.dp).border(3.dp, Color.White.copy(alpha = .9f), RoundedCornerShape(18.dp)))
        Surface(Modifier.align(Alignment.TopCenter).padding(top = 24.dp), color = Color.Black.copy(alpha = .55f), shape = CircleShape) {
            Text(stringResource(R.string.scan_instruction), color = Color.White, modifier = Modifier.padding(horizontal = 16.dp, vertical = 10.dp), style = MaterialTheme.typography.bodyMedium)
        }
        IconButton(
            onClick = {
                captureError = false
                val file = File(context.cacheDir, "scan-${System.currentTimeMillis()}.jpg")
                capture?.takePicture(ImageCapture.OutputFileOptions.Builder(file).build(), ContextCompat.getMainExecutor(context), object : ImageCapture.OnImageSavedCallback {
                    override fun onImageSaved(result: ImageCapture.OutputFileResults) = onCaptured(Uri.fromFile(file).toString())
                    override fun onError(exception: ImageCaptureException) { captureError = true }
                })
            },
            enabled = capture != null,
            modifier = Modifier.align(Alignment.BottomCenter).padding(bottom = 30.dp).size(76.dp).background(Color.White, CircleShape),
        ) { Icon(Icons.Rounded.CameraAlt, stringResource(R.string.capture), tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(34.dp)) }
        if (captureError) {
            Surface(Modifier.align(Alignment.BottomCenter).padding(bottom = 118.dp), color = MaterialTheme.colorScheme.errorContainer, shape = RoundedCornerShape(12.dp)) {
                Text(stringResource(R.string.camera_capture_failed), color = MaterialTheme.colorScheme.onErrorContainer, modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp))
            }
        }
    }
}

@Composable
private fun PermissionContent(onRequest: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(32.dp), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
        Icon(Icons.Rounded.CameraAlt, null, tint = Color.White, modifier = Modifier.size(58.dp))
        Text(stringResource(R.string.camera_permission_title), style = MaterialTheme.typography.headlineMedium, color = Color.White, modifier = Modifier.padding(top = 20.dp))
        Text(stringResource(R.string.camera_permission_body), style = MaterialTheme.typography.bodyLarge, color = Color.White.copy(alpha = .75f), modifier = Modifier.padding(top = 10.dp, bottom = 24.dp))
        Button(onClick = onRequest, colors = ButtonDefaults.buttonColors(containerColor = Color.White, contentColor = MaterialTheme.colorScheme.primary)) { Text(stringResource(R.string.allow_camera)) }
    }
}
