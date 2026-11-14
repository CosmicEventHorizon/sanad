package com.sanad.app

import android.os.Bundle
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.sanad.app.presentation.SanadViewModelFactory
import com.sanad.app.presentation.home.HomeScreen
import com.sanad.app.presentation.records.RecordDetailScreen
import com.sanad.app.presentation.records.RecordsScreen
import com.sanad.app.presentation.records.RecordsViewModel
import com.sanad.app.presentation.review.ReviewScreen
import com.sanad.app.presentation.review.ReviewViewModel
import com.sanad.app.presentation.scanning.CameraScreen
import com.sanad.app.presentation.scanning.ProcessingScreen
import com.sanad.app.presentation.scanning.ScanViewModel
import com.sanad.app.presentation.theme.SanadTheme

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val factory = SanadViewModelFactory((application as SanadApplication).container)
        setContent { SanadTheme { SanadNavigation(factory) } }
    }
}

@Composable
private fun SanadNavigation(factory: SanadViewModelFactory) {
    val nav = rememberNavController()
    var pendingImage by remember { mutableStateOf("") }
    Scaffold(modifier = Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing)) { padding ->
        NavHost(navController = nav, startDestination = "home", modifier = Modifier.padding(padding)) {
            composable("home") {
                HomeScreen(
                    onScan = { nav.navigate("camera") },
                    onUpload = { pendingImage = it; nav.navigate("processing") },
                    onRecords = { nav.navigate("records") },
                )
            }
            composable("camera") {
                CameraScreen(
                    onBack = { nav.popBackStack() },
                    onCaptured = { pendingImage = it; nav.navigate("processing") },
                )
            }
            composable("processing") {
                val vm: ScanViewModel = viewModel(factory = factory)
                ProcessingScreen(
                    imageUri = pendingImage,
                    viewModel = vm,
                    onComplete = { nav.navigate("review") { popUpTo("processing") { inclusive = true } } },
                    onBack = { nav.popBackStack() },
                )
            }
            composable("review") {
                val vm: ReviewViewModel = viewModel(factory = factory)
                ReviewScreen(
                    viewModel = vm,
                    onBack = { nav.popBackStack() },
                    onSaved = { id -> nav.navigate("record/$id") { popUpTo("home") } },
                )
            }
            composable("records") {
                val vm: RecordsViewModel = viewModel(factory = factory)
                RecordsScreen(vm, onBack = { nav.popBackStack() }, onRecord = { nav.navigate("record/$it") })
            }
            composable("record/{id}", arguments = listOf(navArgument("id") { type = NavType.StringType })) { entry ->
                val vm: RecordsViewModel = viewModel(factory = factory)
                RecordDetailScreen(entry.arguments?.getString("id").orEmpty(), vm) { nav.popBackStack() }
            }
        }
    }
}
