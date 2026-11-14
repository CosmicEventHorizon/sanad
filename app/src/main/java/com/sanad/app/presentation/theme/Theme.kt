package com.sanad.app.presentation.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

val SanadTeal = Color(0xFF0B7367)
val SanadTealBright = Color(0xFF14A090)
val DeepIndigo = Color(0xFF243B67)
val WarmAmber = Color(0xFFF2A93B)
val Ink = Color(0xFF172426)
val Mist = Color(0xFFF3F8F7)
val Line = Color(0xFFD9E5E3)

private val LightColors = lightColorScheme(
    primary = SanadTeal,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFD5F1EC),
    onPrimaryContainer = Color(0xFF074D45),
    secondary = DeepIndigo,
    secondaryContainer = Color(0xFFDDE6FB),
    tertiary = WarmAmber,
    background = Color(0xFFF8FBFA),
    surface = Color.White,
    surfaceVariant = Mist,
    outline = Line,
    error = Color(0xFFBA1A1A),
)

private val DarkColors = darkColorScheme(primary = Color(0xFF7DDBCD), secondary = Color(0xFFB7C8F4))

private val SanadTypography = Typography(
    displaySmall = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 32.sp, lineHeight = 38.sp),
    headlineMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.Bold, fontSize = 25.sp, lineHeight = 31.sp),
    titleLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 20.sp),
    titleMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 16.sp),
    bodyLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 16.sp, lineHeight = 24.sp),
    bodyMedium = TextStyle(fontFamily = FontFamily.SansSerif, fontSize = 14.sp, lineHeight = 20.sp),
    labelLarge = TextStyle(fontFamily = FontFamily.SansSerif, fontWeight = FontWeight.SemiBold, fontSize = 14.sp),
)

@Composable
fun SanadTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = LightColors, typography = SanadTypography, content = content)
}
