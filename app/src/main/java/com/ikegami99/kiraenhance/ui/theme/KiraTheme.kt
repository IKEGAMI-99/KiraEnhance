package com.ikegami99.kiraenhance.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val DarkColors = darkColorScheme(
    primary = Color(0xFFE9B8FF),
    onPrimary = Color(0xFF35133F),
    primaryContainer = Color(0xFF51245F),
    onPrimaryContainer = Color(0xFFF8D8FF),
    secondary = Color(0xFFFFB7D5),
    onSecondary = Color(0xFF4D1730),
    tertiary = Color(0xFFC9C1FF),
    background = Color(0xFF120F16),
    onBackground = Color(0xFFF3EDF5),
    surface = Color(0xFF1B171F),
    onSurface = Color(0xFFF3EDF5),
    surfaceVariant = Color(0xFF2A2330),
    onSurfaceVariant = Color(0xFFD8CADC),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF7A438A),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFF8D8FF),
    onPrimaryContainer = Color(0xFF2E0C38),
    secondary = Color(0xFF9A4169),
    onSecondary = Color.White,
    tertiary = Color(0xFF5D5598),
    background = Color(0xFFFFF8FF),
    onBackground = Color(0xFF201A22),
    surface = Color(0xFFFFF8FF),
    onSurface = Color(0xFF201A22),
    surfaceVariant = Color(0xFFF0E5F1),
    onSurfaceVariant = Color(0xFF514651),
)

private val KiraShapes = Shapes(
    extraSmall = RoundedCornerShape(10.dp),
    small = RoundedCornerShape(14.dp),
    medium = RoundedCornerShape(20.dp),
    large = RoundedCornerShape(26.dp),
    extraLarge = RoundedCornerShape(32.dp),
)

@Composable
fun KiraEnhanceTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        typography = Typography(),
        shapes = KiraShapes,
        content = content,
    )
}
