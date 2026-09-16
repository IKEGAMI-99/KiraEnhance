package com.ikegami99.kiraenhance.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

private val DarkColors = darkColorScheme(
    primary = Color(0xFFE7B7FF),
    onPrimary = Color(0xFF35203D),
    primaryContainer = Color(0xFF563866),
    onPrimaryContainer = Color(0xFFF8E9FF),
    secondary = Color(0xFFFFB8D7),
    onSecondary = Color(0xFF4A2031),
    tertiary = Color(0xFFBFD1FF),
    background = Color(0xFF121015),
    onBackground = Color(0xFFF0EAF3),
    surface = Color(0xFF1A171E),
    onSurface = Color(0xFFF0EAF3),
    surfaceVariant = Color(0xFF29242E),
    onSurfaceVariant = Color(0xFFD3C7D8),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF765085),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFF1D8FF),
    onPrimaryContainer = Color(0xFF2D1736),
    secondary = Color(0xFF8D4967),
    onSecondary = Color.White,
    tertiary = Color(0xFF4F638E),
    background = Color(0xFFFFF8FF),
    onBackground = Color(0xFF211D23),
    surface = Color(0xFFFFF8FF),
    onSurface = Color(0xFF211D23),
    surfaceVariant = Color(0xFFF0E8F1),
    onSurfaceVariant = Color(0xFF514A54),
)

private val KiraShapes = Shapes(
    extraSmall = RoundedCornerShape(10.dp),
    small = RoundedCornerShape(14.dp),
    medium = RoundedCornerShape(20.dp),
    large = RoundedCornerShape(28.dp),
    extraLarge = RoundedCornerShape(36.dp),
)

@Composable
fun KiraEnhanceTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        shapes = KiraShapes,
        content = content,
    )
}
