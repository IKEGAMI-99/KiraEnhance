package com.ikegami99.kiraenhance

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.ikegami99.kiraenhance.ui.home.HomeScreen

private const val HOME_ROUTE = "home"
private const val MODELS_ROUTE = "models"

@Composable
fun KiraEnhanceApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = HOME_ROUTE,
    ) {
        composable(HOME_ROUTE) {
            HomeScreen(
                onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
            )
        }
        composable(MODELS_ROUTE) {
            ModelManagerPlaceholder(
                onBack = { navController.popBackStack() },
            )
        }
    }
}

@Composable
private fun ModelManagerPlaceholder(onBack: () -> Unit) {
    Scaffold { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "モデル管理",
                style = MaterialTheme.typography.headlineMedium,
            )
            Text(
                text = "モデルのダウンロードと管理機能を準備中です。",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            TextButton(onClick = onBack) {
                Text("戻る")
            }
        }
    }
}
