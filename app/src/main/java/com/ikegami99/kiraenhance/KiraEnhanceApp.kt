package com.ikegami99.kiraenhance

import android.content.Intent
import android.net.Uri
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.ikegami99.kiraenhance.ui.home.HomeScreen
import com.ikegami99.kiraenhance.ui.models.ModelManagerScreen
import com.ikegami99.kiraenhance.ui.models.ModelManagerViewModel
import com.ikegami99.kiraenhance.ui.models.ModelManagerViewModelFactory

private const val HOME_ROUTE = "home"
private const val MODELS_ROUTE = "models"

@Composable
fun KiraEnhanceApp() {
    val navController = rememberNavController()
    val context = LocalContext.current

    NavHost(
        navController = navController,
        startDestination = HOME_ROUTE,
    ) {
        composable(HOME_ROUTE) {
            HomeScreen(
                onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
                onOpenSmokeTest = {},
            )
        }
        composable(MODELS_ROUTE) {
            val factory = remember(context.applicationContext) {
                ModelManagerViewModelFactory(context.applicationContext)
            }
            val modelManagerViewModel: ModelManagerViewModel = viewModel(factory = factory)
            val state by modelManagerViewModel.state.collectAsStateWithLifecycle()

            ModelManagerScreen(
                state = state,
                onBack = { navController.popBackStack() },
                onDownload = modelManagerViewModel::download,
                onDelete = modelManagerViewModel::delete,
                onWifiOnlyChanged = modelManagerViewModel::setWifiOnly,
                onOpenLicense = { url ->
                    runCatching {
                        context.startActivity(
                            Intent(Intent.ACTION_VIEW, Uri.parse(url)),
                        )
                    }
                },
            )
        }
    }
}
