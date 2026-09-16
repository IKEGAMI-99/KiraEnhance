package com.ikegami99.kiraenhance

import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.ikegami99.kiraenhance.diagnostics.AppDiagnosticLogger
import com.ikegami99.kiraenhance.ui.enhance.EnhanceRoute
import com.ikegami99.kiraenhance.ui.home.HomeScreen
import com.ikegami99.kiraenhance.ui.models.ModelManagerScreen
import com.ikegami99.kiraenhance.ui.models.ModelManagerViewModel
import com.ikegami99.kiraenhance.ui.models.ModelManagerViewModelFactory
import com.ikegami99.kiraenhance.ui.smoke.NcnnSmokeRoute
import com.ikegami99.kiraenhance.ui.update.AppUpdateScreen
import com.ikegami99.kiraenhance.ui.update.AppUpdateViewModel
import com.ikegami99.kiraenhance.ui.update.AppUpdateViewModelFactory

private const val HOME_ROUTE = "home"
private const val ENHANCE_ROUTE = "enhance"
private const val MODELS_ROUTE = "models"
private const val SMOKE_ROUTE = "smoke"
private const val UPDATE_ROUTE = "update"

@Composable
fun KiraEnhanceApp() {
    val navController = rememberNavController()
    val context = LocalContext.current
    val logger = remember(context.applicationContext) {
        AppDiagnosticLogger.get(context.applicationContext)
    }
    val logSaveLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/plain"),
    ) { uri ->
        if (uri == null) {
            logger.log("LogExport", "save cancelled")
        } else {
            logger.exportTo(uri).fold(
                onSuccess = { verifiedBytes ->
                    logger.log(
                        "LogExport",
                        "save complete bytes=$verifiedBytes scheme=${uri.scheme ?: "unknown"}",
                    )
                    Toast.makeText(
                        context,
                        "ログを保存しました (${verifiedBytes} bytes)",
                        Toast.LENGTH_SHORT,
                    ).show()
                },
                onFailure = { error ->
                    logger.log(
                        "LogExport",
                        "save failed type=${error.javaClass.simpleName} message=${error.message ?: "no-message"}",
                    )
                    Toast.makeText(
                        context,
                        "ログの保存に失敗しました: ${error.message ?: "unknown error"}",
                        Toast.LENGTH_LONG,
                    ).show()
                },
            )
        }
    }

    LaunchedEffect(logger) {
        logger.logAppSession()
    }

    NavHost(
        navController = navController,
        startDestination = HOME_ROUTE,
    ) {
        composable(HOME_ROUTE) {
            HomeScreen(
                onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
                onOpenSmokeTest = { navController.navigate(SMOKE_ROUTE) },
                onOpenAppUpdate = { navController.navigate(UPDATE_ROUTE) },
                onSaveLogs = {
                    logger.log("LogExport", "save requested")
                    logSaveLauncher.launch(logger.exportFileName())
                },
                onStartEnhance = { navController.navigate(ENHANCE_ROUTE) },
            )
        }
        composable(ENHANCE_ROUTE) {
            EnhanceRoute(
                onBack = { navController.popBackStack() },
                onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
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
        composable(SMOKE_ROUTE) {
            NcnnSmokeRoute(
                onBack = { navController.popBackStack() },
                onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
            )
        }
        composable(UPDATE_ROUTE) {
            val factory = remember(context.applicationContext) {
                AppUpdateViewModelFactory(context.applicationContext)
            }
            val updateViewModel: AppUpdateViewModel = viewModel(factory = factory)
            val state by updateViewModel.state.collectAsStateWithLifecycle()

            AppUpdateScreen(
                state = state,
                onBack = { navController.popBackStack() },
                onCheck = updateViewModel::checkForUpdates,
                onDownload = updateViewModel::download,
                onInstall = updateViewModel::install,
            )
        }
    }
}
