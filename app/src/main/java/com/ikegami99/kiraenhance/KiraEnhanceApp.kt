package com.ikegami99.kiraenhance

import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.ikegami99.kiraenhance.diagnostics.AppDiagnosticLogger
import com.ikegami99.kiraenhance.diagnostics.MnnRuntimeDiagnostics
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaNativeBridge
import com.ikegami99.kiraenhance.model.EnhancementMode
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
private const val ENHANCE_ROUTE = "enhance/{mode}"
private const val ENHANCE_ROUTE_PREFIX = "enhance"
private const val MODE_ARGUMENT = "mode"
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
        logger.log(
            "MNN",
            MnnRuntimeDiagnostics.capture {
                MnnPisaNativeBridge.runtimeInfo()
            },
        )
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
                onStartEnhance = { mode ->
                    enhancementModeRouteToken(mode)?.let { token ->
                        navController.navigate("$ENHANCE_ROUTE_PREFIX/$token")
                    }
                },
            )
        }
        composable(
            route = ENHANCE_ROUTE,
            arguments = listOf(
                navArgument(MODE_ARGUMENT) { type = NavType.StringType },
            ),
        ) { backStackEntry ->
            val token = backStackEntry.arguments?.getString(MODE_ARGUMENT)
            val mode = enhancementModeFromRouteToken(token)
            if (mode == null) {
                LaunchedEffect(token) {
                    logger.log("Enhance", "invalid route mode=${token ?: "missing"}")
                    navController.popBackStack()
                }
            } else {
                EnhanceRoute(
                    mode = mode,
                    onBack = { navController.popBackStack() },
                    onOpenModelManager = { navController.navigate(MODELS_ROUTE) },
                )
            }
        }
        composable(MODELS_ROUTE) {
            val factory = remember(context.applicationContext) {
                ModelManagerViewModelFactory(context.applicationContext)
            }
            val modelManagerViewModel: ModelManagerViewModel = viewModel(factory = factory)
            val state by modelManagerViewModel.state.collectAsStateWithLifecycle()
            var pendingValidationImport by remember { mutableStateOf<String?>(null) }
            val validationImportLauncher = rememberLauncherForActivityResult(
                contract = ActivityResultContracts.OpenMultipleDocuments(),
            ) { uris ->
                val modelId = pendingValidationImport
                pendingValidationImport = null
                if (modelId != null) {
                    modelManagerViewModel.importValidationArtifacts(modelId, uris)
                }
            }

            ModelManagerScreen(
                state = state,
                onBack = { navController.popBackStack() },
                onDownload = modelManagerViewModel::download,
                onDelete = modelManagerViewModel::delete,
                onWifiOnlyChanged = modelManagerViewModel::setWifiOnly,
                onImportValidation = { modelId ->
                    pendingValidationImport = modelId
                    validationImportLauncher.launch(arrayOf("*/*"))
                },
                onProbeValidation = modelManagerViewModel::probeValidationModel,
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

private fun enhancementModeRouteToken(mode: EnhancementMode): String? = when (mode) {
    EnhancementMode.BALANCED -> "balanced"
    EnhancementMode.ULTRASHARP -> "ultrasharp"
    EnhancementMode.FIDELITY,
    EnhancementMode.DETAIL,
    -> null
}

private fun enhancementModeFromRouteToken(token: String?): EnhancementMode? = when (token) {
    "balanced" -> EnhancementMode.BALANCED
    "ultrasharp" -> EnhancementMode.ULTRASHARP
    else -> null
}
