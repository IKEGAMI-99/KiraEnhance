package com.ikegami99.kiraenhance.ui.enhance

import android.app.ActivityManager
import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import com.ikegami99.kiraenhance.diagnostics.AppDiagnosticLogger
import com.ikegami99.kiraenhance.image.EnhanceOutputNamer
import com.ikegami99.kiraenhance.image.EnhancedImageSaver
import com.ikegami99.kiraenhance.image.SaveResult
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ncnn.NcnnUpscaleEngine
import com.ikegami99.kiraenhance.inference.ncnn.UltraSharpModelRequestFactory
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.model.ModelManifestParser
import java.time.Instant
import java.time.ZoneId
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun EnhanceRoute(
    onBack: () -> Unit,
    onOpenModelManager: () -> Unit,
) {
    val applicationContext = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    val logger = remember(applicationContext) { AppDiagnosticLogger.get(applicationContext) }
    val setup = remember(applicationContext) { createEnhanceSetup(applicationContext) }
    val processor = remember { EnhanceProcessor() }

    var state by remember(setup) {
        mutableStateOf(
            when {
                setup.model == null -> EnhanceUiState(
                    stage = EnhanceStage.ERROR,
                    status = setup.errorMessage ?: "UltraSharpモデル情報を読み込めません",
                )
                !setup.modelInstalled -> EnhanceUiState(
                    stage = EnhanceStage.ERROR,
                    status = "4x-UltraSharpモデルが未インストールです。モデル管理からダウンロードしてください。",
                )
                else -> EnhanceUiState()
            },
        )
    }
    var sourceBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var resultBitmap by remember { mutableStateOf<Bitmap?>(null) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri == null) {
            logger.log("Enhance", "image selection cancelled")
            return@rememberLauncherForActivityResult
        }
        scope.launch {
            state = state.copy(
                stage = EnhanceStage.SELECTING,
                status = "画像を読み込み中...",
                savedLocation = null,
            )
            val decoded = withContext(Dispatchers.IO) {
                runCatching { decodeFullImage(applicationContext, uri) }
            }
            decoded.onSuccess { bitmap ->
                sourceBitmap?.takeIf { it !== bitmap && !it.isRecycled }?.recycle()
                resultBitmap?.takeIf { !it.isRecycled }?.recycle()
                sourceBitmap = bitmap
                resultBitmap = null
                state = EnhanceUiReducer.reduce(
                    EnhanceUiState(),
                    EnhanceEvent.ImageReady(bitmap.width, bitmap.height),
                )
                logger.log(
                    "Enhance",
                    "image ready width=${bitmap.width} height=${bitmap.height} target=${bitmap.width * 4}x${bitmap.height * 4}",
                )
            }.onFailure { error ->
                state = EnhanceUiReducer.reduce(
                    state,
                    EnhanceEvent.Failed("画像を読み込めません: ${error.message ?: "unknown error"}"),
                )
                logger.log(
                    "Enhance",
                    "image decode failed type=${error.javaClass.simpleName} message=${error.message ?: "no-message"}",
                )
            }
        }
    }

    val startEnhance: () -> Unit = {
        val model = setup.model
        val source = sourceBitmap
        if (model == null || !setup.modelInstalled) {
            state = EnhanceUiReducer.reduce(
                state,
                EnhanceEvent.Failed("UltraSharpモデルが利用できません。モデル管理を確認してください。"),
            )
        } else if (source == null) {
            state = EnhanceUiReducer.reduce(
                state,
                EnhanceEvent.Failed("先に画像を選択してください。"),
            )
        } else if (state.stage != EnhanceStage.PROCESSING) {
            val binding = EnhanceEngineBinding(
                engine = NcnnUpscaleEngine(
                    totalRamMbProvider = { totalRamMb(applicationContext) },
                ),
                requestFactory = UltraSharpModelRequestFactory,
                outputScale = 4,
                saveModeName = "UltraSharp",
            )
            resultBitmap?.takeIf { !it.isRecycled }?.recycle()
            resultBitmap = null
            state = EnhanceUiReducer.reduce(state, EnhanceEvent.ProcessingStarted)
            logger.log(
                "Enhance",
                "start model=${model.id} version=${model.version} input=${source.width}x${source.height} scale=${binding.outputScale}",
            )
            scope.launch {
                val result = processor.run(
                    bitmap = source,
                    model = model,
                    binding = binding,
                    artifactPath = { fileName ->
                        setup.store.artifactFile(
                            modelId = model.id,
                            version = model.version,
                            fileName = fileName,
                        ).absolutePath
                    },
                    onProgress = { progress ->
                        state = EnhanceUiReducer.reduce(
                            state,
                            EnhanceEvent.Progress(
                                completedTiles = progress.completedTiles,
                                totalTiles = progress.totalTiles,
                                elapsedMs = progress.elapsedMs,
                            ),
                        )
                    },
                )

                when (result) {
                    is EnhanceProcessResult.Success -> {
                        resultBitmap = result.bitmap
                        state = EnhanceUiReducer.reduce(
                            state,
                            EnhanceEvent.ProcessingCompleted(result.elapsedMs),
                        )
                        logger.log(
                            "Enhance",
                            "success output=${result.bitmap.width}x${result.bitmap.height} usedGpu=${result.usedGpu} elapsedMs=${result.elapsedMs}",
                        )
                    }
                    is EnhanceProcessResult.Failed -> {
                        state = EnhanceUiReducer.reduce(
                            state,
                            if (result.code == EngineErrorCode.CANCELLED) {
                                EnhanceEvent.Cancelled
                            } else {
                                EnhanceEvent.Failed(result.message)
                            },
                        )
                        logger.log(
                            "Enhance",
                            "failed code=${result.code ?: "unknown"} message=${result.message}",
                        )
                    }
                }
            }
        }
    }

    val saveEnhanced: () -> Unit = {
        val bitmap = resultBitmap
        if (bitmap != null && !bitmap.isRecycled) {
            scope.launch {
                val fileName = EnhanceOutputNamer.png(
                    instant = Instant.now(),
                    zoneId = ZoneId.systemDefault(),
                    mode = "UltraSharp",
                    scale = 4,
                )
                logger.log("EnhanceSave", "start file=$fileName size=${bitmap.width}x${bitmap.height}")
                when (val result = EnhancedImageSaver.savePng(applicationContext, bitmap, fileName)) {
                    is SaveResult.Success -> {
                        state = EnhanceUiReducer.reduce(
                            state,
                            EnhanceEvent.Saved(result.location),
                        )
                        logger.log("EnhanceSave", "success location=${result.location}")
                    }
                    is SaveResult.Failed -> {
                        state = EnhanceUiReducer.reduce(
                            state,
                            EnhanceEvent.SaveFailed(result.message),
                        )
                        logger.log("EnhanceSave", "failed message=${result.message}")
                    }
                }
            }
        }
    }

    LaunchedEffect(setup.modelInstalled) {
        if (setup.modelInstalled && sourceBitmap == null) {
            imagePicker.launch("image/*")
        }
    }

    DisposableEffect(processor) {
        onDispose {
            processor.cancel()
            sourceBitmap?.takeIf { !it.isRecycled }?.recycle()
            resultBitmap?.takeIf { !it.isRecycled }?.recycle()
        }
    }

    val sourcePreview = remember(sourceBitmap) { sourceBitmap?.asImageBitmap() }
    val resultPreview = remember(resultBitmap) { resultBitmap?.asImageBitmap() }

    EnhanceScreen(
        state = state,
        sourcePreview = sourcePreview,
        resultPreview = resultPreview,
        onBack = {
            if (state.stage == EnhanceStage.PROCESSING) {
                processor.cancel()
            }
            onBack()
        },
        onSelectImage = { imagePicker.launch("image/*") },
        onEnhance = startEnhance,
        onCancel = {
            logger.log("Enhance", "cancel requested")
            processor.cancel()
        },
        onSave = saveEnhanced,
        onRunAgain = startEnhance,
        onOpenModelManager = onOpenModelManager,
    )
}

private data class EnhanceSetup(
    val model: ModelDescriptor?,
    val store: InstalledModelStore,
    val modelInstalled: Boolean,
    val errorMessage: String? = null,
)

private fun createEnhanceSetup(context: Context): EnhanceSetup {
    val store = InstalledModelStore(context)
    val model = runCatching {
        val json = context.assets.open("model-manifest.json")
            .bufferedReader()
            .use { it.readText() }
        ModelManifestParser()
            .parse(json)
            .models
            .first { it.id == ULTRASHARP_MODEL_ID }
    }.getOrElse { error ->
        return EnhanceSetup(
            model = null,
            store = store,
            modelInstalled = false,
            errorMessage = "UltraSharpモデル情報を読み込めません: ${error.message ?: "unknown error"}",
        )
    }
    return EnhanceSetup(
        model = model,
        store = store,
        modelInstalled = store.isInstalled(model),
    )
}

private fun decodeFullImage(context: Context, uri: Uri): Bitmap {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    return ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
    }
}

private fun totalRamMb(context: Context): Long? {
    val manager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager ?: return null
    val info = ActivityManager.MemoryInfo()
    manager.getMemoryInfo(info)
    return info.totalMem / (1024L * 1024L)
}

private const val ULTRASHARP_MODEL_ID = "ultrasharp"
