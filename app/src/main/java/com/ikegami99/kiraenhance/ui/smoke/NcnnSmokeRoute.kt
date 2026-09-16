package com.ikegami99.kiraenhance.ui.smoke

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import com.ikegami99.kiraenhance.inference.ncnn.NcnnUpscaleEngine
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeCropPlanner
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeImage
import com.ikegami99.kiraenhance.inference.smoke.NcnnSmokeInferenceRunner
import com.ikegami99.kiraenhance.model.InstalledModelStore
import com.ikegami99.kiraenhance.model.ModelDescriptor
import com.ikegami99.kiraenhance.model.ModelManifestParser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun NcnnSmokeRoute(
    onBack: () -> Unit,
    onOpenModelManager: () -> Unit,
) {
    val applicationContext = LocalContext.current.applicationContext
    val scope = rememberCoroutineScope()
    val setup = remember(applicationContext) {
        createSetup(applicationContext)
    }
    val presenter = remember {
        NcnnSmokeInferencePresenter(
            runner = NcnnSmokeInferenceRunner {
                NcnnUpscaleEngine()
            },
        )
    }

    var state by remember(setup) { mutableStateOf(setup.initialState) }
    var selectedImage by remember { mutableStateOf<NcnnSmokeImage?>(null) }
    var inputPreview by remember { mutableStateOf<ImageBitmap?>(null) }
    var outputPreview by remember { mutableStateOf<ImageBitmap?>(null) }

    val imagePicker = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent(),
    ) { uri ->
        if (uri == null || setup.model == null || !state.modelInstalled) {
            return@rememberLauncherForActivityResult
        }

        state = state.copy(
            status = "画像を読み込み中...",
            imageSelected = false,
            isRunning = true,
            inputDescription = null,
            outputDescription = null,
            runtimeDetails = null,
            elapsedMs = null,
        )
        selectedImage = null
        inputPreview = null
        outputPreview = null

        scope.launch {
            val decoded = withContext(Dispatchers.IO) {
                runCatching {
                    decodeSmokeImage(applicationContext, uri)
                }
            }

            decoded.onSuccess { image ->
                val presentation = NcnnSmokeSelectionPresenter.present(image)
                selectedImage = presentation.image
                inputPreview = image.toImageBitmap()
                outputPreview = null
                state = presentation.state
            }.onFailure { error ->
                state = state.copy(
                    status = "画像を読み込めません: ${error.message ?: "unknown error"}",
                    imageSelected = false,
                    isRunning = false,
                    inputDescription = null,
                    outputDescription = null,
                    runtimeDetails = null,
                    elapsedMs = null,
                )
            }
        }
    }

    NcnnSmokeContent(
        state = state,
        onBack = onBack,
        onSelectImage = {
            imagePicker.launch("image/*")
        },
        onRun = {
            val model = setup.model ?: return@NcnnSmokeContent
            val image = selectedImage ?: return@NcnnSmokeContent

            state = state.copy(
                status = "4x処理中...",
                isRunning = true,
                outputDescription = null,
                runtimeDetails = null,
                elapsedMs = null,
            )
            outputPreview = null

            scope.launch {
                val result = withContext(Dispatchers.Default) {
                    presenter.run(
                        model = model,
                        artifactPath = { fileName ->
                            setup.store.artifactFile(
                                modelId = model.id,
                                version = model.version,
                                fileName = fileName,
                            ).absolutePath
                        },
                        image = image,
                    )
                }

                state = result.state
                outputPreview = result.output?.toImageBitmap()
            }
        },
        onOpenModelManager = onOpenModelManager,
        inputPreview = inputPreview,
        outputPreview = outputPreview,
    )
}

private data class NcnnSmokeSetup(
    val model: ModelDescriptor?,
    val store: InstalledModelStore,
    val initialState: NcnnSmokeScreenState,
)

private fun createSetup(context: Context): NcnnSmokeSetup {
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
        return NcnnSmokeSetup(
            model = null,
            store = store,
            initialState = NcnnSmokeScreenState(
                modelInstalled = false,
                status = "4x-UltraSharpモデル情報を読み込めません: ${error.message ?: "unknown error"}",
            ),
        )
    }

    val installed = store.isInstalled(model)
    return NcnnSmokeSetup(
        model = model,
        store = store,
        initialState = NcnnSmokeScreenState(
            modelInstalled = installed,
            status = if (installed) {
                "画像を選択してください"
            } else {
                "4x-UltraSharpモデルが未インストールです"
            },
        ),
    )
}

private fun decodeSmokeImage(
    context: Context,
    uri: Uri,
): NcnnSmokeImage {
    val source = ImageDecoder.createSource(context.contentResolver, uri)
    val bitmap = ImageDecoder.decodeBitmap(source) { decoder, _, _ ->
        decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
    }

    return try {
        val crop = NcnnSmokeCropPlanner.centerSquare(
            imageWidth = bitmap.width,
            imageHeight = bitmap.height,
        )
        val pixels = IntArray(crop.width * crop.height)
        bitmap.getPixels(
            pixels,
            0,
            crop.width,
            crop.left,
            crop.top,
            crop.width,
            crop.height,
        )
        NcnnSmokeImage(
            width = crop.width,
            height = crop.height,
            argbPixels = pixels,
        )
    } finally {
        bitmap.recycle()
    }
}

private fun NcnnSmokeImage.toImageBitmap(): ImageBitmap {
    val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    bitmap.setPixels(
        argbPixels,
        0,
        width,
        0,
        0,
        width,
        height,
    )
    return bitmap.asImageBitmap()
}

private const val ULTRASHARP_MODEL_ID = "ultrasharp"
