package com.ikegami99.kiraenhance.ui.enhance

import android.graphics.Bitmap
import android.os.SystemClock
import com.ikegami99.kiraenhance.image.RgbaPixelCodec
import com.ikegami99.kiraenhance.inference.EngineError
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleProgress
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import com.ikegami99.kiraenhance.inference.ncnn.NcnnUpscaleEngine
import com.ikegami99.kiraenhance.inference.ncnn.UltraSharpModelRequestFactory
import com.ikegami99.kiraenhance.model.ModelDescriptor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface EnhanceProcessResult {
    data class Success(
        val bitmap: Bitmap,
        val usedGpu: Boolean,
        val elapsedMs: Long,
    ) : EnhanceProcessResult

    data class Failed(
        val message: String,
        val code: EngineErrorCode? = null,
    ) : EnhanceProcessResult
}

data class EnhanceProcessProgress(
    val completedTiles: Int,
    val totalTiles: Int,
    val elapsedMs: Long,
)

class EnhanceProcessor(
    private val engineFactory: () -> UpscaleEngine = { NcnnUpscaleEngine() },
) {
    @Volatile
    private var activeEngine: UpscaleEngine? = null

    suspend fun run(
        bitmap: Bitmap,
        model: ModelDescriptor,
        artifactPath: (String) -> String,
        onProgress: (EnhanceProcessProgress) -> Unit,
    ): EnhanceProcessResult = coroutineScope {
        val startedAt = SystemClock.elapsedRealtime()
        val engine = engineFactory()
        activeEngine = engine

        try {
            val request = runCatching {
                UltraSharpModelRequestFactory.create(model, artifactPath)
            }.getOrElse { error ->
                return@coroutineScope EnhanceProcessResult.Failed(
                    message = error.message ?: "UltraSharpモデル設定が不正です",
                    code = EngineErrorCode.INVALID_INPUT,
                )
            }

            val input = withContext(Dispatchers.Default) {
                val pixels = IntArray(bitmap.width * bitmap.height)
                bitmap.getPixels(
                    pixels,
                    0,
                    bitmap.width,
                    0,
                    0,
                    bitmap.width,
                    bitmap.height,
                )
                RgbaPixelCodec.encodeArgb(bitmap.width, bitmap.height, pixels)
            }

            when (val load = withContext(Dispatchers.Default) { engine.load(request) }) {
                is ModelLoadResult.Failed -> {
                    return@coroutineScope EnhanceProcessResult.Failed(
                        message = presentError(load.error),
                        code = load.error.code,
                    )
                }
                is ModelLoadResult.Loaded -> Unit
            }

            val progressJob = launchProgressPolling(engine, startedAt, onProgress)
            val upscaleResult = try {
                withContext(Dispatchers.Default) {
                    engine.upscale(
                        input = input,
                        settings = UpscaleSettings(outputScale = OUTPUT_SCALE),
                    )
                }
            } finally {
                progressJob.cancelAndJoin()
            }

            when (upscaleResult) {
                is UpscaleResult.Failed -> EnhanceProcessResult.Failed(
                    message = presentError(upscaleResult.error),
                    code = upscaleResult.error.code,
                )

                is UpscaleResult.Success -> {
                    val outputBitmap = withContext(Dispatchers.Default) {
                        val output = upscaleResult.output
                        val rendered = Bitmap.createBitmap(
                            output.width,
                            output.height,
                            Bitmap.Config.ARGB_8888,
                        )
                        val row = IntArray(output.width)
                        for (y in 0 until output.height) {
                            RgbaPixelCodec.decodeRow(output, y, row)
                            rendered.setPixels(row, 0, output.width, 0, y, output.width, 1)
                        }
                        rendered
                    }
                    EnhanceProcessResult.Success(
                        bitmap = outputBitmap,
                        usedGpu = upscaleResult.usedGpu,
                        elapsedMs = SystemClock.elapsedRealtime() - startedAt,
                    )
                }
            }
        } catch (_: OutOfMemoryError) {
            EnhanceProcessResult.Failed(
                message = "メモリ不足で画像を処理できませんでした。ほかのアプリを閉じて再試行してください。",
                code = EngineErrorCode.OUT_OF_MEMORY,
            )
        } finally {
            engine.close()
            if (activeEngine === engine) {
                activeEngine = null
            }
        }
    }

    fun cancel() {
        activeEngine?.cancel()
    }

    private fun kotlinx.coroutines.CoroutineScope.launchProgressPolling(
        engine: UpscaleEngine,
        startedAt: Long,
        onProgress: (EnhanceProcessProgress) -> Unit,
    ): Job = launch {
        while (isActive) {
            val progress: UpscaleProgress = engine.progress()
            onProgress(
                EnhanceProcessProgress(
                    completedTiles = progress.completedTiles,
                    totalTiles = progress.totalTiles,
                    elapsedMs = SystemClock.elapsedRealtime() - startedAt,
                ),
            )
            delay(PROGRESS_POLL_MS)
        }
    }

    private fun presentError(error: EngineError): String = when (error.code) {
        EngineErrorCode.MODEL_ARTIFACT_MISSING ->
            "モデルファイルが見つかりません。モデル管理からUltraSharpを再ダウンロードしてください。"
        EngineErrorCode.MODEL_LOAD_FAILED ->
            "UltraSharpモデルを読み込めません。モデル管理から再インストールしてください。"
        EngineErrorCode.GPU_UNAVAILABLE ->
            "この端末では必要なGPU機能を利用できません。"
        EngineErrorCode.INVALID_INPUT ->
            "この画像をUltraSharpで処理できません: ${error.message}"
        EngineErrorCode.CANCELLED ->
            "処理をキャンセルしました"
        EngineErrorCode.OUT_OF_MEMORY ->
            "メモリ不足で処理できませんでした。ほかのアプリを閉じて再試行してください。"
        EngineErrorCode.INFERENCE_FAILED,
        EngineErrorCode.INTERNAL,
        -> "高画質化に失敗しました: ${error.message}"
    }

    private companion object {
        const val OUTPUT_SCALE = 4
        const val PROGRESS_POLL_MS = 100L
    }
}
