package com.ikegami99.kiraenhance.inference.smoke

import com.ikegami99.kiraenhance.inference.EngineError
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import com.ikegami99.kiraenhance.model.ModelDescriptor


data class NcnnSmokeImage(
    val width: Int,
    val height: Int,
    val argbPixels: IntArray,
) {
    init {
        require(width > 0 && height > 0) { "Smoke image dimensions must be positive" }
        require(argbPixels.size == width * height) {
            "Smoke ARGB pixel count does not match image dimensions"
        }
    }
}

sealed interface NcnnSmokeRunResult {
    data class Success(
        val width: Int,
        val height: Int,
        val argbPixels: IntArray,
        val usedGpu: Boolean,
    ) : NcnnSmokeRunResult

    data class Failed(
        val error: EngineError,
    ) : NcnnSmokeRunResult
}

class NcnnSmokeInferenceRunner(
    private val engineFactory: () -> UpscaleEngine,
) {
    fun run(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
        image: NcnnSmokeImage,
    ): NcnnSmokeRunResult {
        var engine: UpscaleEngine? = null
        try {
            engine = engineFactory()
            val loadRequest = NcnnSmokeModelRequestFactory.create(
                model = model,
                artifactPath = artifactPath,
            )
            when (val loadResult = engine.load(loadRequest)) {
                is ModelLoadResult.Failed -> return NcnnSmokeRunResult.Failed(loadResult.error)
                is ModelLoadResult.Loaded -> Unit
            }

            val input = NcnnSmokePixelCodec.encodeArgb(
                width = image.width,
                height = image.height,
                argb = image.argbPixels,
            )
            return when (
                val upscaleResult = engine.upscale(
                    input = input,
                    settings = UpscaleSettings(
                        outputScale = OUTPUT_SCALE,
                        useGpu = true,
                        tileSize = TILE_SIZE,
                    ),
                )
            ) {
                is UpscaleResult.Failed -> NcnnSmokeRunResult.Failed(upscaleResult.error)
                is UpscaleResult.Success -> NcnnSmokeRunResult.Success(
                    width = upscaleResult.output.width,
                    height = upscaleResult.output.height,
                    argbPixels = NcnnSmokePixelCodec.decodeToArgb(upscaleResult.output),
                    usedGpu = upscaleResult.usedGpu,
                )
            }
        } catch (error: OutOfMemoryError) {
            return NcnnSmokeRunResult.Failed(
                EngineError(
                    code = EngineErrorCode.OUT_OF_MEMORY,
                    message = error.message ?: "Smoke inference ran out of memory",
                ),
            )
        } catch (error: LinkageError) {
            return NcnnSmokeRunResult.Failed(
                EngineError(
                    code = EngineErrorCode.INTERNAL,
                    message = error.message ?: "ncnn native runtime could not be loaded",
                ),
            )
        } catch (error: RuntimeException) {
            return NcnnSmokeRunResult.Failed(
                EngineError(
                    code = EngineErrorCode.INTERNAL,
                    message = error.message ?: "Smoke inference failed unexpectedly",
                ),
            )
        } finally {
            runCatching { engine?.close() }
        }
    }

    companion object {
        const val OUTPUT_SCALE = 4
        const val TILE_SIZE = 128
    }
}
