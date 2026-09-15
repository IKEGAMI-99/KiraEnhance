package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.inference.UpscaleInput
import com.ikegami99.kiraenhance.inference.UpscaleResult
import com.ikegami99.kiraenhance.inference.UpscaleSettings
import java.io.File
import java.nio.ByteBuffer
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class NcnnAutoTilingTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun unsetTileSizeUsesConservativeAutomaticTiling() {
        val native = RecordingNcnnNativeApi(scale = 4)
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 129, height = 129),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Success)
        assertEquals(4, native.inferCalls)
        assertEquals(
            listOf(
                129 to 129,
                11 to 129,
                129 to 11,
                11 to 11,
            ),
            native.inputSizes,
        )
    }

    @Test
    fun highMemoryGpuUsesLargerAutomaticTile() {
        val native = RecordingNcnnNativeApi(scale = 4)
        val engine = NcnnUpscaleEngine(
            nativeApi = native,
            totalRamMbProvider = { 12_288L },
        )
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 385, height = 385),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Success)
        assertEquals(4, native.inferCalls)
        assertEquals(
            listOf(
                385 to 385,
                11 to 385,
                385 to 11,
                11 to 11,
            ),
            native.inputSizes,
        )
    }

    private fun validRequest(): ModelLoadRequest {
        val directory = temporaryFolder.newFolder("model-${System.nanoTime()}")
        val param = File(directory, "model.param").apply { writeText("param") }
        val bin = File(directory, "model.bin").apply { writeBytes(byteArrayOf(1)) }
        return ModelLoadRequest(
            modelId = "ultrasharp",
            version = "test",
            artifacts = listOf(
                ModelArtifactFile(param.name, param.absolutePath),
                ModelArtifactFile(bin.name, bin.absolutePath),
            ),
            capabilities = EngineCapabilities(
                nativeScale = 4,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = "data",
                outputBlobName = "output",
                prePadding = 10,
                supportsGpu = true,
            ),
        )
    }

    private fun rgbaInput(width: Int, height: Int): UpscaleInput {
        val rowStride = width * 4
        val pixels = ByteBuffer.allocateDirect(rowStride * height)
        repeat(width * height) {
            pixels.put(0x10)
            pixels.put(0x20)
            pixels.put(0x30)
            pixels.put(0xff.toByte())
        }
        pixels.flip()
        return UpscaleInput(
            width = width,
            height = height,
            rowStrideBytes = rowStride,
            pixelFormat = PixelFormat.RGBA_8888,
            pixels = pixels,
        )
    }

    private class RecordingNcnnNativeApi(
        private val scale: Int,
    ) : NcnnNativeApi {
        var inferCalls = 0
        val inputSizes = mutableListOf<Pair<Int, Int>>()

        override fun runtimeInfo(): NcnnRuntimeInfo = NcnnRuntimeInfo(
            ncnnVersion = "20260526",
            vulkanCompiled = true,
            gpuCount = 1,
        )

        override fun loadModel(
            paramPath: String,
            binPath: String,
            inputBlobName: String,
            outputBlobName: String,
            nativeScale: Int,
            prePadding: Int,
            preferGpu: Boolean,
        ): NcnnNativeLoadResult = NcnnNativeLoadResult(
            handle = 303L,
            errorCode = NcnnNativeError.NONE,
            gpuEnabled = true,
        )

        override fun infer(
            handle: Long,
            inputPixels: ByteBuffer,
            width: Int,
            height: Int,
            inputRowStrideBytes: Int,
            outputPixels: ByteBuffer,
            outputCapacityBytes: Long,
        ): NcnnNativeInferenceResult {
            inferCalls += 1
            inputSizes += width to height
            val outputWidth = width * scale
            val outputHeight = height * scale
            val rowStride = outputWidth * 4
            var offset = 0
            repeat(outputWidth * outputHeight) {
                outputPixels.put(offset, inferCalls.toByte())
                outputPixels.put(offset + 1, 0)
                outputPixels.put(offset + 2, 0)
                outputPixels.put(offset + 3, 0xff.toByte())
                offset += 4
            }
            return NcnnNativeInferenceResult(
                errorCode = NcnnNativeError.NONE,
                outputWidth = outputWidth,
                outputHeight = outputHeight,
                outputRowStrideBytes = rowStride,
                gpuUsed = true,
            )
        }

        override fun unload(handle: Long) = Unit

        override fun cancel(handle: Long) = Unit
    }
}
