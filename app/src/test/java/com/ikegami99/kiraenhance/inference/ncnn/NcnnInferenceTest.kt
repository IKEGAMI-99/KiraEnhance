package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.EngineErrorCode
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

class NcnnInferenceTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun successfulInferenceReturnsNativeOutputAndGpuState() {
        val native = FakeNcnnNativeApi()
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 2, height = 2),
            settings = UpscaleSettings(outputScale = 4, useGpu = true),
        )

        assertTrue(result is UpscaleResult.Success)
        result as UpscaleResult.Success
        assertEquals(8, result.output.width)
        assertEquals(8, result.output.height)
        assertEquals(32, result.output.rowStrideBytes)
        assertEquals(PixelFormat.RGBA_8888, result.output.pixelFormat)
        assertTrue(result.usedGpu)
        assertEquals(1, native.inferCalls)
        assertEquals(0x11.toByte(), result.output.pixels.get(0))
        assertEquals(0x22.toByte(), result.output.pixels.get(1))
        assertEquals(0x33.toByte(), result.output.pixels.get(2))
        assertEquals(0xff.toByte(), result.output.pixels.get(3))
    }

    @Test
    fun explicitTileSizeRunsMultipleInferencesAndComposesCoreRegions() {
        val native = TiledFakeNcnnNativeApi(scale = 4)
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 65, height = 33),
            settings = UpscaleSettings(outputScale = 4, useGpu = true, tileSize = 32),
        )

        assertTrue(result is UpscaleResult.Success)
        result as UpscaleResult.Success
        assertEquals(260, result.output.width)
        assertEquals(132, result.output.height)
        assertEquals(6, native.inferCalls)
        assertEquals(
            listOf(
                42 to 33,
                43 to 33,
                11 to 33,
                42 to 11,
                43 to 11,
                11 to 11,
            ),
            native.inputSizes,
        )

        val pixels = result.output.pixels
        val rowStride = result.output.rowStrideBytes
        assertEquals(1.toByte(), pixels.get(rgbaOffset(x = 0, y = 0, rowStride = rowStride)))
        assertEquals(2.toByte(), pixels.get(rgbaOffset(x = 128, y = 0, rowStride = rowStride)))
        assertEquals(3.toByte(), pixels.get(rgbaOffset(x = 256, y = 0, rowStride = rowStride)))
        assertEquals(4.toByte(), pixels.get(rgbaOffset(x = 0, y = 128, rowStride = rowStride)))
        assertEquals(5.toByte(), pixels.get(rgbaOffset(x = 128, y = 128, rowStride = rowStride)))
        assertEquals(6.toByte(), pixels.get(rgbaOffset(x = 259, y = 131, rowStride = rowStride)))
        assertEquals(0xff.toByte(), pixels.get(rgbaOffset(x = 259, y = 131, rowStride = rowStride) + 3))
    }

    @Test
    fun tiledInferenceRetriesWithSmallerTilesAfterNativeOutOfMemory() {
        val native = TiledFakeNcnnNativeApi(scale = 4, outOfMemoryOnCall = 1)
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 100, height = 100),
            settings = UpscaleSettings(outputScale = 4, useGpu = true, tileSize = 96),
        )

        assertTrue(result is UpscaleResult.Success)
        assertEquals(5, native.inferCalls)
        assertEquals(100 to 100, native.inputSizes[0])
        assertEquals(74 to 74, native.inputSizes[1])
    }

    @Test
    fun nonNativeScaleIsRejectedBeforeNativeInference() {
        val native = FakeNcnnNativeApi()
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 2, height = 2),
            settings = UpscaleSettings(outputScale = 2),
        )

        assertTrue(result is UpscaleResult.Failed)
        assertEquals(
            EngineErrorCode.INVALID_INPUT,
            (result as UpscaleResult.Failed).error.code,
        )
        assertEquals(0, native.inferCalls)
    }

    @Test
    fun oversizedSmokeOutputIsRejectedBeforeNativeInference() {
        val native = FakeNcnnNativeApi()
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 1_025, height = 1_025),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Failed)
        assertEquals(
            EngineErrorCode.OUT_OF_MEMORY,
            (result as UpscaleResult.Failed).error.code,
        )
        assertEquals(0, native.inferCalls)
    }

    @Test
    fun nativeCancellationBecomesTypedCancellation() {
        val native = FakeNcnnNativeApi(
            inferenceResult = NcnnNativeInferenceResult(
                errorCode = NcnnNativeError.CANCELLED,
                outputWidth = 0,
                outputHeight = 0,
                outputRowStrideBytes = 0,
                gpuUsed = false,
            ),
        )
        val engine = NcnnUpscaleEngine(native)
        engine.load(validRequest())

        val result = engine.upscale(
            input = rgbaInput(width = 2, height = 2),
            settings = UpscaleSettings(outputScale = 4),
        )

        assertTrue(result is UpscaleResult.Failed)
        assertEquals(
            EngineErrorCode.CANCELLED,
            (result as UpscaleResult.Failed).error.code,
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

    private fun rgbaOffset(x: Int, y: Int, rowStride: Int): Int =
        y * rowStride + x * 4

    private class FakeNcnnNativeApi(
        private val inferenceResult: NcnnNativeInferenceResult = NcnnNativeInferenceResult(
            errorCode = NcnnNativeError.NONE,
            outputWidth = 8,
            outputHeight = 8,
            outputRowStrideBytes = 32,
            gpuUsed = true,
        ),
    ) : NcnnNativeApi {
        var inferCalls = 0

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
            handle = 99L,
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
            if (inferenceResult.errorCode == NcnnNativeError.NONE) {
                outputPixels.put(0, 0x11)
                outputPixels.put(1, 0x22)
                outputPixels.put(2, 0x33)
                outputPixels.put(3, 0xff.toByte())
            }
            return inferenceResult
        }

        override fun unload(handle: Long) = Unit

        override fun cancel(handle: Long) = Unit
    }

    private class TiledFakeNcnnNativeApi(
        private val scale: Int,
        private val outOfMemoryOnCall: Int? = null,
    ) : NcnnNativeApi {
        var inferCalls: Int = 0
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
            handle = 101L,
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
            if (inferCalls == outOfMemoryOnCall) {
                return NcnnNativeInferenceResult(
                    errorCode = NcnnNativeError.OUT_OF_MEMORY,
                    outputWidth = 0,
                    outputHeight = 0,
                    outputRowStrideBytes = 0,
                    gpuUsed = false,
                )
            }

            val marker = inferCalls.toByte()
            val outputWidth = width * scale
            val outputHeight = height * scale
            val rowStride = outputWidth * 4
            var offset = 0
            repeat(outputWidth * outputHeight) {
                outputPixels.put(offset, marker)
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
