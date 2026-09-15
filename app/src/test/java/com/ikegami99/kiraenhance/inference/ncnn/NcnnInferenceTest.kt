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
}
