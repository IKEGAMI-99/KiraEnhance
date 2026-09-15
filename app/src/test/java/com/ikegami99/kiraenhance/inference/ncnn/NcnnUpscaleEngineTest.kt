package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.EngineErrorCode
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelLoadResult
import com.ikegami99.kiraenhance.inference.PixelFormat
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class NcnnUpscaleEngineTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun resolvesParamAndBinFromOneModelDirectory() {
        val directory = temporaryFolder.newFolder("ultrasharp")
        val param = File(directory, "4x-UltraSharp-fp16.param").apply { writeText("param") }
        val bin = File(directory, "4x-UltraSharp-fp16.bin").apply { writeBytes(byteArrayOf(1, 2, 3)) }

        val bundle = NcnnModelBundle.resolve(
            request(
                ModelArtifactFile(param.name, param.absolutePath),
                ModelArtifactFile(bin.name, bin.absolutePath),
            ),
        )

        assertEquals(param.canonicalPath, bundle.paramFile.canonicalPath)
        assertEquals(bin.canonicalPath, bundle.binFile.canonicalPath)
    }

    @Test
    fun missingArtifactFailsBeforeNativeLoad() {
        val directory = temporaryFolder.newFolder("missing")
        val bin = File(directory, "model.bin").apply { writeBytes(byteArrayOf(1)) }
        val native = FakeNcnnNativeApi()
        val engine = NcnnUpscaleEngine(native)

        val result = engine.load(
            request(
                ModelArtifactFile("model.param", File(directory, "model.param").absolutePath),
                ModelArtifactFile(bin.name, bin.absolutePath),
            ),
        )

        assertTrue(result is ModelLoadResult.Failed)
        assertEquals(
            EngineErrorCode.MODEL_ARTIFACT_MISSING,
            (result as ModelLoadResult.Failed).error.code,
        )
        assertEquals(0, native.loadCalls)
        assertFalse(engine.isLoaded())
    }

    @Test
    fun cpuFallbackIsExplicitInSuccessfulLoadResult() {
        val directory = temporaryFolder.newFolder("cpu-fallback")
        val param = File(directory, "model.param").apply { writeText("param") }
        val bin = File(directory, "model.bin").apply { writeBytes(byteArrayOf(1)) }
        val native = FakeNcnnNativeApi(
            loadResult = NcnnNativeLoadResult(
                handle = 42L,
                errorCode = NcnnNativeError.NONE,
                gpuEnabled = false,
            ),
        )
        val engine = NcnnUpscaleEngine(native)

        val result = engine.load(
            request(
                ModelArtifactFile(param.name, param.absolutePath),
                ModelArtifactFile(bin.name, bin.absolutePath),
            ),
        )

        assertEquals(ModelLoadResult.Loaded(gpuEnabled = false), result)
        assertTrue(engine.isLoaded())
        assertEquals(1, native.loadCalls)
    }

    @Test
    fun nativeParamFailureBecomesTypedLoadFailure() {
        val directory = temporaryFolder.newFolder("bad-param")
        val param = File(directory, "model.param").apply { writeText("bad") }
        val bin = File(directory, "model.bin").apply { writeBytes(byteArrayOf(1)) }
        val native = FakeNcnnNativeApi(
            loadResult = NcnnNativeLoadResult(
                handle = 0L,
                errorCode = NcnnNativeError.LOAD_PARAM_FAILED,
                gpuEnabled = false,
            ),
        )
        val engine = NcnnUpscaleEngine(native)

        val result = engine.load(
            request(
                ModelArtifactFile(param.name, param.absolutePath),
                ModelArtifactFile(bin.name, bin.absolutePath),
            ),
        )

        assertTrue(result is ModelLoadResult.Failed)
        assertEquals(
            EngineErrorCode.MODEL_LOAD_FAILED,
            (result as ModelLoadResult.Failed).error.code,
        )
        assertFalse(engine.isLoaded())
    }

    private fun request(vararg artifacts: ModelArtifactFile): ModelLoadRequest = ModelLoadRequest(
        modelId = "ultrasharp",
        version = "4x-candidate",
        artifacts = artifacts.toList(),
        capabilities = EngineCapabilities(
            nativeScale = 4,
            pixelFormat = PixelFormat.RGBA_8888,
            inputBlobName = "data",
            outputBlobName = "output",
            prePadding = 10,
            supportsGpu = true,
        ),
    )

    private class FakeNcnnNativeApi(
        private val loadResult: NcnnNativeLoadResult = NcnnNativeLoadResult(
            handle = 1L,
            errorCode = NcnnNativeError.NONE,
            gpuEnabled = true,
        ),
    ) : NcnnNativeApi {
        var loadCalls = 0

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
        ): NcnnNativeLoadResult {
            loadCalls += 1
            return loadResult
        }

        override fun unload(handle: Long) = Unit

        override fun cancel(handle: Long) = Unit
    }
}
