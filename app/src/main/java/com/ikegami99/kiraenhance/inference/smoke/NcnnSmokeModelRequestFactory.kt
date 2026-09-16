package com.ikegami99.kiraenhance.inference.smoke

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor

object NcnnSmokeModelRequestFactory {
    fun create(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
    ): ModelLoadRequest {
        require(model.id == ULTRASHARP_MODEL_ID) { "Smoke inference only supports 4x-UltraSharp" }
        require(model.backend == ModelBackend.NCNN) { "UltraSharp smoke model must use the ncnn backend" }
        require(NATIVE_SCALE in model.supportedScales) { "UltraSharp smoke model must support native 4x scale" }

        val artifactNames = model.artifacts.map { it.fileName }.toSet()
        require(PARAM_FILE_NAME in artifactNames && BIN_FILE_NAME in artifactNames) {
            "UltraSharp smoke model requires model.param and model.bin"
        }

        return ModelLoadRequest(
            modelId = model.id,
            version = model.version,
            artifacts = model.artifacts.map { artifact ->
                ModelArtifactFile(
                    fileName = artifact.fileName,
                    absolutePath = artifactPath(artifact.fileName),
                )
            },
            capabilities = EngineCapabilities(
                nativeScale = NATIVE_SCALE,
                pixelFormat = PixelFormat.RGBA_8888,
                inputBlobName = INPUT_BLOB_NAME,
                outputBlobName = OUTPUT_BLOB_NAME,
                prePadding = PRE_PADDING,
                supportsGpu = true,
            ),
        )
    }

    private const val ULTRASHARP_MODEL_ID = "ultrasharp"
    private const val PARAM_FILE_NAME = "model.param"
    private const val BIN_FILE_NAME = "model.bin"
    private const val INPUT_BLOB_NAME = "data"
    private const val OUTPUT_BLOB_NAME = "output"
    private const val NATIVE_SCALE = 4
    private const val PRE_PADDING = 10
}
