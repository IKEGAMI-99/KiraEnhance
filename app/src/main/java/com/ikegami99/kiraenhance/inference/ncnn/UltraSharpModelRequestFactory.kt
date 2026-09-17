package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor

object UltraSharpModelRequestFactory : ModelRequestFactory {
    override fun create(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
    ): ModelLoadRequest {
        require(model.id == ULTRASHARP_MODEL_ID) { "Production UltraSharp inference requires model id ultrasharp" }
        require(model.mode == EnhancementMode.ULTRASHARP) { "Model must be declared as UltraSharp mode" }
        require(model.backend == ModelBackend.NCNN) { "UltraSharp model must use the ncnn backend" }
        require(NATIVE_SCALE in model.supportedScales) { "UltraSharp model must support native 4x scale" }

        val artifactNames = model.artifacts.map { it.fileName }.toSet()
        require(PARAM_FILE_NAME in artifactNames && BIN_FILE_NAME in artifactNames) {
            "UltraSharp requires model.param and model.bin"
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
