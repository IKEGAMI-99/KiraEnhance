package com.ikegami99.kiraenhance.inference.mnn

import com.ikegami99.kiraenhance.inference.EngineCapabilities
import com.ikegami99.kiraenhance.inference.ModelArtifactFile
import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.PixelFormat
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor

object PisaModelRequestFactory : ModelRequestFactory {
    override fun create(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
    ): ModelLoadRequest {
        require(model.id == MODEL_ID) { "PiSA-SR inference requires model id pisa-sr" }
        require(model.mode == EnhancementMode.BALANCED) { "PiSA-SR must use BALANCED mode" }
        require(model.backend == ModelBackend.MNN) { "PiSA-SR must use the MNN backend" }
        require(NATIVE_SCALE in model.supportedScales) { "PiSA-SR must support native 4x scale" }

        val artifactNames = model.artifacts.map { it.fileName }.toSet()
        require(REQUIRED_ARTIFACTS.all { it in artifactNames }) {
            "PiSA-SR requires ${REQUIRED_ARTIFACTS.joinToString()}"
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
                inputBlobName = INPUT_NAME,
                outputBlobName = OUTPUT_NAME,
                prePadding = 0,
                supportsGpu = true,
            ),
        )
    }

    const val VAE_ENCODER_FILE = "vae_encoder.mnn"
    const val UNET_FILE = "unet_default.mnn"
    const val VAE_DECODER_FILE = "vae_decoder.mnn"
    const val EMPTY_PROMPT_FILE = "empty_prompt.fp16"

    val REQUIRED_ARTIFACTS = listOf(
        VAE_ENCODER_FILE,
        UNET_FILE,
        VAE_DECODER_FILE,
        EMPTY_PROMPT_FILE,
    )

    private const val MODEL_ID = "pisa-sr"
    private const val INPUT_NAME = "image"
    private const val OUTPUT_NAME = "output"
    private const val NATIVE_SCALE = 4
}
