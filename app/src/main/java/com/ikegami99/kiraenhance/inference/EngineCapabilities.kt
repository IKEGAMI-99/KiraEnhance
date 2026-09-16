package com.ikegami99.kiraenhance.inference

enum class PixelFormat {
    RGBA_8888,
}

data class EngineCapabilities(
    val nativeScale: Int,
    val pixelFormat: PixelFormat,
    val inputBlobName: String,
    val outputBlobName: String,
    val prePadding: Int,
    val supportsGpu: Boolean,
) {
    init {
        require(nativeScale > 0) { "nativeScale must be positive" }
        require(inputBlobName.isNotBlank()) { "inputBlobName must not be blank" }
        require(outputBlobName.isNotBlank()) { "outputBlobName must not be blank" }
        require(prePadding >= 0) { "prePadding must not be negative" }
    }
}

data class ModelArtifactFile(
    val fileName: String,
    val absolutePath: String,
) {
    init {
        require(SAFE_FILE_NAME.matches(fileName)) { "Unsafe model artifact file name" }
        require(absolutePath.isNotBlank()) { "Model artifact path must not be blank" }
    }

    private companion object {
        val SAFE_FILE_NAME = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}

data class ModelLoadRequest(
    val modelId: String,
    val version: String,
    val artifacts: List<ModelArtifactFile>,
    val capabilities: EngineCapabilities,
) {
    init {
        require(SAFE_SEGMENT.matches(modelId)) { "Unsafe model id" }
        require(SAFE_SEGMENT.matches(version)) { "Unsafe model version" }
        require(artifacts.isNotEmpty()) { "At least one model artifact is required" }

        val duplicateNames = artifacts
            .groupingBy { it.fileName }
            .eachCount()
            .filterValues { it > 1 }
            .keys
        require(duplicateNames.isEmpty()) {
            "Duplicate model artifact names: ${duplicateNames.joinToString()}"
        }
    }

    fun artifact(fileName: String): ModelArtifactFile? =
        artifacts.firstOrNull { it.fileName == fileName }

    private companion object {
        val SAFE_SEGMENT = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}

enum class EngineErrorCode {
    MODEL_ARTIFACT_MISSING,
    MODEL_LOAD_FAILED,
    GPU_UNAVAILABLE,
    INVALID_INPUT,
    INFERENCE_FAILED,
    CANCELLED,
    OUT_OF_MEMORY,
    INTERNAL,
}

data class EngineError(
    val code: EngineErrorCode,
    val message: String,
)

sealed interface ModelLoadResult {
    data class Loaded(
        val gpuEnabled: Boolean,
    ) : ModelLoadResult

    data class Failed(
        val error: EngineError,
    ) : ModelLoadResult
}
