package com.ikegami99.kiraenhance.model

enum class ModelBackend {
    MNN,
    NCNN,
}

enum class EnhancementMode {
    FIDELITY,
    BALANCED,
    DETAIL,
    ULTRASHARP,
}

enum class ControlType {
    SLIDER,
    TOGGLE,
    CHOICE,
}

data class ModelControlDescriptor(
    val id: String,
    val label: String,
    val type: ControlType,
    val min: Double? = null,
    val max: Double? = null,
    val defaultValue: Any? = null,
    val choices: List<String> = emptyList(),
)

data class ModelArtifactDescriptor(
    val fileName: String,
    val downloadUrl: String,
    val fileSizeBytes: Long,
    val sha256: String,
)

data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val mode: EnhancementMode,
    val version: String,
    val backend: ModelBackend,
    val artifacts: List<ModelArtifactDescriptor>,
    val supportedScales: List<Int>,
    val minAppVersion: String,
    val estimatedRamMb: Long,
    val licenseName: String,
    val licenseUrl: String,
    val description: String,
    val community: Boolean = false,
    val controls: List<ModelControlDescriptor> = emptyList(),
) {
    val totalFileSizeBytes: Long
        get() = artifacts.sumOf(ModelArtifactDescriptor::fileSizeBytes)
}

data class ModelManifest(
    val schemaVersion: Int,
    val models: List<ModelDescriptor>,
)
