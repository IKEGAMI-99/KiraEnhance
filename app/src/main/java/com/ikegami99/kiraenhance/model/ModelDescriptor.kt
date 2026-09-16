package com.ikegami99.kiraenhance.model

import com.squareup.moshi.JsonClass

@JsonClass(generateAdapter = false)
data class ModelManifest(
    val schemaVersion: Int,
    val models: List<ModelDescriptor>,
)

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

@JsonClass(generateAdapter = false)
data class ModelControlDescriptor(
    val id: String,
    val displayName: String,
    val type: ControlType,
    val min: Double? = null,
    val max: Double? = null,
    val defaultValue: Double? = null,
    val choices: List<String> = emptyList(),
)

@JsonClass(generateAdapter = false)
data class ModelDescriptor(
    val id: String,
    val displayName: String,
    val mode: EnhancementMode,
    val version: String,
    val backend: ModelBackend,
    val downloadUrl: String,
    val fileSizeBytes: Long,
    val sha256: String,
    val supportedScales: List<Int>,
    val minAppVersion: String,
    val estimatedRamMb: Long,
    val licenseName: String,
    val licenseUrl: String,
    val description: String,
    val community: Boolean = false,
    val controls: List<ModelControlDescriptor> = emptyList(),
)
