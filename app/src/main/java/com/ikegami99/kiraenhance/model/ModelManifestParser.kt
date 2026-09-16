package com.ikegami99.kiraenhance.model

import com.squareup.moshi.JsonDataException
import com.squareup.moshi.Moshi
import com.squareup.moshi.kotlin.reflect.KotlinJsonAdapterFactory

class ModelManifestParser(
    moshi: Moshi = Moshi.Builder()
        .addLast(KotlinJsonAdapterFactory())
        .build(),
) {
    private val adapter = moshi.adapter(ModelManifest::class.java)

    fun parse(json: String): ModelManifest {
        val manifest = try {
            adapter.fromJson(json)
        } catch (error: JsonDataException) {
            throw IllegalArgumentException("Invalid model manifest", error)
        } catch (error: IllegalArgumentException) {
            throw IllegalArgumentException("Invalid model manifest", error)
        } ?: throw IllegalArgumentException("Model manifest is empty")

        validate(manifest)
        return manifest
    }

    private fun validate(manifest: ModelManifest) {
        require(manifest.schemaVersion > 0) { "schemaVersion must be positive" }

        val ids = mutableSetOf<String>()
        manifest.models.forEach { model ->
            require(model.id.isNotBlank()) { "Model id must not be blank" }
            require(ids.add(model.id)) { "Duplicate model id: ${model.id}" }
            require(model.displayName.isNotBlank()) { "Display name must not be blank: ${model.id}" }
            require(model.version.isNotBlank()) { "Version must not be blank: ${model.id}" }
            require(model.downloadUrl.isNotBlank()) { "Download URL must not be blank: ${model.id}" }
            require(model.fileSizeBytes > 0) { "File size must be positive: ${model.id}" }
            require(SHA_256.matches(model.sha256)) { "Invalid SHA-256: ${model.id}" }
            require(model.supportedScales.isNotEmpty()) { "At least one scale is required: ${model.id}" }
            require(model.supportedScales.all { it in SUPPORTED_SCALES }) {
                "Unsupported scale for ${model.id}; only 2x and 4x are supported"
            }
            require(model.estimatedRamMb > 0) { "RAM estimate must be positive: ${model.id}" }
            require(model.licenseName.isNotBlank()) { "License name must not be blank: ${model.id}" }
            require(model.licenseUrl.isNotBlank()) { "License URL must not be blank: ${model.id}" }

            val controlIds = mutableSetOf<String>()
            model.controls.forEach { control ->
                require(control.id.isNotBlank()) { "Control id must not be blank: ${model.id}" }
                require(controlIds.add(control.id)) { "Duplicate control id ${control.id}: ${model.id}" }
                if (control.type == ControlType.SLIDER) {
                    val min = requireNotNull(control.min) { "Slider min is required: ${control.id}" }
                    val max = requireNotNull(control.max) { "Slider max is required: ${control.id}" }
                    val defaultValue = requireNotNull(control.defaultValue) {
                        "Slider defaultValue is required: ${control.id}"
                    }
                    require(min <= max) { "Slider min must be <= max: ${control.id}" }
                    require(defaultValue in min..max) {
                        "Slider defaultValue must be between min and max: ${control.id}"
                    }
                }
            }
        }
    }

    private companion object {
        val SHA_256 = Regex("^[0-9a-fA-F]{64}$")
        val SUPPORTED_SCALES = setOf(2, 4)
    }
}
