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
                ?: throw IllegalArgumentException("Model manifest is empty")
        } catch (error: IllegalArgumentException) {
            throw error
        } catch (error: JsonDataException) {
            throw IllegalArgumentException("Invalid model manifest", error)
        } catch (error: RuntimeException) {
            throw IllegalArgumentException("Invalid model manifest", error)
        }

        validate(manifest)
        return manifest
    }

    private fun validate(manifest: ModelManifest) {
        require(manifest.schemaVersion > 0) { "schemaVersion must be positive" }
        require(manifest.models.isNotEmpty()) { "Manifest must contain at least one model" }

        val duplicateIds = manifest.models
            .groupingBy { it.id }
            .eachCount()
            .filterValues { it > 1 }
            .keys
        require(duplicateIds.isEmpty()) { "Duplicate model ids: ${duplicateIds.joinToString()}" }

        manifest.models.forEach(::validateModel)
    }

    private fun validateModel(model: ModelDescriptor) {
        requireSafeSegment(model.id, "model id")
        require(model.displayName.isNotBlank()) { "displayName must not be blank for ${model.id}" }
        requireSafeSegment(model.version, "model version")
        require(model.downloadUrl.isNotBlank()) { "downloadUrl must not be blank for ${model.id}" }
        require(model.fileSizeBytes > 0) { "fileSizeBytes must be positive for ${model.id}" }
        require(SHA_256.matches(model.sha256)) { "Invalid SHA-256 for ${model.id}" }
        require(model.supportedScales.isNotEmpty()) { "supportedScales must not be empty for ${model.id}" }
        require(model.supportedScales.all { it == 2 || it == 4 }) {
            "Only 2x and 4x scales are supported for ${model.id}"
        }
        require(model.minAppVersion.isNotBlank()) { "minAppVersion must not be blank for ${model.id}" }
        require(model.estimatedRamMb > 0) { "estimatedRamMb must be positive for ${model.id}" }
        require(model.licenseName.isNotBlank()) { "licenseName must not be blank for ${model.id}" }
        require(model.licenseUrl.isNotBlank()) { "licenseUrl must not be blank for ${model.id}" }
        require(model.description.isNotBlank()) { "description must not be blank for ${model.id}" }

        val duplicateControls = model.controls
            .groupingBy { it.id }
            .eachCount()
            .filterValues { it > 1 }
            .keys
        require(duplicateControls.isEmpty()) { "Duplicate control ids in ${model.id}" }

        model.controls.forEach { control -> validateControl(model.id, control) }
    }

    private fun validateControl(modelId: String, control: ModelControlDescriptor) {
        requireSafeSegment(control.id, "control id")
        require(control.label.isNotBlank()) { "Control label must not be blank in $modelId" }

        when (control.type) {
            ControlType.SLIDER -> {
                val min = requireNotNull(control.min) { "Slider min is required for ${control.id}" }
                val max = requireNotNull(control.max) { "Slider max is required for ${control.id}" }
                val default = (control.defaultValue as? Number)?.toDouble()
                    ?: throw IllegalArgumentException("Slider defaultValue must be numeric for ${control.id}")
                require(min <= max) { "Slider min must be <= max for ${control.id}" }
                require(default in min..max) { "Slider defaultValue must be within range for ${control.id}" }
            }

            ControlType.TOGGLE -> {
                require(control.defaultValue == null || control.defaultValue is Boolean) {
                    "Toggle defaultValue must be boolean for ${control.id}"
                }
            }

            ControlType.CHOICE -> {
                require(control.choices.isNotEmpty()) { "Choice options are required for ${control.id}" }
                val default = control.defaultValue
                require(default == null || default is String && default in control.choices) {
                    "Choice defaultValue must be one of the choices for ${control.id}"
                }
            }
        }
    }

    private fun requireSafeSegment(value: String, label: String) {
        require(SAFE_SEGMENT.matches(value)) { "$label contains unsupported characters" }
    }

    private companion object {
        val SHA_256 = Regex("^[A-Fa-f0-9]{64}$")
        val SAFE_SEGMENT = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}
