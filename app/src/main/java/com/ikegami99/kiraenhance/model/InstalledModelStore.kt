package com.ikegami99.kiraenhance.model

import android.content.Context
import java.io.File

class InstalledModelStore(
    private val filesDir: File,
    private val cacheDir: File,
) {
    constructor(context: Context) : this(
        filesDir = context.filesDir,
        cacheDir = context.cacheDir,
    )

    fun modelsDir(): File = File(filesDir, "models")

    fun modelFile(modelId: String, version: String): File {
        validateSegment(modelId, "modelId")
        validateSegment(version, "version")
        return File(modelsDir(), "$modelId/$version/model.bin")
    }

    fun partialFile(modelId: String, version: String): File {
        validateSegment(modelId, "modelId")
        validateSegment(version, "version")
        return File(cacheDir, "model-downloads/$modelId-$version.part")
    }

    fun isInstalled(modelId: String, version: String): Boolean =
        modelFile(modelId, version).isFile

    fun delete(modelId: String, version: String): Boolean {
        val versionDir = modelFile(modelId, version).parentFile ?: return false
        if (!versionDir.exists()) return true
        return versionDir.deleteRecursively()
    }

    private fun validateSegment(value: String, field: String) {
        require(SAFE_SEGMENT.matches(value)) { "$field contains unsupported characters" }
    }

    private companion object {
        val SAFE_SEGMENT = Regex("^[A-Za-z0-9._-]+$")
    }
}
