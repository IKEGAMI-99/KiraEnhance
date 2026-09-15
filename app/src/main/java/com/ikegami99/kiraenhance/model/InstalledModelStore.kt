package com.ikegami99.kiraenhance.model

import android.content.Context
import java.io.File

class InstalledModelStore(
    private val context: Context,
) {
    fun modelsDir(): File = File(context.filesDir, "models").apply { mkdirs() }

    fun modelFile(modelId: String, version: String): File =
        File(File(File(modelsDir(), safeSegment(modelId)), safeSegment(version)), "model.bin")

    fun partialFile(modelId: String, version: String): File {
        val downloadDir = File(context.cacheDir, "model-downloads").apply { mkdirs() }
        return File(downloadDir, "${safeSegment(modelId)}-${safeSegment(version)}.part")
    }

    fun isInstalled(modelId: String, version: String): Boolean =
        modelFile(modelId, version).isFile

    private fun safeSegment(value: String): String {
        require(SAFE_SEGMENT.matches(value)) { "Unsafe model path segment" }
        return value
    }

    private companion object {
        val SAFE_SEGMENT = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}
