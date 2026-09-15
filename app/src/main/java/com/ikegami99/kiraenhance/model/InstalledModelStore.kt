package com.ikegami99.kiraenhance.model

import android.content.Context
import java.io.File

class InstalledModelStore(
    private val context: Context,
) {
    fun modelsDir(): File = File(context.filesDir, "models").apply { mkdirs() }

    fun modelVersionDir(modelId: String, version: String): File =
        File(File(modelsDir(), safeSegment(modelId)), safeSegment(version))

    fun artifactFile(modelId: String, version: String, fileName: String): File =
        File(modelVersionDir(modelId, version), safeSegment(fileName))

    fun partialVersionDir(modelId: String, version: String): File {
        val downloadRoot = File(context.cacheDir, "model-downloads").apply { mkdirs() }
        return File(File(downloadRoot, safeSegment(modelId)), safeSegment(version))
    }

    fun partialFile(modelId: String, version: String, fileName: String): File =
        File(partialVersionDir(modelId, version), "${safeSegment(fileName)}.part")

    fun isInstalled(model: ModelDescriptor): Boolean =
        model.artifacts.isNotEmpty() && model.artifacts.all { artifact ->
            val file = artifactFile(model.id, model.version, artifact.fileName)
            file.isFile && file.length() == artifact.fileSizeBytes
        }

    fun deleteModel(model: ModelDescriptor): Boolean {
        val modelDir = modelVersionDir(model.id, model.version)
        val partialDir = partialVersionDir(model.id, model.version)
        val modelDeleted = !modelDir.exists() || modelDir.deleteRecursively()
        val partialDeleted = !partialDir.exists() || partialDir.deleteRecursively()
        return modelDeleted && partialDeleted
    }

    private fun safeSegment(value: String): String {
        require(SAFE_SEGMENT.matches(value)) { "Unsafe model path segment" }
        return value
    }

    private companion object {
        val SAFE_SEGMENT = Regex("^[A-Za-z0-9][A-Za-z0-9._-]*$")
    }
}
