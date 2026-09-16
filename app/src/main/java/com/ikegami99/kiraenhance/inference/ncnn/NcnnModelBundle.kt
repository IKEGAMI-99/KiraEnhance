package com.ikegami99.kiraenhance.inference.ncnn

import com.ikegami99.kiraenhance.inference.ModelLoadRequest
import java.io.File

data class NcnnModelBundle(
    val paramFile: File,
    val binFile: File,
) {
    companion object {
        fun resolve(request: ModelLoadRequest): NcnnModelBundle {
            val paramArtifacts = request.artifacts.filter { it.fileName.endsWith(".param", ignoreCase = true) }
            val binArtifacts = request.artifacts.filter { it.fileName.endsWith(".bin", ignoreCase = true) }

            require(paramArtifacts.size == 1) {
                "ncnn model requires exactly one .param artifact"
            }
            require(binArtifacts.size == 1) {
                "ncnn model requires exactly one .bin artifact"
            }

            return NcnnModelBundle(
                paramFile = File(paramArtifacts.single().absolutePath),
                binFile = File(binArtifacts.single().absolutePath),
            )
        }
    }
}
