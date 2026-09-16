package com.ikegami99.kiraenhance.model

import java.net.URI

private val SHA_256 = Regex("^[0-9a-fA-F]{64}$")

fun ModelDescriptor.isDownloadableProductionArtifactSet(): Boolean =
    artifacts.isNotEmpty() && artifacts.all { artifact ->
        val uri = runCatching { URI(artifact.downloadUrl) }.getOrNull()
        val host = uri?.host?.lowercase()

        uri?.scheme?.lowercase() == "https" &&
            !host.isNullOrBlank() &&
            host != "example.invalid" &&
            !host.endsWith(".invalid") &&
            artifact.fileSizeBytes > 0L &&
            SHA_256.matches(artifact.sha256) &&
            artifact.sha256.toSet().size > 1
    }

fun ModelDescriptor.requireDownloadable() {
    require(isDownloadableProductionArtifactSet()) {
        "Model $id does not have production download artifacts"
    }
}
