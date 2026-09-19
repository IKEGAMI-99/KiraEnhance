package com.ikegami99.kiraenhance.diagnostics

import com.ikegami99.kiraenhance.inference.mnn.MnnPisaDiagnosticsSnapshot
import com.ikegami99.kiraenhance.inference.mnn.MnnPisaInferenceDiagnostics

object MnnPisaDiagnosticsFormatter {
    fun formatInference(diagnostic: MnnPisaInferenceDiagnostics): String = buildString {
        append("source=")
        append(diagnostic.sourceWidth)
        append("x")
        append(diagnostic.sourceHeight)
        append(" pre=")
        append(diagnostic.preUpscaleWidth)
        append("x")
        append(diagnostic.preUpscaleHeight)
        append(" rawModel=")
        append(diagnostic.rawModelWidth)
        append("x")
        append(diagnostic.rawModelHeight)
        append(" model=")
        append(diagnostic.modelWidth)
        append("x")
        append(diagnostic.modelHeight)
        append(" vae=")
        append(if (diagnostic.segmentedVae) "segmented" else "monolithic")
        append(" output=")
        append(diagnostic.outputWidth)
        append("x")
        append(diagnostic.outputHeight)
        append(" boosted=")
        append(diagnostic.smallInputBoosted)
        append(" seed=")
        append(diagnostic.noiseSeed)
        append(" backend=")
        append(diagnostic.backend?.name?.lowercase() ?: "unknown")
        append(" nativeError=")
        append(diagnostic.nativeError.name.lowercase())
        append(" nativeOutput=")
        append(diagnostic.nativeOutputWidth)
        append("x")
        append(diagnostic.nativeOutputHeight)
        append(" gpu=")
        append(diagnostic.gpuUsed)
        append(" vaePeakBytes=")
        append(diagnostic.nativePeakTrackedBytes)
    }

    fun format(snapshot: MnnPisaDiagnosticsSnapshot): List<String> {
        val session = snapshot.sessionInfo
        val tensors = snapshot.tensorInfo

        val summary = buildString {
            append("modelVersion=")
            append(snapshot.modelVersion)
            append(" backend=")
            append(session?.backend?.name?.lowercase() ?: "unknown")
            append(" gpu=")
            append(session?.gpuEnabled?.toString() ?: "unknown")
            append(" tensorCount=")
            append(tensors?.size?.toString() ?: "unavailable")
        }

        if (tensors == null) {
            return listOf(summary)
        }

        return buildList {
            add(summary)
            tensors.forEach { tensor ->
                val shape = if (tensor.shape.isEmpty()) {
                    "scalar"
                } else {
                    tensor.shape.joinToString("x")
                }

                add(
                    buildString {
                        append("graph=")
                        append(tensor.graph.name.lowercase())
                        append(" role=")
                        append(tensor.role.name.lowercase())
                        append(" name=")
                        append(tensor.name)
                        append(" shape=")
                        append(shape)
                        append(" type=")
                        append(tensor.typeCode)
                        append("/")
                        append(tensor.typeBits)
                        append("/")
                        append(tensor.typeLanes)
                        append(" dim=")
                        append(tensor.dimensionType)
                    },
                )
            }
        }
    }
}
