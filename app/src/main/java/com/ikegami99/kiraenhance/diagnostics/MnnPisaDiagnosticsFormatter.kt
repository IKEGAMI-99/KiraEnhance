package com.ikegami99.kiraenhance.diagnostics

import com.ikegami99.kiraenhance.inference.mnn.MnnPisaDiagnosticsSnapshot

object MnnPisaDiagnosticsFormatter {
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
