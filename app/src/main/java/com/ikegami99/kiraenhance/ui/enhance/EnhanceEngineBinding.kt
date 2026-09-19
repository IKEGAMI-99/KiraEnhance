package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.UpscaleEngine

data class EnhanceOutputSize(
    val width: Int,
    val height: Int,
)

fun interface EnhanceOutputSizePlanner {
    fun plan(inputWidth: Int, inputHeight: Int, outputScale: Int): EnhanceOutputSize?
}

private val ExactScaleOutputSizePlanner = EnhanceOutputSizePlanner { width, height, scale ->
    if (width <= 0 || height <= 0 || scale <= 0) {
        null
    } else {
        val outputWidth = width.toLong() * scale.toLong()
        val outputHeight = height.toLong() * scale.toLong()
        if (
            outputWidth !in 1..Int.MAX_VALUE.toLong() ||
            outputHeight !in 1..Int.MAX_VALUE.toLong()
        ) {
            null
        } else {
            EnhanceOutputSize(
                width = outputWidth.toInt(),
                height = outputHeight.toInt(),
            )
        }
    }
}

data class EnhanceEngineBinding(
    val engine: UpscaleEngine,
    val requestFactory: ModelRequestFactory,
    val outputScale: Int,
    val saveModeName: String,
    val outputSizePlanner: EnhanceOutputSizePlanner = ExactScaleOutputSizePlanner,
) {
    fun plannedOutputSize(inputWidth: Int, inputHeight: Int): EnhanceOutputSize? =
        outputSizePlanner.plan(inputWidth, inputHeight, outputScale)
}
