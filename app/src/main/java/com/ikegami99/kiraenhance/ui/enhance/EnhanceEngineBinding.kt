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

data class EnhanceEngineBinding(
    val engine: UpscaleEngine,
    val requestFactory: ModelRequestFactory,
    val outputScale: Int,
    val saveModeName: String,
    val outputSizePlanner: EnhanceOutputSizePlanner,
) {
    fun plannedOutputSize(inputWidth: Int, inputHeight: Int): EnhanceOutputSize? =
        outputSizePlanner.plan(inputWidth, inputHeight, outputScale)
}
