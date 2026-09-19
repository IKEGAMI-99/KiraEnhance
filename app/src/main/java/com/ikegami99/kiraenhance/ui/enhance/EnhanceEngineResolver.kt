package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.UpscaleEngine
import com.ikegami99.kiraenhance.inference.mnn.PisaResizeGeometryPlanner
import com.ikegami99.kiraenhance.inference.ncnn.UltraSharpModelRequestFactory
import com.ikegami99.kiraenhance.model.EnhancementMode
import com.ikegami99.kiraenhance.model.ModelBackend
import com.ikegami99.kiraenhance.model.ModelDescriptor

class EnhanceEngineResolver(
    private val ncnnProvider: () -> UpscaleEngine,
    private val mnnPisaProvider: () -> UpscaleEngine,
    private val pisaRequestFactory: ModelRequestFactory,
) {
    fun resolve(model: ModelDescriptor): EnhanceEngineBinding = when {
        model.mode == EnhancementMode.ULTRASHARP && model.backend == ModelBackend.NCNN ->
            EnhanceEngineBinding(
                engine = ncnnProvider(),
                requestFactory = UltraSharpModelRequestFactory,
                outputScale = 4,
                saveModeName = "UltraSharp",
                outputSizePlanner = EnhanceOutputSizePlanner { width, height, scale ->
                    exactScaledOutput(width, height, scale)
                },
            )

        model.mode == EnhancementMode.BALANCED && model.backend == ModelBackend.MNN ->
            EnhanceEngineBinding(
                engine = mnnPisaProvider(),
                requestFactory = pisaRequestFactory,
                outputScale = 4,
                saveModeName = "PiSA-SR",
                outputSizePlanner = EnhanceOutputSizePlanner { width, height, scale ->
                    PisaResizeGeometryPlanner.build(
                        sourceWidth = width,
                        sourceHeight = height,
                        upscale = scale,
                    )?.let { geometry ->
                        EnhanceOutputSize(
                            width = geometry.outputWidth,
                            height = geometry.outputHeight,
                        )
                    }
                },
            )

        else -> throw IllegalArgumentException(
            "Unsupported enhancement model: ${model.id} (${model.mode}/${model.backend})",
        )
    }
}


private fun exactScaledOutput(
    width: Int,
    height: Int,
    scale: Int,
): EnhanceOutputSize? {
    if (width <= 0 || height <= 0 || scale <= 0) {
        return null
    }
    val outputWidth = width.toLong() * scale.toLong()
    val outputHeight = height.toLong() * scale.toLong()
    if (
        outputWidth !in 1..Int.MAX_VALUE.toLong() ||
        outputHeight !in 1..Int.MAX_VALUE.toLong()
    ) {
        return null
    }
    return EnhanceOutputSize(
        width = outputWidth.toInt(),
        height = outputHeight.toInt(),
    )
}
