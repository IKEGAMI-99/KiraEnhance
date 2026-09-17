package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.UpscaleEngine
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
            )

        model.mode == EnhancementMode.BALANCED && model.backend == ModelBackend.MNN ->
            EnhanceEngineBinding(
                engine = mnnPisaProvider(),
                requestFactory = pisaRequestFactory,
                outputScale = 4,
                saveModeName = "PiSA-SR",
            )

        else -> throw IllegalArgumentException(
            "Unsupported enhancement model: ${model.id} (${model.mode}/${model.backend})",
        )
    }
}
