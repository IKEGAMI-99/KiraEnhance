package com.ikegami99.kiraenhance.ui.enhance

import com.ikegami99.kiraenhance.inference.ModelRequestFactory
import com.ikegami99.kiraenhance.inference.UpscaleEngine

data class EnhanceEngineBinding(
    val engine: UpscaleEngine,
    val requestFactory: ModelRequestFactory,
    val outputScale: Int,
    val saveModeName: String,
)
