package com.ikegami99.kiraenhance.inference

import com.ikegami99.kiraenhance.model.ModelDescriptor

fun interface ModelRequestFactory {
    fun create(
        model: ModelDescriptor,
        artifactPath: (String) -> String,
    ): ModelLoadRequest
}
