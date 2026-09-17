package com.ikegami99.kiraenhance.inference.mnn

enum class MnnPisaBackend {
    OPENCL,
    VULKAN,
    CPU;

    companion object {
        fun preference(preferGpu: Boolean): List<MnnPisaBackend> =
            if (preferGpu) {
                listOf(OPENCL, VULKAN, CPU)
            } else {
                listOf(CPU)
            }
    }
}
