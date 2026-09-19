package com.ikegami99.kiraenhance.inference.mnn

data class MnnPisaSessionInfo(
    val backend: MnnPisaBackend,
    val gpuEnabled: Boolean,
) {
    companion object {
        fun parse(raw: String): MnnPisaSessionInfo? {
            if (raw.isBlank()) {
                return null
            }

            val fields = LinkedHashMap<String, String>()
            for (part in raw.split(';')) {
                val separator = part.indexOf('=')
                if (separator <= 0 || separator == part.lastIndex) {
                    return null
                }
                val key = part.substring(0, separator)
                val value = part.substring(separator + 1)
                if (fields.put(key, value) != null) {
                    return null
                }
            }

            if (fields.keys != setOf("backend", "gpu")) {
                return null
            }

            val backend = when (fields["backend"]) {
                "opencl" -> MnnPisaBackend.OPENCL
                "vulkan" -> MnnPisaBackend.VULKAN
                "cpu" -> MnnPisaBackend.CPU
                else -> return null
            }
            val gpuEnabled = when (fields["gpu"]) {
                "1" -> true
                "0" -> false
                else -> return null
            }

            val expectedGpu = backend != MnnPisaBackend.CPU
            if (gpuEnabled != expectedGpu) {
                return null
            }

            return MnnPisaSessionInfo(
                backend = backend,
                gpuEnabled = gpuEnabled,
            )
        }
    }
}
