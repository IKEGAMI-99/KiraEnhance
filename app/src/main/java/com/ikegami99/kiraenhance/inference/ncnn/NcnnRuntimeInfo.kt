package com.ikegami99.kiraenhance.inference.ncnn

data class NcnnRuntimeInfo(
    val ncnnVersion: String,
    val vulkanCompiled: Boolean,
    val gpuCount: Int,
) {
    val hasUsableVulkanGpu: Boolean
        get() = vulkanCompiled && gpuCount > 0

    companion object {
        fun parse(raw: String): NcnnRuntimeInfo {
            val values = linkedMapOf<String, String>()

            raw.split(';').forEach { token ->
                val separator = token.indexOf('=')
                require(separator > 0 && separator < token.lastIndex) {
                    "Malformed ncnn runtime info"
                }

                val key = token.substring(0, separator).trim()
                val value = token.substring(separator + 1).trim()
                require(key.isNotEmpty() && value.isNotEmpty()) {
                    "Malformed ncnn runtime info"
                }
                require(values.put(key, value) == null) {
                    "Duplicate ncnn runtime info key: $key"
                }
            }

            val version = values["ncnn"]?.takeIf { it.isNotBlank() }
                ?: throw IllegalArgumentException("Missing ncnn version")

            val vulkan = when (values["vulkan"]) {
                "0" -> false
                "1" -> true
                else -> throw IllegalArgumentException("Invalid Vulkan flag")
            }

            val gpuCount = values["gpuCount"]?.toIntOrNull()
                ?: throw IllegalArgumentException("Invalid GPU count")
            require(gpuCount >= 0) { "GPU count must not be negative" }

            val supportedKeys = setOf("ncnn", "vulkan", "gpuCount")
            require(values.keys.all { it in supportedKeys }) {
                "Unknown ncnn runtime info key"
            }

            return NcnnRuntimeInfo(
                ncnnVersion = version,
                vulkanCompiled = vulkan,
                gpuCount = gpuCount,
            )
        }
    }
}
