package com.ikegami99.kiraenhance.inference.mnn

data class MnnRuntimeInfo(
    val mnnVersion: String,
    val linked: Boolean,
) {
    companion object {
        fun parse(raw: String): MnnRuntimeInfo {
            val values = linkedMapOf<String, String>()

            raw.split(';').forEach { token ->
                val separator = token.indexOf('=')
                require(separator > 0 && separator < token.lastIndex) {
                    "Malformed MNN runtime info"
                }

                val key = token.substring(0, separator).trim()
                val value = token.substring(separator + 1).trim()
                require(key.isNotEmpty() && value.isNotEmpty()) {
                    "Malformed MNN runtime info"
                }
                require(values.put(key, value) == null) {
                    "Duplicate MNN runtime info key: $key"
                }
            }

            val version = values["mnn"]?.takeIf { it.isNotBlank() }
                ?: throw IllegalArgumentException("Missing MNN version")
            val linked = when (values["linked"]) {
                "0" -> false
                "1" -> true
                else -> throw IllegalArgumentException("Invalid MNN linked flag")
            }

            val supportedKeys = setOf("mnn", "linked")
            require(values.keys.all { it in supportedKeys }) {
                "Unknown MNN runtime info key"
            }

            return MnnRuntimeInfo(
                mnnVersion = version,
                linked = linked,
            )
        }
    }
}
