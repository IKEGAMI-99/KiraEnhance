package com.ikegami99.kiraenhance.inference.mnn

import java.nio.ByteBuffer
import java.nio.charset.CodingErrorAction

enum class MnnPisaGraph {
    VAE_ENCODER,
    UNET,
    VAE_DECODER,
}

enum class MnnTensorRole {
    INPUT,
    OUTPUT,
}

data class MnnPisaTensorInfo(
    val graph: MnnPisaGraph,
    val role: MnnTensorRole,
    val name: String,
    val shape: List<Int>,
    val typeCode: Int,
    val typeBits: Int,
    val typeLanes: Int,
    val dimensionType: Int,
)

object MnnPisaTensorInfoCodec {
    private val expectedFields = setOf(
        "graph",
        "role",
        "nameHex",
        "shape",
        "type",
        "dim",
    )

    fun decode(raw: String): List<MnnPisaTensorInfo>? {
        if (raw.isEmpty()) {
            return emptyList()
        }

        val result = ArrayList<MnnPisaTensorInfo>()
        for (line in raw.split('\n')) {
            if (line.isEmpty()) {
                return null
            }
            result += decodeLine(line) ?: return null
        }
        return result
    }

    private fun decodeLine(line: String): MnnPisaTensorInfo? {
        val fields = LinkedHashMap<String, String>()
        for (part in line.split(';')) {
            val separator = part.indexOf('=')
            if (separator <= 0) {
                return null
            }

            val key = part.substring(0, separator)
            val value = part.substring(separator + 1)
            if (fields.put(key, value) != null) {
                return null
            }
        }

        if (fields.keys != expectedFields) {
            return null
        }

        val graph = when (fields["graph"]) {
            "vae_encoder" -> MnnPisaGraph.VAE_ENCODER
            "unet" -> MnnPisaGraph.UNET
            "vae_decoder" -> MnnPisaGraph.VAE_DECODER
            else -> return null
        }

        val role = when (fields["role"]) {
            "input" -> MnnTensorRole.INPUT
            "output" -> MnnTensorRole.OUTPUT
            else -> return null
        }

        val name = decodeUtf8Hex(fields["nameHex"] ?: return null) ?: return null
        val shape = decodeShape(fields["shape"] ?: return null) ?: return null
        val type = decodeType(fields["type"] ?: return null) ?: return null

        val dimensionType = fields["dim"]?.toIntOrNull() ?: return null
        if (dimensionType !in 0..2) {
            return null
        }

        return MnnPisaTensorInfo(
            graph = graph,
            role = role,
            name = name,
            shape = shape,
            typeCode = type[0],
            typeBits = type[1],
            typeLanes = type[2],
            dimensionType = dimensionType,
        )
    }

    private fun decodeShape(raw: String): List<Int>? {
        if (raw.isEmpty()) {
            return emptyList()
        }
        return raw.split(',').map { it.toIntOrNull() ?: return null }
    }

    private fun decodeType(raw: String): IntArray? {
        val parts = raw.split(',')
        if (parts.size != 3) {
            return null
        }

        val code = parts[0].toIntOrNull() ?: return null
        val bits = parts[1].toIntOrNull() ?: return null
        val lanes = parts[2].toIntOrNull() ?: return null

        if (code !in 0..255 || bits !in 1..255 || lanes !in 1..65535) {
            return null
        }

        return intArrayOf(code, bits, lanes)
    }

    private fun decodeUtf8Hex(raw: String): String? {
        if (raw.isEmpty() || raw.length % 2 != 0) {
            return null
        }

        val bytes = ByteArray(raw.length / 2)
        for (index in bytes.indices) {
            val offset = index * 2
            val value = raw.substring(offset, offset + 2).toIntOrNull(16)
                ?: return null
            bytes[index] = value.toByte()
        }

        return runCatching {
            Charsets.UTF_8
                .newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        }.getOrNull()
    }
}
