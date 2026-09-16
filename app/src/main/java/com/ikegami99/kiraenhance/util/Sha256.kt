package com.ikegami99.kiraenhance.util

import java.io.File
import java.security.MessageDigest

object Sha256 {
    private const val BUFFER_SIZE = 128 * 1024
    private val HEX = "0123456789abcdef".toCharArray()

    fun hex(file: File): String = digest(file).toHex()

    fun matches(file: File, expectedHex: String): Boolean {
        val expected = expectedHex.hexToBytesOrNull() ?: return false
        return MessageDigest.isEqual(digest(file), expected)
    }

    private fun digest(file: File): ByteArray {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered(BUFFER_SIZE).use { input ->
            val buffer = ByteArray(BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read < 0) break
                if (read > 0) digest.update(buffer, 0, read)
            }
        }
        return digest.digest()
    }

    private fun ByteArray.toHex(): String {
        val chars = CharArray(size * 2)
        forEachIndexed { index, byte ->
            val value = byte.toInt() and 0xff
            chars[index * 2] = HEX[value ushr 4]
            chars[index * 2 + 1] = HEX[value and 0x0f]
        }
        return String(chars)
    }

    private fun String.hexToBytesOrNull(): ByteArray? {
        val normalized = lowercase()
        if (normalized.length != 64 || normalized.any { it !in '0'..'9' && it !in 'a'..'f' }) return null

        return ByteArray(normalized.length / 2) { index ->
            val high = normalized[index * 2].digitToInt(16)
            val low = normalized[index * 2 + 1].digitToInt(16)
            ((high shl 4) or low).toByte()
        }
    }
}
