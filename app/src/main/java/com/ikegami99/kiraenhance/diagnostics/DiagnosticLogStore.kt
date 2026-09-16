package com.ikegami99.kiraenhance.diagnostics

import java.io.File
import java.io.OutputStream
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

class DiagnosticLogStore(
    private val file: File,
    private val nowMillis: () -> Long = System::currentTimeMillis,
) {
    fun append(tag: String, message: String) {
        val line = buildString {
            append(TIMESTAMP_FORMATTER.format(Instant.ofEpochMilli(nowMillis())))
            append(" [")
            append(sanitize(tag))
            append("] ")
            append(sanitize(message))
            append('\n')
        }

        synchronized(FILE_LOCK) {
            file.parentFile?.mkdirs()
            file.appendText(line, Charsets.UTF_8)
        }
    }

    fun copyTo(output: OutputStream): Long {
        synchronized(FILE_LOCK) {
            if (!file.isFile) {
                val fallback = "KiraEnhance diagnostic log is empty.\n".toByteArray(Charsets.UTF_8)
                output.write(fallback)
                return fallback.size.toLong()
            }
            return file.inputStream().use { input -> input.copyTo(output) }
        }
    }

    companion object {
        private val FILE_LOCK = Any()
        private val TIMESTAMP_FORMATTER = DateTimeFormatter
            .ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSSXXX")
            .withZone(ZoneId.systemDefault())
        private val FILE_NAME_FORMATTER = DateTimeFormatter
            .ofPattern("yyyyMMdd-HHmmss")
            .withZone(ZoneId.systemDefault())

        fun exportFileName(nowMillis: Long = System.currentTimeMillis()): String =
            "KiraEnhance-log-${FILE_NAME_FORMATTER.format(Instant.ofEpochMilli(nowMillis))}.txt"

        private fun sanitize(value: String): String = value
            .replace("\r", "\\r")
            .replace("\n", "\\n")
    }
}
