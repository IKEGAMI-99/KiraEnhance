package com.ikegami99.kiraenhance.diagnostics

import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.io.path.createTempDirectory
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticLogStoreTest {
    @Test
    fun appendPersistsTimestampedEventAndExportCopiesIt() {
        val dir = createTempDirectory(prefix = "kira-log-test-").toFile()
        val file = File(dir, "kiraenhance.log")
        val store = DiagnosticLogStore(
            file = file,
            nowMillis = { 1_700_000_000_000L },
        )

        store.append("ModelDownloadWorker", "HTTP 200 model.bin")

        val output = ByteArrayOutputStream()
        val written = store.copyTo(output)
        val text = output.toString(Charsets.UTF_8.name())

        assertTrue(written > 0L)
        assertEquals(output.size().toLong(), written)
        assertTrue(text.contains("ModelDownloadWorker"))
        assertTrue(text.contains("HTTP 200 model.bin"))
        assertTrue(text.lines().first().matches(Regex("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}.*")))
    }

    @Test
    fun missingLogStillExportsReadableNonEmptyText() {
        val dir = createTempDirectory(prefix = "kira-empty-log-test-").toFile()
        val store = DiagnosticLogStore(
            file = File(dir, "missing.log"),
            nowMillis = { 1_700_000_000_000L },
        )
        val output = ByteArrayOutputStream()

        val written = store.copyTo(output)
        val text = output.toString(Charsets.UTF_8.name())

        assertTrue(written > 0L)
        assertEquals(output.size().toLong(), written)
        assertTrue(text.contains("diagnostic log is empty"))
    }
}
