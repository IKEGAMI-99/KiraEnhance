package com.ikegami99.kiraenhance.diagnostics

import java.io.ByteArrayOutputStream
import java.io.File
import org.junit.Assert.assertTrue
import org.junit.Test

class DiagnosticLogStoreTest {
    @Test
    fun appendPersistsTimestampedEventAndExportCopiesIt() {
        val dir = createTempDir(prefix = "kira-log-test-")
        val file = File(dir, "kiraenhance.log")
        val store = DiagnosticLogStore(
            file = file,
            nowMillis = { 1_700_000_000_000L },
        )

        store.append("ModelDownloadWorker", "HTTP 200 model.bin")

        val output = ByteArrayOutputStream()
        store.copyTo(output)
        val text = output.toString(Charsets.UTF_8.name())

        assertTrue(text.contains("ModelDownloadWorker"))
        assertTrue(text.contains("HTTP 200 model.bin"))
        assertTrue(text.lines().first().matches(Regex("\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}\\.\\d{3}.*")))
    }
}
