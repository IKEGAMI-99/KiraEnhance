package com.ikegami99.kiraenhance.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class InstalledModelStoreTest {
    @get:Rule
    val temporaryFolder = TemporaryFolder()

    @Test
    fun `builds final and partial model paths without creating fake installed state`() {
        val filesDir = temporaryFolder.newFolder("files")
        val cacheDir = temporaryFolder.newFolder("cache")
        val store = InstalledModelStore(filesDir = filesDir, cacheDir = cacheDir)

        assertEquals(
            filesDir.resolve("models/pisa-sr/1.0.0/model.bin"),
            store.modelFile("pisa-sr", "1.0.0"),
        )
        assertEquals(
            cacheDir.resolve("model-downloads/pisa-sr-1.0.0.part"),
            store.partialFile("pisa-sr", "1.0.0"),
        )
        assertFalse(store.isInstalled("pisa-sr", "1.0.0"))
    }
}
