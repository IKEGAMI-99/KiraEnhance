package com.ikegami99.kiraenhance.util

import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class Sha256Test {
    @Test
    fun hashesFilesWithoutLoadingThemAsText() {
        val file = Files.createTempFile("kira-sha", ".bin").toFile()
        try {
            file.writeBytes("abc".toByteArray(Charsets.UTF_8))

            assertEquals(
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                Sha256.hex(file),
            )
        } finally {
            file.delete()
        }
    }

    @Test
    fun verifiesDigestCaseInsensitively() {
        val file = Files.createTempFile("kira-sha", ".bin").toFile()
        try {
            file.writeBytes("abc".toByteArray(Charsets.UTF_8))

            assertTrue(
                Sha256.matches(
                    file,
                    "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
                ),
            )
            assertFalse(Sha256.matches(file, "0".repeat(64)))
        } finally {
            file.delete()
        }
    }
}
