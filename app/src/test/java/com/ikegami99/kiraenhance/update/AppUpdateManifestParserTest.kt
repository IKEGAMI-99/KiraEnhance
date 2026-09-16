package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class AppUpdateManifestParserTest {
    private val parser = AppUpdateManifestParser()

    @Test
    fun parsesValidManifest() {
        val info = parser.parse(
            """
            {
              "versionName": "0.1.0-alpha02",
              "versionCode": 2,
              "releaseNotes": "Updater test build",
              "apkUrl": "https://github.com/IKEGAMI-99/KiraEnhance/releases/download/v0.1.0-alpha02/KiraEnhance-0.1.0-alpha02.apk",
              "apkSha256": "${"a".repeat(64)}"
            }
            """.trimIndent(),
        )

        assertEquals("0.1.0-alpha02", info.versionName)
        assertEquals(2, info.versionCode)
        assertEquals("a".repeat(64), info.apkSha256)
    }

    @Test
    fun rejectsNonHttpsApkUrl() {
        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(
                """
                {
                  "versionName": "0.1.0-alpha02",
                  "versionCode": 2,
                  "releaseNotes": "test",
                  "apkUrl": "http://example.invalid/app.apk",
                  "apkSha256": "${"a".repeat(64)}"
                }
                """.trimIndent(),
            )
        }
    }

    @Test
    fun rejectsInvalidShaAndVersionCode() {
        assertThrows(IllegalArgumentException::class.java) {
            parser.parse(
                """
                {
                  "versionName": "0.1.0-alpha02",
                  "versionCode": 0,
                  "releaseNotes": "test",
                  "apkUrl": "https://example.com/app.apk",
                  "apkSha256": "bad"
                }
                """.trimIndent(),
            )
        }
    }
}
