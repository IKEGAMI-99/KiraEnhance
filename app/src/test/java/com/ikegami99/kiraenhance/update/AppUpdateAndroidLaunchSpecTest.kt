package com.ikegami99.kiraenhance.update

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class AppUpdateAndroidLaunchSpecTest {
    @Test
    fun `unknown sources settings targets this package`() {
        val spec = AppUpdateAndroidLaunchSpec.unknownSources("com.ikegami99.kiraenhance")

        assertEquals("android.settings.MANAGE_UNKNOWN_APP_SOURCES", spec.action)
        assertEquals("package:com.ikegami99.kiraenhance", spec.data)
        assertNull(spec.mimeType)
        assertTrue(spec.newTask)
        assertEquals(false, spec.grantReadPermission)
    }

    @Test
    fun `package installer uses apk mime and temporary read permission`() {
        val spec = AppUpdateAndroidLaunchSpec.packageInstaller("content://kira/update.apk")

        assertEquals("android.intent.action.VIEW", spec.action)
        assertEquals("content://kira/update.apk", spec.data)
        assertEquals("application/vnd.android.package-archive", spec.mimeType)
        assertTrue(spec.newTask)
        assertTrue(spec.grantReadPermission)
    }
}
