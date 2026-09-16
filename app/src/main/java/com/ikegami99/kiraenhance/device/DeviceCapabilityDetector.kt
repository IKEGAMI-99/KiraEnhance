package com.ikegami99.kiraenhance.device

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.StatFs

class DeviceCapabilityDetector(
    private val context: Context,
) {
    fun detect(): DeviceCapabilities {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val memoryInfo = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val totalRamMb = memoryInfo.totalMem / BYTES_PER_MB

        val statFs = StatFs(context.filesDir.absolutePath)
        val availableStorageMb = statFs.availableBytes / BYTES_PER_MB

        val packageManager = context.packageManager
        val hasVulkan12 = packageManager.hasSystemFeature(
            PackageManager.FEATURE_VULKAN_HARDWARE_VERSION,
            VULKAN_1_2,
        )
        val is64Bit = Build.SUPPORTED_64_BIT_ABIS.isNotEmpty()

        return DeviceCapabilities(
            totalRamMb = totalRamMb,
            availableStorageMb = availableStorageMb,
            hasVulkan12 = hasVulkan12,
            is64Bit = is64Bit,
            deviceName = listOf(Build.MANUFACTURER, Build.MODEL)
                .filter { it.isNotBlank() }
                .joinToString(" ")
                .trim(),
            androidApi = Build.VERSION.SDK_INT,
            supportTier = classifyDevice(
                totalRamMb = totalRamMb,
                hasVulkan12 = hasVulkan12,
                is64Bit = is64Bit,
            ),
        )
    }

    private companion object {
        const val BYTES_PER_MB = 1024L * 1024L
        const val VULKAN_1_2 = (1 shl 22) or (2 shl 12)
    }
}
