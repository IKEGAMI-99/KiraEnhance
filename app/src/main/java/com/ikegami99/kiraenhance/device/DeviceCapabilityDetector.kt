package com.ikegami99.kiraenhance.device

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.StatFs
import java.util.Locale

class DeviceCapabilityDetector(
    private val context: Context,
) {
    fun detect(): DeviceCapabilities {
        val activityManager = context.getSystemService(ActivityManager::class.java)
        val memoryInfo = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val totalRamMb = memoryInfo.totalMem / BYTES_PER_MIB

        val statFs = StatFs(context.filesDir.absolutePath)
        val availableStorageMb = statFs.availableBytes / BYTES_PER_MIB

        val packageManager = context.packageManager
        val vulkanFeatureVersion = packageManager.systemAvailableFeatures
            ?.firstOrNull { it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION }
            ?.version
            ?: 0
        val hasVulkan12 = packageManager.hasSystemFeature(
            PackageManager.FEATURE_VULKAN_HARDWARE_VERSION,
            VULKAN_1_2_VERSION,
        )

        val is64Bit = Build.SUPPORTED_64_BIT_ABIS.isNotEmpty()

        return DeviceCapabilities(
            totalRamMb = totalRamMb,
            availableStorageMb = availableStorageMb,
            vulkanVersion = formatVulkanVersion(vulkanFeatureVersion),
            deviceName = buildDeviceName(),
            androidApi = Build.VERSION.SDK_INT,
            supportTier = classifyDevice(
                totalRamMb = totalRamMb,
                hasVulkan12 = hasVulkan12,
                is64Bit = is64Bit,
            ),
        )
    }

    private fun buildDeviceName(): String {
        val manufacturer = Build.MANUFACTURER.orEmpty().trim()
        val model = Build.MODEL.orEmpty().trim()
        if (manufacturer.isBlank()) return model.ifBlank { "Unknown Android device" }
        if (model.isBlank()) return manufacturer

        return if (model.startsWith(manufacturer, ignoreCase = true)) {
            model
        } else {
            "${manufacturer.replaceFirstChar { it.titlecase(Locale.ROOT) }} $model"
        }
    }

    private fun formatVulkanVersion(version: Int): String {
        if (version <= 0) return "Unavailable"
        val major = version ushr 22
        val minor = (version ushr 12) and 0x3ff
        val patch = version and 0xfff
        return if (patch == 0) "$major.$minor" else "$major.$minor.$patch"
    }

    private companion object {
        const val BYTES_PER_MIB = 1024L * 1024L
        const val VULKAN_1_2_VERSION = (1 shl 22) or (2 shl 12)
    }
}
