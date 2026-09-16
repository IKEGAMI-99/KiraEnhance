package com.ikegami99.kiraenhance.device

enum class SupportTier {
    RECOMMENDED,
    SUPPORTED,
    NOT_RECOMMENDED,
    UNSUPPORTED,
}

data class DeviceCapabilities(
    val totalRamMb: Long,
    val availableStorageMb: Long,
    val hasVulkan12: Boolean,
    val is64Bit: Boolean,
    val deviceName: String,
    val androidApi: Int,
    val supportTier: SupportTier,
)

fun classifyDevice(
    totalRamMb: Long,
    hasVulkan12: Boolean,
    is64Bit: Boolean,
): SupportTier = when {
    !is64Bit || !hasVulkan12 -> SupportTier.UNSUPPORTED
    totalRamMb >= 12_000 -> SupportTier.RECOMMENDED
    totalRamMb >= 8_000 -> SupportTier.SUPPORTED
    else -> SupportTier.NOT_RECOMMENDED
}
