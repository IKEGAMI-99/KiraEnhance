package com.ikegami99.kiraenhance.image

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

object EnhanceOutputNamer {
    fun png(
        instant: Instant,
        zoneId: ZoneId,
        mode: String,
        scale: Int,
    ): String {
        require(scale > 0) { "scale must be positive" }
        val safeMode = mode
            .trim()
            .replace(Regex("[^A-Za-z0-9_-]+"), "_")
            .trim('_')
            .ifBlank { "AI" }
        val timestamp = FORMATTER.withZone(zoneId).format(instant)
        return "KiraEnhance_${timestamp}_${safeMode}_${scale}x.png"
    }

    private val FORMATTER = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")
}
