package com.ikegami99.kiraenhance.inference.tile

object AutoTileSizer {
    fun initial(
        totalRamMb: Long?,
        hasUsableGpu: Boolean,
    ): Int {
        if (!hasUsableGpu || totalRamMb == null || totalRamMb <= 0L) {
            return 128
        }

        return when {
            totalRamMb >= 12_288L -> 384
            totalRamMb >= 8_192L -> 256
            totalRamMb >= 6_144L -> 192
            else -> 128
        }
    }
}
