package com.ikegami99.kiraenhance.download

internal object RangeResumePolicy {
    fun shouldRestartFromZero(responseCode: Int, resumeOffset: Long): Boolean =
        responseCode == 416 && resumeOffset > 0L
}
