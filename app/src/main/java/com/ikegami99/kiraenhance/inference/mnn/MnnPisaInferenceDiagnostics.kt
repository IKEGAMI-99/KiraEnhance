package com.ikegami99.kiraenhance.inference.mnn

data class MnnPisaInferenceDiagnostics(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val preUpscaleWidth: Int,
    val preUpscaleHeight: Int,
    val rawModelWidth: Int,
    val rawModelHeight: Int,
    val modelWidth: Int,
    val modelHeight: Int,
    val outputWidth: Int,
    val outputHeight: Int,
    val smallInputBoosted: Boolean,
    val noiseSeed: Long,
    val backend: MnnPisaBackend?,
    val nativeError: MnnPisaNativeError,
    val nativeOutputWidth: Int,
    val nativeOutputHeight: Int,
    val gpuUsed: Boolean,
    val segmentedVae: Boolean = false,
    val nativePeakTrackedBytes: Long = 0L,
)
