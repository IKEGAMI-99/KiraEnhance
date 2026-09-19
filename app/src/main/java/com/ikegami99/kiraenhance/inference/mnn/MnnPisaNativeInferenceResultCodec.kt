package com.ikegami99.kiraenhance.inference.mnn

object MnnPisaNativeInferenceResultCodec {
    fun decode(payload: LongArray): MnnPisaNativeInferenceResult {
        if (payload.size != PAYLOAD_SIZE) {
            return internalFailure()
        }

        val errorCode = decodeError(payload[ERROR_INDEX]) ?: return internalFailure()
        val gpuUsed = when (payload[GPU_INDEX]) {
            0L -> false
            1L -> true
            else -> return internalFailure()
        }

        val width = payload[WIDTH_INDEX].toPositiveIntOrZero() ?: return internalFailure()
        val height = payload[HEIGHT_INDEX].toPositiveIntOrZero() ?: return internalFailure()
        val rowStride = payload[ROW_STRIDE_INDEX].toPositiveIntOrZero() ?: return internalFailure()
        val segmentedVaePeakTrackedBytes =
            payload[SEGMENTED_VAE_PEAK_BYTES_INDEX].takeIf { it >= 0L }
                ?: return internalFailure()

        if (errorCode == MnnPisaNativeError.NONE) {
            if (width <= 0 || height <= 0 || rowStride < width * BYTES_PER_PIXEL) {
                return internalFailure()
            }
        }

        return MnnPisaNativeInferenceResult(
            errorCode = errorCode,
            outputWidth = width,
            outputHeight = height,
            outputRowStrideBytes = rowStride,
            gpuUsed = gpuUsed,
            segmentedVaePeakTrackedBytes = segmentedVaePeakTrackedBytes,
        )
    }

    private fun decodeError(value: Long): MnnPisaNativeError? {
        if (value < 0L || value > Int.MAX_VALUE.toLong()) {
            return null
        }
        return MnnPisaNativeError.entries.getOrNull(value.toInt())
    }

    private fun Long.toPositiveIntOrZero(): Int? {
        if (this < 0L || this > Int.MAX_VALUE.toLong()) {
            return null
        }
        return toInt()
    }

    private fun internalFailure() = MnnPisaNativeInferenceResult(
        errorCode = MnnPisaNativeError.INTERNAL,
        outputWidth = 0,
        outputHeight = 0,
        outputRowStrideBytes = 0,
        gpuUsed = false,
    )

    private const val PAYLOAD_SIZE = 6
    private const val ERROR_INDEX = 0
    private const val WIDTH_INDEX = 1
    private const val HEIGHT_INDEX = 2
    private const val ROW_STRIDE_INDEX = 3
    private const val GPU_INDEX = 4
    private const val SEGMENTED_VAE_PEAK_BYTES_INDEX = 5
    private const val BYTES_PER_PIXEL = 4
}
