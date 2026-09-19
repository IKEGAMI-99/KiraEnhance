package com.ikegami99.kiraenhance.inference.mnn

object MnnPisaNativePrepareResultCodec {
    fun decode(payload: LongArray): MnnPisaNativePrepareResult {
        if (payload.size != PAYLOAD_SIZE) {
            return internalFailure()
        }

        val errorOrdinal = payload[ERROR_INDEX]
        if (errorOrdinal < 0L || errorOrdinal > Int.MAX_VALUE.toLong()) {
            return internalFailure()
        }
        val errorCode = MnnPisaNativeError.entries.getOrNull(errorOrdinal.toInt())
            ?: return internalFailure()

        val imageWidth = payload[IMAGE_WIDTH_INDEX].toNonNegativeInt()
            ?: return internalFailure()
        val imageHeight = payload[IMAGE_HEIGHT_INDEX].toNonNegativeInt()
            ?: return internalFailure()
        val latentWidth = payload[LATENT_WIDTH_INDEX].toNonNegativeInt()
            ?: return internalFailure()
        val latentHeight = payload[LATENT_HEIGHT_INDEX].toNonNegativeInt()
            ?: return internalFailure()

        if (
            errorCode == MnnPisaNativeError.NONE &&
            (imageWidth <= 0 || imageHeight <= 0 || latentWidth <= 0 || latentHeight <= 0)
        ) {
            return internalFailure()
        }

        return MnnPisaNativePrepareResult(
            errorCode = errorCode,
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            latentWidth = latentWidth,
            latentHeight = latentHeight,
        )
    }

    private fun Long.toNonNegativeInt(): Int? =
        takeIf { it in 0L..Int.MAX_VALUE.toLong() }?.toInt()

    private fun internalFailure() = MnnPisaNativePrepareResult(
        errorCode = MnnPisaNativeError.INTERNAL,
        imageWidth = 0,
        imageHeight = 0,
        latentWidth = 0,
        latentHeight = 0,
    )

    private const val PAYLOAD_SIZE = 5
    private const val ERROR_INDEX = 0
    private const val IMAGE_WIDTH_INDEX = 1
    private const val IMAGE_HEIGHT_INDEX = 2
    private const val LATENT_WIDTH_INDEX = 3
    private const val LATENT_HEIGHT_INDEX = 4
}
