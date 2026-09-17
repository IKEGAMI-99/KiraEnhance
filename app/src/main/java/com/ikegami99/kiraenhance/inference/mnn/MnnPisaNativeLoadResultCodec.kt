package com.ikegami99.kiraenhance.inference.mnn

object MnnPisaNativeLoadResultCodec {
    fun decode(payload: LongArray): MnnPisaNativeLoadResult {
        if (payload.size != PAYLOAD_SIZE) {
            return internalFailure()
        }

        val errorOrdinal = payload[ERROR_INDEX]
        if (errorOrdinal < 0L || errorOrdinal > Int.MAX_VALUE.toLong()) {
            return internalFailure()
        }
        val errorCode = MnnPisaNativeError.entries.getOrNull(errorOrdinal.toInt())
            ?: return internalFailure()

        val gpuEnabled = when (payload[GPU_INDEX]) {
            0L -> false
            1L -> true
            else -> return internalFailure()
        }

        val handle = payload[HANDLE_INDEX]
        if (errorCode == MnnPisaNativeError.NONE && handle == 0L) {
            return internalFailure()
        }

        return MnnPisaNativeLoadResult(
            handle = handle,
            errorCode = errorCode,
            gpuEnabled = gpuEnabled,
        )
    }

    private fun internalFailure() = MnnPisaNativeLoadResult(
        handle = 0L,
        errorCode = MnnPisaNativeError.INTERNAL,
        gpuEnabled = false,
    )

    private const val PAYLOAD_SIZE = 3
    private const val HANDLE_INDEX = 0
    private const val ERROR_INDEX = 1
    private const val GPU_INDEX = 2
}
