package com.ikegami99.kiraenhance.inference.mnn

data class MnnPisaNativeSmokeResult(
    val errorCode: MnnPisaNativeError,
    val completedStages: Int,
    val outputFinite: Boolean,
)

object MnnPisaNativeSmokeResultCodec {
    fun decode(payload: LongArray): MnnPisaNativeSmokeResult {
        if (payload.size != PAYLOAD_SIZE) {
            return internalFailure()
        }

        val errorOrdinal = payload[ERROR_INDEX]
        if (errorOrdinal < 0L || errorOrdinal > Int.MAX_VALUE.toLong()) {
            return internalFailure()
        }
        val errorCode = MnnPisaNativeError.entries.getOrNull(errorOrdinal.toInt())
            ?: return internalFailure()

        val completedStagesLong = payload[COMPLETED_STAGES_INDEX]
        if (completedStagesLong !in 0L..MAX_STAGES.toLong()) {
            return internalFailure()
        }

        val finiteFlag = payload[OUTPUT_FINITE_INDEX]
        if (finiteFlag != 0L && finiteFlag != 1L) {
            return internalFailure()
        }

        val result = MnnPisaNativeSmokeResult(
            errorCode = errorCode,
            completedStages = completedStagesLong.toInt(),
            outputFinite = finiteFlag == 1L,
        )

        if (
            result.errorCode == MnnPisaNativeError.NONE &&
            (
                result.completedStages != MAX_STAGES ||
                !result.outputFinite
            )
        ) {
            return internalFailure()
        }

        return result
    }

    private fun internalFailure() = MnnPisaNativeSmokeResult(
        errorCode = MnnPisaNativeError.INTERNAL,
        completedStages = 0,
        outputFinite = false,
    )

    const val MAX_STAGES = 3

    private const val PAYLOAD_SIZE = 3
    private const val ERROR_INDEX = 0
    private const val COMPLETED_STAGES_INDEX = 1
    private const val OUTPUT_FINITE_INDEX = 2
}
