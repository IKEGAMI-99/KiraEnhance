package com.ikegami99.kiraenhance.ui.enhance

enum class EnhanceStage {
    SELECTING,
    READY,
    PROCESSING,
    COMPLETED,
    SAVED,
    ERROR,
}

data class EnhanceUiState(
    val stage: EnhanceStage = EnhanceStage.SELECTING,
    val sourceWidth: Int? = null,
    val sourceHeight: Int? = null,
    val targetWidth: Int? = null,
    val targetHeight: Int? = null,
    val progressFraction: Float = 0f,
    val progressText: String? = null,
    val elapsedMs: Long = 0L,
    val status: String = "画像を選択してください",
    val savedLocation: String? = null,
)

sealed interface EnhanceEvent {
    data class ImageReady(val width: Int, val height: Int) : EnhanceEvent
    data object ProcessingStarted : EnhanceEvent
    data class Progress(
        val completedTiles: Int,
        val totalTiles: Int,
        val elapsedMs: Long,
    ) : EnhanceEvent
    data class ProcessingCompleted(
        val elapsedMs: Long,
        val outputWidth: Int,
        val outputHeight: Int,
    ) : EnhanceEvent
    data object Cancelled : EnhanceEvent
    data class Failed(val message: String) : EnhanceEvent
    data class Saved(val location: String) : EnhanceEvent
    data class SaveFailed(val message: String) : EnhanceEvent
}

object EnhanceUiReducer {
    fun reduce(state: EnhanceUiState, event: EnhanceEvent): EnhanceUiState = when (event) {
        is EnhanceEvent.ImageReady -> {
            require(event.width > 0 && event.height > 0) { "Image dimensions must be positive" }
            state.copy(
                stage = EnhanceStage.READY,
                sourceWidth = event.width,
                sourceHeight = event.height,
                targetWidth = event.width * OUTPUT_SCALE,
                targetHeight = event.height * OUTPUT_SCALE,
                progressFraction = 0f,
                progressText = null,
                elapsedMs = 0L,
                status = "UltraSharp 4x の準備ができました",
                savedLocation = null,
            )
        }

        EnhanceEvent.ProcessingStarted -> state.copy(
            stage = EnhanceStage.PROCESSING,
            progressFraction = 0f,
            progressText = null,
            elapsedMs = 0L,
            status = "4x高画質化を処理中...",
            savedLocation = null,
        )

        is EnhanceEvent.Progress -> {
            val safeTotal = event.totalTiles.coerceAtLeast(0)
            val safeCompleted = if (safeTotal == 0) {
                0
            } else {
                event.completedTiles.coerceIn(0, safeTotal)
            }
            state.copy(
                progressFraction = if (safeTotal == 0) 0f else safeCompleted.toFloat() / safeTotal.toFloat(),
                progressText = if (safeTotal == 0) null else "$safeCompleted / $safeTotal tiles",
                elapsedMs = event.elapsedMs.coerceAtLeast(0L),
                status = "4x高画質化を処理中...",
            )
        }

        is EnhanceEvent.ProcessingCompleted -> {
            require(event.outputWidth > 0 && event.outputHeight > 0) {
                "Output dimensions must be positive"
            }
            state.copy(
                stage = EnhanceStage.COMPLETED,
                targetWidth = event.outputWidth,
                targetHeight = event.outputHeight,
                progressFraction = 1f,
                elapsedMs = event.elapsedMs.coerceAtLeast(0L),
                status = "高画質化が完了しました",
            )
        }

        EnhanceEvent.Cancelled -> state.copy(
            stage = EnhanceStage.READY,
            progressFraction = 0f,
            progressText = null,
            status = "処理をキャンセルしました",
        )

        is EnhanceEvent.Failed -> state.copy(
            stage = EnhanceStage.ERROR,
            status = event.message,
        )

        is EnhanceEvent.Saved -> state.copy(
            stage = EnhanceStage.SAVED,
            status = "画像を保存しました",
            savedLocation = event.location,
        )

        is EnhanceEvent.SaveFailed -> state.copy(
            stage = EnhanceStage.COMPLETED,
            status = "保存に失敗しました: ${event.message}",
        )
    }

    private const val OUTPUT_SCALE = 4
}
