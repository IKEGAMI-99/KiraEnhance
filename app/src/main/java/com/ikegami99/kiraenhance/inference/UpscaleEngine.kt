package com.ikegami99.kiraenhance.inference

import java.nio.ByteBuffer

data class UpscaleInput(
    val width: Int,
    val height: Int,
    val rowStrideBytes: Int,
    val pixelFormat: PixelFormat,
    val pixels: ByteBuffer,
) {
    init {
        require(width > 0 && height > 0) { "Input dimensions must be positive" }
        require(rowStrideBytes >= width * BYTES_PER_RGBA_PIXEL) {
            "rowStrideBytes is smaller than one RGBA row"
        }
        require(pixels.remaining().toLong() >= rowStrideBytes.toLong() * height.toLong()) {
            "Pixel buffer is smaller than the declared image"
        }
    }

    private companion object {
        const val BYTES_PER_RGBA_PIXEL = 4
    }
}

data class UpscaleSettings(
    val outputScale: Int,
    val useGpu: Boolean = true,
    val tileSize: Int? = null,
) {
    init {
        require(outputScale > 0) { "outputScale must be positive" }
        require(tileSize == null || tileSize >= 32) { "tileSize must be at least 32 when set" }
    }
}

data class UpscaleProgress(
    val completedTiles: Int = 0,
    val totalTiles: Int = 0,
) {
    init {
        require(completedTiles >= 0) { "completedTiles must not be negative" }
        require(totalTiles >= 0) { "totalTiles must not be negative" }
        require(completedTiles <= totalTiles) { "completedTiles must not exceed totalTiles" }
    }

    val fraction: Float
        get() = if (totalTiles == 0) 0f else completedTiles.toFloat() / totalTiles.toFloat()
}

data class UpscaleOutput(
    val width: Int,
    val height: Int,
    val rowStrideBytes: Int,
    val pixelFormat: PixelFormat,
    val pixels: ByteBuffer,
) {
    init {
        require(width > 0 && height > 0) { "Output dimensions must be positive" }
    }
}

sealed interface UpscaleResult {
    data class Success(
        val output: UpscaleOutput,
        val usedGpu: Boolean,
    ) : UpscaleResult

    data class Failed(
        val error: EngineError,
    ) : UpscaleResult
}

interface UpscaleEngine : AutoCloseable {
    fun load(request: ModelLoadRequest): ModelLoadResult

    fun isLoaded(): Boolean

    fun progress(): UpscaleProgress

    fun upscale(
        input: UpscaleInput,
        settings: UpscaleSettings,
    ): UpscaleResult

    fun cancel()

    fun unload()

    override fun close() {
        unload()
    }
}
