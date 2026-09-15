package com.ikegami99.kiraenhance.inference.tile

data class IntRect(
    val left: Int,
    val top: Int,
    val right: Int,
    val bottom: Int,
) {
    init {
        require(left >= 0) { "left must be non-negative" }
        require(top >= 0) { "top must be non-negative" }
        require(right >= left) { "right must be >= left" }
        require(bottom >= top) { "bottom must be >= top" }
    }

    val width: Int
        get() = right - left

    val height: Int
        get() = bottom - top
}

data class TileRegion(
    val core: IntRect,
    val input: IntRect,
    val cropFromUpscaledTile: IntRect,
    val outputCore: IntRect,
)

object TilePlanner {
    fun plan(
        imageWidth: Int,
        imageHeight: Int,
        tileSize: Int,
        padding: Int,
        scale: Int,
    ): List<TileRegion> {
        require(imageWidth > 0) { "imageWidth must be positive" }
        require(imageHeight > 0) { "imageHeight must be positive" }
        require(tileSize > 0) { "tileSize must be positive" }
        require(padding >= 0) { "padding must be non-negative" }
        require(scale > 0) { "scale must be positive" }

        val tiles = ArrayList<TileRegion>()
        var top = 0
        while (top < imageHeight) {
            val bottom = minOf(top + tileSize, imageHeight)
            var left = 0

            while (left < imageWidth) {
                val right = minOf(left + tileSize, imageWidth)
                val core = IntRect(left, top, right, bottom)

                val input = IntRect(
                    left = maxOf(0, core.left - padding),
                    top = maxOf(0, core.top - padding),
                    right = minOf(imageWidth, core.right + padding),
                    bottom = minOf(imageHeight, core.bottom + padding),
                )

                val cropLeft = (core.left - input.left) * scale
                val cropTop = (core.top - input.top) * scale
                val cropFromUpscaledTile = IntRect(
                    left = cropLeft,
                    top = cropTop,
                    right = cropLeft + core.width * scale,
                    bottom = cropTop + core.height * scale,
                )

                val outputCore = IntRect(
                    left = core.left * scale,
                    top = core.top * scale,
                    right = core.right * scale,
                    bottom = core.bottom * scale,
                )

                tiles += TileRegion(
                    core = core,
                    input = input,
                    cropFromUpscaledTile = cropFromUpscaledTile,
                    outputCore = outputCore,
                )

                left = right
            }

            top = bottom
        }

        return tiles
    }
}

object AdaptiveTileSizer {
    private val supportedSizes = intArrayOf(512, 384, 256, 192, 128, 96, 64)

    fun nextSmaller(current: Int): Int? =
        supportedSizes.firstOrNull { it < current }
}
