package com.ikegami99.kiraenhance.inference.smoke


data class NcnnSmokeCropRect(
    val left: Int,
    val top: Int,
    val width: Int,
    val height: Int,
)

object NcnnSmokeCropPlanner {
    fun centerSquare(
        imageWidth: Int,
        imageHeight: Int,
        maxSide: Int = DEFAULT_MAX_SIDE,
    ): NcnnSmokeCropRect {
        require(imageWidth > 0 && imageHeight > 0) { "Image dimensions must be positive" }
        require(maxSide > 0) { "maxSide must be positive" }

        val side = minOf(imageWidth, imageHeight, maxSide)
        return NcnnSmokeCropRect(
            left = (imageWidth - side) / 2,
            top = (imageHeight - side) / 2,
            width = side,
            height = side,
        )
    }

    const val DEFAULT_MAX_SIDE = 256
}
