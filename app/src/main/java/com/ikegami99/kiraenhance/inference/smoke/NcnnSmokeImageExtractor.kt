package com.ikegami99.kiraenhance.inference.smoke

object NcnnSmokeImageExtractor {
    fun crop(
        image: NcnnSmokeImage,
        rect: NcnnSmokeCropRect,
    ): NcnnSmokeImage {
        require(rect.left >= 0 && rect.top >= 0) { "Crop origin must not be negative" }
        require(rect.width > 0 && rect.height > 0) { "Crop dimensions must be positive" }
        require(rect.left + rect.width <= image.width) { "Crop exceeds image width" }
        require(rect.top + rect.height <= image.height) { "Crop exceeds image height" }

        val output = IntArray(rect.width * rect.height)
        repeat(rect.height) { row ->
            val sourceOffset = (rect.top + row) * image.width + rect.left
            val outputOffset = row * rect.width
            image.argbPixels.copyInto(
                destination = output,
                destinationOffset = outputOffset,
                startIndex = sourceOffset,
                endIndex = sourceOffset + rect.width,
            )
        }

        return NcnnSmokeImage(
            width = rect.width,
            height = rect.height,
            argbPixels = output,
        )
    }
}
