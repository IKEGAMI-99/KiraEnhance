package com.ikegami99.kiraenhance.inference.mnn

internal data class PisaResizeGeometry(
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
) {
    val modelPixels: Long
        get() = modelWidth.toLong() * modelHeight.toLong()

    val outputPixels: Long
        get() = outputWidth.toLong() * outputHeight.toLong()
}

internal object PisaResizeGeometryPlanner {
    fun build(
        sourceWidth: Int,
        sourceHeight: Int,
        processSize: Int = 512,
        upscale: Int = 4,
        modelMultiple: Int = 8,
    ): PisaResizeGeometry? {
        if (
            sourceWidth <= 0 ||
            sourceHeight <= 0 ||
            processSize <= 0 ||
            upscale <= 0 ||
            modelMultiple <= 0 ||
            processSize < upscale
        ) {
            return null
        }

        val minimumPreUpscale = processSize / upscale
        if (minimumPreUpscale <= 0) {
            return null
        }

        val smallInputBoosted =
            sourceWidth < minimumPreUpscale ||
                sourceHeight < minimumPreUpscale

        var preWidth = sourceWidth
        var preHeight = sourceHeight
        if (smallInputBoosted) {
            val minimumSide = minOf(sourceWidth, sourceHeight)
            val scale = minimumPreUpscale.toDouble() / minimumSide.toDouble()
            val scaledWidth = scale * sourceWidth.toDouble()
            val scaledHeight = scale * sourceHeight.toDouble()
            if (
                !scaledWidth.isFinite() ||
                !scaledHeight.isFinite() ||
                scaledWidth > Int.MAX_VALUE.toDouble() ||
                scaledHeight > Int.MAX_VALUE.toDouble()
            ) {
                return null
            }

            // Match Python int(...) truncation for the positive dimensions
            // used by the upstream PiSA-SR preprocessing path.
            preWidth = scaledWidth.toInt()
            preHeight = scaledHeight.toInt()
            if (preWidth <= 0 || preHeight <= 0) {
                return null
            }
        }

        val rawModelWidth = checkedScale(preWidth, upscale) ?: return null
        val rawModelHeight = checkedScale(preHeight, upscale) ?: return null
        val modelWidth = floorToMultiple(rawModelWidth, modelMultiple)
        val modelHeight = floorToMultiple(rawModelHeight, modelMultiple)
        if (modelWidth <= 0 || modelHeight <= 0) {
            return null
        }

        val outputWidth: Int
        val outputHeight: Int
        if (smallInputBoosted) {
            outputWidth = checkedScale(sourceWidth, upscale) ?: return null
            outputHeight = checkedScale(sourceHeight, upscale) ?: return null
        } else {
            outputWidth = modelWidth
            outputHeight = modelHeight
        }

        return PisaResizeGeometry(
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            preUpscaleWidth = preWidth,
            preUpscaleHeight = preHeight,
            rawModelWidth = rawModelWidth,
            rawModelHeight = rawModelHeight,
            modelWidth = modelWidth,
            modelHeight = modelHeight,
            outputWidth = outputWidth,
            outputHeight = outputHeight,
            smallInputBoosted = smallInputBoosted,
        )
    }

    private fun checkedScale(value: Int, scale: Int): Int? {
        val scaled = value.toLong() * scale.toLong()
        return if (scaled in 1..Int.MAX_VALUE.toLong()) {
            scaled.toInt()
        } else {
            null
        }
    }

    private fun floorToMultiple(value: Int, multiple: Int): Int =
        value - value % multiple
}
