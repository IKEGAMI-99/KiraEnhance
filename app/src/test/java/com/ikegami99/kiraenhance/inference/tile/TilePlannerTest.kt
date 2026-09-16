package com.ikegami99.kiraenhance.inference.tile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TilePlannerTest {
    @Test
    fun exactTileBoundaryProducesFourCoreTiles() {
        val tiles = TilePlanner.plan(
            imageWidth = 512,
            imageHeight = 512,
            tileSize = 256,
            padding = 10,
            scale = 4,
        )

        assertEquals(4, tiles.size)
        assertEquals(IntRect(0, 0, 256, 256), tiles[0].core)
        assertEquals(IntRect(256, 0, 512, 256), tiles[1].core)
        assertEquals(IntRect(0, 256, 256, 512), tiles[2].core)
        assertEquals(IntRect(256, 256, 512, 512), tiles[3].core)
    }

    @Test
    fun oddDimensionsKeepOnePixelEdgeTiles() {
        val tiles = TilePlanner.plan(
            imageWidth = 513,
            imageHeight = 257,
            tileSize = 256,
            padding = 10,
            scale = 4,
        )

        assertEquals(6, tiles.size)
        val bottomRight = tiles.last()
        assertEquals(IntRect(512, 256, 513, 257), bottomRight.core)
        assertEquals(1, bottomRight.core.width)
        assertEquals(1, bottomRight.core.height)
        assertEquals(4, bottomRight.outputCore.width)
        assertEquals(4, bottomRight.outputCore.height)
    }

    @Test
    fun cornerAndInteriorTilesClampOnlyAtImageEdges() {
        val tiles = TilePlanner.plan(
            imageWidth = 600,
            imageHeight = 600,
            tileSize = 256,
            padding = 10,
            scale = 4,
        )

        val topLeft = tiles.first()
        assertEquals(IntRect(0, 0, 266, 266), topLeft.input)
        assertEquals(IntRect(0, 0, 1024, 1024), topLeft.cropFromUpscaledTile)

        val center = tiles.first { it.core.left == 256 && it.core.top == 256 }
        assertEquals(IntRect(246, 246, 522, 522), center.input)
        assertEquals(IntRect(40, 40, 1064, 1064), center.cropFromUpscaledTile)
        assertEquals(IntRect(1024, 1024, 2048, 2048), center.outputCore)
    }

    @Test
    fun plannedCoreRegionsCoverOutputWithoutOverlapOrGaps() {
        val scale = 4
        val imageWidth = 517
        val imageHeight = 333
        val tiles = TilePlanner.plan(
            imageWidth = imageWidth,
            imageHeight = imageHeight,
            tileSize = 192,
            padding = 12,
            scale = scale,
        )

        val coveredPixels = tiles.sumOf { it.outputCore.width.toLong() * it.outputCore.height.toLong() }
        val expectedPixels = imageWidth.toLong() * scale * imageHeight.toLong() * scale
        assertEquals(expectedPixels, coveredPixels)

        tiles.forEach { tile ->
            assertTrue(tile.input.left <= tile.core.left)
            assertTrue(tile.input.top <= tile.core.top)
            assertTrue(tile.input.right >= tile.core.right)
            assertTrue(tile.input.bottom >= tile.core.bottom)
            assertEquals(tile.core.width * scale, tile.cropFromUpscaledTile.width)
            assertEquals(tile.core.height * scale, tile.cropFromUpscaledTile.height)
        }
    }

    @Test
    fun adaptiveSizerStepsDownToSafeMinimum() {
        assertEquals(384, AdaptiveTileSizer.nextSmaller(512))
        assertEquals(256, AdaptiveTileSizer.nextSmaller(384))
        assertEquals(192, AdaptiveTileSizer.nextSmaller(256))
        assertEquals(128, AdaptiveTileSizer.nextSmaller(192))
        assertEquals(96, AdaptiveTileSizer.nextSmaller(128))
        assertEquals(64, AdaptiveTileSizer.nextSmaller(96))
        assertNull(AdaptiveTileSizer.nextSmaller(64))
    }
}
