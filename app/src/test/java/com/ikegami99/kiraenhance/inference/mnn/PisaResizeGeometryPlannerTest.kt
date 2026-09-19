package com.ikegami99.kiraenhance.inference.mnn

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PisaResizeGeometryPlannerTest {
    @Test
    fun `regular aligned input stays exact four-x`() {
        val plan = requireNotNull(PisaResizeGeometryPlanner.build(1920, 1080))

        assertFalse(plan.smallInputBoosted)
        assertEquals(7680, plan.modelWidth)
        assertEquals(4320, plan.modelHeight)
        assertEquals(7680, plan.outputWidth)
        assertEquals(4320, plan.outputHeight)
    }

    @Test
    fun `regular odd input keeps upstream eight-pixel alignment`() {
        val plan = requireNotNull(PisaResizeGeometryPlanner.build(257, 256))

        assertFalse(plan.smallInputBoosted)
        assertEquals(1028, plan.rawModelWidth)
        assertEquals(1024, plan.rawModelHeight)
        assertEquals(1024, plan.modelWidth)
        assertEquals(1024, plan.modelHeight)
        assertEquals(1024, plan.outputWidth)
        assertEquals(1024, plan.outputHeight)
        assertEquals(1024L * 1024L, plan.modelPixels)
    }

    @Test
    fun `small input models boosted geometry but returns source four-x`() {
        val plan = requireNotNull(PisaResizeGeometryPlanner.build(64, 96))

        assertTrue(plan.smallInputBoosted)
        assertEquals(128, plan.preUpscaleWidth)
        assertEquals(192, plan.preUpscaleHeight)
        assertEquals(512, plan.modelWidth)
        assertEquals(768, plan.modelHeight)
        assertEquals(256, plan.outputWidth)
        assertEquals(384, plan.outputHeight)
        assertEquals(512L * 768L, plan.modelPixels)
        assertEquals(256L * 384L, plan.outputPixels)
    }

    @Test
    fun `small non-square input matches Python truncation`() {
        val plan = requireNotNull(PisaResizeGeometryPlanner.build(100, 73))

        assertEquals(175, plan.preUpscaleWidth)
        assertEquals(128, plan.preUpscaleHeight)
        assertEquals(700, plan.rawModelWidth)
        assertEquals(512, plan.rawModelHeight)
        assertEquals(696, plan.modelWidth)
        assertEquals(512, plan.modelHeight)
        assertEquals(400, plan.outputWidth)
        assertEquals(292, plan.outputHeight)
    }

    @Test
    fun `extreme aspect ratio exposes boosted model cost separately from output`() {
        val plan = requireNotNull(PisaResizeGeometryPlanner.build(64, 1000))

        assertTrue(plan.smallInputBoosted)
        assertEquals(512, plan.modelWidth)
        assertEquals(8000, plan.modelHeight)
        assertEquals(256, plan.outputWidth)
        assertEquals(4000, plan.outputHeight)
        assertTrue(plan.modelPixels > 1024L * 1024L)
        assertTrue(plan.outputPixels <= 1024L * 1024L)
    }

    @Test
    fun `invalid and overflowing dimensions are rejected`() {
        assertNull(PisaResizeGeometryPlanner.build(0, 100))
        assertNull(PisaResizeGeometryPlanner.build(100, 0))
        assertNull(PisaResizeGeometryPlanner.build(Int.MAX_VALUE, Int.MAX_VALUE))
        assertNull(
            PisaResizeGeometryPlanner.build(
                sourceWidth = 100,
                sourceHeight = 100,
                processSize = 3,
                upscale = 4,
            ),
        )
    }
}
