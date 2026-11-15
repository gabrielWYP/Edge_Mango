package com.edgemango.app.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RoiTest {
    @Test
    fun cropWindowMatchesRoiCropRuleOfTraining() {
        val cases = Fixtures.json("roi/expected.json")["crop_windows"]!!.jsonArray
        for (case in cases) {
            val c = case.jsonObject
            val frame = c["frame"]!!.ints()
            val b = c["box"]!!.ints()
            val w = c["window"]!!.ints()
            val got = RoiRule.window(b[0].toDouble(), b[1].toDouble(), b[2].toDouble(), b[3].toDouble(), frame[0], frame[1])
            assertEquals("frame=$frame box=$b", IntBox(w[0], w[1], w[2], w[3]), got)
        }
    }

    @Test
    fun boxOutsideTheFrameFallsBackToNoWindow() {
        assertNull(RoiRule.window(2000.0, 2000.0, 2100.0, 2100.0, 1280, 720))
    }

    @Test
    fun centerSquareIsTheLargestCenteredSquare() {
        assertEquals(IntBox(160, 0, 1120, 960), RoiRule.centerSquare(1280, 960))
        assertEquals(IntBox(0, 160, 960, 1120), RoiRule.centerSquare(960, 1280))
    }

    @Test
    fun fullFrameWindowCoversEverything() {
        assertEquals(IntBox(0, 0, 640, 480), RoiRule.full(640, 480))
    }

    @Test
    fun maskBoxIsScaledBackToTheFullFrame() {
        val small = IntBox(100, 50, 200, 150)
        val scaled = RoiRule.scaleBox(small, 4, 4)
        assertEquals(IntBox(400, 200, 800, 600), scaled)
    }
}
