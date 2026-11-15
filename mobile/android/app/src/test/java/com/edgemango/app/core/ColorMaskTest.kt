package com.edgemango.app.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ColorMaskTest {
    private val expected = Fixtures.json("roi/expected.json")

    @Test
    fun kotlinMaskBboxMatchesPythonRgbMangoMask() {
        val cases = expected["cases"]!!.jsonArray
        assertEquals(3, cases.size)
        for (case in cases) {
            val c = case.jsonObject
            val img = Fixtures.image("roi/" + c["file"]!!.jsonPrimitive.content)
            val mask = MangoColorSegmenter.segment(img)
            val box = mask.bbox() ?: error("no mango found in ${c["file"]}")
            val exp = c["bbox"]!!.ints()
            val iou = box.iou(IntBox(exp[0], exp[1], exp[2], exp[3]))
            println("ROI ${c["file"]}: IoU=$iou kotlin=$box python=$exp")
            assertTrue("${c["file"]} bbox IoU $iou, got $box expected $exp", iou >= 0.95)
            val area = c["mask_area"]!!.jsonPrimitive.content.toInt()
            assertEquals("mask area of ${c["file"]}", area.toDouble(), mask.area.toDouble(), area * 0.05)
        }
    }

    @Test
    fun emptyOrDarkImageGivesNoBox() {
        val dark = RgbImage(64, 48, IntArray(64 * 48) { 0xFF000000.toInt() })
        assertNull(MangoColorSegmenter.segment(dark).bbox())
    }

    @Test
    fun ellipseKernelMatchesOpenCvShape() {
        // cv2.getStructuringElement(MORPH_ELLIPSE, (7, 7)) row run lengths.
        val runs = MangoColorSegmenter.ellipseRuns(7).map { it.second - it.first + 1 }
        assertEquals(listOf(1, 5, 7, 7, 7, 5, 1), runs)
        val runs5 = MangoColorSegmenter.ellipseRuns(5).map { it.second - it.first + 1 }
        assertEquals(listOf(1, 5, 5, 5, 1), runs5)
        val runs13 = MangoColorSegmenter.ellipseRuns(13).map { it.second - it.first + 1 }
        assertEquals(listOf(1, 7, 9, 11, 13, 13, 13, 13, 13, 11, 9, 7, 1), runs13)
    }

    @Test
    fun oddRoundsLikeThePythonHelper() {
        assertEquals(7, MangoColorSegmenter.odd(0.022 * 320))
        assertEquals(13, MangoColorSegmenter.odd(0.037 * 320))
        assertEquals(3, MangoColorSegmenter.odd(0.5))
    }
}
