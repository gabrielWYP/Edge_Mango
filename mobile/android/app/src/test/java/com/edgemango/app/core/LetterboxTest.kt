package com.edgemango.app.core

import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LetterboxTest {
    private val spec = Fixtures.json("roi/expected.json")["letterbox"]!!.jsonObject
    private val source = Fixtures.image("roi/" + spec["source"]!!.jsonPrimitive.content)

    @Test
    fun geometryMatchesTheDatasetLetterbox() {
        for (case in spec["cases"]!!.jsonArray) {
            val c = case.jsonObject
            val win = c["window"]!!.takeIf { it !is JsonNull }?.ints() ?: listOf(0, 0, source.width, source.height)
            val g = LetterboxGeometry.of(win[2] - win[0], win[3] - win[1], 640)
            assertEquals(c["scale"]!!.jsonPrimitive.double, g.scale, 1e-12)
            assertEquals(c["pad_x"]!!.jsonPrimitive.content.toInt(), g.padX)
            assertEquals(c["pad_y"]!!.jsonPrimitive.content.toInt(), g.padY)
            assertEquals(c["new_w"]!!.jsonPrimitive.content.toInt(), g.newW)
            assertEquals(c["new_h"]!!.jsonPrimitive.content.toInt(), g.newH)
        }
    }

    @Test
    fun resizedPixelsMatchCv2WithinOneLevel() {
        for (case in spec["cases"]!!.jsonArray) {
            val c = case.jsonObject
            val win = c["window"]!!.takeIf { it !is JsonNull }?.ints() ?: listOf(0, 0, source.width, source.height)
            val g = LetterboxGeometry.of(win[2] - win[0], win[3] - win[1], 640)
            val canvas = Letterbox.toCanvas(source, IntBox(win[0], win[1], win[2], win[3]), g, 114)
            val expected = Fixtures.image("roi/" + c["file"]!!.jsonPrimitive.content)
            var sum = 0L
            var max = 0
            for (i in canvas.pixels.indices) {
                val a = canvas.pixels[i]
                val b = expected.pixels[i]
                for (shift in intArrayOf(16, 8, 0)) {
                    val d = Math.abs(((a shr shift) and 0xFF) - ((b shr shift) and 0xFF))
                    sum += d
                    if (d > max) max = d
                }
            }
            val mean = sum.toDouble() / (canvas.pixels.size * 3)
            assertTrue("${c["name"]}: mean abs diff $mean", mean < 0.5)
            assertTrue("${c["name"]}: max abs diff $max", max <= 2)
        }
    }

    @Test
    fun normalizationUsesImageNetStats() {
        val canvas = RgbImage(2, 1, intArrayOf(0xFF000000.toInt() or (114 shl 16) or (114 shl 8) or 114, 0xFF000000.toInt() or (255 shl 16)))
        val out = FloatArray(2 * 3)
        Letterbox.normalizeInto(canvas, out, Normalization.IMAGENET)
        assertEquals((114 / 255f - 0.485f) / 0.229f, out[0], 1e-6f)
        assertEquals((114 / 255f - 0.456f) / 0.224f, out[1], 1e-6f)
        assertEquals((114 / 255f - 0.406f) / 0.225f, out[2], 1e-6f)
        assertEquals((1f - 0.485f) / 0.229f, out[3], 1e-6f)
        assertEquals((0f - 0.456f) / 0.224f, out[4], 1e-6f)
    }

    @Test
    fun unletterboxMapsBackToFrameCoordinates() {
        val g = LetterboxGeometry.of(480, 360, 640)
        // The centre of the canvas is the centre of the crop, whatever the crop origin is.
        val (x, y) = g.toSource(320.0, 320.0)
        assertEquals(240.0, x, 1e-9)
        assertEquals(180.0, y, 1e-9)
    }
}
