package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UltralyticsDecoderTest {
    private val n = 8400

    /** Box of a background anchor: real exports hold arbitrary boxes there, with a near-zero score. */
    private val BG = floatArrayOf(0.5f, 0.5f, 0.3f, 0.3f)
    private val params = DecodeParams(confThreshold = 0.25f, nmsIouThreshold = 0.5f, imageSize = 640, numClasses = 1)

    /** `(1, 4+nc, N)` tensor, row-major, background score 0.01. Anchors are `[cx, cy, w, h, score...]`. */
    private fun channelsFirst(nc: Int, vararg anchors: Pair<Int, FloatArray>): FloatArray {
        val out = FloatArray((4 + nc) * n)
        for (a in 0 until n) {
            out[a] = BG[0]; out[n + a] = BG[1]; out[2 * n + a] = BG[2]; out[3 * n + a] = BG[3]
            for (c in 0 until nc) out[(4 + c) * n + a] = 0.01f
        }
        for ((a, v) in anchors) for (c in v.indices) out[c * n + a] = v[c]
        return out
    }

    private fun channelsLast(nc: Int, vararg anchors: Pair<Int, FloatArray>): FloatArray {
        val k = 4 + nc
        val out = FloatArray(k * n)
        for (a in 0 until n) {
            for (c in 0 until 4) out[a * k + c] = BG[c]
            for (c in 0 until nc) out[a * k + 4 + c] = 0.01f
        }
        for ((a, v) in anchors) for (c in v.indices) out[a * k + c] = v[c]
        return out
    }

    @Test
    fun normalizedBoxesAreScaledToInputPixels() {
        val raw = channelsFirst(1, 10 to floatArrayOf(0.5f, 0.25f, 0.2f, 0.4f, 0.9f))
        val dets = UltralyticsDecoder.decode(raw, params)
        assertEquals(1, dets.size)
        val d = dets[0]
        assertEquals(0, d.cls)
        assertEquals(0.9f, d.score, 1e-6f)
        assertEquals(320f, d.cx, 1e-3f)
        assertEquals(160f, d.cy, 1e-3f)
        assertEquals(128f, d.w, 1e-3f)
        assertEquals(256f, d.h, 1e-3f)
    }

    @Test
    fun pixelBoxesAreLeftAlone() {
        val raw = channelsFirst(1, 10 to floatArrayOf(320f, 160f, 128f, 256f, 0.9f))
        val d = UltralyticsDecoder.decode(raw, params).single()
        assertEquals(320f, d.cx, 1e-3f)
        assertEquals(256f, d.h, 1e-3f)
    }

    @Test
    fun channelsLastGivesTheSameDetections() {
        val a = floatArrayOf(0.5f, 0.25f, 0.2f, 0.4f, 0.9f)
        val first = UltralyticsDecoder.decode(channelsFirst(1, 7 to a), params, OutputLayout.CHANNELS_FIRST)
        val last = UltralyticsDecoder.decode(channelsLast(1, 7 to a), params, OutputLayout.CHANNELS_LAST)
        assertEquals(first.map { it.copy(anchor = 0) }, last.map { it.copy(anchor = 0) })
    }

    @Test
    fun autoLayoutRecognisesBothLayoutsInNormalizedAndPixelSpace() {
        for (box in listOf(floatArrayOf(0.5f, 0.25f, 0.2f, 0.4f, 0.9f), floatArrayOf(320f, 160f, 128f, 256f, 0.9f))) {
            // pixel-space export: the background boxes are in pixels too
            val first = UltralyticsDecoder.decode(channelsFirst(1, 7 to box), params, OutputLayout.AUTO)
            val last = UltralyticsDecoder.decode(channelsLast(1, 7 to box), params, OutputLayout.AUTO)
            assertEquals(1, first.size)
            assertEquals(1, last.size)
            assertEquals(first.single().cx, last.single().cx, 1e-3f)
            assertEquals(first.single().score, last.single().score, 1e-6f)
        }
    }

    @Test
    fun confidenceThresholdAndNmsApply() {
        val raw = channelsFirst(
            1,
            1 to floatArrayOf(0.5f, 0.5f, 0.3f, 0.3f, 0.9f),
            2 to floatArrayOf(0.51f, 0.5f, 0.3f, 0.3f, 0.8f), // overlaps #1, suppressed
            3 to floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f, 0.7f), // far away, kept
            4 to floatArrayOf(0.9f, 0.9f, 0.1f, 0.1f, 0.2f), // below conf
        )
        val dets = UltralyticsDecoder.decode(raw, params)
        assertEquals(listOf(0.9f, 0.7f), dets.map { it.score })
        assertTrue(UltralyticsDecoder.decode(raw, params.copy(confThreshold = 0.95f)).isEmpty())
    }

    @Test
    fun emptyWhenNothingPasses() {
        assertTrue(UltralyticsDecoder.decode(channelsFirst(1), params).isEmpty())
    }

    @Test
    fun multiClassTakesTheBestClassPerAnchor() {
        val p = params.copy(numClasses = 2)
        val raw = channelsFirst(2, 5 to floatArrayOf(0.5f, 0.5f, 0.2f, 0.2f, 0.3f, 0.8f))
        val d = UltralyticsDecoder.decode(raw, p).single()
        assertEquals(1, d.cls)
        assertEquals(0.8f, d.score, 1e-6f)
    }

    @Test
    fun rejectsATensorThatDoesNotMatchTheClassCount() {
        assertTrue(runCatching { UltralyticsDecoder.decode(FloatArray(8403), params) }.isFailure)
    }
}
