package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TwoStagePipelineTest {
    // 640x480: scale 1, padY 80, so a canvas point (u, v) is the frame point (u, v - 80).
    private val frame = RgbImage(640, 480, IntArray(640 * 480) { 0xFF808080.toInt() })
    private val detSidecar = ModelSidecar.parse(
        """{"format":"ultralytics","task":"detect","input":{"shape":[1,640,640,3]},"class_names":["mango"]}""",
    )
    private val dmgSidecar = ModelSidecar.parse(String(Fixtures.bytes("sidecar_segment_fp32.json")))

    /** Stage 1 stub: reports the given normalized mango boxes `(cx, cy, w, h, score)` on the 640 canvas. */
    private class FakeDetector(var boxes: List<FloatArray>) : Inferencer {
        var calls = 0
        override fun infer(input: FloatArray): RawOutputs {
            calls++
            val n = 8400
            val out = FloatArray(5 * n)
            for (a in 0 until n) { // background anchors: arbitrary box, near-zero score
                out[a] = 0.3f; out[n + a] = 0.3f; out[2 * n + a] = 0.1f; out[3 * n + a] = 0.1f; out[4 * n + a] = 0.01f
            }
            for ((i, b) in boxes.withIndex()) for (c in 0 until 5) out[c * n + 100 + i] = b[c]
            return RawOutputs(out, FloatArray(0), null, null)
        }
    }

    /** Stage 2 stub: damage detections at fixed canvas boxes `(cx, cy, w, h)` in input pixels, score 0.8. */
    private class FakeDamage(val canvasBoxes: List<FloatArray>) : Inferencer {
        var calls = 0
        override fun infer(input: FloatArray): RawOutputs {
            calls++
            val boxes = FloatArray(8400 * 4)
            val cls = FloatArray(8400 * 2) { -10f }
            for ((i, b) in canvasBoxes.withIndex()) {
                val gx = 10 + i * 5
                val gy = 40
                val a = gy * 80 + gx
                boxes[a * 4] = b[0] - (gx + 0.5f) * 8
                boxes[a * 4 + 1] = b[1] - (gy + 0.5f) * 8
                boxes[a * 4 + 2] = Math.log(b[2].toDouble()).toFloat()
                boxes[a * 4 + 3] = Math.log(b[3].toDouble()).toFloat()
                cls[a * 2 + 1] = 1.4f // sigmoid = 0.80
            }
            return RawOutputs(boxes, cls, FloatArray(8400 * 32), FloatArray(32 * 160 * 160))
        }
    }

    private val mangoA = floatArrayOf(0.30f, 0.50f, 0.20f, 0.20f, 0.95f) // canvas 192,320 size 128 -> frame x 128..256, y 176..304
    private val mangoB = floatArrayOf(0.70f, 0.50f, 0.20f, 0.20f, 0.70f)
    private val weak = floatArrayOf(0.50f, 0.20f, 0.10f, 0.10f, 0.30f)

    private fun pipeline(det: FakeDetector, dmg: FakeDamage) =
        TwoStagePipeline(FramePipeline(detSidecar, det), FramePipeline(dmgSidecar, dmg))

    @Test
    fun noMangoMeansStageTwoDoesNotRun() {
        val dmg = FakeDamage(listOf(floatArrayOf(320f, 320f, 100f, 100f)))
        val p = pipeline(FakeDetector(listOf(weak)), dmg) // 0.30 < 0.5
        val r = p.process(frame, TwoStageConfig())
        assertTrue(r.noMango)
        assertEquals(0, dmg.calls)
        assertEquals(0f, r.maxDamageScore, 0f)
    }

    @Test
    fun eachMangoGetsItsOwnCropAndOnlyDamageInsideItsBox() {
        // Canvas centre (320, 320) is the crop centre = the mango centre (symmetric margin, nothing clipped).
        val dmg = FakeDamage(listOf(floatArrayOf(320f, 320f, 100f, 100f), floatArrayOf(10f, 320f, 20f, 20f)))
        val p = pipeline(FakeDetector(listOf(mangoA, mangoB)), dmg)
        val r = p.process(frame, TwoStageConfig(damageThreshold = 0.5f, computeMasks = false))

        assertEquals(2, r.mangos.size)
        assertEquals(2, dmg.calls)
        assertEquals(listOf(0.95f, 0.70f), r.mangos.map { it.score }) // best first
        val a = r.mangos[0]
        assertEquals(128.0, a.box.x1, 1.0)
        assertEquals(256.0, a.box.x2, 1.0)
        assertEquals(176.0, a.box.y1, 1.0)
        assertEquals(RoiRule.window(a.box.x1, a.box.y1, a.box.x2, a.box.y2, 640, 480), a.window)
        // The second damage stub box sits in the crop margin, outside the mango box: dropped.
        assertEquals(1, a.damages.size)
        assertEquals(a.box.centerX, a.damages[0].box.centerX, 1.0)
        assertEquals(0.8f, a.maxDamageScore, 0.01f)
        assertNull(a.maskLayer)
    }

    @Test
    fun maxMangosCapsTheCropsAndKeepsTheBest() {
        val dmg = FakeDamage(emptyList())
        val p = pipeline(FakeDetector(listOf(mangoB, mangoA)), dmg)
        val r = p.process(frame, TwoStageConfig(maxMangos = 1))
        assertEquals(1, r.mangos.size)
        assertEquals(0.95f, r.mangos[0].score, 1e-6f)
        assertEquals(1, dmg.calls)
    }

    @Test
    fun mangoThresholdIsConfigurable() {
        val p = pipeline(FakeDetector(listOf(mangoA, mangoB)), FakeDamage(emptyList()))
        assertEquals(2, p.process(frame, TwoStageConfig(mangoThreshold = 0.5f)).mangos.size)
        assertEquals(1, p.process(frame, TwoStageConfig(mangoThreshold = 0.8f)).mangos.size)
    }

    @Test
    fun trackIdsAreStableAcrossFrames() {
        val det = FakeDetector(listOf(mangoA, mangoB))
        val p = pipeline(det, FakeDamage(emptyList()))
        val first = p.process(frame, TwoStageConfig()).mangos.associate { it.score to it.trackId }
        // Both mangos drift by a few percent: same ids.
        det.boxes = listOf(
            floatArrayOf(0.31f, 0.50f, 0.20f, 0.20f, 0.96f),
            floatArrayOf(0.71f, 0.51f, 0.20f, 0.20f, 0.71f),
        )
        val second = p.process(frame, TwoStageConfig()).mangos
        assertEquals(first[0.95f], second[0].trackId)
        assertEquals(first[0.70f], second[1].trackId)
        assertTrue(second[0].trackId != second[1].trackId)
    }

    @Test
    fun polygonsAreTracedForDamageWhenRequested() {
        // One damage box at the crop centre; coefficients all zero means an empty mask, so patch the protos instead:
        // sigmoid(coef . proto) > 0.5 needs a positive sum, which the stub provides through a constant coef/proto pair.
        val inner = FakeDamage(listOf(floatArrayOf(320f, 320f, 100f, 100f)))
        val withMask = object : Inferencer {
            override fun infer(input: FloatArray): RawOutputs {
                val r = inner.infer(input)
                val coefs = r.coefs!!
                val protos = r.protos!!
                for (a in 0 until 8400) coefs[a * 32] = 1f
                for (i in 0 until 160 * 160) protos[i] = 1f // channel 0 all ones: every pixel inside the box is "on"
                return r
            }
        }
        val p = TwoStagePipeline(FramePipeline(detSidecar, FakeDetector(listOf(mangoA))), FramePipeline(dmgSidecar, withMask))
        val r = p.process(frame, TwoStageConfig(damageThreshold = 0.5f, computeMasks = true, computePolygons = true))
        val d = r.mangos.single().damages.single()
        val poly = assertNotNull(d.polygon).let { d.polygon!! }
        assertTrue(poly.size >= 8)
        val xs = (poly.indices step 2).map { poly[it] }
        val ys = (1 until poly.size step 2).map { poly[it] }
        val box = d.box
        // The outline is the mask clipped to the damage box (prototype resolution = 4 input px).
        assertTrue(xs.min() >= box.x1 - 6 && xs.max() <= box.x2 + 6)
        assertTrue(ys.min() >= box.y1 - 6 && ys.max() <= box.y2 + 6)
        assertNotNull(r.mangos.single().maskLayer)
    }
}
