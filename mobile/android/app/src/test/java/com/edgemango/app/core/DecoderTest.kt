package com.edgemango.app.core

import java.util.Base64
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DecoderTest {
    private val expected = Fixtures.json("decode/expected.json")
    private val boxesRaw = Fixtures.floats("decode/boxes_raw.bin")
    private val clsLogits = Fixtures.floats("decode/cls_logits.bin")
    private val coefs = Fixtures.quantized("decode/mask_coefs_q16.bin", 16f)
    private val protos = Fixtures.quantized("decode/protos_q8.bin", 8f)
    private val params = DecodeParams(
        confThreshold = expected["conf"]!!.jsonPrimitive.double.toFloat(),
        nmsIouThreshold = expected["iou"]!!.jsonPrimitive.double.toFloat(),
        maxDetections = expected["max_det"]!!.jsonPrimitive.content.toInt(),
    )

    @Test
    fun decodeMatchesPythonReferenceExactly() {
        val dets = Decoder.decode(boxesRaw, clsLogits, params)
        val exp = expected["detections"]!!.jsonArray
        assertEquals(exp.size, dets.size)
        assertTrue("fixture should exercise the max-detections cap", expected["n_candidates_over_conf"]!!.jsonPrimitive.content.toInt() > dets.size)
        for ((i, e) in exp.withIndex()) {
            val o = e.jsonObject
            val d = dets[i]
            assertEquals("class #$i", o["cls"]!!.jsonPrimitive.content.toInt(), d.cls)
            assertEquals("score #$i", o["score"]!!.jsonPrimitive.double, d.score.toDouble(), 1e-4)
            val box = o["box_cxcywh"]!!.doubles()
            assertEquals(box[0], d.cx.toDouble(), 1e-4)
            assertEquals(box[1], d.cy.toDouble(), 1e-4)
            assertEquals(box[2], d.w.toDouble(), 1e-4)
            assertEquals(box[3], d.h.toDouble(), 1e-4)
        }
    }

    @Test
    fun letterboxInverseMatchesPythonReference() {
        val dets = Decoder.decode(boxesRaw, clsLogits, params)
        val crop = expected["crop"]!!.ints()
        val g = LetterboxGeometry.of(crop[2], crop[3], 640)
        assertEquals(expected["scale"]!!.jsonPrimitive.double, g.scale, 1e-12)
        assertEquals(expected["pad_x"]!!.jsonPrimitive.content.toInt(), g.padX)
        assertEquals(expected["pad_y"]!!.jsonPrimitive.content.toInt(), g.padY)
        for ((i, e) in expected["detections"]!!.jsonArray.withIndex()) {
            val o = e.jsonObject
            val box = dets[i].toSourceBox(g, crop[0], crop[1], crop[2], crop[3])
            val cropBox = o["box_crop_xyxy"]!!.doubles()
            val frameBox = o["box_frame_xyxy"]!!.doubles()
            val got = doubleArrayOf(box.x1, box.y1, box.x2, box.y2)
            for (k in 0..3) {
                assertEquals("frame box #$i coord $k", frameBox[k], got[k], 1e-4)
                assertEquals("crop box #$i coord $k", cropBox[k], got[k] - (if (k % 2 == 0) crop[0] else crop[1]), 1e-4)
            }
        }
    }

    @Test
    fun masksMatchPythonReferencePixelForPixel() {
        val dets = Decoder.decode(boxesRaw, clsLogits, params)
        val nm = 32
        val ps = 160
        for ((i, e) in expected["detections"]!!.jsonArray.withIndex()) {
            val o = e.jsonObject
            val mask = MaskAssembler.assemble(dets[i], coefs, protos, nm, ps, ps, 640)
            val bits = Base64.getDecoder().decode(o["mask_bits"]!!.jsonPrimitive.content)
            var mismatches = 0
            var count = 0
            for (p in 0 until ps * ps) {
                val want = (bits[p / 8].toInt() shr (7 - p % 8)) and 1 == 1
                val got = mask[p]
                if (got) count++
                if (want != got) mismatches++
            }
            assertEquals("mask #$i mismatching pixels", 0, mismatches)
            assertEquals(o["mask_true"]!!.jsonPrimitive.content.toInt(), count)
        }
    }

    @Test
    fun scoreMustBeStrictlyAboveTheThreshold() {
        val boxes = FloatArray(8400 * 4) { if (it % 4 >= 2) Math.log(20.0).toFloat() else 0f }
        val logits = FloatArray(8400 * 2) { -10f }
        logits[0] = 0f // sigmoid = 0.5 exactly
        val at = Decoder.decode(boxes, logits, DecodeParams(confThreshold = 0.5f))
        assertEquals(0, at.size)
        val below = Decoder.decode(boxes, logits, DecodeParams(confThreshold = 0.49f))
        assertEquals(1, below.size)
    }

    @Test
    fun anchorGeometryFollowsTheThreeLevels() {
        val boxes = FloatArray(8400 * 4) { if (it % 4 >= 2) Math.log(10.0).toFloat() else 0f }
        val logits = FloatArray(8400 * 2) { -10f }
        // anchor 81 -> level 0 (stride 8) gx=1 gy=1 ; anchor 6400+41 -> level 1 gx=1 gy=1 ; 8000+21 -> level 2 gx=1 gy=1
        logits[81 * 2 + 1] = 5f
        logits[(6400 + 41) * 2] = 5f
        logits[(8000 + 21) * 2 + 1] = 5f
        val dets = Decoder.decode(boxes, logits, DecodeParams()).sortedBy { it.cx }
        assertEquals(listOf(12f, 24f, 48f), dets.map { it.cx })
        assertEquals(listOf(12f, 24f, 48f), dets.map { it.cy })
        assertEquals(listOf(1, 0, 1), dets.map { it.cls })
    }

    @Test
    fun nmsIsPerClassAndGreedyBySuppressingTheLowerScore() {
        val boxes = FloatArray(8400 * 4) { if (it % 4 >= 2) Math.log(40.0).toFloat() else 0f }
        val logits = FloatArray(8400 * 2) { -10f }
        // anchors 0 and 1 are neighbours (8 px apart, IoU(40x40 boxes) = 32*40/(2*1600-32*40) = 0.6667 > 0.5).
        logits[0] = 4f          // class 0, score high
        logits[1 * 2] = 3f      // class 0, overlaps -> suppressed
        logits[1 * 2 + 1] = 3f  // class 1 on the same box -> kept (different class)
        val dets = Decoder.decode(boxes, logits, DecodeParams())
        assertEquals(listOf(0 to 4f, 1 to 3f).map { it.first }, dets.map { it.cls })
        assertTrue(dets[0].score > dets[1].score)
    }

    @Test
    fun maxDetectionsKeepsTheHighestScores() {
        val boxes = FloatArray(8400 * 4) { if (it % 4 >= 2) Math.log(2.0).toFloat() else 0f }
        val logits = FloatArray(8400 * 2) { -10f }
        for (a in 0 until 50) logits[a * 2] = 1f + a * 0.01f
        val dets = Decoder.decode(boxes, logits, DecodeParams(maxDetections = 5))
        assertEquals(5, dets.size)
        assertEquals(49, dets[0].anchor)
        assertEquals(45, dets[4].anchor)
    }

    @Test
    fun emptyMaskWhenTheBoxIsOutsideThePrototypes() {
        val det = Detection(0, 0.9f, -100f, -100f, 10f, 10f, 0)
        val mask = MaskAssembler.assemble(det, FloatArray(32) { 1f }, FloatArray(32 * 160 * 160) { 1f }, 32, 160, 160, 640)
        assertEquals(0, mask.count { it })
    }
}
