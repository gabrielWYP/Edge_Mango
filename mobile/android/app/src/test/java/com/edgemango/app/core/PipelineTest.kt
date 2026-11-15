package com.edgemango.app.core

import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class PipelineTest {
    private val frame = Fixtures.image("roi/frame0_320.rgb.gz")
    private val expectedBox = Fixtures.json("roi/expected.json")["cases"]!!.jsonArray[0].jsonObject["bbox"]!!.ints()
    private val sidecar = ModelSidecar.parse(String(Fixtures.bytes("sidecar_segment_fp32.json")))

    /** A model that sees the canvas and answers with one box covering the whole letterboxed image area. */
    private class FillBoxModel(val g: () -> LetterboxGeometry) : Inferencer {
        var lastInput: FloatArray? = null
        var calls = 0
        override fun infer(input: FloatArray): RawOutputs {
            calls++
            lastInput = input
            val geo = g()
            val boxes = FloatArray(8400 * 4) { if (it % 4 >= 2) 0f else 0f }
            val cls = FloatArray(8400 * 2) { -10f }
            val gx = 40
            val gy = 40
            val a = gy * 80 + gx
            val cx = geo.padX + geo.newW / 2f
            val cy = geo.padY + geo.newH / 2f
            boxes[a * 4] = cx - (gx + 0.5f) * 8
            boxes[a * 4 + 1] = cy - (gy + 0.5f) * 8
            boxes[a * 4 + 2] = Math.log(geo.newW.toDouble()).toFloat()
            boxes[a * 4 + 3] = Math.log(geo.newH.toDouble()).toFloat()
            cls[a * 2 + 1] = 3f
            return RawOutputs(boxes, cls, FloatArray(8400 * 32), FloatArray(32 * 160 * 160))
        }
    }

    @Test
    fun colorMaskModeCropsToTheMangoPlusMarginAndMapsBoxesBack() {
        var geo: LetterboxGeometry? = null
        val model = FillBoxModel { geo!! }
        val pipeline = FramePipeline(sidecar, model)
        val prepared = pipeline.prepare(frame, PipelineConfig(RoiMode.COLOR_MASK))
        geo = prepared.geometry
        val expectedWindow = RoiRule.window(
            expectedBox[0].toDouble(), expectedBox[1].toDouble(), expectedBox[2].toDouble(), expectedBox[3].toDouble(), 320, 240,
        )!!
        assertEquals(expectedWindow, prepared.window)
        assertEquals(IntBox(expectedBox[0], expectedBox[1], expectedBox[2], expectedBox[3]), prepared.mangoBox)

        val result = pipeline.process(frame, PipelineConfig(RoiMode.COLOR_MASK, computeMasks = false))
        assertEquals(1, result.detections.size)
        val d = result.detections[0]
        assertEquals(1, d.cls)
        // The model boxed the whole letterboxed content, which is exactly the crop window.
        assertEquals(expectedWindow.x1.toDouble(), d.box.x1, 0.6)
        assertEquals(expectedWindow.y1.toDouble(), d.box.y1, 0.6)
        assertEquals(expectedWindow.x2.toDouble(), d.box.x2, 0.6)
        assertEquals(expectedWindow.y2.toDouble(), d.box.y2, 0.6)
        assertEquals(d.score.toDouble(), result.maxDamageScore.toDouble(), 1e-9)
        assertNull(result.maskLayer)
        assertEquals(640 * 640 * 3, model.lastInput!!.size)
    }

    @Test
    fun fullFrameModeUsesTheWholeFrame() {
        val pipeline = FramePipeline(sidecar, FillBoxModel { LetterboxGeometry.of(320, 240, 640) })
        val result = pipeline.process(frame, PipelineConfig(RoiMode.FULL_FRAME, computeMasks = false))
        assertEquals(IntBox(0, 0, 320, 240), result.window)
        assertNull(result.mangoBox)
        assertEquals(0.0, result.detections[0].box.x1, 0.6)
        assertEquals(320.0, result.detections[0].box.x2, 0.6)
    }

    @Test
    fun centerSquareMode() {
        val pipeline = FramePipeline(sidecar, FillBoxModel { LetterboxGeometry.of(240, 240, 640) })
        val result = pipeline.process(frame, PipelineConfig(RoiMode.CENTER_SQUARE, computeMasks = false))
        assertEquals(IntBox(40, 0, 280, 240), result.window)
    }

    @Test
    fun noMangoFallsBackToTheFullFrame() {
        val dark = RgbImage(320, 240, IntArray(320 * 240) { 0xFF000000.toInt() })
        val pipeline = FramePipeline(sidecar, FillBoxModel { LetterboxGeometry.of(320, 240, 640) })
        val result = pipeline.process(dark, PipelineConfig(RoiMode.COLOR_MASK, computeMasks = false))
        assertEquals(IntBox(0, 0, 320, 240), result.window)
        assertNull(result.mangoBox)
    }

    @Test
    fun maskLayerIsBuiltOnlyForSegmentModelsAndPaintsTheDamageClass() {
        val model = FillBoxModel { LetterboxGeometry.of(320, 240, 640) }
        val pipeline = FramePipeline(sidecar, model)
        val result = pipeline.process(frame, PipelineConfig(RoiMode.FULL_FRAME, computeMasks = true))
        val layer = result.maskLayer ?: error("segment model should produce a mask layer")
        assertEquals(160, layer.width)
        // coefs are all zero -> sigmoid(0) = 0.5 is not above the threshold -> nothing painted.
        assertEquals(0, layer.argb.count { it != 0 })

        val detectOnly = FramePipeline(ModelSidecar.default(ModelTask.DETECT), object : Inferencer {
            override fun infer(input: FloatArray) = RawOutputs(FloatArray(8400 * 4), FloatArray(8400 * 2) { -10f }, null, null)
        })
        assertNull(detectOnly.process(frame, PipelineConfig(RoiMode.FULL_FRAME, computeMasks = true)).maskLayer)
    }

    @Test
    fun maskLayerAffineMapsPrototypePixelsToFrameCoordinates() {
        val g = LetterboxGeometry.of(480, 360, 640) // scale 4/3, padY 80
        val window = IntBox(300, 120, 780, 480)
        val m = MaskLayer.affine(g, window, protoW = 160, protoH = 160, imageSize = 640)
        // proto pixel 80 is the middle of the canvas (x = 320): the middle of the crop in x.
        assertEquals(300.0 + 240.0, m.sx * 80 + m.tx, 1e-9)
        assertEquals(120.0 + 180.0, m.sy * 80 + m.ty, 1e-9)
    }

    @Test
    fun disablingConfidenceFiltersDetections() {
        val pipeline = FramePipeline(sidecar, FillBoxModel { LetterboxGeometry.of(320, 240, 640) })
        val none = pipeline.process(frame, PipelineConfig(RoiMode.FULL_FRAME, confThreshold = 0.99f, computeMasks = false))
        assertTrue(none.detections.isEmpty())
        assertEquals(0f, none.maxDamageScore, 0f)
    }
}
