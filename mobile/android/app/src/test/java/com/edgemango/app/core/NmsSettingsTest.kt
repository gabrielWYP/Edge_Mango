package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Test

class NmsSettingsTest {
    private fun det(cls: Int, score: Float, cx: Float, cy: Float = 100f, w: Float = 100f, h: Float = 100f) =
        Detection(cls, score, cx, cy, w, h, anchor = 0)

    @Test
    fun iouThresholdDecidesWhichOverlapsSurvive() {
        // Two 100x100 boxes shifted by 25 px: IoU = 75/125 = 0.6.
        val boxes = listOf(det(0, 0.9f, 100f), det(0, 0.8f, 125f))
        val base = DecodeParams(numClasses = 1)
        assertEquals(1, Decoder.nms(boxes, base.copy(nmsIouThreshold = 0.5f)).size)
        assertEquals(2, Decoder.nms(boxes, base.copy(nmsIouThreshold = 0.7f)).size)
        // The comparison is strict (suppress when IoU > threshold): at exactly 0.6 both survive.
        assertEquals(2, Decoder.nms(boxes, base.copy(nmsIouThreshold = 0.6f)).size)
    }

    @Test
    fun perClassNmsKeepsOverlapsOfDifferentClasses() {
        val boxes = listOf(det(0, 0.9f, 100f), det(1, 0.8f, 100f))
        val p = DecodeParams(numClasses = 2, nmsIouThreshold = 0.5f)
        assertEquals(2, Decoder.nms(boxes, p).size)
    }

    @Test
    fun agnosticNmsSuppressesAcrossClasses() {
        val boxes = listOf(det(0, 0.9f, 100f), det(1, 0.8f, 100f), det(1, 0.7f, 400f))
        val p = DecodeParams(numClasses = 2, nmsIouThreshold = 0.5f, agnosticNms = true)
        val kept = Decoder.nms(boxes, p)
        assertEquals(listOf(0.9f, 0.7f), kept.map { it.score })
    }

    @Test
    fun agnosticNmsKeepsTheHigherScoreRegardlessOfClass() {
        val boxes = listOf(det(1, 0.95f, 100f), det(0, 0.5f, 105f))
        val kept = Decoder.nms(boxes, DecodeParams(numClasses = 2, agnosticNms = true))
        assertEquals(1, kept.size)
        assertEquals(1, kept[0].cls)
    }

    // ---- through the pipeline: the config replaces the sidecar's IoU ------------------------------------------------

    private val frame = RgbImage(640, 640, IntArray(640 * 640) { 0xFF808080.toInt() })
    private val sidecar = ModelSidecar.parse(
        """{"format":"ultralytics","task":"detect","input":{"shape":[1,640,640,3]},"class_names":["a","b"],
            "decode":{"nms":{"iou_threshold":0.5}}}""",
    )

    private fun model(vararg anchors: FloatArray) = object : Inferencer {
        override fun infer(input: FloatArray): RawOutputs {
            val n = 8400
            val out = FloatArray(6 * n)
            for (a in 0 until n) { out[a] = 0.9f; out[n + a] = 0.9f; out[2 * n + a] = 0.1f; out[3 * n + a] = 0.1f; out[4 * n + a] = 0.01f; out[5 * n + a] = 0.01f }
            for ((i, v) in anchors.withIndex()) for (c in 0 until 6) out[c * n + i] = v[c]
            return RawOutputs(out, FloatArray(0), null, null)
        }
    }

    // cx, cy, w, h normalized; two class scores. 0.2 wide boxes shifted by 0.05: IoU 0.6.
    private val first = floatArrayOf(0.30f, 0.30f, 0.2f, 0.2f, 0.9f, 0.0f)
    private val second = floatArrayOf(0.35f, 0.30f, 0.2f, 0.2f, 0.8f, 0.0f)
    private val otherClass = floatArrayOf(0.30f, 0.30f, 0.2f, 0.2f, 0.0f, 0.85f)

    @Test
    fun pipelineUsesTheConfiguredIouInsteadOfTheSidecarOne() {
        val p = FramePipeline(sidecar, model(first, second))
        val cfg = PipelineConfig(RoiMode.FULL_FRAME, 0.5f, computeMasks = false)
        assertEquals(1, p.process(frame, cfg).detections.size) // sidecar 0.5
        assertEquals(1, p.process(frame, cfg.copy(nmsIou = 0.5f)).detections.size)
        assertEquals(2, p.process(frame, cfg.copy(nmsIou = 0.7f)).detections.size)
    }

    @Test
    fun pipelineAgnosticToggle() {
        val p = FramePipeline(sidecar, model(first, otherClass))
        val cfg = PipelineConfig(RoiMode.FULL_FRAME, 0.5f, computeMasks = false)
        // One candidate per anchor, so two anchors with different best classes: kept by per-class NMS.
        assertEquals(2, p.process(frame, cfg).detections.size)
        assertEquals(1, p.process(frame, cfg.copy(agnosticNms = true)).detections.size)
    }
}
