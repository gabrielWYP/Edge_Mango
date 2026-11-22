package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Test

class UltralyticsPipelineTest {
    private val frame = RgbImage(640, 480, IntArray(640 * 480) { 0xFF808080.toInt() })
    private val sidecar = ModelSidecar.parse(
        """{"format":"ultralytics","task":"detect","input":{"shape":[1,640,640,3]},"class_names":["mango"]}""",
    )

    @Test
    fun fullFrameBoxesComeBackInFramePixels() {
        // Frame 640x480 -> scale 1, padY 80. The model reports a normalized box covering the middle of the canvas.
        val model = object : Inferencer {
            override fun infer(input: FloatArray): RawOutputs {
                val n = 8400
                val out = FloatArray(5 * n)
                for (a in 0 until n) { // background anchors: arbitrary box, near-zero score
                    out[a] = 0.3f; out[n + a] = 0.3f; out[2 * n + a] = 0.1f; out[3 * n + a] = 0.1f; out[4 * n + a] = 0.01f
                }
                out[0 * n + 3] = 0.5f; out[1 * n + 3] = 0.5f; out[2 * n + 3] = 0.5f; out[3 * n + 3] = 0.5f; out[4 * n + 3] = 0.9f
                return RawOutputs(out, FloatArray(0), null, null)
            }
        }
        val result = FramePipeline(sidecar, model).process(frame, PipelineConfig(RoiMode.FULL_FRAME, 0.5f, computeMasks = false))
        val d = result.detections.single()
        assertEquals(0, d.cls)
        assertEquals(160.0, d.box.x1, 1.0)
        assertEquals(480.0, d.box.x2, 1.0)
        assertEquals(80.0, d.box.y1, 1.0) // canvas y 160 -> frame y 80
        assertEquals(400.0, d.box.y2, 1.0) // canvas y 480 -> frame y 400
    }
}
