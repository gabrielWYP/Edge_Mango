package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UltralyticsSidecarTest {
    private val template = """
        {
          "format": "ultralytics",
          "task": "detect",
          "input": {"shape": [1, 640, 640, 3], "layout": "NHWC", "dtype": "float32"},
          "class_names": ["mango"],
          "preprocessing": {
            "normalization": {"mean": [0, 0, 0], "std": [1, 1, 1]},
            "letterbox": {"target_size": 640, "pad_value": 114}
          },
          "decode": {"conf_threshold": 0.25, "nms": {"iou_threshold": 0.5}, "max_detections": 300}
        }
    """.trimIndent()

    @Test
    fun parsesTheDocumentedTemplate() {
        val s = ModelSidecar.parse(template)
        assertEquals(ModelFormat.ULTRALYTICS, s.format)
        assertTrue(s.isUltralytics)
        assertEquals(ModelTask.DETECT, s.task)
        assertEquals(640, s.inputSize)
        assertEquals(listOf("mango"), s.classNames)
        assertEquals(Normalization.UNIT, s.normalization)
        assertEquals(114, s.padValue)
        assertEquals(OutputLayout.AUTO, s.outputLayout)
        assertEquals(1, s.decodeParams().numClasses)
    }

    @Test
    fun formatDefaultsToTheProjectContract() {
        val s = ModelSidecar.parse("""{"task":"detect","input":{"shape":[1,320,320,3]}}""")
        assertEquals(ModelFormat.EDGE_MANGO, s.format)
        assertEquals(Normalization.IMAGENET, s.normalization)
    }

    @Test
    fun ultralyticsWithoutNormalizationUsesUnitScaling() {
        val s = ModelSidecar.parse("""{"format":"ultralytics","task":"detect","input_size":640}""")
        assertEquals(Normalization.UNIT, s.normalization)
        assertEquals(listOf("mango"), s.classNames)
    }

    @Test
    fun explicitOutputLayoutAndUnknownValues() {
        assertEquals(
            OutputLayout.CHANNELS_LAST,
            ModelSidecar.parse("""{"format":"ultralytics","task":"detect","input_size":640,"output_layout":"channels_last"}""").outputLayout,
        )
        assertTrue(runCatching { ModelSidecar.parse("""{"format":"onnx","task":"detect","input_size":640}""") }.isFailure)
    }
}
