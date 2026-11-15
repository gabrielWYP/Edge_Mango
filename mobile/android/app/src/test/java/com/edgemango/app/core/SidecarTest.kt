package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SidecarTest {
    @Test
    fun parsesTheRealExporterSidecar() {
        val s = ModelSidecar.parse(String(Fixtures.bytes("sidecar_segment_fp32.json")))
        assertEquals(ModelTask.SEGMENT, s.task)
        assertEquals(640, s.inputSize)
        assertEquals(listOf("mango", "damage"), s.classNames)
        assertEquals(3354818L, s.numParameters)
        assertEquals("fp32", s.precision)
        assertEquals(0.25f, s.decode.confThreshold, 0f)
        assertEquals(0.5f, s.decode.nmsIouThreshold, 0f)
        assertEquals(300, s.decode.maxDetections)
        assertEquals(listOf(8, 16, 32), s.decode.strides)
        assertEquals(listOf("boxes_raw", "cls_logits", "mask_coefs", "protos"), s.outputOrder)
        assertEquals(32, s.numPrototypes)
        assertEquals(114, s.padValue)
        assertEquals(13653616L, s.fileSizeBytes)
        assertNull(s.flops)
    }

    @Test
    fun onlyTaskAndInputSizeAreRequired() {
        val s = ModelSidecar.parse("""{"task":"detect","input":{"shape":[1,320,320,3]}}""")
        assertEquals(ModelTask.DETECT, s.task)
        assertEquals(320, s.inputSize)
        assertEquals(0.25f, s.decode.confThreshold, 0f)
        assertEquals(listOf(8, 16, 32), s.decode.strides)
        assertEquals(Normalization.IMAGENET, s.normalization)
        assertEquals(114, s.padValue)
        assertNull(s.numParameters)
    }

    @Test
    fun readsFlopsFromSeveralPlausibleSpellings() {
        val a = ModelSidecar.parse("""{"task":"detect","input_size":640,"model":{"flops":8.7e9,"num_parameters":3000000}}""")
        assertEquals(8.7e9, a.flops!!, 1.0)
        val b = ModelSidecar.parse("""{"task":"segment","image_size":640,"complexity":{"gflops":9.5,"params":3354818}}""")
        assertEquals(9.5e9, b.flops!!, 1.0)
        assertEquals(3354818L, b.numParameters)
        val c = ModelSidecar.parse("""{"task":"detect","input_size":640,"gmacs":4.0}""")
        assertEquals(8.0e9, c.flops!!, 1.0) // 1 MAC = 2 FLOPs
    }

    @Test
    fun missingTaskOrSizeIsAnError() {
        val noTask = runCatching { ModelSidecar.parse("""{"input_size":640}""") }
        assertTrue(noTask.isFailure)
        val noSize = runCatching { ModelSidecar.parse("""{"task":"detect"}""") }
        assertTrue(noSize.isFailure)
    }

    @Test
    fun resolvesOutputsByElementCountNotByIndex() {
        val sizes = intArrayOf(32 * 160 * 160, 8400 * 2, 8400 * 4, 8400 * 32)
        val roles = OutputRoles.resolve(sizes, ModelTask.SEGMENT, anchors = 8400, numClasses = 2, numPrototypes = 32, protoCells = 160 * 160)
        assertEquals(2, roles.boxes)
        assertEquals(1, roles.cls)
        assertEquals(3, roles.coefs)
        assertEquals(0, roles.protos)
    }

    @Test
    fun resolvesDetectOutputs() {
        val roles = OutputRoles.resolve(intArrayOf(8400 * 2, 8400 * 4), ModelTask.DETECT, 8400, 2, 32, 160 * 160)
        assertEquals(1, roles.boxes)
        assertEquals(0, roles.cls)
        assertNull(roles.coefs)
    }

    @Test
    fun ambiguousSizesFallBackToTheSidecarOrder() {
        // 4 classes would make cls_logits the same size as boxes_raw.
        val roles = OutputRoles.resolve(
            intArrayOf(8400 * 4, 8400 * 4), ModelTask.DETECT, 8400, 4, 32, 160 * 160,
            sidecarOrder = listOf("boxes_raw", "cls_logits"),
        )
        assertEquals(0, roles.boxes)
        assertEquals(1, roles.cls)
    }
}
