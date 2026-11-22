package com.edgemango.app.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionJsonTest {
    private val ctx = FrameContext(
        roiMode = "DETECTOR", confThreshold = 0.25f, mangoThreshold = 0.5f, nmsIou = 0.5f, agnosticNms = false, maxMangos = 10,
        accelerator = "CPU x4", detectorModel = "mango_det.tflite", damageModel = "student_seg.tflite",
        keepThreshold = 0.15f, holdFrames = 5,
    )
    private val timings = FrameTimings(0.0, 4.0, 31.5, 6.25)

    private fun twoStage(): TwoStageResult {
        val damage = FrameDetection(1, 0.8123456f, SourceBox(150.04, 200.0, 180.0, 230.0), doubleArrayOf(150.0, 200.0, 180.0, 200.0, 165.0, 230.0))
        val m1 = MangoDetection(3, SourceBox(100.0, 150.0, 300.0, 350.0), 0.93f, IntBox(80, 130, 320, 370), listOf(damage), null, timings)
        val m2 = MangoDetection(4, SourceBox(400.0, 150.0, 500.0, 250.0), 0.61f, IntBox(390, 140, 510, 260), emptyList(), null, timings, held = true)
        return TwoStageResult(640, 480, listOf(m1, m2), 12.0, FrameTimings(0.0, 10.0, 70.0, 12.0))
    }

    @Test
    fun twoStageLineHasEveryFieldTheEvaluatorReads() {
        val line = SessionJson.frameLine(SessionJson.fromTwoStage(7, 1_700_000_000_123, 987_654_321L, "frames/f_000007.jpg", ctx, twoStage()))
        assertFalse("one JSON line", line.contains('\n'))
        val o = Json.parseToJsonElement(line).jsonObject
        assertEquals(7, o["index"]!!.jsonPrimitive.content.toInt())
        assertEquals("1700000000123", o["wallMs"]!!.jsonPrimitive.content)
        assertEquals("987654321", o["cameraTimestampNs"]!!.jsonPrimitive.content)
        assertEquals("640", o["frameW"]!!.jsonPrimitive.content)
        assertEquals("DETECTOR", o["roiMode"]!!.jsonPrimitive.content)
        assertEquals("frames/f_000007.jpg", o["frame"]!!.jsonPrimitive.content)
        assertEquals("mango_det.tflite", o["detectorModel"]!!.jsonPrimitive.content)
        assertEquals("false", o["noMango"]!!.jsonPrimitive.content)
        assertEquals("0.5", o["mangoThreshold"]!!.jsonPrimitive.content)
        assertEquals("0.5", o["nmsIou"]!!.jsonPrimitive.content)
        assertEquals("0.15", o["keepThreshold"]!!.jsonPrimitive.content)
        assertEquals("5", o["holdFrames"]!!.jsonPrimitive.content)
        assertEquals("false", o["mangos"]!!.jsonArray[0].jsonObject["held"]!!.jsonPrimitive.content)
        assertEquals("true", o["mangos"]!!.jsonArray[1].jsonObject["held"]!!.jsonPrimitive.content)
        assertNull("single-crop field absent in DETECTOR mode", o["detections"])

        val mangos = o["mangos"]!!.jsonArray
        assertEquals(2, mangos.size)
        val m = mangos[0].jsonObject
        assertEquals("3", m["trackId"]!!.jsonPrimitive.content)
        assertEquals(listOf(100.0, 150.0, 300.0, 350.0), m["box"]!!.jsonArray.map { it.jsonPrimitive.content.toDouble() })
        assertEquals("0.93", m["score"]!!.jsonPrimitive.content)
        assertEquals("0.8123", m["maxDamage"]!!.jsonPrimitive.content)
        val d = (m["damages"] as JsonArray)[0].jsonObject
        assertEquals(listOf(150.0, 200.0, 180.0, 230.0), d["box"]!!.jsonArray.map { it.jsonPrimitive.content.toDouble() })
        assertEquals(6, d["polygon"]!!.jsonArray.size)
        assertEquals(0.0, ((mangos[1].jsonObject["damages"]) as JsonArray).size.toDouble(), 0.0)

        val t = o["timings"]!!.jsonObject
        assertEquals("92.0", t["totalMs"]!!.jsonPrimitive.content)
        assertEquals("12.0", t["detectorMs"]!!.jsonPrimitive.content)
    }

    @Test
    fun noMangoFrameIsARecordWithAnEmptyMangoList() {
        val r = TwoStageResult(640, 480, emptyList(), 11.0, FrameTimings(0.0, 4.0, 30.0, 1.0))
        val parsed = SessionJson.parseFrame(SessionJson.frameLine(SessionJson.fromTwoStage(0, 1L, 2L, null, ctx, r)))
        assertEquals(true, parsed.noMango)
        assertTrue(parsed.mangos.isEmpty())
        assertNull(parsed.frame)
    }

    @Test
    fun lineRoundTrips() {
        val rec = SessionJson.fromTwoStage(7, 5L, 6L, "frames/f_000007.jpg", ctx, twoStage())
        assertEquals(rec, SessionJson.parseFrame(SessionJson.frameLine(rec)))
    }

    @Test
    fun singleCropModesListTheirDetections() {
        val r = FrameResult(
            640, 480, IntBox(0, 0, 640, 480), null,
            listOf(FrameDetection(0, 0.9f, SourceBox(1.0, 2.0, 3.0, 4.0)), FrameDetection(1, 0.4f, SourceBox(1.0, 2.0, 2.0, 3.0))),
            null, timings,
        )
        val rec = SessionJson.parseFrame(SessionJson.frameLine(SessionJson.fromSingleCrop(1, 1L, 2L, "frames/f_000001.jpg", ctx.copy(roiMode = "FULL_FRAME"), r)))
        assertNull(rec.noMango)
        assertTrue(rec.mangos.isEmpty())
        assertEquals(listOf(0, 1), rec.detections!!.map { it.cls })
        assertEquals(0.4, rec.detections!![1].score, 1e-9)
    }

    @Test
    fun sessionInfoRoundTripsWithTheSidecarEmbedded() {
        val sidecar = Json.parseToJsonElement("""{"format":"ultralytics","task":"detect"}""")
        val info = SessionInfo(
            "session_20261121_101500", "0.1.0", "Xiaomi 23078PND5G", "MediaTek MT6897", "15 (API 35)", 1L, 2L, 120, 118, 2, 0,
            SessionSettings("DETECTOR", "BOTH", 0.25, 0.5, 10, 0.5, false, 1280, "CPU_4"),
            ModelInfo("student_seg.tflite", "CPU x4", null), ModelInfo("mango_det.tflite", "CPU x4", sidecar),
        )
        val parsed = SessionJson.parseSession(SessionJson.sessionJson(info))
        assertEquals(info, parsed)
        val o = Json.parseToJsonElement(SessionJson.sessionJson(info)).jsonObject
        assertNotNull((o["detectorModel"] as JsonObject)["sidecar"])
        assertEquals(120, o["frameCount"]!!.jsonPrimitive.content.toInt())
    }
}
