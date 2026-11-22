package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ResultsTest {
    private val sample = BenchRecord(
        timestamp = "2026-11-14T20:00:00Z",
        appVersion = "0.1.0",
        device = "Xiaomi 2407FPN8EG",
        soc = "MediaTek MT6897",
        androidVersion = "15 (API 35)",
        modelName = "student_seg_nano_random_fp32.tflite",
        task = "segment",
        precision = "fp32",
        modelSizeMb = 13.02,
        numParameters = 3354818,
        flops = null,
        accelerator = "CPU x4",
        accelerationNote = "",
        input = "sample_crop.png",
        warmUp = 10,
        runs = 100,
        loadMs = 120.5,
        inference = LatencyStats.of(doubleArrayOf(10.0, 12.0, 11.0)),
        endToEnd = LatencyStats.of(doubleArrayOf(20.0, 22.0, 21.0)),
        peakRamMb = 310.25,
        peakRamSource = "VmHWM",
        ramBeforeLoadMb = 200.0,
        nativeHeapMb = 90.0,
    )

    @Test
    fun csvRowMatchesTheHeaderWidth() {
        assertEquals(BenchRecord.CSV_HEADER.split(",").size, sample.toCsvRow().split(",").size)
    }

    @Test
    fun jsonRoundTripKeepsEverything() {
        val back = BenchRecord.fromJson(sample.toJson())
        assertEquals(sample, back)
        assertNull(back.flops)
    }

    @Test
    fun procStatusParsing() {
        val text = "Name:\tcom.edgemango.app\nVmPeak:\t 9000000 kB\nVmHWM:\t  312345 kB\nVmRSS:\t  300000 kB\nThreads:\t40\n"
        assertEquals(312345L, ProcStatus.kb(text, "VmHWM"))
        assertEquals(300000L, ProcStatus.kb(text, "VmRSS"))
        assertNull(ProcStatus.kb(text, "VmSwap"))
    }

    @Test
    fun rssHwmReadsBothLinesOrNull() {
        val text = "VmHWM:\t  312345 kB\nVmRSS:\t  300000 kB\n"
        assertEquals(300000L to 312345L, ProcStatus.rssHwm(text))
        assertNull(ProcStatus.rssHwm("VmRSS:\t 1 kB\n"))
    }
}
