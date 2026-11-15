package com.edgemango.app.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class LatencyStatsTest {
    @Test
    fun summarizesATextbookSample() {
        val s = LatencyStats.of(doubleArrayOf(10.0, 20.0, 30.0, 40.0, 50.0))
        assertEquals(5, s.count)
        assertEquals(30.0, s.meanMs, 1e-9)
        assertEquals(30.0, s.p50Ms, 1e-9)
        assertEquals(10.0, s.minMs, 1e-9)
        assertEquals(50.0, s.maxMs, 1e-9)
        // linear interpolation like numpy.percentile: 0.95 * 4 = 3.8 -> 40 + 0.8 * 10
        assertEquals(48.0, s.p95Ms, 1e-9)
        assertEquals(1000.0 / 30.0, s.fps, 1e-9)
        assertEquals(Math.sqrt(250.0), s.stdMs, 1e-9) // sample std (n - 1)
    }

    @Test
    fun orderOfSamplesDoesNotMatter() {
        val a = LatencyStats.of(doubleArrayOf(5.0, 1.0, 3.0, 2.0, 4.0))
        val b = LatencyStats.of(doubleArrayOf(1.0, 2.0, 3.0, 4.0, 5.0))
        assertEquals(a, b)
    }

    @Test
    fun singleSampleHasZeroSpread() {
        val s = LatencyStats.of(doubleArrayOf(8.0))
        assertEquals(8.0, s.p95Ms, 1e-9)
        assertEquals(0.0, s.stdMs, 1e-9)
        assertEquals(125.0, s.fps, 1e-9)
    }

    @Test(expected = IllegalArgumentException::class)
    fun emptySampleIsRejected() {
        LatencyStats.of(doubleArrayOf())
    }

    @Test
    fun benchmarkRunnerDiscardsWarmUpRuns() {
        var calls = 0
        val times = mutableListOf<Double>()
        val result = BenchmarkRunner.run(warmUp = 3, runs = 4, clockMs = { calls.toDouble() }) {
            calls += 1
            times += calls.toDouble()
        }
        assertEquals(7, calls)
        assertEquals(4, result.count)
        assertEquals(3, result.warmUp)
        assertTrue(result.stats.meanMs >= 0.0)
    }

    @Test
    fun csvEscapesCommasQuotesAndNewlines() {
        assertEquals("a,\"b,c\",\"d\"\"e\",\"f\ng\"", Csv.row(listOf("a", "b,c", "d\"e", "f\ng")))
        assertEquals("1.5,2,x", Csv.row(listOf(1.5, 2, "x")))
    }
}
