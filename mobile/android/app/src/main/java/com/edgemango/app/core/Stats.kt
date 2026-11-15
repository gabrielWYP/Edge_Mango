package com.edgemango.app.core

import kotlin.math.floor
import kotlin.math.sqrt

/**
 * Latency summary of one benchmark run, in milliseconds.
 * Percentiles use linear interpolation between order statistics (numpy's default), the standard deviation is the
 * sample one (n - 1), and `fps = 1000 / mean`, i.e. throughput of sequential single-image inference.
 */
@kotlinx.serialization.Serializable
data class LatencyStats(
    val count: Int,
    val meanMs: Double,
    val stdMs: Double,
    val minMs: Double,
    val maxMs: Double,
    val p50Ms: Double,
    val p95Ms: Double,
    val fps: Double,
) {
    companion object {
        fun of(samplesMs: DoubleArray): LatencyStats {
            require(samplesMs.isNotEmpty()) { "no samples" }
            val sorted = samplesMs.sortedArray()
            val n = sorted.size
            val mean = sorted.sum() / n
            val variance = if (n > 1) sorted.sumOf { (it - mean) * (it - mean) } / (n - 1) else 0.0
            return LatencyStats(n, mean, sqrt(variance), sorted.first(), sorted.last(), percentile(sorted, 0.50), percentile(sorted, 0.95), 1000.0 / mean)
        }

        private fun percentile(sorted: DoubleArray, p: Double): Double {
            val pos = p * (sorted.size - 1)
            val lo = floor(pos).toInt()
            val hi = minOf(lo + 1, sorted.size - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
        }
    }
}

class BenchmarkResult(val warmUp: Int, val samplesMs: DoubleArray, val stats: LatencyStats) {
    val count: Int get() = samplesMs.size
}

object BenchmarkRunner {
    /** Runs [block] `warmUp` times (discarded) then `runs` times, timing each run with [clockMs]. */
    fun run(
        warmUp: Int,
        runs: Int,
        clockMs: () -> Double = { System.nanoTime() / 1e6 },
        onProgress: (done: Int, total: Int) -> Unit = { _, _ -> },
        block: () -> Unit,
    ): BenchmarkResult {
        require(runs > 0) { "runs must be > 0" }
        repeat(warmUp) { block(); onProgress(it + 1, warmUp + runs) }
        val samples = DoubleArray(runs)
        for (i in 0 until runs) {
            val t0 = clockMs()
            block()
            samples[i] = clockMs() - t0
            onProgress(warmUp + i + 1, warmUp + runs)
        }
        return BenchmarkResult(warmUp, samples, LatencyStats.of(samples))
    }
}

object Csv {
    fun row(values: List<Any?>): String = values.joinToString(",") { escape(it?.toString() ?: "") }

    private fun escape(s: String): String =
        if (s.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
}
