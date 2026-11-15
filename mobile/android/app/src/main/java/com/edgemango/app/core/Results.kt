package com.edgemango.app.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** One benchmark run, as saved to JSON and appended to `benchmarks.csv`. */
@Serializable
data class BenchRecord(
    val timestamp: String,
    val appVersion: String,
    val device: String,
    val soc: String,
    val androidVersion: String,
    val modelName: String,
    val task: String,
    val precision: String,
    val modelSizeMb: Double,
    val numParameters: Long?,
    val flops: Double?,
    val accelerator: String,
    val accelerationNote: String,
    val input: String,
    val warmUp: Int,
    val runs: Int,
    val loadMs: Double,
    /** Model only: copy input in, run, copy outputs out. */
    val inference: LatencyStats,
    /** ROI + preprocessing + inference + decode + masks, per frame. */
    val endToEnd: LatencyStats,
    val peakRamMb: Double,
    val peakRamSource: String,
    val ramBeforeLoadMb: Double,
    val nativeHeapMb: Double,
) {
    fun toJson(): String = JSON.encodeToString(serializer(), this)

    fun toCsvRow(): String = Csv.row(
        listOf(
            timestamp, appVersion, device, soc, androidVersion, modelName, task, precision, fmt(modelSizeMb), numParameters, flops,
            accelerator, accelerationNote, input, warmUp, runs, fmt(loadMs),
            fmt(inference.meanMs), fmt(inference.stdMs), fmt(inference.p50Ms), fmt(inference.p95Ms), fmt(inference.minMs), fmt(inference.maxMs), fmt(inference.fps),
            fmt(endToEnd.meanMs), fmt(endToEnd.p50Ms), fmt(endToEnd.p95Ms), fmt(endToEnd.fps),
            fmt(peakRamMb), peakRamSource, fmt(ramBeforeLoadMb), fmt(nativeHeapMb),
        ),
    )

    companion object {
        private val JSON = Json { prettyPrint = true; encodeDefaults = true }

        fun fromJson(text: String): BenchRecord = JSON.decodeFromString(serializer(), text)

        private fun fmt(v: Double) = String.format(java.util.Locale.US, "%.3f", v)

        val CSV_HEADER = listOf(
            "timestamp", "app_version", "device", "soc", "android", "model", "task", "precision", "model_size_mb", "num_parameters", "flops",
            "accelerator", "acceleration_note", "input", "warmup", "runs", "load_ms",
            "infer_mean_ms", "infer_std_ms", "infer_p50_ms", "infer_p95_ms", "infer_min_ms", "infer_max_ms", "infer_fps",
            "e2e_mean_ms", "e2e_p50_ms", "e2e_p95_ms", "e2e_fps",
            "peak_ram_mb", "peak_ram_source", "ram_before_load_mb", "native_heap_mb",
        ).joinToString(",")
    }
}

/** `/proc/self/status` helpers. */
object ProcStatus {
    /** Value in kB of a `Key:  1234 kB` line, or null when the key is absent. */
    fun kb(status: String, key: String): Long? {
        for (line in status.lineSequence()) {
            if (line.startsWith("$key:")) {
                return line.substringAfter(':').trim().substringBefore(' ').toLongOrNull()
            }
        }
        return null
    }
}
