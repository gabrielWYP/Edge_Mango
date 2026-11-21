package com.edgemango.app.data

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.edgemango.app.core.BenchRecord
import com.edgemango.app.core.Csv
import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Local result log: everything lives in `<app files>/results/` (the thesis' "registro local de resultados"). */
class ResultStore(private val context: Context) {
    val dir = File(context.filesDir, "results").apply { mkdirs() }
    val benchmarksCsv: File get() = File(dir, "benchmarks.csv")

    @Synchronized
    fun saveBenchmark(record: BenchRecord): File {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val json = File(dir, "bench_$stamp.json")
        json.writeText(record.toJson())
        val csv = benchmarksCsv
        if (!csv.exists()) csv.writeText(BenchRecord.CSV_HEADER + "\n")
        csv.appendText(record.toCsvRow() + "\n")
        return json
    }

    fun benchmarkJsons(): List<File> =
        dir.listFiles { f -> f.name.startsWith("bench_") && f.extension == "json" }?.sortedByDescending { it.name }.orEmpty()

    fun liveLogs(): List<File> =
        dir.listFiles { f -> f.name.startsWith("live_") && f.extension == "csv" }?.sortedByDescending { it.name }.orEmpty()

    fun startLiveLog(): LiveLog {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return LiveLog(File(dir, "live_$stamp.csv"))
    }

    /** Creates `session_<yyyyMMdd_HHmmss>/` for a field-test session. */
    fun startSession(): SessionRecorder {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        return SessionRecorder(File(dir, "session_$stamp"))
    }

    fun share(files: List<File>, title: String, mime: String = "text/plain"): Intent {
        val uris = files.filter { it.exists() }.map { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", it) }
        val intent = if (uris.size == 1) {
            Intent(Intent.ACTION_SEND).putExtra(Intent.EXTRA_STREAM, uris[0])
        } else {
            Intent(Intent.ACTION_SEND_MULTIPLE).putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
        }
        intent.type = mime
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        return Intent.createChooser(intent, title)
    }
}

/** Per-frame CSV of a live session. */
class LiveLog(val file: File) : AutoCloseable {
    private val writer: BufferedWriter = file.bufferedWriter().also {
        it.write(HEADER)
        it.newLine()
    }
    var rows = 0
        private set

    @Synchronized
    fun append(values: List<Any?>) {
        writer.write(Csv.row(values))
        writer.newLine()
        rows++
        if (rows % 30 == 0) writer.flush()
    }

    @Synchronized
    override fun close() {
        runCatching { writer.flush() }
        runCatching { writer.close() }
    }

    companion object {
        val HEADER = listOf(
            "timestamp_ms", "model", "accelerator", "roi_mode", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
            "n_detections", "n_mango", "n_damage", "max_damage_score", "confidence",
            "roi_ms", "pre_ms", "infer_ms", "post_ms", "total_ms", "live_fps",
        ).joinToString(",")
    }
}
