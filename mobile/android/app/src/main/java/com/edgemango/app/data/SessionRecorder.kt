package com.edgemango.app.data

import android.graphics.Bitmap
import android.util.Log
import com.edgemango.app.core.FrameRecord
import com.edgemango.app.core.MemRecord
import com.edgemango.app.core.MemoryStats
import com.edgemango.app.core.SessionMemory
import com.edgemango.app.device.MemoryProbe
import com.edgemango.app.core.RgbImage
import com.edgemango.app.core.SessionInfo
import com.edgemango.app.core.SessionJson
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

/**
 * One field-test session on disk:
 *
 * ```
 * session_<yyyyMMdd_HHmmss>/
 *   frames/f_000000.jpg ...   the upright analysed frame, JPEG
 *   predictions.jsonl         one FrameRecord per processed frame
 *   session.json              SessionInfo, written by [finish]
 * ```
 *
 * [record] is called on the camera analysis thread and never blocks on disk for frames: JPEG encoding and writing run on
 * one background thread fed through a bounded queue. When the queue is full the frame is dropped (the prediction line is
 * still written, with `frame = null`), counted in [droppedWrites] and logged.
 */
class SessionRecorder(
    val dir: File,
    queueCapacity: Int = 6,
    private val jpegQuality: Int = 90,
) : AutoCloseable {
    private class Job(val file: File, val frame: RgbImage)

    private val framesDir = File(dir, "frames").apply { mkdirs() }
    private val predictions: BufferedWriter = File(dir, "predictions.jsonl").bufferedWriter()
    private val queue = ArrayBlockingQueue<Job>(queueCapacity)
    private val dropped = AtomicInteger()
    private val saved = AtomicInteger()
    private val failed = AtomicInteger()
    private var count = 0
    private val memStats = MemoryStats()
    @Volatile private var lastRssKb = 0L
    @Volatile private var closing = false

    val startedAtMs = System.currentTimeMillis()

    /** Frames processed so far (prediction lines written). */
    val frames: Int @Synchronized get() = count
    /** Latest `VmRSS` in kB (0 before the first frame), for the HUD. */
    val rssKb: Long get() = lastRssKb
    val framesSaved: Int get() = saved.get()
    val droppedWrites: Int get() = dropped.get()
    val failedWrites: Int get() = failed.get()

    private val worker = Thread({
        while (true) {
            val job = queue.poll(100, TimeUnit.MILLISECONDS)
            if (job == null) {
                if (closing && queue.isEmpty()) break
                continue
            }
            try {
                val bmp = Bitmap.createBitmap(job.frame.pixels, job.frame.width, job.frame.height, Bitmap.Config.ARGB_8888)
                FileOutputStream(job.file).use { bmp.compress(Bitmap.CompressFormat.JPEG, jpegQuality, it) }
                bmp.recycle()
                saved.incrementAndGet()
            } catch (t: Throwable) {
                failed.incrementAndGet()
                Log.w(TAG, "frame write failed: ${job.file.name}: ${t.message}")
            }
        }
    }, "session-writer").apply { isDaemon = true; start() }

    /**
     * Logs one processed frame. [frame] is the upright analysed frame; it is copied only when the queue has room (the
     * converter reuses its buffer). [makeRecord] builds the prediction line from the frame's index and relative path.
     */
    @Synchronized
    fun record(frame: RgbImage, makeRecord: (index: Int, frameFile: String?) -> FrameRecord) {
        check(!closing) { "session is closed" }
        val index = count++
        val name = String.format(Locale.US, "f_%06d.jpg", index)
        var relative: String? = null
        if (queue.remainingCapacity() > 0) {
            val copy = RgbImage(frame.width, frame.height, frame.pixels.copyOf(frame.width * frame.height))
            if (queue.offer(Job(File(framesDir, name), copy))) relative = "frames/$name"
        }
        if (relative == null) {
            val n = dropped.incrementAndGet()
            if (n == 1 || n % 25 == 0) Log.w(TAG, "write queue full: $n frame(s) dropped so far (frame $index)")
        }
        val mem = MemoryProbe.rssHwmKb()?.let { (rss, hwm) ->
            val pss = if (memStats.wantsPss(index)) runCatching { MemoryProbe.pssKb() }.getOrNull() else null
            lastRssKb = rss
            memStats.add(rss, hwm, pss)
            MemRecord(rss, hwm, pss)
        }
        predictions.write(SessionJson.frameLine(makeRecord(index, relative).copy(mem = mem)))
        predictions.newLine()
        if (index % 20 == 19) predictions.flush()
    }

    /**
     * Drains the write queue, closes `predictions.jsonl`, writes `session.json` from [info] (its counters are filled in
     * here) and zips the folder next to it. Slow (waits for the queue, zips): call it off the main thread.
     */
    fun finish(
        ramBeforeLoadKb: Long,
        info: (frames: Int, saved: Int, dropped: Int, failed: Int, endedAtMs: Long, memory: SessionMemory?) -> SessionInfo,
    ): File {
        close()
        val memory = synchronized(this) { memStats.summary(ramBeforeLoadKb) }
        val session = info(frames, framesSaved, droppedWrites, failedWrites, System.currentTimeMillis(), memory)
        File(dir, "session.json").writeText(SessionJson.sessionJson(session))
        return SessionZip.zip(dir, File(dir.parentFile, dir.name + ".zip"))
    }

    @Synchronized
    override fun close() {
        if (closing) return
        closing = true
        runCatching { worker.join() }
        runCatching { predictions.flush() }
        runCatching { predictions.close() }
    }

    companion object {
        private const val TAG = "SessionRecorder"
    }
}
