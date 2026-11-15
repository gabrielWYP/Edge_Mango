package com.edgemango.app.ui.benchmark

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.edgemango.app.AppGraph
import com.edgemango.app.core.BenchRecord
import com.edgemango.app.core.BenchmarkRunner
import com.edgemango.app.core.FramePipeline
import com.edgemango.app.core.PipelineConfig
import com.edgemango.app.core.RgbImage
import com.edgemango.app.core.RoiMode
import com.edgemango.app.data.ModelRef
import com.edgemango.app.device.DeviceInfo
import com.edgemango.app.device.MemoryProbe
import com.edgemango.app.inference.AcceleratorChoice
import com.edgemango.app.inference.LiteRtSession
import java.time.Instant
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

data class BenchConfig(
    val model: ModelRef,
    val accelerator: AcceleratorChoice,
    val warmUp: Int,
    val runs: Int,
    val useLastFrame: Boolean,
    val roiMode: RoiMode,
)

/**
 * The thesis benchmark. For one model and accelerator it measures, on one fixed input:
 *
 * - **model load time** (open + first verification run);
 * - **inference latency**: copy input in + run + copy outputs out, `runs` times after `warmUp` discarded runs;
 * - **end-to-end latency**: ROI + preprocessing + inference + decode + masks, same W and N;
 * - **peak RAM** over load + all runs (see [MemoryProbe]);
 * and records model size, parameters/FLOPs from the sidecar, device and Android version.
 */
class BenchmarkEngine(private val context: Context, private val graph: AppGraph) {

    suspend fun run(config: BenchConfig, onProgress: (phase: String, done: Int, total: Int) -> Unit): BenchRecord =
        withContext(Dispatchers.Default) {
            val job = currentCoroutineContext()[Job]
            val (frame, inputName, roi) = loadInput(config)

            System.gc()
            val peakReset = MemoryProbe.resetPeak()
            val before = MemoryProbe.snapshot()

            onProgress("Loading model", 0, 1)
            val session = LiteRtSession.open(context, config.model.file.absolutePath, config.model.sidecar, config.accelerator)
            session.use {
                val pipeline = FramePipeline(config.model.sidecar, session)
                val cfg = PipelineConfig(roi, config.model.sidecar.decode.confThreshold, computeMasks = true)
                pipeline.prepare(frame, cfg)

                val total = 2 * (config.warmUp + config.runs)
                val inference = BenchmarkRunner.run(
                    config.warmUp, config.runs,
                    onProgress = { done, _ -> job?.ensureActive(); onProgress("Inference (model only)", done, total) },
                ) { session.infer(pipeline.input) }

                val endToEnd = BenchmarkRunner.run(
                    config.warmUp, config.runs,
                    onProgress = { done, _ -> job?.ensureActive(); onProgress("End to end (ROI, pre, infer, post)", config.warmUp + config.runs + done, total) },
                ) { pipeline.process(frame, cfg) }

                val after = MemoryProbe.snapshot()
                val sidecar = config.model.sidecar
                BenchRecord(
                    timestamp = Instant.now().toString(),
                    appVersion = appVersion(),
                    device = DeviceInfo.device,
                    soc = DeviceInfo.soc,
                    androidVersion = DeviceInfo.android,
                    modelName = config.model.name,
                    task = sidecar.task.name.lowercase(),
                    precision = sidecar.precision ?: "unknown",
                    modelSizeMb = config.model.sizeMb,
                    numParameters = sidecar.numParameters,
                    flops = sidecar.flops,
                    accelerator = session.label,
                    accelerationNote = session.note,
                    input = inputName,
                    warmUp = config.warmUp,
                    runs = config.runs,
                    loadMs = session.loadMs,
                    inference = inference.stats,
                    endToEnd = endToEnd.stats,
                    peakRamMb = after.peakRssMb,
                    peakRamSource = if (peakReset) "VmHWM since just before model load" else "VmHWM since process start (reset refused)",
                    ramBeforeLoadMb = before.rssMb,
                    nativeHeapMb = after.nativeHeapMb,
                )
            }
        }

    private fun loadInput(config: BenchConfig): Triple<RgbImage, String, RoiMode> {
        if (config.useLastFrame) {
            val f = graph.lastFrame.get() ?: error("no camera frame yet: open the Live tab first")
            return Triple(f, "last camera frame ${f.width}x${f.height}", config.roiMode)
        }
        val bmp = context.assets.open(SAMPLE_ASSET).use { BitmapFactory.decodeStream(it) }
        val rgb = bitmapToRgb(bmp)
        // The sample is already an ROI crop, so it goes through the pipeline as a whole frame.
        return Triple(rgb, "$SAMPLE_ASSET ${rgb.width}x${rgb.height}", RoiMode.FULL_FRAME)
    }

    private fun bitmapToRgb(bmp: Bitmap): RgbImage {
        val pixels = IntArray(bmp.width * bmp.height)
        bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
        return RgbImage(bmp.width, bmp.height, pixels)
    }

    private fun appVersion(): String =
        runCatching { context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "?" }.getOrDefault("?")

    companion object {
        const val SAMPLE_ASSET = "sample_crop.png"
    }
}
