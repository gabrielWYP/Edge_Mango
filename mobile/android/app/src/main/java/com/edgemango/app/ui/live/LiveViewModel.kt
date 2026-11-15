package com.edgemango.app.ui.live

import android.app.Application
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgemango.app.EdgeMangoApplication
import com.edgemango.app.camera.FrameConverter
import com.edgemango.app.camera.copyOf
import com.edgemango.app.core.FramePipeline
import com.edgemango.app.core.FrameResult
import com.edgemango.app.core.PipelineConfig
import com.edgemango.app.core.RoiMode
import com.edgemango.app.data.AppSettings
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.data.LiveLog
import com.edgemango.app.inference.AcceleratorChoice
import com.edgemango.app.inference.LiteRtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

sealed interface ModelStatus {
    data object Loading : ModelStatus
    data class Ready(val modelName: String, val accelerator: String, val note: String, val isSegment: Boolean, val loadMs: Double) : ModelStatus
    data class Failed(val message: String) : ModelStatus
}

/** Everything the live screen shows; plain data, so the presentational composables stay stateless. */
data class LiveUiState(
    val settings: AppSettings = AppSettings(),
    val model: ModelStatus = ModelStatus.Loading,
    val result: FrameResult? = null,
    val liveFps: Double = 0.0,
    val recording: Boolean = false,
    val recordedRows: Int = 0,
    val recordPath: String? = null,
)

/** Owns the model session and runs the per-frame pipeline on the camera analysis thread. */
class LiveViewModel(application: Application) : AndroidViewModel(application) {
    private val graph = (application as EdgeMangoApplication).graph
    private val converter = FrameConverter()
    private val lock = Any()

    @Volatile private var pipeline: FramePipeline? = null
    @Volatile private var session: LiteRtSession? = null
    @Volatile private var modelName: String = ""
    @Volatile private var recorder: LiveLog? = null
    private var lastFrameAt = 0L
    private var lastSnapshotAt = 0L
    private var fpsEma = 0.0

    private val active = MutableStateFlow(false)
    private val _state = MutableStateFlow(LiveUiState(settings = graph.settings.state.value))
    val state: StateFlow<LiveUiState> = _state.asStateFlow()

    val analyzer = ImageAnalysis.Analyzer { image -> analyze(image) }

    init {
        viewModelScope.launch {
            graph.settings.state.collect { s -> _state.update { it.copy(settings = s) } }
        }
        viewModelScope.launch {
            combine(graph.settings.state.map { it.modelId to it.accelerator }.distinctUntilChanged(), active) { key, on -> key to on }
                .collect { (key, on) -> if (on) load(key.first, key.second) else unload() }
        }
    }

    /** The live screen is on screen: load the model. Off screen the session is released so other screens (benchmark) measure alone. */
    fun setActive(on: Boolean) {
        active.value = on
    }

    // ---- user actions -----------------------------------------------------------------------------------------

    fun setDisplayMode(mode: DisplayMode) = graph.settings.update { copy(displayMode = mode) }
    fun setRoiMode(mode: RoiMode) = graph.settings.update { copy(roiMode = mode) }
    fun setConfidence(value: Float) = graph.settings.update { copy(confidence = value) }

    fun toggleRecording() {
        if (recorder == null) {
            val log = graph.results.startLiveLog()
            recorder = log
            _state.update { it.copy(recording = true, recordedRows = 0, recordPath = log.file.absolutePath) }
        } else {
            stopRecording()
        }
    }

    private fun stopRecording() {
        val log = recorder ?: return
        recorder = null
        log.close()
        _state.update { it.copy(recording = false, recordedRows = log.rows) }
    }

    // ---- model lifecycle --------------------------------------------------------------------------------------

    private suspend fun unload() {
        withContext(Dispatchers.Default) {
            val old = synchronized(lock) { pipeline = null; session.also { session = null } }
            old?.close()
        }
        stopRecording()
        _state.update { it.copy(model = ModelStatus.Loading, result = null) }
    }

    private suspend fun load(modelId: String, accelerator: AcceleratorChoice) {
        _state.update { it.copy(model = ModelStatus.Loading, result = null) }
        withContext(Dispatchers.Default) {
            val old = synchronized(lock) { pipeline = null; session.also { session = null } }
            old?.close()
            val ref = graph.catalog.find(modelId) ?: graph.catalog.list().firstOrNull()
            if (ref == null) {
                _state.update { it.copy(model = ModelStatus.Failed("no model found")) }
                return@withContext
            }
            try {
                val s = LiteRtSession.open(getApplication(), ref.file.absolutePath, ref.sidecar, accelerator)
                synchronized(lock) {
                    session = s
                    modelName = ref.name
                    pipeline = FramePipeline(ref.sidecar, s)
                }
                _state.update {
                    it.copy(model = ModelStatus.Ready(ref.name, s.label, s.note, ref.sidecar.task == com.edgemango.app.core.ModelTask.SEGMENT, s.loadMs))
                }
            } catch (t: Throwable) {
                _state.update { it.copy(model = ModelStatus.Failed(t.message ?: t.toString())) }
            }
        }
    }

    // ---- per-frame work (camera analysis thread) -----------------------------------------------------------------

    private fun analyze(image: ImageProxy) {
        val frame = try {
            converter.toUpright(image)
        } finally {
            image.close()
        }
        val s = _state.value.settings
        val config = PipelineConfig(s.roiMode, s.confidence, computeMasks = s.displayMode.showMask)
        val result = synchronized(lock) {
            val p = pipeline ?: return
            val r = try {
                p.process(frame, config)
            } catch (t: Throwable) {
                _state.update { it.copy(model = ModelStatus.Failed("inference failed: ${t.message}")) }
                pipeline = null
                return
            }
            r
        }

        val now = System.nanoTime()
        if (lastFrameAt != 0L) {
            val dt = (now - lastFrameAt) / 1e6
            fpsEma = if (fpsEma == 0.0) 1000.0 / dt else 0.85 * fpsEma + 0.15 * (1000.0 / dt)
        }
        lastFrameAt = now
        if (now - lastSnapshotAt > 1_000_000_000L) {
            lastSnapshotAt = now
            graph.lastFrame.set(frame.copyOf())
        }

        recorder?.let { log ->
            val w = result.window
            val t = result.timings
            log.append(
                listOf(
                    System.currentTimeMillis(), modelName, session?.label, s.roiMode.name, w.x1, w.y1, w.x2, w.y2,
                    result.detections.size, result.detections.count { it.cls == FrameResult.MANGO_CLASS },
                    result.detections.count { it.cls == FrameResult.DAMAGE_CLASS },
                    "%.4f".format(java.util.Locale.US, result.maxDamageScore), s.confidence,
                    "%.2f".format(java.util.Locale.US, t.roiMs), "%.2f".format(java.util.Locale.US, t.preMs),
                    "%.2f".format(java.util.Locale.US, t.inferMs), "%.2f".format(java.util.Locale.US, t.postMs),
                    "%.2f".format(java.util.Locale.US, t.totalMs), "%.1f".format(java.util.Locale.US, fpsEma),
                ),
            )
            _state.update { it.copy(recordedRows = log.rows) }
        }
        _state.update { it.copy(result = result, liveFps = fpsEma) }
    }

    override fun onCleared() {
        stopRecording()
        synchronized(lock) { pipeline = null }
        session?.close()
        session = null
    }
}
