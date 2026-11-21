package com.edgemango.app.ui.live

import android.app.Application
import android.content.Intent
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgemango.app.EdgeMangoApplication
import com.edgemango.app.camera.FrameConverter
import com.edgemango.app.camera.copyOf
import com.edgemango.app.core.FrameContext
import com.edgemango.app.core.FramePipeline
import com.edgemango.app.core.FrameResult
import com.edgemango.app.core.ModelInfo
import com.edgemango.app.core.ModelTask
import com.edgemango.app.core.PipelineConfig
import com.edgemango.app.core.RgbImage
import com.edgemango.app.core.RoiMode
import com.edgemango.app.core.SessionInfo
import com.edgemango.app.core.SessionJson
import com.edgemango.app.core.SessionSettings
import com.edgemango.app.core.TwoStageConfig
import com.edgemango.app.core.TwoStagePipeline
import com.edgemango.app.core.TwoStageResult
import com.edgemango.app.data.AppSettings
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.data.LiveLog
import com.edgemango.app.data.ModelRef
import com.edgemango.app.data.SessionRecorder
import com.edgemango.app.device.DeviceInfo
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
import kotlinx.serialization.json.Json
import java.io.File
import java.util.Locale

sealed interface ModelStatus {
    data object Loading : ModelStatus
    data class Ready(
        val modelName: String,
        val accelerator: String,
        val note: String,
        val isSegment: Boolean,
        val loadMs: Double,
        /** Stage-1 model, when one is configured and loaded. */
        val detectorName: String? = null,
        /** Why the stage-1 model is not usable, when it failed to load. */
        val detectorNote: String = "",
    ) : ModelStatus
    data class Failed(val message: String) : ModelStatus
}

/** Field-test session state shown in the HUD and controls. */
data class SessionUi(
    val active: Boolean = false,
    val frames: Int = 0,
    val saved: Int = 0,
    val dropped: Int = 0,
    val finishing: Boolean = false,
    /** The folder of the running or last session. */
    val path: String? = null,
    /** The zip of the last finished session, ready to share. */
    val zip: File? = null,
)

/** Everything the live screen shows; plain data, so the presentational composables stay stateless. */
data class LiveUiState(
    val settings: AppSettings = AppSettings(),
    val model: ModelStatus = ModelStatus.Loading,
    /** Result of the single-crop ROI modes (null in DETECTOR mode). */
    val result: FrameResult? = null,
    /** Result of the DETECTOR mode (null in the other modes). */
    val twoStage: TwoStageResult? = null,
    val liveFps: Double = 0.0,
    val recording: Boolean = false,
    val recordedRows: Int = 0,
    val recordPath: String? = null,
    val session: SessionUi = SessionUi(),
) {
    /** DETECTOR mode is selected but no usable stage-1 model is loaded. */
    val detectorMissing: Boolean
        get() = settings.roiMode == RoiMode.DETECTOR && (model as? ModelStatus.Ready)?.detectorName == null
}

/** Owns the model sessions and runs the per-frame pipeline on the camera analysis thread. */
class LiveViewModel(application: Application) : AndroidViewModel(application) {
    private val graph = (application as EdgeMangoApplication).graph
    private val converter = FrameConverter()
    private val lock = Any()

    @Volatile private var pipeline: FramePipeline? = null
    @Volatile private var twoStage: TwoStagePipeline? = null
    @Volatile private var session: LiteRtSession? = null
    @Volatile private var detSession: LiteRtSession? = null
    @Volatile private var modelName: String = ""
    @Volatile private var damageRef: ModelRef? = null
    @Volatile private var detectorRef: ModelRef? = null
    @Volatile private var recorder: LiveLog? = null
    @Volatile private var sessionRec: SessionRecorder? = null
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
            combine(
                graph.settings.state.map { Triple(it.modelId, it.accelerator, it.detectorModelId) }.distinctUntilChanged(),
                active,
            ) { key, on -> key to on }
                .collect { (key, on) -> if (on) load(key.first, key.second, key.third) else unload() }
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
    fun setNmsIou(value: Float) = graph.settings.update { copy(nmsIou = value) }
    fun setAgnosticNms(on: Boolean) = graph.settings.update { copy(agnosticNms = on) }

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

    fun toggleSession() {
        if (sessionRec == null) startSession() else stopSession()
    }

    private fun startSession() {
        if (_state.value.model !is ModelStatus.Ready) return
        twoStage?.resetTracking()
        val rec = graph.results.startSession()
        sessionRec = rec
        _state.update { it.copy(session = SessionUi(active = true, path = rec.dir.absolutePath)) }
    }

    /** Ends the session: the writer drains, `session.json` is written and the folder is zipped off the main thread. */
    private fun stopSession() {
        val rec = sessionRec ?: return
        sessionRec = null
        val settings = _state.value.settings
        val damage = damageRef
        val detector = detectorRef
        val damageAccel = session?.label
        val detAccel = detSession?.label
        _state.update { it.copy(session = it.session.copy(active = false, finishing = true)) }
        viewModelScope.launch(Dispatchers.IO) {
            val zip = runCatching {
                rec.finish { frames, saved, dropped, failed, endedAt ->
                    SessionInfo(
                        session = rec.dir.name,
                        appVersion = appVersion(),
                        device = DeviceInfo.device,
                        soc = DeviceInfo.soc,
                        android = DeviceInfo.android,
                        startedAtMs = rec.startedAtMs,
                        endedAtMs = endedAt,
                        frameCount = frames,
                        framesSaved = saved,
                        droppedWrites = dropped,
                        failedWrites = failed,
                        settings = SessionSettings(
                            settings.roiMode.name, settings.displayMode.name, settings.confidence.toDouble(),
                            settings.mangoThreshold.toDouble(), settings.maxMangos, settings.nmsIou.toDouble(),
                            settings.agnosticNms, settings.analysisWidth, settings.accelerator.name,
                        ),
                        damageModel = modelInfo(damage, damageAccel) ?: ModelInfo(modelName),
                        detectorModel = modelInfo(detector, detAccel),
                    )
                }
            }.getOrNull()
            _state.update {
                it.copy(session = it.session.copy(finishing = false, frames = rec.frames, saved = rec.framesSaved, dropped = rec.droppedWrites, zip = zip))
            }
        }
    }

    /** Share sheet for the last finished session zip, or null when there is none. */
    fun shareIntent(): Intent? {
        val zip = _state.value.session.zip?.takeIf { it.exists() } ?: return null
        return graph.results.share(listOf(zip), "Share field-test session", "application/zip")
    }

    private fun modelInfo(ref: ModelRef?, accelerator: String?): ModelInfo? = ref?.let {
        ModelInfo(it.name, accelerator, it.sidecarText?.let { text -> runCatching { Json.parseToJsonElement(text) }.getOrNull() })
    }

    private fun appVersion(): String = runCatching {
        getApplication<Application>().packageManager.getPackageInfo(getApplication<Application>().packageName, 0).versionName
    }.getOrNull() ?: "?"

    // ---- model lifecycle --------------------------------------------------------------------------------------

    private fun releaseSessions() {
        val (old, oldDet) = synchronized(lock) {
            pipeline = null
            twoStage = null
            Pair(session.also { session = null }, detSession.also { detSession = null })
        }
        old?.close()
        oldDet?.close()
    }

    private suspend fun unload() {
        stopRecording()
        stopSession() // before the sessions are released: it records which accelerators were running
        withContext(Dispatchers.Default) { releaseSessions() }
        _state.update { it.copy(model = ModelStatus.Loading, result = null, twoStage = null) }
    }

    private suspend fun load(modelId: String, accelerator: AcceleratorChoice, detectorId: String) {
        _state.update { it.copy(model = ModelStatus.Loading, result = null, twoStage = null) }
        withContext(Dispatchers.Default) {
            releaseSessions()
            val ref = graph.catalog.find(modelId) ?: graph.catalog.list().firstOrNull()
            if (ref == null) {
                _state.update { it.copy(model = ModelStatus.Failed("no model found")) }
                return@withContext
            }
            try {
                val s = LiteRtSession.open(getApplication(), ref.file.absolutePath, ref.sidecar, accelerator)
                var detNote = ""
                var det: LiteRtSession? = null
                var detRef: ModelRef? = null
                if (detectorId.isNotEmpty()) {
                    val dr = graph.catalog.find(detectorId)
                    if (dr == null) {
                        detNote = "detector model not found"
                    } else {
                        try {
                            det = LiteRtSession.open(getApplication(), dr.file.absolutePath, dr.sidecar, accelerator)
                            detRef = dr
                        } catch (t: Throwable) {
                            detNote = "detector failed: ${t.message ?: t.toString()}"
                        }
                    }
                }
                val damagePipeline = FramePipeline(ref.sidecar, s)
                synchronized(lock) {
                    session = s
                    detSession = det
                    modelName = ref.name
                    damageRef = ref
                    detectorRef = detRef
                    pipeline = damagePipeline
                    twoStage = if (det != null && detRef != null) TwoStagePipeline(FramePipeline(detRef.sidecar, det), damagePipeline) else null
                }
                _state.update {
                    it.copy(
                        model = ModelStatus.Ready(
                            ref.name, s.label, s.note, ref.sidecar.task == ModelTask.SEGMENT, s.loadMs,
                            detectorName = detRef?.name, detectorNote = detNote,
                        ),
                    )
                }
            } catch (t: Throwable) {
                _state.update { it.copy(model = ModelStatus.Failed(t.message ?: t.toString())) }
            }
        }
    }

    // ---- per-frame work (camera analysis thread) -----------------------------------------------------------------

    private fun analyze(image: ImageProxy) {
        val cameraNs = image.imageInfo.timestamp
        val frame = try {
            converter.toUpright(image)
        } finally {
            image.close()
        }
        val s = _state.value.settings
        if (s.roiMode == RoiMode.DETECTOR) analyzeTwoStage(frame, s, cameraNs) else analyzeSingle(frame, s, cameraNs)
    }

    private fun context(s: AppSettings): FrameContext = FrameContext(
        roiMode = s.roiMode.name,
        confThreshold = s.confidence,
        mangoThreshold = s.mangoThreshold,
        nmsIou = s.nmsIou,
        agnosticNms = s.agnosticNms,
        maxMangos = s.maxMangos,
        accelerator = session?.label ?: "?",
        detectorModel = if (s.roiMode == RoiMode.DETECTOR) detectorRef?.name else null,
        damageModel = modelName,
    )

    private fun analyzeSingle(frame: RgbImage, s: AppSettings, cameraNs: Long) {
        val config = PipelineConfig(
            s.roiMode, s.confidence, computeMasks = s.displayMode.showMask,
            computePolygons = sessionRec != null, nmsIou = s.nmsIou, agnosticNms = s.agnosticNms,
        )
        val result = synchronized(lock) {
            val p = pipeline ?: return
            try {
                p.process(frame, config)
            } catch (t: Throwable) {
                _state.update { it.copy(model = ModelStatus.Failed("inference failed: ${t.message}")) }
                pipeline = null
                return
            }
        }
        tick(frame)
        logCsv(s, result.window.x1, result.window.y1, result.window.x2, result.window.y2, result.detections.size,
            result.detections.count { it.cls == FrameResult.MANGO_CLASS }, result.detections.count { it.cls == FrameResult.DAMAGE_CLASS },
            result.maxDamageScore, result.timings)
        recordSession(frame) { index, file ->
            SessionJson.fromSingleCrop(index, System.currentTimeMillis(), cameraNs, file, context(s), result)
        }
        _state.update { it.copy(result = result, twoStage = null, liveFps = fpsEma) }
    }

    private fun analyzeTwoStage(frame: RgbImage, s: AppSettings, cameraNs: Long) {
        val config = TwoStageConfig(
            mangoThreshold = s.mangoThreshold,
            maxMangos = s.maxMangos,
            damageThreshold = s.confidence,
            computeMasks = s.displayMode.showMask,
            computePolygons = sessionRec != null,
            nmsIou = s.nmsIou,
            agnosticNms = s.agnosticNms,
        )
        val result = synchronized(lock) {
            val p = twoStage
            if (p == null) {
                _state.update { it.copy(result = null, twoStage = null) }
                return
            }
            try {
                p.process(frame, config)
            } catch (t: Throwable) {
                _state.update { it.copy(model = ModelStatus.Failed("inference failed: ${t.message}")) }
                twoStage = null
                pipeline = null
                return
            }
        }
        tick(frame)
        val first = result.mangos.firstOrNull()?.window
        logCsv(s, first?.x1 ?: 0, first?.y1 ?: 0, first?.x2 ?: frame.width, first?.y2 ?: frame.height,
            result.mangos.sumOf { it.damages.size } + result.mangos.size, result.mangos.size, result.mangos.sumOf { it.damages.size },
            result.maxDamageScore, result.timings)
        recordSession(frame) { index, file ->
            SessionJson.fromTwoStage(index, System.currentTimeMillis(), cameraNs, file, context(s), result)
        }
        _state.update { it.copy(result = null, twoStage = result, liveFps = fpsEma) }
    }

    private fun tick(frame: RgbImage) {
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
    }

    private fun recordSession(frame: RgbImage, record: (index: Int, frameFile: String?) -> com.edgemango.app.core.FrameRecord) {
        val rec = sessionRec ?: return
        try {
            rec.record(frame, record)
        } catch (_: IllegalStateException) {
            return // the session was stopped while this frame was in flight
        }
        _state.update { it.copy(session = it.session.copy(frames = rec.frames, saved = rec.framesSaved, dropped = rec.droppedWrites)) }
    }

    private fun logCsv(
        s: AppSettings, x1: Int, y1: Int, x2: Int, y2: Int, nDetections: Int, nMango: Int, nDamage: Int,
        maxDamage: Float, t: com.edgemango.app.core.FrameTimings,
    ) {
        val log = recorder ?: return
        log.append(
            listOf(
                System.currentTimeMillis(), modelName, session?.label, s.roiMode.name, x1, y1, x2, y2,
                nDetections, nMango, nDamage,
                "%.4f".format(Locale.US, maxDamage), s.confidence,
                "%.2f".format(Locale.US, t.roiMs), "%.2f".format(Locale.US, t.preMs),
                "%.2f".format(Locale.US, t.inferMs), "%.2f".format(Locale.US, t.postMs),
                "%.2f".format(Locale.US, t.totalMs), "%.1f".format(Locale.US, fpsEma),
            ),
        )
        _state.update { it.copy(recordedRows = log.rows) }
    }

    override fun onCleared() {
        stopRecording()
        sessionRec?.let { runCatching { it.close() } }
        sessionRec = null
        synchronized(lock) { pipeline = null; twoStage = null }
        session?.close()
        session = null
        detSession?.close()
        detSession = null
    }
}
