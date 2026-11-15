package com.edgemango.app.ui.benchmark

import android.app.Application
import android.content.Intent
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgemango.app.EdgeMangoApplication
import com.edgemango.app.core.BenchRecord
import com.edgemango.app.data.ModelRef
import com.edgemango.app.device.DeviceInfo
import com.edgemango.app.inference.AcceleratorChoice
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class BenchUiState(
    val models: List<ModelRef> = emptyList(),
    val selectedModelId: String? = null,
    val accelerator: AcceleratorChoice = AcceleratorChoice.CPU_4,
    val warmUp: String = "10",
    val runs: String = "100",
    val useLastFrame: Boolean = false,
    val lastFrameAvailable: Boolean = false,
    val running: Boolean = false,
    val phase: String = "",
    val progress: Float = 0f,
    val result: BenchRecord? = null,
    val savedFile: String? = null,
    val message: String? = null,
    val device: String = "",
    val externalModelsDir: String = "",
) {
    val selected: ModelRef? get() = models.firstOrNull { it.id == selectedModelId }
    val warmUpValue: Int? get() = warmUp.toIntOrNull()?.takeIf { it >= 0 }
    val runsValue: Int? get() = runs.toIntOrNull()?.takeIf { it > 0 }
    val canRun: Boolean get() = !running && selected != null && warmUpValue != null && runsValue != null && (!useLastFrame || lastFrameAvailable)
}

class BenchmarkViewModel(application: Application) : AndroidViewModel(application) {
    private val graph = (application as EdgeMangoApplication).graph
    private val engine = BenchmarkEngine(application, graph)
    private var job: Job? = null

    private val _state = MutableStateFlow(
        BenchUiState(
            device = "${DeviceInfo.device} | ${DeviceInfo.soc} | Android ${DeviceInfo.android}",
            externalModelsDir = graph.catalog.externalDirPath,
        ),
    )
    val state: StateFlow<BenchUiState> = _state.asStateFlow()

    init {
        refresh(initial = true)
    }

    /** Called whenever the screen is shown, so models added with adb or the last camera frame show up. */
    fun refresh(initial: Boolean = false) {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) { graph.catalog.list() }
            val settings = graph.settings.state.value
            _state.update { s ->
                s.copy(
                    models = models,
                    selectedModelId = s.selectedModelId?.takeIf { id -> models.any { it.id == id } }
                        ?: models.firstOrNull { it.id == settings.modelId }?.id ?: models.firstOrNull()?.id,
                    accelerator = if (initial) settings.accelerator else s.accelerator,
                    lastFrameAvailable = graph.lastFrame.get() != null,
                )
            }
        }
    }

    fun selectModel(id: String) = _state.update { it.copy(selectedModelId = id) }
    fun selectAccelerator(a: AcceleratorChoice) = _state.update { it.copy(accelerator = a) }
    fun setWarmUp(text: String) = _state.update { it.copy(warmUp = text.filter(Char::isDigit)) }
    fun setRuns(text: String) = _state.update { it.copy(runs = text.filter(Char::isDigit)) }
    fun setUseLastFrame(v: Boolean) = _state.update { it.copy(useLastFrame = v) }

    fun importModels(uris: List<Uri>) {
        viewModelScope.launch {
            val names = withContext(Dispatchers.IO) { graph.catalog.import(uris) }
            _state.update { it.copy(message = if (names.isEmpty()) "Nothing imported (pick a .tflite and its .json)" else "Imported: ${names.joinToString()}") }
            refresh()
        }
    }

    fun run() {
        val s = _state.value
        val model = s.selected ?: return
        val w = s.warmUpValue ?: return
        val n = s.runsValue ?: return
        if (!s.canRun) return
        _state.update { it.copy(running = true, result = null, savedFile = null, message = null, progress = 0f, phase = "Starting") }
        job = viewModelScope.launch {
            try {
                val config = BenchConfig(model, s.accelerator, w, n, s.useLastFrame, graph.settings.state.value.roiMode)
                val record = engine.run(config) { phase, done, total ->
                    _state.update { it.copy(phase = phase, progress = if (total > 0) done.toFloat() / total else 0f) }
                }
                val file = withContext(Dispatchers.IO) { graph.results.saveBenchmark(record) }
                _state.update { it.copy(running = false, result = record, savedFile = file.absolutePath, phase = "Done", progress = 1f) }
            } catch (c: CancellationException) {
                _state.update { it.copy(running = false, phase = "Cancelled", message = "Benchmark cancelled") }
                throw c
            } catch (t: Throwable) {
                _state.update { it.copy(running = false, phase = "Failed", message = "Benchmark failed: ${t.message ?: t}") }
            }
        }
    }

    fun cancel() {
        job?.cancel()
    }

    /** Intent that shares the last run's JSON and the accumulated CSV. */
    fun shareIntent(): Intent? {
        val json = _state.value.savedFile?.let { java.io.File(it) }
        val files = listOfNotNull(json, graph.results.benchmarksCsv.takeIf { it.exists() })
        return if (files.isEmpty()) null else graph.results.share(files, "Share benchmark results")
    }
}
