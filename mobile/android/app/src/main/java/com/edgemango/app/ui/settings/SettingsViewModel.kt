package com.edgemango.app.ui.settings

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgemango.app.EdgeMangoApplication
import com.edgemango.app.core.RoiMode
import com.edgemango.app.data.AppSettings
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.data.ModelRef
import com.edgemango.app.inference.AcceleratorChoice
import com.edgemango.app.inference.LiteRtSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class SettingsUiState(
    val settings: AppSettings = AppSettings(),
    val models: List<ModelRef> = emptyList(),
    val externalModelsDir: String = "",
    val resultsDir: String = "",
    val availableAccelerators: String = "",
    val message: String? = null,
)

class SettingsViewModel(application: Application) : AndroidViewModel(application) {
    private val graph = (application as EdgeMangoApplication).graph
    private val _state = MutableStateFlow(
        SettingsUiState(
            settings = graph.settings.state.value,
            externalModelsDir = graph.catalog.externalDirPath,
            resultsDir = graph.results.dir.absolutePath,
        ),
    )
    val state: StateFlow<SettingsUiState> = _state.asStateFlow()

    init {
        viewModelScope.launch { graph.settings.state.collect { s -> _state.update { it.copy(settings = s) } } }
        refresh()
    }

    fun refresh() {
        viewModelScope.launch {
            val models = withContext(Dispatchers.IO) { graph.catalog.list() }
            _state.update { it.copy(models = models) }
        }
    }

    fun queryAccelerators() {
        viewModelScope.launch {
            val text = withContext(Dispatchers.Default) { LiteRtSession.available(getApplication()) }
            _state.update { it.copy(availableAccelerators = text) }
        }
    }

    fun importModels(uris: List<Uri>) {
        viewModelScope.launch {
            val names = withContext(Dispatchers.IO) { graph.catalog.import(uris) }
            _state.update { it.copy(message = if (names.isEmpty()) "Nothing imported (pick a .tflite and its .json)" else "Imported: ${names.joinToString()}") }
            refresh()
        }
    }

    fun setDetectorModel(id: String) = graph.settings.update { copy(detectorModelId = id) }
    fun setMangoThreshold(v: Float) = graph.settings.update { copy(mangoThreshold = v) }
    fun setKeepThreshold(v: Float) = graph.settings.update { copy(keepThreshold = v) }
    fun setHoldFrames(n: Int) = graph.settings.update { copy(holdFrames = n) }
    fun setMaxMangos(n: Int) = graph.settings.update { copy(maxMangos = n) }
    fun setModel(id: String) = graph.settings.update { copy(modelId = id) }
    fun setAccelerator(a: AcceleratorChoice) = graph.settings.update { copy(accelerator = a) }
    fun setRoiMode(m: RoiMode) = graph.settings.update { copy(roiMode = m) }
    fun setNmsIou(v: Float) = graph.settings.update { copy(nmsIou = v) }
    fun setAgnosticNms(on: Boolean) = graph.settings.update { copy(agnosticNms = on) }
    fun setDisplayMode(m: DisplayMode) = graph.settings.update { copy(displayMode = m) }
    fun setAnalysisWidth(w: Int) = graph.settings.update { copy(analysisWidth = w) }
}
