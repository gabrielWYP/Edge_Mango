package com.edgemango.app.data

import android.content.Context
import com.edgemango.app.core.RoiMode
import com.edgemango.app.inference.AcceleratorChoice
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

enum class DisplayMode(val label: String) {
    BOXES("Boxes"),
    MASK("Mask"),
    BOTH("Both");

    val showBoxes: Boolean get() = this != MASK
    val showMask: Boolean get() = this != BOXES
}

data class AppSettings(
    val modelId: String = ModelCatalog.DEFAULT_ID,
    val accelerator: AcceleratorChoice = AcceleratorChoice.CPU_4,
    val roiMode: RoiMode = RoiMode.COLOR_MASK,
    val displayMode: DisplayMode = DisplayMode.BOTH,
    val confidence: Float = 0.25f,
    /** NMS IoU threshold ("Jaccard") applied to every model instead of the sidecar's value. */
    val nmsIou: Float = 0.5f,
    /** Stage-2 NMS also suppresses overlapping boxes of different classes. */
    val agnosticNms: Boolean = false,
    /** Camera analysis width in pixels (4:3 frames). Training frames were 3264x2448 crops, so more is closer to training. */
    val analysisWidth: Int = 1280,
)

class SettingsRepository(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(load())
    val state: StateFlow<AppSettings> = _state.asStateFlow()

    fun update(change: AppSettings.() -> AppSettings) {
        val next = _state.value.change()
        _state.value = next
        save(next)
    }

    private fun load(): AppSettings {
        val d = AppSettings()
        return AppSettings(
            modelId = prefs.getString("modelId", d.modelId) ?: d.modelId,
            accelerator = enumOr(prefs.getString("accelerator", null), d.accelerator),
            roiMode = enumOr(prefs.getString("roiMode", null), d.roiMode),
            displayMode = enumOr(prefs.getString("displayMode", null), d.displayMode),
            confidence = prefs.getFloat("confidence", d.confidence),
            nmsIou = prefs.getFloat("nmsIou", d.nmsIou),
            agnosticNms = prefs.getBoolean("agnosticNms", d.agnosticNms),
            analysisWidth = prefs.getInt("analysisWidth", d.analysisWidth),
        )
    }

    private fun save(s: AppSettings) {
        prefs.edit()
            .putString("modelId", s.modelId)
            .putString("accelerator", s.accelerator.name)
            .putString("roiMode", s.roiMode.name)
            .putString("displayMode", s.displayMode.name)
            .putFloat("confidence", s.confidence)
            .putFloat("nmsIou", s.nmsIou)
            .putBoolean("agnosticNms", s.agnosticNms)
            .putInt("analysisWidth", s.analysisWidth)
            .apply()
    }

    private inline fun <reified E : Enum<E>> enumOr(name: String?, default: E): E =
        enumValues<E>().firstOrNull { it.name == name } ?: default
}
