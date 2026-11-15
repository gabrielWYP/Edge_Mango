package com.edgemango.app.ui.settings

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.edgemango.app.core.RoiMode
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.inference.AcceleratorChoice
import com.edgemango.app.ui.common.ChoiceChips
import com.edgemango.app.ui.common.DropdownField

data class SettingsActions(
    val onModel: (String) -> Unit,
    val onAccelerator: (AcceleratorChoice) -> Unit,
    val onRoiMode: (RoiMode) -> Unit,
    val onDisplayMode: (DisplayMode) -> Unit,
    val onAnalysisWidth: (Int) -> Unit,
    val onPickModel: () -> Unit,
    val onQueryAccelerators: () -> Unit,
)

val ANALYSIS_WIDTHS = listOf(640, 1280, 1920, 2560)

/** Presentational settings screen for the live view. (The benchmark screen has its own model/accelerator pickers.) */
@Composable
fun SettingsContent(state: SettingsUiState, actions: SettingsActions, modifier: Modifier = Modifier) {
    val s = state.settings
    Column(
        modifier.verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Live settings", style = MaterialTheme.typography.headlineSmall)

        DropdownField(
            label = "Model",
            selected = state.models.firstOrNull { it.id == s.modelId },
            options = state.models,
            optionLabel = { "${it.name}  (${"%.1f".format(it.sizeMb)} MB, ${it.origin})" },
            onSelect = { actions.onModel(it.id) },
        )
        state.models.firstOrNull { it.id == s.modelId }?.warning?.takeIf { it.isNotEmpty() }?.let {
            Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        }
        OutlinedButton(onClick = actions.onPickModel) { Text("Pick model file(s)... (.tflite + .json)") }
        Text("Or: adb push the files to ${state.externalModelsDir}", style = MaterialTheme.typography.bodySmall)
        state.message?.let { Text(it, style = MaterialTheme.typography.bodySmall) }

        DropdownField(
            label = "Accelerator",
            selected = s.accelerator,
            options = AcceleratorChoice.entries,
            optionLabel = { it.label },
            onSelect = actions.onAccelerator,
        )
        Text(
            "GPU and NPU fall back to CPU x4 if the model cannot run on them; the HUD shows what is really running.",
            style = MaterialTheme.typography.bodySmall,
        )
        OutlinedButton(onClick = actions.onQueryAccelerators) { Text("Query available accelerators") }
        if (state.availableAccelerators.isNotEmpty()) Text("Runtime reports: ${state.availableAccelerators}", style = MaterialTheme.typography.bodySmall)

        Text("ROI mode", style = MaterialTheme.typography.labelMedium)
        ChoiceChips(RoiMode.entries, s.roiMode, { it.label }, actions.onRoiMode)

        Text("Overlay", style = MaterialTheme.typography.labelMedium)
        ChoiceChips(DisplayMode.entries, s.displayMode, { it.label }, actions.onDisplayMode)

        DropdownField(
            label = "Camera analysis width (4:3)",
            selected = s.analysisWidth,
            options = ANALYSIS_WIDTHS,
            optionLabel = { "$it x ${it * 3 / 4}" },
            onSelect = actions.onAnalysisWidth,
        )
        Text(
            "The model was trained on crops of 3264x2448 frames: the more pixels the mango has, the closer to training. " +
                "Higher widths cost camera and ROI time.",
            style = MaterialTheme.typography.bodySmall,
        )

        Text("Results are stored in ${state.resultsDir}", style = MaterialTheme.typography.bodySmall)
    }
}
