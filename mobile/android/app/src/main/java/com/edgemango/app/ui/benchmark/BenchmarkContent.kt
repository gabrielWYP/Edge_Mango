package com.edgemango.app.ui.benchmark

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import com.edgemango.app.core.BenchRecord
import com.edgemango.app.core.LatencyStats
import com.edgemango.app.inference.AcceleratorChoice
import com.edgemango.app.ui.common.DropdownField
import com.edgemango.app.ui.common.LabeledValue

data class BenchActions(
    val onSelectModel: (String) -> Unit,
    val onSelectAccelerator: (AcceleratorChoice) -> Unit,
    val onWarmUp: (String) -> Unit,
    val onRuns: (String) -> Unit,
    val onUseLastFrame: (Boolean) -> Unit,
    val onPickModel: () -> Unit,
    val onRun: () -> Unit,
    val onCancel: () -> Unit,
    val onShare: () -> Unit,
)

/** Presentational benchmark screen: renders [BenchUiState] and reports intents through [BenchActions]. */
@Composable
fun BenchmarkContent(state: BenchUiState, actions: BenchActions, modifier: Modifier = Modifier) {
    Column(
        modifier.verticalScroll(rememberScrollState()).padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("Benchmark", style = MaterialTheme.typography.headlineSmall)
        Text(state.device, style = MaterialTheme.typography.bodySmall)

        DropdownField(
            label = "Model",
            selected = state.selected,
            options = state.models,
            optionLabel = { "${it.name}  (${"%.1f".format(it.sizeMb)} MB, ${it.origin})" },
            onSelect = { actions.onSelectModel(it.id) },
        )
        state.selected?.warning?.takeIf { it.isNotEmpty() }?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        OutlinedButton(onClick = actions.onPickModel, enabled = !state.running) { Text("Pick model file(s)... (.tflite + .json)") }
        Text(
            "Or copy files with adb to ${state.externalModelsDir} and reopen this screen.",
            style = MaterialTheme.typography.bodySmall,
        )

        DropdownField(
            label = "Accelerator",
            selected = state.accelerator,
            options = AcceleratorChoice.entries,
            optionLabel = { it.label },
            onSelect = actions.onSelectAccelerator,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedTextField(
                value = state.warmUp, onValueChange = actions.onWarmUp, label = { Text("Warm-up runs W") },
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true, modifier = Modifier.weight(1f), enabled = !state.running,
            )
            OutlinedTextField(
                value = state.runs, onValueChange = actions.onRuns, label = { Text("Measured runs N") },
                keyboardOptions = androidx.compose.foundation.text.KeyboardOptions(keyboardType = KeyboardType.Number),
                singleLine = true, modifier = Modifier.weight(1f), enabled = !state.running,
            )
        }

        Text("Input", style = MaterialTheme.typography.labelMedium)
        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(selected = !state.useLastFrame, onClick = { actions.onUseLastFrame(false) }, enabled = !state.running)
            Text("Bundled sample crop (fixed, reproducible)")
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(
                selected = state.useLastFrame, onClick = { actions.onUseLastFrame(true) },
                enabled = !state.running && state.lastFrameAvailable,
            )
            Text(if (state.lastFrameAvailable) "Last camera frame (current ROI mode)" else "Last camera frame (open the Live tab first)")
        }

        if (state.running) {
            Button(onClick = actions.onCancel, modifier = Modifier.fillMaxWidth()) { Text("Cancel") }
            LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
            Text(state.phase, style = MaterialTheme.typography.bodySmall)
        } else {
            Button(onClick = actions.onRun, enabled = state.canRun, modifier = Modifier.fillMaxWidth()) { Text("Run benchmark") }
        }
        state.message?.let { Text(it, color = MaterialTheme.colorScheme.error) }

        state.result?.let { ResultCard(it, state.savedFile, actions.onShare) }

        MetricsHelp()
    }
}

@Composable
private fun ResultCard(r: BenchRecord, savedFile: String?, onShare: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("Result", style = MaterialTheme.typography.titleMedium)
            LabeledValue("Model", r.modelName)
            LabeledValue("Accelerator", r.accelerator)
            if (r.accelerationNote.isNotEmpty()) Text("note: ${r.accelerationNote}", style = MaterialTheme.typography.bodySmall)
            LabeledValue("Input", r.input)
            LabeledValue("Runs", "${r.runs} after ${r.warmUp} warm-up")
            LabeledValue("Model load", "%.0f ms".format(r.loadMs))

            Text("Inference (model only)", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 8.dp))
            StatsRows(r.inference)
            Text("End to end (ROI + pre + infer + post)", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 8.dp))
            StatsRows(r.endToEnd)

            Text("Memory and model", style = MaterialTheme.typography.titleSmall, modifier = Modifier.padding(top = 8.dp))
            LabeledValue("Peak RAM (VmHWM)", "%.1f MB".format(r.peakRamMb))
            LabeledValue("RSS before model load", "%.1f MB".format(r.ramBeforeLoadMb))
            LabeledValue("Native heap after", "%.1f MB".format(r.nativeHeapMb))
            Text(r.peakRamSource, style = MaterialTheme.typography.bodySmall)
            LabeledValue("Model size on disk", "%.2f MB".format(r.modelSizeMb))
            LabeledValue("Parameters", r.numParameters?.let { "%.3f M".format(it / 1e6) } ?: "not in sidecar")
            LabeledValue("FLOPs", r.flops?.let { "%.2f G".format(it / 1e9) } ?: "not in sidecar")
            LabeledValue("Precision / task", "${r.precision} / ${r.task}")
            LabeledValue("Device", r.device)
            LabeledValue("SoC", r.soc)
            LabeledValue("Android", r.androidVersion)
            savedFile?.let { Text("saved: $it", style = MaterialTheme.typography.bodySmall) }
            Button(onClick = onShare, modifier = Modifier.padding(top = 8.dp)) { Text("Share JSON + CSV") }
        }
    }
}

@Composable
private fun StatsRows(s: LatencyStats) {
    LabeledValue("mean", "%.2f ms".format(s.meanMs))
    LabeledValue("std", "%.2f ms".format(s.stdMs))
    LabeledValue("p50 / p95", "%.2f / %.2f ms".format(s.p50Ms, s.p95Ms))
    LabeledValue("min / max", "%.2f / %.2f ms".format(s.minMs, s.maxMs))
    LabeledValue("FPS (1 / mean)", "%.1f".format(s.fps))
}

@Composable
private fun MetricsHelp() {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("How each number is measured", style = MaterialTheme.typography.titleSmall)
            Text(
                "Latency: W warm-up runs are discarded, then N runs are timed one by one with System.nanoTime on the same preprocessed input. " +
                    "'Inference' copies the input in, runs the model and copies every output out; 'end to end' adds the colour-mask ROI, " +
                    "crop/letterbox/normalize, decode, NMS and mask assembly.",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "FPS = 1000 / mean latency, i.e. sequential single-image throughput (not the live camera FPS).",
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                "Peak RAM: VmHWM from /proc/self/status, the high-water mark of the resident memory of the whole app process. " +
                    "It is reset to the current value just before the model loads (when the kernel allows it), so it covers model load + all runs. " +
                    "It includes the app's own baseline, shown as 'RSS before model load'. Native heap is Debug.getNativeHeapAllocatedSize().",
                style = MaterialTheme.typography.bodySmall,
            )
            Text("Model size is the .tflite file size on disk; parameters and FLOPs come from the sidecar JSON.", style = MaterialTheme.typography.bodySmall)
        }
    }
}
