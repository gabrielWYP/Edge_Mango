package com.edgemango.app.ui.live

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.edgemango.app.core.FrameResult
import com.edgemango.app.core.RoiMode
import com.edgemango.app.data.DisplayMode
import com.edgemango.app.ui.common.ChoiceChips

/** User intents of the live screen; the container maps them onto the view model. */
data class LiveActions(
    val onDisplayMode: (DisplayMode) -> Unit,
    val onRoiMode: (RoiMode) -> Unit,
    val onConfidence: (Float) -> Unit,
    val onToggleRecording: () -> Unit,
)

/**
 * Presentational live screen: HUD, overlay and controls drawn over a camera preview that the container supplies as
 * [preview]. It holds no state of its own.
 */
@Composable
fun LiveContent(
    state: LiveUiState,
    actions: LiveActions,
    preview: @Composable () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(modifier.fillMaxSize().background(Color.Black)) {
        preview()
        DetectionOverlay(
            result = state.result,
            displayMode = state.settings.displayMode,
            roiMode = state.settings.roiMode,
            modifier = Modifier.fillMaxSize(),
        )
        Hud(state, Modifier.align(Alignment.TopStart).padding(8.dp))
        Controls(state, actions, Modifier.align(Alignment.BottomCenter))
    }
}

@Composable
private fun Hud(state: LiveUiState, modifier: Modifier) {
    val r = state.result
    val t = r?.timings
    Column(
        modifier.background(Color(0x99000000)).padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(2.dp),
    ) {
        val line = @Composable { text: String, color: Color ->
            Text(text, color = color, fontSize = 12.sp, fontFamily = FontFamily.Monospace)
        }
        when (val m = state.model) {
            is ModelStatus.Loading -> line("loading model...", Color.Yellow)
            is ModelStatus.Failed -> line("model error: ${m.message}", Color.Red)
            is ModelStatus.Ready -> {
                line("${m.modelName}", Color.White)
                line("accel: ${m.accelerator}", Color.Cyan)
                if (m.note.isNotEmpty()) line("note: ${m.note}", Color.Yellow)
            }
        }
        line("live FPS: %.1f".format(state.liveFps), Color.White)
        if (t != null) {
            line("infer: %.1f ms (%.1f FPS)".format(t.inferMs, 1000.0 / t.inferMs), Color.White)
            line("roi %.1f | pre %.1f | post %.1f ms".format(t.roiMs, t.preMs, t.postMs), Color.White)
            line("total: %.1f ms".format(t.totalMs), Color.White)
            line("detections: ${r.detections.size}", Color.White)
            val damage = r.maxDamageScore
            line("max damage: %.2f".format(damage), if (damage > 0f) Color(0xFFFF5252) else Color.White)
        }
        if (state.recording) line("REC ${state.recordedRows} rows", Color(0xFFFF5252))
    }
}

@Composable
private fun Controls(state: LiveUiState, actions: LiveActions, modifier: Modifier) {
    val s = state.settings
    Column(
        modifier.fillMaxWidth().background(Color(0xAA000000)).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            ChoiceChips(DisplayMode.entries, s.displayMode, { it.label }, actions.onDisplayMode)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("ROI", color = Color.White, modifier = Modifier.padding(end = 8.dp))
            ChoiceChips(RoiMode.entries, s.roiMode, { it.label }, actions.onRoiMode)
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("conf %.2f".format(s.confidence), color = Color.White, modifier = Modifier.padding(end = 8.dp))
            Slider(
                value = s.confidence,
                onValueChange = actions.onConfidence,
                valueRange = 0.05f..0.95f,
                modifier = Modifier.weight(1f),
            )
            Text("record", color = Color.White, modifier = Modifier.padding(horizontal = 8.dp))
            Switch(checked = state.recording, onCheckedChange = { actions.onToggleRecording() })
        }
        if (!state.recording && state.recordPath != null) {
            Text("saved: ${state.recordPath} (${state.recordedRows} rows)", color = Color.LightGray, fontSize = 10.sp)
        }
    }
}

/** Shown instead of the live screen until the camera permission is granted. */
@Composable
fun CameraPermissionPrompt(onRequest: () -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("The live view needs the camera.", style = MaterialTheme.typography.titleMedium)
        Button(onClick = onRequest, modifier = Modifier.padding(top = 16.dp)) { Text("Grant camera permission") }
    }
}
