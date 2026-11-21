package com.edgemango.app.ui.live

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
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
    val onToggleSession: () -> Unit,
    val onShareSession: () -> Unit,
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
            twoStage = state.twoStage,
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
        if (state.settings.roiMode == RoiMode.DETECTOR) {
            val detector = (state.model as? ModelStatus.Ready)?.let { it.detectorName ?: "none (set one in Settings)" }
            line("detector: ${detector ?: "-"}", if (state.detectorMissing) Color(0xFFFF5252) else Color.Cyan)
            (state.model as? ModelStatus.Ready)?.detectorNote?.takeIf { it.isNotEmpty() }?.let { line(it, Color.Yellow) }
            val ts = state.twoStage
            if (ts != null) {
                line("stage 1: %.1f ms".format(ts.detectorMs), Color.White)
                line("total: %.1f ms".format(ts.timings.totalMs), Color.White)
                if (ts.noMango) {
                    line("mangos: 0 (no mango, damage model skipped)", Color.Yellow)
                } else {
                    line("mangos: ${ts.mangos.size}", Color.White)
                    for (m in ts.mangos.take(8)) {
                        line("  #${m.trackId} damage max %.2f".format(m.maxDamageScore), if (m.maxDamageScore > 0f) Color(0xFFFF5252) else Color.White)
                    }
                    if (ts.mangos.size > 8) line("  ... +${ts.mangos.size - 8}", Color.LightGray)
                }
            }
        }
        if (state.recording) line("REC ${state.recordedRows} rows", Color(0xFFFF5252))
        val ses = state.session
        if (ses.active) line("SESSION ${ses.frames} frames, ${ses.saved} saved, ${ses.dropped} dropped", Color(0xFFFF5252))
        if (ses.finishing) line("session: writing and zipping...", Color.Yellow)
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
            ChoiceChips(RoiMode.entries, s.roiMode, { it.label }, actions.onRoiMode, Modifier.horizontalScroll(rememberScrollState()))
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
        val ses = state.session
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("session", color = Color.White, modifier = Modifier.padding(end = 8.dp))
            Switch(checked = ses.active, onCheckedChange = { actions.onToggleSession() }, enabled = !ses.finishing)
            if (!ses.active && !ses.finishing && ses.zip != null) {
                Button(onClick = actions.onShareSession, modifier = Modifier.padding(start = 12.dp)) { Text("Share session") }
            }
            if (ses.active) Text("  ${ses.frames} frames, ${ses.saved} saved", color = Color.White, fontSize = 12.sp)
        }
        if (!ses.active && ses.path != null) {
            val tail = if (ses.zip != null) " (${ses.frames} frames, ${ses.saved} saved, ${ses.dropped} dropped) + .zip" else ""
            Text("session: ${ses.path}$tail", color = Color.LightGray, fontSize = 10.sp)
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
