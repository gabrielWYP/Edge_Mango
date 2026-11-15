package com.edgemango.app.ui.live

import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.foundation.layout.fillMaxSize
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgemango.app.camera.CameraPreview

/** Container: binds [LiveViewModel] to the presentational [LiveContent] and supplies the camera. */
@Composable
fun LiveScreen(hasCameraPermission: Boolean, onRequestPermission: () -> Unit, modifier: Modifier = Modifier) {
    if (!hasCameraPermission) {
        CameraPermissionPrompt(onRequestPermission, modifier)
        return
    }
    val vm: LiveViewModel = viewModel()
    val state by vm.state.collectAsState()

    DisposableEffect(vm) {
        vm.setActive(true)
        onDispose { vm.setActive(false) }
    }

    LiveContent(
        state = state,
        actions = LiveActions(
            onDisplayMode = vm::setDisplayMode,
            onRoiMode = vm::setRoiMode,
            onConfidence = vm::setConfidence,
            onToggleRecording = vm::toggleRecording,
        ),
        preview = {
            CameraPreview(
                analysisWidth = state.settings.analysisWidth,
                analyzer = vm.analyzer,
                modifier = Modifier.fillMaxSize(),
            )
        },
        modifier = modifier,
    )
}
