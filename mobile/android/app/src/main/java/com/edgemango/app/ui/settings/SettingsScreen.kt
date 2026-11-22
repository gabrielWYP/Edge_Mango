package com.edgemango.app.ui.settings

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel

/** Container for the settings screen. */
@Composable
fun SettingsScreen(modifier: Modifier = Modifier) {
    val vm: SettingsViewModel = viewModel()
    val state by vm.state.collectAsState()
    LaunchedEffect(vm) { vm.refresh() }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.importModels(uris)
    }

    SettingsContent(
        state = state,
        actions = SettingsActions(
            onModel = vm::setModel,
            onAccelerator = vm::setAccelerator,
            onRoiMode = vm::setRoiMode,
            onDisplayMode = vm::setDisplayMode,
            onAnalysisWidth = vm::setAnalysisWidth,
            onDetectorModel = vm::setDetectorModel,
            onMangoThreshold = vm::setMangoThreshold,
            onKeepThreshold = vm::setKeepThreshold,
            onHoldFrames = vm::setHoldFrames,
            onMaxMangos = vm::setMaxMangos,
            onNmsIou = vm::setNmsIou,
            onAgnosticNms = vm::setAgnosticNms,
            onPickModel = { picker.launch(arrayOf("*/*")) },
            onQueryAccelerators = vm::queryAccelerators,
        ),
        modifier = modifier,
    )
}
